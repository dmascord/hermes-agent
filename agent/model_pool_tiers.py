"""Automatic model pool tier system.

Sorts available models into capability tiers so the passthrough chain
picks the best model for each request type. This replaces the static
env-var-ordered fallback chain with a capability-aware selection.

Usage:
    from agent.model_pool_tiers import build_tiered_pool, ModelTier

    pool = build_tiered_pool(
        available_models=["minimax/MiniMax-M3", "zai/glm-4.7", ...],
        needs_tools=True,
        needs_vision=False,
        estimated_tokens=150_000,
    )
    # pool is ordered: Tier 0 (best) -> Tier N (fallback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

logger = logging.getLogger(__name__)


class ModelTier(IntEnum):
    """Priority tiers for model selection. Lower = preferred."""
    TIER0_LARGE_RELIABLE = 0   # >=200K context, reliable tool calling
    TIER1_MEDIUM_RELIABLE = 1  # 128K-200K context, reliable tool calling
    TIER2_LARGE_UNRELIABLE = 2  # >=200K context, unreliable tool calling
    TIER3_SMALL_FAST = 3       # <128K context, reliable
    TIER4_LIMITED = 4          # Free tier, rate-limited, or fragile
    TIER5_SKIPPED = 5          # Skip entirely (context too small, etc)


# Static model classification — ground truth for pool tier assignment.
# Models not listed default to TIER4 (limited).
_MODEL_TIERS: Dict[str, ModelTier] = {
    # Tier 0: Large context + reliable tools
    "github-copilot-enterprise/gpt-5.4-mini": ModelTier.TIER0_LARGE_RELIABLE,
    "github-copilot-enterprise/claude-sonnet-4.6": ModelTier.TIER0_LARGE_RELIABLE,
    "github-copilot-enterprise/claude-opus-4.6": ModelTier.TIER0_LARGE_RELIABLE,
    "minimax/MiniMax-M3": ModelTier.TIER0_LARGE_RELIABLE,
    "opencode-go/mimo-v2.5": ModelTier.TIER0_LARGE_RELIABLE,
    "opencode-go/deepseek-v4-pro": ModelTier.TIER0_LARGE_RELIABLE,
    "opencode-go/qwen3.6-plus": ModelTier.TIER0_LARGE_RELIABLE,
    "google/gemini-2.5-flash": ModelTier.TIER0_LARGE_RELIABLE,
    "openai-codex/gpt-5.5": ModelTier.TIER0_LARGE_RELIABLE,
    "openai-codex/gpt-5.4": ModelTier.TIER0_LARGE_RELIABLE,
    "openai-codex/gpt-5.4-mini": ModelTier.TIER0_LARGE_RELIABLE,
    "openai-codex/gpt-5.3-codex": ModelTier.TIER0_LARGE_RELIABLE,
    "openai-codex/gpt-5.3-codex-spark": ModelTier.TIER0_LARGE_RELIABLE,

    # Tier 1: Medium context + reliable tools
    "zai/glm-4.7": ModelTier.TIER1_MEDIUM_RELIABLE,
    "minimax/MiniMax-M2.7": ModelTier.TIER1_MEDIUM_RELIABLE,
    "minimax/MiniMax-M2.5": ModelTier.TIER1_MEDIUM_RELIABLE,
    "opencode-go/glm-5": ModelTier.TIER1_MEDIUM_RELIABLE,
    "opencode-go/kimi-k2.6": ModelTier.TIER1_MEDIUM_RELIABLE,

    # Tier 2: Large context but unreliable tool calling
    "opencode-zen/mimo-v2.5-free": ModelTier.TIER2_LARGE_UNRELIABLE,
    "opencode-zen/deepseek-v4-flash-free": ModelTier.TIER2_LARGE_UNRELIABLE,
    "opencode-zen/big-pickle": ModelTier.TIER2_LARGE_UNRELIABLE,
    "opencode-go/deepseek-v4-flash": ModelTier.TIER2_LARGE_UNRELIABLE,
    "ollama/kimi-k2-thinking": ModelTier.TIER2_LARGE_UNRELIABLE,
    "ollama/qwen3-coder-next": ModelTier.TIER2_LARGE_UNRELIABLE,

    # Tier 3: Small context but reliable
    "ollama/glm-5.1": ModelTier.TIER3_SMALL_FAST,
    "ollama/deepseek-v4-flash": ModelTier.TIER3_SMALL_FAST,
    "groq/llama-3.3-70b-versatile": ModelTier.TIER3_SMALL_FAST,

    # Tier 4: Limited (free, rate-limited, fragile)
    "nous/stepfun/step-3.7-flash:free": ModelTier.TIER4_LIMITED,
    "nous/nvidia/nemotron-3-ultra:free": ModelTier.TIER4_LIMITED,
    "arliai/Mistral-Medium-3.5-128B": ModelTier.TIER4_LIMITED,
    "arliai/GLM-4.6-Derestricted-v5": ModelTier.TIER4_LIMITED,
    "arliai/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted": ModelTier.TIER4_LIMITED,
    "arliai/Qwen3.5-27B-BlueStar-v3-Derestricted-Lite": ModelTier.TIER4_LIMITED,
}

# Provider-level tier boost/penalty. Applied to all models from a provider.
_PROVIDER_TIER_ADJUST: Dict[str, int] = {
    "opencode-zen": +1,  # Free tier, bump down
    "arliai": +1,        # Fragile, bump down
    "nous": +1,          # Free tier, bump down
    "groq": +1,          # Rate-limited, bump down
}


@dataclass
class TieredModel:
    model: str
    tier: ModelTier
    context_length: int
    provider: str = ""


def _get_provider(model: str) -> str:
    if "/" in model:
        return model.split("/")[0]
    return ""


def _get_context_length(model: str) -> int:
    try:
        from agent.model_metadata import get_model_context_length
        return get_model_context_length(model) or 0
    except Exception:
        return 0


def _effective_tier(model: str) -> ModelTier:
    """Compute the effective tier for a model, applying provider adjustments."""
    base_tier = _MODEL_TIERS.get(model, ModelTier.TIER4_LIMITED)
    prov = _get_provider(model)
    adj = _PROVIDER_TIER_ADJUST.get(prov, 0)
    adjusted = int(base_tier) + adj
    adjusted = max(0, min(adjusted, int(ModelTier.TIER5_SKIPPED)))
    return ModelTier(adjusted)


def build_tiered_pool(
    available_models: List[str],
    *,
    needs_tools: bool = True,
    needs_vision: bool = False,
    estimated_tokens: int = 0,
    max_context: int = 0,
) -> List[str]:
    """Build an ordered model pool sorted by capability tier.

    Models are grouped by tier, then within each tier ordered by context
    length (largest first). Models that cannot handle the request are
    excluded.

    Args:
        available_models: All models available (from env vars, etc.)
        needs_tools: Whether the request requires tool calling
        needs_vision: Whether the request requires vision support
        estimated_tokens: Estimated token count for the request
        max_context: If set, skip models with context < max_context

    Returns:
        Ordered list of model strings, best first.
    """
    tiered: List[TieredModel] = []

    for model in available_models:
        if not model or "/" not in model:
            continue

        tier = _effective_tier(model)
        ctx = _get_context_length(model)

        # Skip models with insufficient context (85% margin like _model_can_handle_context)
        if estimated_tokens > 0 and ctx > 0 and ctx * 0.85 < estimated_tokens:
            tier = ModelTier.TIER5_SKIPPED

        if ctx > 0 and max_context > 0 and ctx < max_context:
            tier = ModelTier.TIER5_SKIPPED

        if tier == ModelTier.TIER5_SKIPPED:
            continue

        tiered.append(TieredModel(
            model=model,
            tier=tier,
            context_length=ctx,
            provider=_get_provider(model),
        ))

    # Sort: tier first (ascending), then context_length descending
    tiered.sort(key=lambda m: (m.tier, -m.context_length))

    result = [m.model for m in tiered]

    if result:
        tier_counts: Dict[int, int] = {}
        for m in tiered:
            tier_counts[int(m.tier)] = tier_counts.get(int(m.tier), 0) + 1
        logger.debug(
            "[model_pool_tiers] tiered pool: %d models, tiers=%s",
            len(result), tier_counts,
        )

    return result


def model_tier_summary() -> Dict[str, Dict[str, str]]:
    """Return a human-readable summary of all known model tiers."""
    result: Dict[str, Dict[str, str]] = {}
    for model, tier in sorted(_MODEL_TIERS.items(), key=lambda x: x[1]):
        ctx = _get_context_length(model)
        result[model] = {
            "tier": f"T{int(tier)}",
            "context": f"{ctx:,}" if ctx > 0 else "unknown",
            "provider": _get_provider(model),
        }
    return result