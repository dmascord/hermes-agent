"""Rolling monthly AIU spend tracker for GitHub Copilot Enterprise.

GHE returns a ``copilot_usage`` field on every chat completion response:

    {
      "copilot_usage": {
        "token_details": [...],
        "total_nano_aiu": 1518000
      }
    }

This module:
  1. Extracts ``total_nano_aiu`` from ``response_obj`` and records it.
  2. Maintains a rolling list of (timestamp, nano_aiu) events in a JSON file
     so the monthly window is always accurate regardless of restart.
  3. Compares the rolling 30-day total against a configurable AIU limit
     (env ``COPILOT_MONTHLY_AIU_LIMIT``, default 8000 AIU).
  4. When the limit is exceeded, calls ``mark_model_cooldown`` for every
     ``github-copilot-enterprise/*`` model with a cooldown that expires at
     the end of the current 30-day rolling window.

Unit: 1 AIU = 1_000_000_000 nano-AIU (confirmed from copilot billing docs).

Thread-safety: a module-level lock guards all reads and writes; the JSON
file is written atomically so concurrent processes also see consistent state.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_NANO_AIU_PER_AIU: int = 1_000_000_000
_ROLLING_WINDOW_SECONDS: float = 30 * 24 * 3600.0  # 30 days
_DEFAULT_MONTHLY_AIU_LIMIT: float = 8000.0

# All GHE provider prefixes that share the same spend pool.
_GHE_PROVIDER = "github-copilot-enterprise"


def _db_path() -> Path:
    return get_hermes_home() / "copilot_spend.json"


def _monthly_limit_aiu() -> float:
    raw = os.getenv("COPILOT_MONTHLY_AIU_LIMIT", "").strip()
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_MONTHLY_AIU_LIMIT
    except (ValueError, TypeError):
        return _DEFAULT_MONTHLY_AIU_LIMIT


def _empty_db() -> dict[str, Any]:
    return {"version": 1, "events": []}


def _load_unlocked() -> dict[str, Any]:
    path = _db_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_db()
    except Exception as exc:
        logger.warning("[copilot_spend] failed to read %s: %s", path, exc)
        return _empty_db()
    if not isinstance(data, dict) or not isinstance(data.get("events"), list):
        return _empty_db()
    return data


def _save_unlocked(data: dict[str, Any]) -> None:
    path = _db_path()
    atomic_json_write(path, data, indent=2)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _prune_old_events(events: list, now: float) -> list:
    """Drop events older than the rolling window."""
    cutoff = now - _ROLLING_WINDOW_SECONDS
    return [e for e in events if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] >= cutoff]


def _rolling_total_nano_aiu(events: list) -> int:
    return sum(int(e[1]) for e in events if isinstance(e, (list, tuple)) and len(e) >= 2)


def extract_nano_aiu(response_obj: Any) -> int:
    """Return ``total_nano_aiu`` from a GHE ChatCompletion response, or 0."""
    try:
        cu = getattr(response_obj, "copilot_usage", None)
        if cu is None:
            # Fall back to model_extra for forward-compat with SDK changes.
            me = getattr(response_obj, "model_extra", None) or {}
            cu = me.get("copilot_usage")
        if not isinstance(cu, dict):
            return 0
        v = cu.get("total_nano_aiu", 0)
        return int(v) if isinstance(v, (int, float)) and v > 0 else 0
    except Exception:
        return 0


def record_and_check(
    response_obj: Any,
    *,
    provider_model: str = "",
    base_url: str = "",
) -> None:
    """Record AIU spend from *response_obj* and enforce monthly limit.

    Call this immediately after every successful GHE response.  No-ops
    silently when the response has no ``copilot_usage`` field (non-GHE
    providers).

    Args:
        response_obj: The ChatCompletion object returned by the OpenAI SDK.
        provider_model: Full model string, e.g. ``"github-copilot-enterprise/gpt-5.4"``.
        base_url: The GHE base URL used for the request (for cooldown scoping).
    """
    nano_aiu = extract_nano_aiu(response_obj)
    if nano_aiu <= 0:
        return

    now = time.time()
    limit_nano = int(_monthly_limit_aiu() * _NANO_AIU_PER_AIU)

    with _LOCK:
        data = _load_unlocked()
        events: list = data.get("events", [])
        events = _prune_old_events(events, now)
        events.append([now, nano_aiu])
        data["events"] = events

        rolling_nano = _rolling_total_nano_aiu(events)
        rolling_aiu = rolling_nano / _NANO_AIU_PER_AIU

        logger.info(
            "[copilot_spend] recorded %.4f AIU (%.4f AIU rolling/30d, limit=%.0f AIU) model=%s",
            nano_aiu / _NANO_AIU_PER_AIU,
            rolling_aiu,
            _monthly_limit_aiu(),
            provider_model,
        )

        _save_unlocked(data)

    if rolling_nano > limit_nano:
        _enforce_cooldown(rolling_nano, limit_nano, now, base_url=base_url)


def _enforce_cooldown(
    rolling_nano: int,
    limit_nano: int,
    now: float,
    *,
    base_url: str = "",
) -> None:
    """Place all GHE models into cooldown until the rolling window clears.

    The cooldown duration is set to expire when the oldest event in the
    window would fall out of the 30-day look-back, making the monthly total
    drop below the limit again.  We conservatively add 60 s of padding.
    """
    try:
        from agent.model_cooldown_db import mark_model_cooldown
    except Exception as exc:
        logger.error("[copilot_spend] cannot import mark_model_cooldown: %s", exc)
        return

    # Determine when the window clears: oldest event timestamp + 30 days.
    with _LOCK:
        data = _load_unlocked()
        events = _prune_old_events(data.get("events", []), now)

    # Find the oldest event that is still pushing us over the limit.
    # Walk from oldest → newest, dropping events until we're under limit.
    rolling = _rolling_total_nano_aiu(events)
    window_clears_at = now + _ROLLING_WINDOW_SECONDS  # pessimistic default
    for e in sorted(events, key=lambda x: x[0]):
        rolling -= int(e[1])
        if rolling <= limit_nano:
            # Once this event drops out of the window we'd be under limit.
            window_clears_at = e[0] + _ROLLING_WINDOW_SECONDS
            break

    cooldown_seconds = max(60.0, window_clears_at - now + 60.0)
    rolling_aiu = rolling_nano / _NANO_AIU_PER_AIU
    limit_aiu = limit_nano / _NANO_AIU_PER_AIU

    logger.warning(
        "[copilot_spend] monthly AIU limit exceeded: %.2f / %.0f AIU — "
        "placing github-copilot-enterprise into cooldown for %.0fs (%.1f days)",
        rolling_aiu,
        limit_aiu,
        cooldown_seconds,
        cooldown_seconds / 3600 / 24,
    )

    # Cooldown every known GHE model so the router skips the entire pool.
    _GHE_MODELS = [
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-5-mini",
        "gpt-5.4",
        "gpt-5.3-codex",
        "claude-sonnet-4.6",
        "claude-opus-4.6",
        # Wildcard sentinel: the router checks cooldown_remaining for the
        # raw provider_model string too, so record a global provider-level
        # entry using an empty model to catch anything we don't list here.
        "",
    ]
    for model in _GHE_MODELS:
        try:
            mark_model_cooldown(
                _GHE_PROVIDER,
                model or "*",
                base_url=base_url,
                cooldown_seconds=cooldown_seconds,
                reason="monthly_aiu_limit_exceeded",
            )
        except Exception as exc:
            logger.error("[copilot_spend] mark_model_cooldown failed model=%s: %s", model, exc)


def rolling_spend_aiu() -> float:
    """Return the current rolling 30-day AIU spend (read-only, no locking needed for monitoring)."""
    now = time.time()
    with _LOCK:
        data = _load_unlocked()
        events = _prune_old_events(data.get("events", []), now)
    return _rolling_total_nano_aiu(events) / _NANO_AIU_PER_AIU
