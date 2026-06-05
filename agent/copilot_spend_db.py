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
# Separate AIU limits by endpoint:
# - Public: https://api.githubcopilot.com (dmascord's GitHub Copilot)
# - Enterprise: https://copilot-api.sita.ghe.com (SITA GHE, separate budget)
_GHE_PUBLIC_BASE = "https://api.githubcopilot.com"
_GHE_ENTERPRISE_BASE = "https://copilot-api.sita.ghe.com"


def _db_path() -> Path:
    return get_hermes_home() / "copilot_spend.json"


# Return limit for a given base_url.
# - Public (api.githubcopilot.com): 3000 AIU (dmascord's GitHub Copilot)
# - Enterprise (copilot-api.sita.ghe.com): 3900 AIU (SITA GHE)
def _monthly_limit_for(base_url: str) -> float:
    base = (base_url or "").rstrip("/")
    if base == _GHE_PUBLIC_BASE:
        raw = os.getenv("COPILOT_PUBLIC_MONTHLY_AIU_LIMIT", "").strip()
        if raw:
            try:
                return max(0, float(raw))
            except (ValueError, TypeError):
                pass
        return 3000.0
    elif base == _GHE_ENTERPRISE_BASE:
        raw = os.getenv("COPILOT_ENTERPRISE_MONTHLY_AIU_LIMIT", "").strip()
        if raw:
            try:
                return max(0, float(raw))
            except (ValueError, TypeError):
                pass
        return 3900.0
    # Fallback: legacy env var
    raw = os.getenv("COPILOT_MONTHLY_AIU_LIMIT", "").strip()
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_MONTHLY_AIU_LIMIT
    except (ValueError, TypeError):
        return _DEFAULT_MONTHLY_AIU_LIMIT
# Legacy function for backward compatibility.
_monthly_limit_aiu = _monthly_limit_for  # noqa: E731


def _empty_db() -> dict[str, Any]:
    return {"version": 1, "events": {}}


def _load_unlocked() -> dict[str, Any]:
    path = _db_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_db()
    except Exception as exc:
        logger.warning("[copilot_spend] failed to read %s: %s", path, exc)
        return _empty_db()
    # Migrate legacy list format to per-base dict.
    raw_events = data.get("events")
    if isinstance(raw_events, list):
        # Legacy: all events in one list. Assign to enterprise base (existing
        # behaviour for pre-split deployments), then save new dict format.
        events = {_GHE_ENTERPRISE_BASE: raw_events}
        data["events"] = events
        _save_unlocked(data)
    elif not isinstance(raw_events, dict):
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
            me = getattr(response_obj, "model_extra", None) or {}
            cu = me.get("copilot_usage")
        if not isinstance(cu, dict):
            return 0
        v = cu.get("total_nano_aiu", 0)
        return int(v) if isinstance(v, (int, float)) and v > 0 else 0
    except Exception:
        return 0
def _normalize_base(base_url: str) -> str:
    """Return the canonical bucket key for *base_url*."""
    base = (base_url or "").rstrip("/")
    if base == _GHE_PUBLIC_BASE:
        return _GHE_PUBLIC_BASE
    if base == _GHE_ENTERPRISE_BASE:
        return _GHE_ENTERPRISE_BASE
    return base or _GHE_ENTERPRISE_BASE

def record_and_check(
    response_obj: Any,
    *,
    provider_model: str = "",
    base_url: str = "",
) -> None:
    """Record AIU spend from *response_obj* and enforce per-account monthly limit.
    The spend is bucketed by base_url so the public GitHub Copilot account
    (dmascord, 3000 AIU) and SITA GHE account (3900 AIU) are tracked separately.
    Args:
        response_obj: The ChatCompletion object returned by the OpenAI SDK.
        provider_model: Full model string, e.g. ``"github-copilot-enterprise/gpt-5.4"``.
        base_url: The GHE base URL used for the request.
    """
    nano_aiu = extract_nano_aiu(response_obj)
    if nano_aiu <= 0:
        return
    now = time.time()
    bucket = _normalize_base(base_url)
    limit_aiu = _monthly_limit_for(bucket)
    limit_nano = int(limit_aiu * _NANO_AIU_PER_AIU)
    with _LOCK:
        data = _load_unlocked()
        events_by_base: dict = data.get("events", {})
        bucket_events: list = _prune_old_events(events_by_base.get(bucket, []), now)
        bucket_events.append([now, nano_aiu])
        events_by_base[bucket] = bucket_events
        data["events"] = events_by_base
        rolling_nano = _rolling_total_nano_aiu(bucket_events)
        rolling_aiu = rolling_nano / _NANO_AIU_PER_AIU
        logger.info(
            "[copilot_spend] recorded %.4f AIU (%.4f AIU rolling/30d, limit=%.0f AIU) model=%s base=%s",
            nano_aiu / _NANO_AIU_PER_AIU,
            rolling_aiu,
            limit_aiu,
            provider_model,
            bucket,
        )
        _save_unlocked(data)
    if rolling_nano > limit_nano:
        _enforce_cooldown(bucket, rolling_nano, limit_nano, now)


def _enforce_cooldown(
    base: str,
    rolling_nano: int,
    limit_nano: int,
    now: float,
) -> None:
    """Place GHE models for *base* into cooldown until the rolling window clears."""
    try:
        from agent.model_cooldown_db import mark_model_cooldown
    except Exception as exc:
        logger.error("[copilot_spend] cannot import mark_model_cooldown: %s", exc)
        return
    with _LOCK:
        data = _load_unlocked()
        events_by_base: dict = data.get("events", {})
        events = _prune_old_events(events_by_base.get(base, []), now)
    rolling = _rolling_total_nano_aiu(events)
    window_clears_at = now + _ROLLING_WINDOW_SECONDS
    for e in sorted(events, key=lambda x: x[0]):
        rolling -= int(e[1])
        if rolling <= limit_nano:
            window_clears_at = e[0] + _ROLLING_WINDOW_SECONDS
            break
    cooldown_seconds = max(60.0, window_clears_at - now + 60.0)
    rolling_aiu = rolling_nano / _NANO_AIU_PER_AIU
    limit_aiu = limit_nano / _NANO_AIU_PER_AIU
    label = "public" if base == _GHE_PUBLIC_BASE else "enterprise"
    logger.warning(
        "[copilot_spend] monthly AIU limit exceeded (%s): %.2f / %.0f AIU — "
        "placing github-copilot-enterprise into cooldown for %.0fs (%.1f days)",
        label,
        rolling_aiu,
        limit_aiu,
        cooldown_seconds,
        cooldown_seconds / 3600 / 24,
    )
    _GHE_MODELS = [
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-5-mini",
        "gpt-5.4",
        "gpt-5.3-codex",
        "claude-sonnet-4.6",
        "claude-opus-4.6",
        "",
    ]
    for model in _GHE_MODELS:
        try:
            mark_model_cooldown(
                _GHE_PROVIDER,
                model or "*",
                base_url=base,
                cooldown_seconds=cooldown_seconds,
                reason="monthly_aiu_limit_exceeded",
            )
        except Exception as exc:
            logger.error("[copilot_spend] mark_model_cooldown failed model=%s: %s", model, exc)


def rolling_spend_aiu(base_url: str = "") -> float:
    """Return the current rolling 30-day AIU spend for the given *base_url*.
    When *base_url* is empty/omitted, returns the combined total (legacy
    monitoring). Call with the specific base to get per-account spend.
    """
    now = time.time()
    bucket = _normalize_base(base_url) if base_url else "*"
    with _LOCK:
        data = _load_unlocked()
        events_by_base: dict = data.get("events", {})
        if bucket == "*":
            total = sum(
                _rolling_total_nano_aiu(_prune_old_events(evts, now))
                for evts in events_by_base.values()
            )
            return total / _NANO_AIU_PER_AIU
        events = _prune_old_events(events_by_base.get(bucket, []), now)
    return _rolling_total_nano_aiu(events) / _NANO_AIU_PER_AIU
