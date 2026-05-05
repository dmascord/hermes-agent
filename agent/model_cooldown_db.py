"""Persistent model cooldown hooks for gateway routing.

The gateway asks :func:`model_cooldown_remaining` before selecting a swarm
model.  Slow or stale provider calls can use :func:`record_model_latency` and
:func:`mark_model_cooldown` to temporarily "sin-bin" a provider/model/base URL
combination so future routing skips it for a short period.
"""

from __future__ import annotations

import hashlib
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
_SCHEMA_VERSION = 1
_MAX_RECORDS = 512
_DEFAULT_SLOW_THRESHOLD_SECONDS = 75.0
_DEFAULT_COOLDOWN_SECONDS = 600.0


def _db_path() -> Path:
    return get_hermes_home() / "model_cooldowns.json"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def slow_threshold_seconds() -> float:
    return _env_float("HERMES_SLOW_MODEL_THRESHOLD_SECONDS", _DEFAULT_SLOW_THRESHOLD_SECONDS)


def slow_cooldown_seconds() -> float:
    return _env_float("HERMES_SLOW_MODEL_COOLDOWN_SECONDS", _DEFAULT_COOLDOWN_SECONDS)


def _normalize(value: str) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _key(provider: str, model: str, base_url: str = "") -> str:
    raw = "\x1f".join((_normalize(provider), _normalize(model), _normalize(base_url)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _empty_db() -> dict[str, Any]:
    return {"version": _SCHEMA_VERSION, "records": {}}


def _repair_path_permissions(path: Path) -> bool:
    """Best-effort repair for cooldown files created by another container user."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
        if path.exists():
            try:
                path.chmod(0o600)
            except OSError:
                return False
        return True
    except OSError:
        return False


def _load_db_unlocked() -> dict[str, Any]:
    path = _db_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_db()
    except PermissionError as exc:
        if _repair_path_permissions(path):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as retry_exc:
                logger.warning("Failed to read model cooldown DB %s after permission repair: %s", path, retry_exc)
                return _empty_db()
        else:
            logger.warning("Failed to read model cooldown DB %s: %s", path, exc)
            return _empty_db()
    except Exception as exc:
        logger.warning("Failed to read model cooldown DB %s: %s", path, exc)
        return _empty_db()
    if not isinstance(data, dict):
        return _empty_db()
    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    data["version"] = _SCHEMA_VERSION
    return data


def _save_db_unlocked(data: dict[str, Any]) -> None:
    records = data.setdefault("records", {})
    if isinstance(records, dict) and len(records) > _MAX_RECORDS:
        ordered = sorted(
            records.items(),
            key=lambda item: float((item[1] or {}).get("updated_at") or 0),
            reverse=True,
        )[:_MAX_RECORDS]
        data["records"] = dict(ordered)
    path = _db_path()
    atomic_json_write(path, data, indent=2, sort_keys=True)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def mark_model_cooldown(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    cooldown_seconds: float | None = None,
    reason: str = "slow_response",
    latency_seconds: float | None = None,
) -> float:
    """Put provider/model/base_url into cooldown and return seconds applied."""
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return 0.0
    cooldown = slow_cooldown_seconds() if cooldown_seconds is None else max(0.0, float(cooldown_seconds))
    if cooldown <= 0:
        return 0.0
    now = time.time()
    rec_key = _key(provider_n, model_n, base_url)
    with _LOCK:
        data = _load_db_unlocked()
        records = data.setdefault("records", {})
        records[rec_key] = {
            "provider": provider_n,
            "model": model_n,
            "base_url": _normalize(base_url),
            "cooldown_until": now + cooldown,
            "reason": str(reason or "slow_response"),
            "latency_seconds": float(latency_seconds) if latency_seconds is not None else None,
            "updated_at": now,
        }
        _save_db_unlocked(data)
    logger.info(
        "Model cooldown set provider=%s model=%s base_url=%s reason=%s latency=%s cooldown=%.0fs",
        provider_n,
        model_n,
        _normalize(base_url),
        reason,
        latency_seconds,
        cooldown,
    )
    return cooldown


def record_model_latency(
    provider: str,
    model: str,
    latency_seconds: float,
    *,
    base_url: str = "",
    threshold_seconds: float | None = None,
    cooldown_seconds: float | None = None,
    reason: str = "slow_response",
) -> bool:
    """Record a completed call latency; cooldown if it exceeds threshold.

    Returns True when a cooldown was applied.
    """
    threshold = slow_threshold_seconds() if threshold_seconds is None else float(threshold_seconds)
    try:
        latency = float(latency_seconds)
    except (TypeError, ValueError):
        return False
    if threshold <= 0 or latency < threshold:
        return False
    return mark_model_cooldown(
        provider,
        model,
        base_url=base_url,
        cooldown_seconds=cooldown_seconds,
        reason=reason,
        latency_seconds=latency,
    ) > 0


def model_cooldown_remaining(provider: str, model: str, *, base_url: str = "") -> float:
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return 0.0
    now = time.time()
    rec_key = _key(provider_n, model_n, base_url)
    changed = False
    with _LOCK:
        data = _load_db_unlocked()
        records = data.setdefault("records", {})
        rec = records.get(rec_key)
        if not isinstance(rec, dict):
            return 0.0
        until = float(rec.get("cooldown_until") or 0)
        remaining = until - now
        if remaining <= 0:
            records.pop(rec_key, None)
            changed = True
        if changed:
            try:
                _save_db_unlocked(data)
            except Exception:
                pass
        return max(0.0, remaining)


def clear_model_cooldowns() -> None:
    """Clear all cooldowns. Intended for tests/admin repair."""
    with _LOCK:
        _save_db_unlocked(_empty_db())
