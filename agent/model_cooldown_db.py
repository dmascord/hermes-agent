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
_SCHEMA_VERSION = 2
_MAX_RECORDS = 512
_DEFAULT_SLOW_THRESHOLD_SECONDS = 75.0
_DEFAULT_COOLDOWN_SECONDS = 600.0
_DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 3
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300.0
_DEFAULT_CIRCUIT_BREAKER_WINDOW_SECONDS = 300.0


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


def circuit_breaker_threshold() -> int:
    raw = os.getenv("HERMES_CIRCUIT_BREAKER_THRESHOLD", "")
    if raw and raw.strip():
        try:
            return max(1, int(raw.strip()))
        except (TypeError, ValueError):
            pass
    return _DEFAULT_CIRCUIT_BREAKER_THRESHOLD


def circuit_breaker_cooldown_seconds() -> float:
    return _env_float("HERMES_CIRCUIT_BREAKER_COOLDOWN_SECONDS", _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS)


def circuit_breaker_window_seconds() -> float:
    return _env_float("HERMES_CIRCUIT_BREAKER_WINDOW_SECONDS", _DEFAULT_CIRCUIT_BREAKER_WINDOW_SECONDS)


def _normalize(value: str) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _key(provider: str, model: str, base_url: str = "", credential_id: str = "") -> str:
    raw = "\x1f".join((
        _normalize(provider),
        _normalize(model),
        _normalize(base_url),
        _normalize(credential_id or ""),
    ))
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
    credential_id: str = "",
    cooldown_seconds: float | None = None,
    reason: str = "slow_response",
    latency_seconds: float | None = None,
) -> float:
    """Put provider/model/base_url/credential into cooldown and return seconds applied.

    When ``credential_id`` is provided (e.g. ``"codex-1"``), the cooldown is
    scoped to that specific credential.  Other credentials in the same pool
    can still be used.  When omitted, the cooldown applies to the provider/
    model combination globally (all credentials).
    """
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return 0.0
    cooldown = slow_cooldown_seconds() if cooldown_seconds is None else max(0.0, float(cooldown_seconds))
    if cooldown <= 0:
        return 0.0
    now = time.time()
    rec_key = _key(provider_n, model_n, base_url, credential_id)
    with _LOCK:
        data = _load_db_unlocked()
        records = data.setdefault("records", {})
        records[rec_key] = {
            "provider": provider_n,
            "model": model_n,
            "base_url": _normalize(base_url),
            "credential_id": _normalize(credential_id) or None,
            "cooldown_until": now + cooldown,
            "reason": str(reason or "slow_response"),
            "latency_seconds": float(latency_seconds) if latency_seconds is not None else None,
            "updated_at": now,
        }
        _save_db_unlocked(data)
    logger.info(
        "Model cooldown set provider=%s model=%s base_url=%s credential_id=%s reason=%s latency=%s cooldown=%.0fs",
        provider_n,
        model_n,
        _normalize(base_url),
        _normalize(credential_id) or "(global)",
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
    credential_id: str = "",
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
        credential_id=credential_id,
        cooldown_seconds=cooldown_seconds,
        reason=reason,
        latency_seconds=latency,
    ) > 0


def model_cooldown_remaining(provider: str, model: str, *, base_url: str = "", credential_id: str = "") -> float:
    """Return seconds remaining until cooldown expires for this combination.

    When ``credential_id`` is provided, only the cooldown for that specific
    credential is checked.  When omitted, only pool-level (global) cooldowns
    are checked — per-credential cooldowns are invisible to this call.
    """
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return 0.0
    now = time.time()
    rec_key = _key(provider_n, model_n, base_url, credential_id)
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


def model_cooldown_remaining_for_pool(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    credential_ids: list[str] | None = None,
) -> float:
    """Return seconds remaining for a credential pool to recover.

    Scans all cooldown records matching provider+model+base_url.  Returns 0
    (no cooldown) if ANY credential in the pool is *not* in cooldown —
    meaning at least one account can still be used.  Returns the *minimum*
    remaining time when ALL credentials are in cooldown.

    ``credential_ids`` should list all known credential IDs for the pool.
    When provided, only those IDs are checked; pool-level (global) cooldowns
    for the same provider+model+base_url are also considered.
    """
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return 0.0
    now = time.time()
    with _LOCK:
        data = _load_db_unlocked()
        records = data.setdefault("records", {})

        # ── Compute the key prefix for this provider+model+base_url ──
        # Walk all records and group by credential_id.
        prefix_raw = "\x1f".join((provider_n, _normalize(model), _normalize(base_url)))
        prefix = hashlib.sha256(prefix_raw.encode("utf-8")).hexdigest()[:24]

        # Collect cooldowns: pool-level + per-credential
        pool_remaining: float = 0.0
        cred_remaining: dict[str, float] = {}
        creds_seen: set[str] = set()

        for rec_key, rec in list(records.items()):
            if not rec_key.startswith(prefix):
                continue
            if not isinstance(rec, dict):
                continue
            rec_cred = _normalize(rec.get("credential_id") or "")
            until = float(rec.get("cooldown_until") or 0)
            remaining = max(0.0, until - now)

            if remaining <= 0:
                # Clean up expired entries
                records.pop(rec_key, None)
                continue

            if not rec_cred:
                # Pool-level cooldown — applies to all credentials
                pool_remaining = remaining if not pool_remaining else min(pool_remaining, remaining)
            else:
                cred_remaining[rec_cred] = remaining
                creds_seen.add(rec_cred)

        # Clean up if we modified records
        try:
            _save_db_unlocked(data)
        except Exception:
            pass

    # If a pool-level cooldown exists, the entire pool is blocked regardless
    if pool_remaining > 0:
        return pool_remaining

    # If we know all credential IDs, check if any are missing from cooldown
    if credential_ids:
        cred_ids_set = set(_normalize(c or "") for c in credential_ids if c)
        not_in_cooldown = cred_ids_set - creds_seen
        if not_in_cooldown:
            return 0.0  # At least one credential has no cooldown

        # All known credentials are in cooldown — return shortest remaining
        if cred_remaining:
            return min(cred_remaining.values())
        return 0.0

    # No credential_ids provided: if ALL records for this prefix are per-cred,
    # assume pool can still try (some untracked credential might work).
    if cred_remaining and not pool_remaining:
        return 0.0

    return pool_remaining


def clear_model_cooldowns() -> None:
    """Clear all cooldowns. Intended for tests/admin repair."""
    with _LOCK:
        _save_db_unlocked(_empty_db())


_CIRCUIT_BREAKER_KEY_PREFIX = "cb::"


def _cb_key(provider: str, model: str, base_url: str = "") -> str:
    return _CIRCUIT_BREAKER_KEY_PREFIX + _key(provider, model, base_url, credential_id="")


def mark_provider_failure(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    reason: str = "request_failed",
) -> bool:
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return False
    now = time.time()
    cbk = _cb_key(provider_n, model_n, base_url)
    threshold = circuit_breaker_threshold()
    cooldown_seconds = circuit_breaker_cooldown_seconds()
    cutoff = now - circuit_breaker_window_seconds()

    with _LOCK:
        data = _load_db_unlocked()
        records = data.setdefault("records", {})
        rec = records.get(cbk)
        if not isinstance(rec, dict):
            rec = {"failures": [], "updated_at": now}
            records[cbk] = rec
        failures = rec.get("failures", [])
        if not isinstance(failures, list):
            failures = []
        failures = [ts for ts in failures if isinstance(ts, (int, float)) and ts > cutoff]
        failures.append(now)
        rec["failures"] = failures
        rec["updated_at"] = now
        rec["last_failure_reason"] = str(reason or "request_failed")

        count = len(failures)
        cooldown_applied = False
        if count >= threshold and cooldown_seconds > 0:
            cooldown_key = _key(provider_n, model_n, base_url, credential_id="")
            records[cooldown_key] = {
                "provider": provider_n,
                "model": model_n,
                "base_url": _normalize(base_url),
                "credential_id": None,
                "cooldown_until": now + cooldown_seconds,
                "reason": f"circuit_breaker:{reason}:{count}_failures",
                "latency_seconds": None,
                "updated_at": now,
            }
            rec["failures"] = []
            cooldown_applied = True

        _save_db_unlocked(data)

    if cooldown_applied:
        logger.warning(
            "Circuit breaker opened provider=%s model=%s base_url=%s failures=%d cooldown=%.0fs reason=%s",
            provider_n, model_n, _normalize(base_url), count, cooldown_seconds, reason,
        )
    return cooldown_applied


def mark_provider_success(
    provider: str,
    model: str,
    *,
    base_url: str = "",
) -> None:
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return
    now = time.time()
    cbk = _cb_key(provider_n, model_n, base_url)

    with _LOCK:
        data = _load_db_unlocked()
        records = data.setdefault("records", {})
        rec = records.get(cbk)
        if isinstance(rec, dict):
            failures = rec.get("failures", [])
            if isinstance(failures, list) and failures:
                rec["failures"] = []
                rec["updated_at"] = now
                _save_db_unlocked(data)


def provider_failure_count(
    provider: str,
    model: str,
    *,
    base_url: str = "",
) -> int:
    provider_n = _normalize(provider)
    model_n = _normalize(model)
    if not provider_n or not model_n:
        return 0
    now = time.time()
    cutoff = now - circuit_breaker_window_seconds()
    cbk = _cb_key(provider_n, model_n, base_url)

    with _LOCK:
        data = _load_db_unlocked()
        rec = data.setdefault("records", {}).get(cbk)
        if not isinstance(rec, dict):
            return 0
        failures = rec.get("failures", [])
        if not isinstance(failures, list):
            return 0
        return len([ts for ts in failures if isinstance(ts, (int, float)) and ts > cutoff])
