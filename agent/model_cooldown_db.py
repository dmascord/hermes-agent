"""Persistent model cooldown hooks for gateway routing (SQLite backend).

The gateway asks :func:`model_cooldown_remaining` before selecting a swarm
model.  Slow or stale provider calls can use :func:`record_model_latency` and
:func:`mark_model_cooldown` to temporarily "sin-bin" a provider/model/base URL
combination so future routing skips it for a short period.

Storage
-------
Backed by a SQLite database at ``<HERMES_HOME>/model_cooldowns.db`` (WAL mode)
with two tables:

* **cooldowns** — provider/model/base_url/credential scoped cooldown entries.
* **circuit_breakers** — rolling failure counters per (provider, model, base_url).

On first use the module automatically migrates any existing data from the
legacy ``model_cooldowns.json`` file, then renames it to
``model_cooldowns.json.migrated``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SCHEMA_VERSION = 3  # bumped from 2 for SQLite migration
_MAX_RECORDS = 512
_DEFAULT_SLOW_THRESHOLD_SECONDS = 75.0
_DEFAULT_COOLDOWN_SECONDS = 600.0
_DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 3
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300.0
_DEFAULT_CIRCUIT_BREAKER_WINDOW_SECONDS = 300.0
_DEFAULT_MAX_COOLDOWN_SECONDS = 3600.0  # 1 hour — universal cap for all cooldowns

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_connection: sqlite3.Connection | None = None
_connection_lock = threading.Lock()


def _db_path() -> Path:
    return get_hermes_home() / "model_cooldowns.db"


def _legacy_json_path() -> Path:
    return get_hermes_home() / "model_cooldowns.json"


def _get_conn() -> sqlite3.Connection:
    """Return the singleton WAL-mode SQLite connection (creating it on first call)."""
    global _connection
    if _connection is not None:
        return _connection
    with _connection_lock:
        if _connection is not None:
            return _connection
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")       # balance safety/speed
        conn.execute("PRAGMA busy_timeout=5000")         # wait up to 5s on lock
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")   # reclaim space
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        _migrate_from_json(conn)
        _connection = conn
        return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cooldowns (
            key             TEXT PRIMARY KEY,
            provider        TEXT NOT NULL,
            model           TEXT NOT NULL,
            base_url        TEXT NOT NULL DEFAULT '',
            credential_id   TEXT,
            cooldown_until  REAL NOT NULL,
            reason          TEXT NOT NULL DEFAULT 'slow_response',
            latency_seconds REAL,
            updated_at      REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cooldowns_provider_model
            ON cooldowns(provider, model, base_url);

        CREATE TABLE IF NOT EXISTS circuit_breakers (
            key                 TEXT PRIMARY KEY,
            provider            TEXT NOT NULL,
            model               TEXT NOT NULL,
            base_url            TEXT NOT NULL DEFAULT '',
            failures            TEXT NOT NULL DEFAULT '[]',
            last_failure_reason TEXT,
            updated_at          REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cb_provider_model
            ON circuit_breakers(provider, model, base_url);
    """)
    conn.commit()


def _migrate_from_json(conn: sqlite3.Connection) -> None:
    """One-time migration from legacy ``model_cooldowns.json`` to SQLite."""
    json_path = _legacy_json_path()
    if not json_path.exists():
        return

    # Check the metadata table to see if migration already happened.
    already = conn.execute(
        "SELECT 1 FROM cooldowns LIMIT 1"
    ).fetchone()
    if already:
        # Rows exist — assume migration was done (or DB was seeded manually).
        # Still rename the JSON file so we don't re-read it next startup.
        try:
            json_path.rename(json_path.with_suffix(".json.migrated"))
        except OSError:
            pass
        return

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[cooldown] legacy JSON migration: cannot read %s: %s", json_path, exc)
        return

    records = raw.get("records") if isinstance(raw, dict) else None
    if not isinstance(records, dict) or not records:
        # Empty or missing — just rename and move on.
        try:
            json_path.rename(json_path.with_suffix(".json.migrated"))
        except OSError:
            pass
        return

    # Separate normal cooldowns from circuit-breaker entries.
    cd_rows: list[tuple[str, str, str, str, str | None, float, str, float | None, float]] = []
    cb_rows: list[tuple[str, str, str, str, str, str, float]] = []
    now = time.time()
    migrated_cd = 0
    migrated_cb = 0

    for rec_key, rec in records.items():
        if not isinstance(rec, dict):
            continue
        if rec_key.startswith("cb::"):
            # Circuit breaker entry
            failures = rec.get("failures", [])
            if isinstance(failures, list):
                failures = [ts for ts in failures if isinstance(ts, (int, float))]
            cb_rows.append((
                rec_key,
                str(rec.get("provider", "")),
                str(rec.get("model", "")),
                str(rec.get("base_url", "")),
                json.dumps(failures),
                str(rec.get("last_failure_reason", "")),
                float(rec.get("updated_at", now)),
            ))
            migrated_cb += 1
        else:
            # Cooldown entry
            cooldown_until = float(rec.get("cooldown_until") or 0)
            if cooldown_until <= now:
                continue  # skip already-expired
            cd_rows.append((
                rec_key,
                str(rec.get("provider", "")),
                str(rec.get("model", "")),
                str(rec.get("base_url", "")),
                rec.get("credential_id"),           # None is fine for TEXT? use None
                cooldown_until,
                str(rec.get("reason", "slow_response")),
                rec.get("latency_seconds"),         # may be None
                float(rec.get("updated_at", now)),
            ))
            migrated_cd += 1

    if cd_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO cooldowns "
            "(key, provider, model, base_url, credential_id, cooldown_until, "
            " reason, latency_seconds, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            cd_rows,
        )
    if cb_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO circuit_breakers "
            "(key, provider, model, base_url, failures, last_failure_reason, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            cb_rows,
        )
    conn.commit()

    # Rename legacy file to prevent re-migration.
    try:
        json_path.rename(json_path.with_suffix(".json.migrated"))
    except OSError:
        pass

    if migrated_cd or migrated_cb:
        logger.info(
            "[cooldown] migrated %d cooldowns + %d circuit breakers from %s to SQLite",
            migrated_cd, migrated_cb, json_path.name,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def max_cooldown_seconds() -> float:
    """Universal cap for all cooldown durations (env: HERMES_MAX_COOLDOWN_SECONDS).

    Prevents runaway cooldowns from 429 quota parsing (e.g. 23h+ from UTC
    reset-time or weekly/daily quota hints) from blocking the entire
    passthrough fallback chain.  Auth-failure cooldowns (token_invalidated)
    are exempt — they need manual re-auth regardless of duration.
    """
    return _env_float("HERMES_MAX_COOLDOWN_SECONDS", _DEFAULT_MAX_COOLDOWN_SECONDS)


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


def _gc_cooldowns(conn: sqlite3.Connection, now: float) -> None:
    """Remove expired cooldowns, stale circuit breakers, and the oldest records when over _MAX_RECORDS."""
    conn.execute("DELETE FROM cooldowns WHERE cooldown_until <= ?", (now,))
    # Prune circuit_breaker entries whose failures have all aged out of the window.
    cutoff = now - circuit_breaker_window_seconds()
    stale_cb = conn.execute(
        "SELECT key, failures FROM circuit_breakers"
    ).fetchall()
    for row in stale_cb:
        try:
            failures = json.loads(row["failures"]) if row["failures"] else []
        except (json.JSONDecodeError, TypeError):
            failures = []
        active = [ts for ts in failures if isinstance(ts, (int, float)) and ts > cutoff]
        if not active:
            conn.execute("DELETE FROM circuit_breakers WHERE key = ?", (row["key"],))
        elif len(active) < len(failures):
            conn.execute(
                "UPDATE circuit_breakers SET failures = ?, updated_at = ? WHERE key = ?",
                (json.dumps(active), now, row["key"]),
            )
    # Keep only the _MAX_RECORDS most-recently-updated cooldowns.
    excess = conn.execute(
        "SELECT COUNT(*) FROM cooldowns"
    ).fetchone()[0] - _MAX_RECORDS
    if excess > 0:
        conn.execute(
            "DELETE FROM cooldowns WHERE key IN ("
            "  SELECT key FROM cooldowns ORDER BY updated_at ASC LIMIT ?"
            ")",
            (excess,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API — cooldowns
# ---------------------------------------------------------------------------

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
    # ── Universal cooldown cap ──────────────────────────────────────────────
    # Prevent runaway cooldowns (e.g. 23h+ from 429 quota parsing) from
    # blocking the entire passthrough fallback chain.  Auth-failure
    # cooldowns (token_invalidated) are exempt — they need manual re-auth.
    _max = max_cooldown_seconds()
    _reason_lower = str(reason or "").lower()
    if _max > 0 and cooldown > _max and "token_invalidated" not in _reason_lower:
        logger.info(
            "Cooldown capped: provider=%s model=%s reason=%s original=%.0fs → capped=%.0fs",
            provider_n, model_n, reason, cooldown, _max,
        )
        cooldown = _max
    now = time.time()
    rec_key = _key(provider_n, model_n, base_url, credential_id)

    with _LOCK:
        conn = _get_conn()
        try:
            _gc_cooldowns(conn, now)
            conn.execute(
                """INSERT OR REPLACE INTO cooldowns
                   (key, provider, model, base_url, credential_id,
                    cooldown_until, reason, latency_seconds, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    rec_key, provider_n, model_n, _normalize(base_url),
                    _normalize(credential_id) or None,
                    now + cooldown,
                    str(reason or "slow_response"),
                    float(latency_seconds) if latency_seconds is not None else None,
                    now,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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

    with _LOCK:
        conn = _get_conn()
        _gc_cooldowns(conn, now)
        row = conn.execute(
            "SELECT cooldown_until FROM cooldowns WHERE key = ?",
            (rec_key,),
        ).fetchone()
        if row is None:
            return 0.0
        remaining = row["cooldown_until"] - now
        if remaining <= 0:
            conn.execute("DELETE FROM cooldowns WHERE key = ?", (rec_key,))
            conn.commit()
            return 0.0
        return remaining


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
        conn = _get_conn()
        _gc_cooldowns(conn, now)
        # ── Pool-level (global) cooldown ──────────────────────────────
        pool_row = conn.execute(
            "SELECT cooldown_until FROM cooldowns"
            " WHERE provider = ? AND model = ? AND base_url = ?"
            "   AND (credential_id IS NULL OR credential_id = '')",
            (provider_n, model_n, _normalize(base_url)),
        ).fetchone()
        pool_remaining = 0.0
        if pool_row is not None:
            pool_remaining = max(0.0, pool_row["cooldown_until"] - now)

        # ── Per-credential cooldowns ──────────────────────────────────
        rows = conn.execute(
            "SELECT credential_id, cooldown_until FROM cooldowns"
            " WHERE provider = ? AND model = ? AND base_url = ?"
            "   AND credential_id IS NOT NULL AND credential_id != ''",
            (provider_n, model_n, _normalize(base_url)),
        ).fetchall()
        cred_remaining: dict[str, float] = {}
        creds_seen: set[str] = set()
        seen_stale = False
        for row in rows:
            cred_id = _normalize(row["credential_id"] or "")
            if not cred_id:
                continue
            remaining = max(0.0, row["cooldown_until"] - now)
            if remaining <= 0:
                seen_stale = True
                continue
            cred_remaining[cred_id] = remaining
            creds_seen.add(cred_id)

        # GC stale entries if any were found.
        if seen_stale:
            conn.execute(
                "DELETE FROM cooldowns WHERE provider = ? AND model = ? AND base_url = ?"
                "  AND credential_id IS NOT NULL AND credential_id != ''"
                "  AND cooldown_until <= ?",
                (provider_n, model_n, _normalize(base_url), now),
            )
            conn.commit()

    # If a pool-level cooldown exists, the entire pool is blocked regardless.
    if pool_remaining > 0:
        return pool_remaining

    # If we know all credential IDs, check if any are missing from cooldown.
    if credential_ids:
        cred_ids_set = set(_normalize(c or "") for c in credential_ids if c)
        not_in_cooldown = cred_ids_set - creds_seen
        if not_in_cooldown:
            return 0.0  # At least one credential has no cooldown.

        # All known credentials are in cooldown — return shortest remaining.
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
        conn = _get_conn()
        conn.execute("DELETE FROM cooldowns")
        conn.execute("DELETE FROM circuit_breakers")
        conn.commit()


# ---------------------------------------------------------------------------
# Public API — circuit breakers
# ---------------------------------------------------------------------------

_CIRCUIT_BREAKER_KEY_PREFIX = "cb::"


def _cb_key(provider: str, model: str, base_url: str = "") -> str:
    return _CIRCUIT_BREAKER_KEY_PREFIX + _key(provider, model, base_url, credential_id="")


def _is_transient_failure(reason: str) -> bool:
    """Return True for failure reasons that reflect network/transient issues
    (TCP reset, read timeout, Cloudflare 524) rather than provider health.

    Transient failures should still be recorded so the failure window is
    accurate, but they should NOT trip the circuit breaker on their own —
    a 3-failures-in-5-min threshold that lumps transient TCP resets together
    with provider quota exhaustion incorrectly cooldowns a working model
    after one bad network blip.
    """
    r = (reason or "").lower()
    transient_markers = (
        "timed out",
        "timeout",
        "connection reset",
        "reset by peer",
        "broken pipe",
        "connection aborted",
        "connection refused",
        "bad file descriptor",
        "eof",
        "stream ended",
        "first event",
        "upstream timeout",
        "upstream gateway timeout",
        "cloudflare 524",
        "cloudflare 5",
        "passthrough_error",
        "transient",
    )
    return any(marker in r for marker in transient_markers)


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
    cooldown_applied = False
    transient = _is_transient_failure(reason)

    with _LOCK:
        conn = _get_conn()
        _gc_cooldowns(conn, now)
        try:
            # Load existing failures.
            row = conn.execute(
                "SELECT failures FROM circuit_breakers WHERE key = ?", (cbk,)
            ).fetchone()
            failures: list[float] = []
            if row is not None:
                try:
                    failures = json.loads(row["failures"])
                except (json.JSONDecodeError, TypeError):
                    failures = []
                if not isinstance(failures, list):
                    failures = []

            # Prune outside window and append.
            failures = [ts for ts in failures if isinstance(ts, (int, float)) and ts > cutoff]
            # For transient failures, record but don't append to the
            # circuit-breaker count.  The reason is recorded on the row
            # so the operator can see the failure happened, but it does
            # not contribute to tripping the breaker.
            if not transient:
                failures.append(now)
            count = len(failures)

            # UPSERT circuit breaker row.
            conn.execute(
                """INSERT OR REPLACE INTO circuit_breakers
                   (key, provider, model, base_url, failures, last_failure_reason, updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    cbk, provider_n, model_n, _normalize(base_url),
                    json.dumps(failures),
                    str(reason or "request_failed"),
                    now,
                ),
            )

            # Circuit breaker trip — only on non-transient failures.
            if not transient and count >= threshold and cooldown_seconds > 0:
                cooldown_key = _key(provider_n, model_n, base_url, credential_id="")
                conn.execute(
                    """INSERT OR REPLACE INTO cooldowns
                       (key, provider, model, base_url, credential_id,
                        cooldown_until, reason, latency_seconds, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        cooldown_key, provider_n, model_n, _normalize(base_url),
                        None,
                        now + cooldown_seconds,
                        f"circuit_breaker:{reason}:{count}_failures",
                        None,
                        now,
                    ),
                )
                # Reset failures after trip.
                conn.execute(
                    "UPDATE circuit_breakers SET failures = '[]', updated_at = ? WHERE key = ?",
                    (now, cbk),
                )
                cooldown_applied = True

            conn.commit()
        except Exception:
            conn.rollback()
            raise

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
    # Key used by mark_model_cooldown() — same identity, no credential_id.
    cooldown_key = _key(provider_n, model_n, base_url, credential_id="")

    with _LOCK:
        conn = _get_conn()
        # Clear the circuit breaker failures (existing behaviour).
        row = conn.execute(
            "SELECT failures FROM circuit_breakers WHERE key = ?", (cbk,)
        ).fetchone()
        cb_changed = False
        if row is not None:
            try:
                failures = json.loads(row["failures"])
            except (json.JSONDecodeError, TypeError):
                failures = []
            if isinstance(failures, list) and failures:
                conn.execute(
                    "UPDATE circuit_breakers SET failures = '[]', updated_at = ? WHERE key = ?",
                    (now, cbk),
                )
                cb_changed = True
        # Also clear transient cooldowns in the cooldowns table for this
        # provider/model.  Without this, a model that hit 3 transient
        # failures (TCP reset, brief 524) gets a 5-minute circuit-breaker
        # cooldown that the very next successful call cannot clear.
        # We only clear cooldowns whose reason indicates a transient /
        # circuit-breaker trigger — quota/usage-limit cooldowns are left
        # alone because the next successful call doesn't reset the quota
        # window on the upstream provider.
        cd_rows = conn.execute(
            "SELECT cooldown_until, reason FROM cooldowns WHERE key = ?",
            (cooldown_key,),
        ).fetchall()
        cd_changed = False
        if cd_rows:
            _transient_reasons = (
                "circuit_breaker:",
                "passthrough_error",
                "hermes_code_stream_",
                "transient_",
            )
            for cd_row in cd_rows:
                _reason = str(cd_row["reason"] or "").lower()
                _is_transient = any(t in _reason for t in _transient_reasons)
                if _is_transient:
                    conn.execute(
                        "DELETE FROM cooldowns WHERE key = ?",
                        (cooldown_key,),
                    )
                    cd_changed = True
        if cb_changed or cd_changed:
            conn.commit()
            if cd_changed:
                logger.info(
                    "provider success cleared transient cooldowns provider=%s model=%s",
                    provider_n, model_n,
                )


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
        conn = _get_conn()
        row = conn.execute(
            "SELECT failures FROM circuit_breakers WHERE key = ?", (cbk,)
        ).fetchone()
        if row is None:
            return 0
        try:
            failures = json.loads(row["failures"])
        except (json.JSONDecodeError, TypeError):
            return 0
        if not isinstance(failures, list):
            return 0
        return len([ts for ts in failures if isinstance(ts, (int, float)) and ts > cutoff])
