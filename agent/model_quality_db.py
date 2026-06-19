"""Model quality tracking — SQLite-backed runtime quality metrics.

Records per-model success/failure rates, latency, and text-only rates.
The passthrough chain uses this data to sort models by current quality
rather than rotating through all models to discover which ones work.

Storage: ``<HERMES_HOME>/model_quality.db`` (WAL mode, single connection).

Schema::

    model_metrics:
        key             TEXT PRIMARY KEY   -- "provider/model" or "provider/model::base_url"
        provider        TEXT NOT NULL
        model           TEXT NOT NULL
        base_url        TEXT NOT NULL DEFAULT ''
        total_calls     INTEGER NOT NULL DEFAULT 0
        success_calls   INTEGER NOT NULL DEFAULT 0
        failure_calls   INTEGER NOT NULL DEFAULT 0
        text_only_calls INTEGER NOT NULL DEFAULT 0   -- tools provided, text returned
        ctx_overflow_calls INTEGER NOT NULL DEFAULT 0  -- context too large errors
        avg_latency_ms  REAL NOT NULL DEFAULT 0       -- rolling average
        quality_score   REAL NOT NULL DEFAULT 50.0    -- 0-100, computed from rates
        last_success_at REAL,
        last_failure_at REAL,
        updated_at      REAL NOT NULL
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_connection: Optional[sqlite3.Connection] = None
_connection_lock = threading.Lock()

_DB_NAME = "model_quality.db"
_QUALITY_WINDOW = 100  # rolling window for quality score calculation
_DECAY_HALF_LIFE_HOURS = float(os.getenv("HERMES_QUALITY_DECAY_HALF_LIFE_HOURS", "24"))  # 24h default


def _db_path() -> "os.PathLike":
    return get_hermes_home() / _DB_NAME


def _get_conn() -> sqlite3.Connection:
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
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        _connection = conn
        return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            key                 TEXT PRIMARY KEY,
            provider            TEXT NOT NULL,
            model               TEXT NOT NULL,
            base_url            TEXT NOT NULL DEFAULT '',
            total_calls         INTEGER NOT NULL DEFAULT 0,
            success_calls       INTEGER NOT NULL DEFAULT 0,
            failure_calls       INTEGER NOT NULL DEFAULT 0,
            text_only_calls     INTEGER NOT NULL DEFAULT 0,
            ctx_overflow_calls  INTEGER NOT NULL DEFAULT 0,
            avg_latency_ms      REAL NOT NULL DEFAULT 0,
            quality_score       REAL NOT NULL DEFAULT 50.0,
            last_success_at     REAL,
            last_failure_at     REAL,
            updated_at          REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mm_provider_model
            ON model_metrics(provider, model);

        CREATE TABLE IF NOT EXISTS model_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            provider        TEXT NOT NULL,
            model           TEXT NOT NULL,
            event_type      TEXT NOT NULL,  -- 'success', 'failure', 'text_only', 'ctx_overflow'
            latency_ms      REAL,
            error_code      INTEGER,
            error_message   TEXT,
            created_at      REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_me_provider_model
            ON model_events(provider, model, created_at);

        -- Keep only last 1000 events per provider/model
        CREATE INDEX IF NOT EXISTS idx_me_created
            ON model_events(created_at);
    """)
    _seed_default_quality_entries(conn)
    conn.commit()


# Models that should be present in the quality DB with a healthy initial score so
# they pass the quality floor (default 60.0) on first deployment, rather than
# being dropped from the fallback chain while they accumulate their first 100
# calls.  An entry here does NOT mean "this model is always good" — once real
# calls start landing in record_success / record_failure the score is updated
# normally and these seeds stop mattering.
_QUALITY_SEED_ENTRIES: List[Tuple[str, str, float]] = [
    # provider, model, initial_score
    ("claude-code-cli", "sonnet", 90.0),
    ("claude-code-cli", "opus", 90.0),
    ("claude-code-cli", "haiku", 85.0),
    ("openai-codex", "gpt-5.5", 90.0),
    ("openai-codex", "gpt-5.4", 90.0),
    ("openai-codex", "gpt-5.4-mini", 85.0),
    ("openai-codex", "gpt-5.3-codex", 85.0),
    ("openai-codex", "gpt-5.3-codex-spark", 80.0),
]


def _seed_default_quality_entries(conn: sqlite3.Connection) -> None:
    """Insert initial quality rows for known-bridge models if absent.

    Uses INSERT OR IGNORE so it never overwrites a row that already exists
    (the real score is built up by record_success/record_failure).
    """
    for provider, model, initial_score in _QUALITY_SEED_ENTRIES:
        key = _make_key(provider, model, "")
        try:
            conn.execute(
                """INSERT OR IGNORE INTO model_metrics
                   (key, provider, model, base_url, total_calls, success_calls,
                    quality_score, last_success_at, updated_at)
                   VALUES (?, ?, ?, '', 0, 0, ?, NULL, ?)""",
                (key, provider, model, initial_score, time.time()),
            )
        except sqlite3.Error as _seed_exc:
            logger.debug("[model-quality] seed insert failed for %s/%s: %s", provider, model, _seed_exc)

    conn.commit()


# ── Scoring ────────────────────────────────────────────────────────────

def _compute_quality_score(
    success_calls: int,
    failure_calls: int,
    text_only_calls: int,
    avg_latency_ms: float,
) -> float:
    """Compute a 0-100 quality score from metrics.

    Score components:
    - Success rate: 0-80 points (80 * success_rate)
    - Latency: 0-20 points (20 * latency_bonus)
    - Text-only responses are recorded but no longer penalise quality (ambiguous signal).
    - Cooldown (2 min) still applies to prevent wasted retries on the current request.

    A model with 100% success, 0% text-only, and under 5 seconds latency gets 100.
    A model with 0% success gets 0.
    """
    total = success_calls + failure_calls
    if total == 0:
        return 50.0  # unknown — neutral

    success_rate = success_calls / total
    # text_only_rate is relative to successful calls with tools
    text_only_rate = text_only_calls / max(success_calls, 1)

    score = 0.0
    # Success rate component (50 points max)
    score += 80.0 * success_rate
    # Latency component (20 points max) — under 5s = 20, over 30s = 0
    latency_s = avg_latency_ms / 1000.0
    latency_bonus = max(0.0, min(1.0, (30.0 - latency_s) / 25.0))
    score += 20.0 * latency_bonus

    return round(score, 2)


def _rolling_avg_latency(old_avg: float, old_count: int, new_latency_ms: float) -> float:
    """Compute rolling average latency."""
    if old_count == 0:
        return new_latency_ms
    # Exponential moving average with alpha=0.3
    alpha = 0.3
    return alpha * new_latency_ms + (1.0 - alpha) * old_avg


def _compute_decayed_score(
    provider: str,
    model: str,
    conn: sqlite3.Connection,
    half_life_hours: float = _DECAY_HALF_LIFE_HOURS,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute decayed success/failure/latency/score from recent events.

    Returns (decayed_success, decayed_failures, decayed_latency_ms, score) or None
    if no events found.  Exponential decay with configurable half-life.
    """
    if "/" in model:
        model = model.split("/", 1)[1]
    rows = conn.execute(
        """SELECT event_type, latency_ms, created_at
           FROM model_events
           WHERE provider=? AND model=?
           ORDER BY created_at DESC LIMIT ?""",
        (provider, model, _QUALITY_WINDOW),
    ).fetchall()
    if not rows:
        return None

    now = time.time()
    decay_constant = half_life_hours * 3600.0 / math.log(2.0)

    decayed_success = 0.0
    decayed_failure = 0.0
    decayed_lat_sum = 0.0
    decayed_lat_weight = 0.0

    for row in rows:
        age = max(0.0, now - row["created_at"])
        weight = math.exp(-age / decay_constant)
        etype = row["event_type"]
        if etype in ("success", "text_only"):
            decayed_success += weight
        elif etype == "failure":
            decayed_failure += weight
        if row["latency_ms"] and row["latency_ms"] > 0:
            decayed_lat_sum += row["latency_ms"] * weight
            decayed_lat_weight += weight

    decayed_latency = decayed_lat_sum / decayed_lat_weight if decayed_lat_weight > 0 else 0.0
    score = _compute_quality_score(decayed_success, decayed_failure, 0, decayed_latency)
    return (decayed_success, decayed_failure, decayed_latency, score)


# ── Recording ──────────────────────────────────────────────────────────

def record_success(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    latency_ms: float = 0.0,
) -> None:
    """Record a successful API call."""
    key = _make_key(provider, model, base_url)
    # Strip provider prefix for model column storage (e.g. "minimax/MiniMax-M2.7" → "MiniMax-M2.7")
    stored_model = model.split("/", 1)[1] if "/" in model else model
    now = time.time()
    with _LOCK:
        conn = _get_conn()
        row = conn.execute(
            "SELECT total_calls, success_calls, text_only_calls, avg_latency_ms FROM model_metrics WHERE key=?",
            (key,),
        ).fetchone()
        if row:
            total = row["total_calls"] + 1
            success = row["success_calls"] + 1
            text_only = row["text_only_calls"]  # unchanged
            avg_lat = _rolling_avg_latency(row["avg_latency_ms"], row["total_calls"], latency_ms)
            score = _compute_quality_score(success, total - success, text_only, avg_lat)
            conn.execute(
                """UPDATE model_metrics
                   SET total_calls=?, success_calls=?, avg_latency_ms=?, quality_score=?,
                       last_success_at=?, updated_at=?
                   WHERE key=?""",
                (total, success, avg_lat, score, now, now, key),
            )
        else:
            score = _compute_quality_score(1, 0, 0, latency_ms)
            conn.execute(
                """INSERT INTO model_metrics
                   (key, provider, model, base_url, total_calls, success_calls,
                    avg_latency_ms, quality_score, last_success_at, updated_at)
                   VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, ?)""",
                (key, provider, stored_model, base_url, latency_ms, score, now, now),
            )
        # Record event
        conn.execute(
            "INSERT INTO model_events (provider, model, event_type, latency_ms, created_at) VALUES (?, ?, 'success', ?, ?)",
            (provider, stored_model, latency_ms, now),
        )
        _gc_events(conn, provider, stored_model)
        conn.commit()


def record_failure(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    latency_ms: float = 0.0,
    error_code: int = 0,
    error_message: str = "",
) -> None:
    """Record a failed API call."""
    key = _make_key(provider, model, base_url)
    # Strip provider prefix for model column storage (e.g. "minimax/MiniMax-M2.7" → "MiniMax-M2.7")
    stored_model = model.split("/", 1)[1] if "/" in model else model
    now = time.time()
    with _LOCK:
        conn = _get_conn()
        row = conn.execute(
            "SELECT total_calls, success_calls, failure_calls, text_only_calls, avg_latency_ms FROM model_metrics WHERE key=?",
            (key,),
        ).fetchone()
        if row:
            total = row["total_calls"] + 1
            failures = row["failure_calls"] + 1
            text_only = row["text_only_calls"]
            avg_lat = _rolling_avg_latency(row["avg_latency_ms"], row["total_calls"], latency_ms)
            score = _compute_quality_score(row["success_calls"], failures, text_only, avg_lat)
            conn.execute(
                """UPDATE model_metrics
                   SET total_calls=?, failure_calls=?, avg_latency_ms=?, quality_score=?,
                       last_failure_at=?, updated_at=?
                   WHERE key=?""",
                (total, failures, avg_lat, score, now, now, key),
            )
        else:
            score = _compute_quality_score(0, 1, 0, latency_ms)
            conn.execute(
                """INSERT INTO model_metrics
                   (key, provider, model, base_url, total_calls, failure_calls,
                    avg_latency_ms, quality_score, last_failure_at, updated_at)
                   VALUES (?, ?, ?, ?, 1, 1, ?, ?, ?, ?)""",
                (key, provider, stored_model, base_url, latency_ms, score, now, now),
            )
        conn.execute(
            "INSERT INTO model_events (provider, model, event_type, latency_ms, error_code, error_message, created_at) VALUES (?, ?, 'failure', ?, ?, ?, ?)",
            (provider, stored_model, latency_ms, error_code, error_message[:500], now),
        )
        _gc_events(conn, provider, stored_model)
        conn.commit()


def record_text_only(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    latency_ms: float = 0.0,
) -> None:
    """Record a text-only response when tools were provided.

    This is a successful API call that failed to produce tool calls,
    so it counts as success for API reliability but failure for tool support.
    """
    key = _make_key(provider, model, base_url)
    # Strip provider prefix for model column storage
    stored_model = model.split("/", 1)[1] if "/" in model else model
    now = time.time()
    with _LOCK:
        conn = _get_conn()
        row = conn.execute(
            "SELECT total_calls, success_calls, failure_calls, text_only_calls, avg_latency_ms FROM model_metrics WHERE key=?",
            (key,),
        ).fetchone()
        if row:
            total = row["total_calls"] + 1
            success = row["success_calls"] + 1  # API call succeeded
            text_only = row["text_only_calls"] + 1
            avg_lat = _rolling_avg_latency(row["avg_latency_ms"], row["total_calls"], latency_ms)
            score = _compute_quality_score(success, row["failure_calls"], text_only, avg_lat)
            conn.execute(
                """UPDATE model_metrics
                   SET total_calls=?, success_calls=?, text_only_calls=?,
                       avg_latency_ms=?, quality_score=?, last_success_at=?, updated_at=?
                   WHERE key=?""",
                (total, success, text_only, avg_lat, score, now, now, key),
            )
        else:
            score = _compute_quality_score(1, 0, 1, latency_ms)
            conn.execute(
                """INSERT INTO model_metrics
                   (key, provider, model, base_url, total_calls, success_calls, text_only_calls,
                    avg_latency_ms, quality_score, last_success_at, updated_at)
                   VALUES (?, ?, ?, ?, 1, 1, 1, ?, ?, ?, ?)""",
                (key, provider, stored_model, base_url, latency_ms, score, now, now),
            )
        conn.execute(
            "INSERT INTO model_events (provider, model, event_type, latency_ms, created_at) VALUES (?, ?, 'text_only', ?, ?)",
            (provider, stored_model, latency_ms, now),
        )
        _gc_events(conn, provider, stored_model)
        conn.commit()


# ── Querying ───────────────────────────────────────────────────────────

def get_quality_score(provider: str, model: str, base_url: str = "") -> float:
    """Return the quality score (0-100) for a model, or 50.0 if unknown.

    When base_url is empty, computes a decayed score from recent events
    (exponential decay with configurable half-life, default 24h) and updates
    the cached quality_score.  This ensures recent performance matters more
    than historical failures (e.g. peak-hour rate limiting).

    When base_url is provided, returns the cached score for that endpoint.
    """
    with _LOCK:
        conn = _get_conn()
        if base_url:
            key = _make_key(provider, model, base_url)
            row = conn.execute(
                "SELECT quality_score FROM model_metrics WHERE key=?", (key,)
            ).fetchone()
            return row["quality_score"] if row else 50.0
        # No base_url — compute decayed score from events, then return best
        stripped = model.split("/", 1)[1] if "/" in model else model
        result = _compute_decayed_score(provider, stripped, conn)
        if result is not None:
            decayed_score = result[3]
            # Update cached score for all base_urls of this model
            conn.execute(
                "UPDATE model_metrics SET quality_score=?, updated_at=? WHERE provider=? AND model=?",
                (decayed_score, time.time(), provider, stripped),
            )
            conn.commit()
            return decayed_score
        # Fallback to cached score
        rows = conn.execute(
            "SELECT quality_score FROM model_metrics WHERE provider=? AND model=? ORDER BY quality_score DESC",
            (provider, stripped),
        ).fetchall()
        return rows[0]["quality_score"] if rows else 50.0

def get_text_only_rate(provider: str, model: str, base_url: str = "") -> float:
    """Return the text-only rate (0.0-1.0) for a model, or 0.0 if unknown.

    A low rate means the model reliably produces tool calls when tools are
    provided. A high rate means the model frequently returns text instead.
    """
    key = _make_key(provider, model, base_url)
    with _LOCK:
        conn = _get_conn()
        row = conn.execute(
            "SELECT text_only_calls, total_calls FROM model_metrics WHERE key=?", (key,)
        ).fetchone()
        if not row or (row["total_calls"] or 0) < 3:
            return 0.0  # Unknown / insufficient data — assume good
        return (row["text_only_calls"] or 0) / (row["total_calls"] or 1)


def get_all_quality_scores() -> Dict[str, Dict]:
    """Return all model metrics as a dict keyed by model key."""
    with _LOCK:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM model_metrics ORDER BY quality_score DESC"
        ).fetchall()
        result = {}
        for row in rows:
            result[row["key"]] = dict(row)
        return result


def get_quality_sorted_models(models: List[str]) -> List[str]:
    """Return models sorted by quality score (best first).

    Models with no data get score 50.0 (neutral).
    """
    scores = {}
    for model in models:
        if "/" not in model:
            scores[model] = 50.0
            continue
        provider = model.split("/")[0]
        scores[model] = get_quality_score(provider, model)

    return sorted(models, key=lambda m: scores.get(m, 50.0), reverse=True)


def get_model_events(
    provider: str,
    model: str,
    *,
    limit: int = 20,
) -> List[Dict]:
    """Return recent events for a model."""
    with _LOCK:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM model_events WHERE provider=? AND model=? ORDER BY created_at DESC LIMIT ?",
            (provider, model, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def reset_model(provider: str, model: str, base_url: str = "") -> None:
    """Reset metrics for a specific model."""
    key = _make_key(provider, model, base_url)
    stored_model = model.split("/", 1)[1] if "/" in model else model
    with _LOCK:
        conn = _get_conn()
        conn.execute("DELETE FROM model_metrics WHERE key=?", (key,))
        conn.execute("DELETE FROM model_events WHERE provider=? AND model IN (?, ?)",
                     (provider, model, stored_model))
        conn.commit()


def reset_all() -> None:
    """Reset all metrics."""
    with _LOCK:
        conn = _get_conn()
        conn.execute("DELETE FROM model_metrics")
        conn.execute("DELETE FROM model_events")
        conn.commit()


# ── Internal ───────────────────────────────────────────────────────────

def _make_key(provider: str, model: str, base_url: str = "") -> str:
    # Strip provider prefix if model already includes it (e.g. "opencode-go/mimo-v2.5")
    if "/" in model:
        model = model.split("/", 1)[1]
    if base_url:
        return f"{provider}/{model}::{base_url}"
    return f"{provider}/{model}"


def _gc_events(conn: sqlite3.Connection, provider: str, model: str) -> None:
    """Keep only the last _QUALITY_WINDOW events per provider/model."""
    conn.execute(
        """DELETE FROM model_events
           WHERE provider=? AND model=? AND id NOT IN (
               SELECT id FROM model_events WHERE provider=? AND model=?
               ORDER BY created_at DESC LIMIT ?
           )""",
        (provider, model, provider, model, _QUALITY_WINDOW),
    )
