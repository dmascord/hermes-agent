"""Persistent provider/model cooldown tracking.

Stores upstream cooldown windows in SQLite so future sessions can avoid
re-selecting a provider/model pair until the provider's Retry-After window
has elapsed.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


class ModelCooldownDB:
    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path or (get_hermes_home() / "cooldowns.db"))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_cooldowns (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                base_url TEXT,
                cooldown_until REAL NOT NULL,
                reason TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (provider, model, base_url)
            )
            """
        )
        self._conn.commit()

    def record(self, provider: str, model: str, *, cooldown_until: float, base_url: str = "", reason: str = "") -> None:
        if not provider or not model or cooldown_until <= time.time():
            return
        self._conn.execute(
            """
            INSERT INTO model_cooldowns(provider, model, base_url, cooldown_until, reason, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, model, base_url)
            DO UPDATE SET cooldown_until=excluded.cooldown_until,
                          reason=excluded.reason,
                          updated_at=excluded.updated_at
            """,
            (provider, model, base_url or "", float(cooldown_until), reason or "", time.time()),
        )
        self._conn.commit()

    def remaining(self, provider: str, model: str, *, base_url: str = "") -> Optional[float]:
        if not provider or not model:
            return None
        row = self._conn.execute(
            "SELECT cooldown_until FROM model_cooldowns WHERE provider=? AND model=? AND base_url=?",
            (provider, model, base_url or ""),
        ).fetchone()
        if not row:
            return None
        remaining = float(row[0]) - time.time()
        if remaining > 0:
            return remaining
        self.clear(provider, model, base_url=base_url)
        return None

    def clear(self, provider: str, model: str, *, base_url: str = "") -> None:
        if not provider or not model:
            return
        self._conn.execute(
            "DELETE FROM model_cooldowns WHERE provider=? AND model=? AND base_url=?",
            (provider, model, base_url or ""),
        )
        self._conn.commit()


_DB: Optional[ModelCooldownDB] = None


def _db() -> ModelCooldownDB:
    global _DB
    if _DB is None:
        _DB = ModelCooldownDB()
    return _DB


def record_model_cooldown(provider: str, model: str, *, cooldown_until: float, base_url: str = "", reason: str = "") -> None:
    try:
        _db().record(provider, model, cooldown_until=cooldown_until, base_url=base_url, reason=reason)
    except Exception as exc:
        logger.debug("Failed to record model cooldown: %s", exc)


def model_cooldown_remaining(provider: str, model: str, *, base_url: str = "") -> Optional[float]:
    try:
        return _db().remaining(provider, model, base_url=base_url)
    except Exception as exc:
        logger.debug("Failed to read model cooldown: %s", exc)
        return None


def clear_model_cooldown(provider: str, model: str, *, base_url: str = "") -> None:
    try:
        _db().clear(provider, model, base_url=base_url)
    except Exception as exc:
        logger.debug("Failed to clear model cooldown: %s", exc)
