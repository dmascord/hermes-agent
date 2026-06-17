"""Background sweeper for Codex credential pool token refresh.

Periodically walks the ``openai-codex`` credential pool and attempts
to refresh entries whose tokens were server-side invalidated (HTTP 401
``token_invalidated``).  This prevents the pool from getting stuck with
``exhausted`` entries whose JWTs are still technically valid but have
been rejected by the upstream, and whose refresh tokens can produce a
fresh working token.

The sweeper runs as an asyncio task started by the API server at startup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace

logger = logging.getLogger(__name__)

_SWEEP_INTERVAL_SECONDS = 300.0  # 5 minutes
_COOLDOWN_SECONDS_ON_FAILURE = 86400.0  # 24 h — don't hammer a dead refresh token


async def _codex_token_sweeper_task() -> None:
    """Background coroutine: sweep the Codex pool every 5 minutes.

    For each pool entry whose last_error_code is 401 and whose refresh_token
    is present, attempt an OAuth token refresh.  On success, clear the
    exhausted state.  On failure, set a 24-hour cooldown.

    Runs inside the event loop as a fire-and-forget task.  The synchronous
    sweep body is dispatched to a thread-pool executor so it does not block
    the event loop during httpx calls.
    """
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _sweep_codex_pool_once)
        except Exception as exc:
            logger.warning(
                "codex_sweeper: unhandled error in sweep cycle: %s", exc,
            )


def _sweep_codex_pool_once() -> None:
    """Synchronous sweep body — one pass over the Codex credential pool."""
    try:
        from agent.credential_pool import load_pool
        from hermes_cli.auth import refresh_codex_oauth_pure
    except Exception as exc:
        logger.debug("codex_sweeper: imports failed: %s", exc)
        return

    try:
        pool = load_pool("openai-codex")
    except Exception as exc:
        logger.debug("codex_sweeper: cannot load openai-codex pool: %s", exc)
        return

    refreshed_count = 0
    for entry in pool.entries():
        if entry.last_error_code != 401 or not entry.refresh_token:
            continue

        label = entry.label or "?"

        # Cooldown after previous failed refresh
        reset_at = entry.last_error_reset_at or 0.0
        if reset_at > time.time():
            continue

        logger.info(
            "codex_sweeper: attempting refresh for pool entry %s (%s)",
            label, entry.id,
        )
        try:
            refreshed = refresh_codex_oauth_pure(
                entry.access_token or "",
                entry.refresh_token,
            )
            new_at = refreshed.get("access_token", "").strip()
            new_rt = refreshed.get("refresh_token", "").strip()
            if not new_at:
                logger.warning(
                    "codex_sweeper: refresh for %s returned empty access_token",
                    label,
                )
                pool._mark_exhausted(entry, 401, {"reason": "refresh_empty_token"})
                continue

            # Update entry with fresh tokens and clear exhausted state
            updated = replace(
                entry,
                access_token=new_at,
                refresh_token=new_rt,
                last_refresh=refreshed.get("last_refresh"),
                expires_at_ms=refreshed.get("expires_at_ms"),
                last_status=None,
                last_status_at=None,
                last_error_code=None,
                last_error_reason=None,
                last_error_message=None,
                last_error_reset_at=None,
            )
            pool._replace_entry(entry, updated)
            pool._persist()
            # Sync refreshed tokens back to auth.json so
            # _seed_from_singletons() on the next load_pool() sees
            # fresh state instead of re-seeding stale tokens.
            if updated.source == "device_code":
                pool._sync_device_code_entry_to_auth_store(updated)
            refreshed_count += 1
            logger.info(
                "codex_sweeper: refresh SUCCESS for %s (new token expires_at_ms=%s)",
                label, refreshed.get("expires_at_ms"),
            )
        except Exception as exc:
            logger.warning(
                "codex_sweeper: refresh FAILED for %s (%s: %s) — "
                "setting 24h cooldown",
                label, type(exc).__name__, exc,
            )
            pool._mark_exhausted(
                entry, 401,
                {"reason": f"refresh_failed:{exc!s}",
                 "reset_at": time.time() + _COOLDOWN_SECONDS_ON_FAILURE},
            )

    if refreshed_count:
        logger.info("codex_sweeper: refreshed %d exhausted token(s)", refreshed_count)
    else:
        logger.debug("codex_sweeper: no stale tokens found")