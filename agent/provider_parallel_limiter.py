"""Provider parallel stream limit enforcement.

Some providers (e.g. arliai) enforce hard limits on concurrent streams.
This module provides per-provider semaphore tracking so that requests
wait or fail-fast depending on the provider's limit policy.

Usage:
    from agent.provider_parallel_limiter import acquire_stream, release_stream

    if not acquire_stream("arliai"):
        # Provider at capacity — return 429 immediately or queue
    try:
        # ... make API call ...
    finally:
        release_stream("arliai")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Default parallel limits per provider.
# Providers not listed here default to unlimited (0).
_DEFAULT_LIMITS: Dict[str, int] = {
    "arliai": 2,          # arliai enforces max 2 concurrent streams
    # Add other providers here as limits are discovered.
}

# Env var override pattern: HERMES_PARALLEL_LIMIT_<PROVIDER>
# e.g. HERMES_PARALLEL_LIMIT_ARLIAI=2
# Set to 0 to disable limiting for a provider.

_LIMITS: Dict[str, int] = {}
_SEMAPHORES: Dict[str, threading.Semaphore] = {}
_LOCK = threading.Lock()

# Track active count per provider for monitoring/debugging
_ACTIVE_COUNTS: Dict[str, int] = {}
_ACTIVE_LOCK = threading.Lock()


def _resolve_limit(provider: str) -> int:
    """Resolve the parallel limit for a provider.

    Check env var HERMES_PARALLEL_LIMIT_<PROVIDER> first, then
    _DEFAULT_LIMITS, then 0 (unlimited).
    """
    env_key = f"HERMES_PARALLEL_LIMIT_{provider.upper().replace('-', '_')}"
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        try:
            return max(0, int(env_val))
        except ValueError:
            pass
    return _DEFAULT_LIMITS.get(provider, 0)


def _get_semaphore(provider: str) -> threading.Semaphore:
    """Get or create the semaphore for a provider."""
    limit = _resolve_limit(provider)
    with _LOCK:
        if provider not in _SEMAPHORES:
            _SEMAPHORES[provider] = threading.Semaphore(limit if limit > 0 else 1_000_000_000)
        return _SEMAPHORES[provider]


def acquire_stream(
    provider: str,
    timeout: float = 0.0,
    wait: bool = True,
) -> bool:
    """Attempt to acquire a parallel stream slot for a provider.

    Args:
        provider: Provider name (e.g. "arliai", "opencode-go")
        timeout: Maximum seconds to wait for a slot. 0 = no wait.
        wait: If False, return False immediately when no slot is available
              instead of blocking.

    Returns:
        True if the slot was acquired. Caller MUST call release_stream()
        when done, even on exception.

        False if wait=True and timeout elapsed, or wait=False and no
        slot was immediately available. Caller must NOT call release_stream()
        in this case.
    """
    limit = _resolve_limit(provider)
    if limit == 0:
        # Unlimited — always succeed
        return True

    sem = _get_semaphore(provider)
    acquired = sem.acquire(timeout=timeout if not wait else 0.0)
    if acquired:
        with _ACTIVE_LOCK:
            _ACTIVE_COUNTS[provider] = _ACTIVE_COUNTS.get(provider, 0) + 1
        logger.debug(
            "[parallel_limiter] acquired slot for provider=%s (active=%d limit=%d)",
            provider,
            _ACTIVE_COUNTS.get(provider, 0),
            limit,
        )
    else:
        logger.warning(
            "[parallel_limiter] no slot available for provider=%s (active=%d limit=%d wait=%.1fs)",
            provider,
            _ACTIVE_COUNTS.get(provider, 0),
            limit,
            timeout,
        )
    return acquired


def release_stream(provider: str) -> None:
    """Release a previously acquired stream slot.

    Must be called exactly once for each successful acquire_stream() call.
    """
    limit = _resolve_limit(provider)
    if limit == 0:
        return

    with _ACTIVE_LOCK:
        count = _ACTIVE_COUNTS.get(provider, 0)
        if count > 0:
            _ACTIVE_COUNTS[provider] = count - 1
        else:
            logger.warning(
                "[parallel_limiter] release_stream called but no active slots for provider=%s",
                provider,
            )

    _get_semaphore(provider).release()

    logger.debug(
        "[parallel_limiter] released slot for provider=%s (active=%d limit=%d)",
        provider,
        _ACTIVE_COUNTS.get(provider, 0),
        limit,
    )


def active_count(provider: str) -> int:
    """Return the number of currently active streams for a provider."""
    with _ACTIVE_LOCK:
        return _ACTIVE_COUNTS.get(provider, 0)


def cooldown_seconds_for_parallel_limit(provider: str) -> float:
    """Return how long to cooldown when parallel limit is hit.

    When a provider returns 429 due to parallel limit, we cooldown
    briefly so subsequent requests wait rather than immediately retry.
    """
    env_key = f"HERMES_PARALLEL_COOLDOWN_{provider.upper().replace('-', '_')}"
    env_val = os.getenv(env_key, "").strip()
    if env_val:
        try:
            return max(1.0, float(env_val))
        except ValueError:
            pass
    # Default: 30s cooldown for parallel limit violations
    return 30.0


def reset_provider(provider: Optional[str] = None) -> None:
    """Reset tracking state for a provider (for testing/admin)."""
    with _LOCK:
        if provider:
            _SEMAPHORES.pop(provider, None)
            with _ACTIVE_LOCK:
                _ACTIVE_COUNTS.pop(provider, None)
        else:
            _SEMAPHORES.clear()
            with _ACTIVE_LOCK:
                _ACTIVE_COUNTS.clear()