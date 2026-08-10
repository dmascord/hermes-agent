"""
OpenAI-compatible API server platform adapter.

Exposes an HTTP server with endpoints:
- POST /v1/chat/completions        — OpenAI Chat Completions format (stateless; opt-in session continuity via X-Hermes-Session-Id header)
- POST /v1/responses               — OpenAI Responses API format (stateful via previous_response_id)
- GET  /v1/responses/{response_id} — Retrieve a stored response
- DELETE /v1/responses/{response_id} — Delete a stored response
- GET  /v1/models                  — lists hermes-agent as an available model
- POST /v1/runs                    — start a run, returns run_id immediately (202)
- GET  /v1/runs/{run_id}/events    — SSE stream of structured lifecycle events
- GET  /health                     — health check
- GET  /health/detailed            — rich status for cross-container dashboard probing

Any OpenAI-compatible frontend (Open WebUI, LobeChat, LibreChat,
AnythingLLM, NextChat, ChatBox, etc.) can connect to hermes-agent
through this adapter by pointing at http://localhost:8642/v1.

Requires:
- aiohttp (already available in the gateway)
"""

import asyncio
import ast
import hashlib
import hmac
import ipaddress
import json
import logging
import os
# Add timestamp formatting to both root and this module's logger.
_fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
if not logging.root.handlers:
    _root_sh = logging.StreamHandler()
    _root_sh.setFormatter(_fmt)
    logging.root.addHandler(_root_sh)
    logging.root.setLevel(logging.INFO)
# Also ensure this module's logger uses timestamps
_gw_logger = logging.getLogger("gateway.platforms.api_server")
if not _gw_logger.handlers:
    _gw_sh = logging.StreamHandler()
    _gw_sh.setFormatter(_fmt)
    _gw_logger.addHandler(_gw_sh)
else:
    for _h in _gw_logger.handlers:
        _h.setFormatter(_fmt)
import socket as _socket
import re
import sqlite3
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Module-level timestamp for correlating per-request logs across the
# many entry points (chat_completions, /pending-tool-calls, /tool_responses).
# Reset at the start of each request; not used as a true global.
_REQ_START_TS: dict[str, float] = {}
_req_start_global = time.monotonic()

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
)
logger = logging.getLogger(__name__)

def _cooldown_seconds_for_429(exc: Exception) -> float:
    """Return an appropriate cooldown duration given a 429 exception.

    Inspects the Retry-After response header first, then falls back to
    keyword matching on the error message body so that quota-exhaustion
    errors (weekly / daily / N-hour limits) are cooled down for the full
    reset window instead of the default 10-minute generic cooldown.
    """
    import os as _os
    import re as _re
    import time as _time
    from datetime import datetime as _datetime, timezone as _timezone, timedelta as _timedelta

    has_response = getattr(exc, "response", None) is not None
    status_code = _exception_status_code(exc)
    is_http_429 = status_code == 429

    # 1. Honour Retry-After header if the underlying HTTP response is attached.
    # Check error body first to determine if this is a weekly/daily quota.
    body = str(exc).lower()
    _is_quota_limit = any(kw in body for kw in ["weekly", "daily", "usage limit", "limitname", "session limit"])
    _max = float(_os.getenv("HERMES_MAX_CIRCUIT_BREAKER_COOLDOWN", "0") or "0")
    try:
        retry_after = exc.response.headers.get("Retry-After") or exc.response.headers.get("retry-after")  # type: ignore[union-attr]
        if retry_after:
            _val = max(60.0, float(retry_after))
            # For weekly/daily quotas, respect the full Retry-After value (providers like
            # opencode-go return the exact reset time in this header). Only cap generic
            # rate limits which have small Retry-After values (typically < 3600s).
            if _is_quota_limit:
                return _val
            return _val if _max <= 0 else min(_val, _max)
    except Exception:
        pass

    # 2. Check for X-RateLimit-Reset header (Unix timestamp).
    try:
        reset_ts = exc.response.headers.get("X-RateLimit-Reset") or exc.response.headers.get("x-ratelimit-reset")  # type: ignore[union-attr]
        if reset_ts:
            _remaining = max(60.0, float(reset_ts) - _time.time())
            return _remaining
    except Exception:
        pass

    # Synthetic RuntimeError(text) calls from _skip_provider_exhaustion_content()
    # do not carry HTTP headers. Keep those cooldowns conservative instead of
    # interpreting arbitrary assistant text as an exact reset window.
    #
    # Bare substring matches on "weekly"/"day" are false-positive prone (any
    # message containing "this week" or "yesterday" would match). Require a
    # quota/limit context keyword before treating the body as a multi-day
    # rate-limit signal, and log the body so future false positives are
    # diagnosable from logs.
    _quota_kw = ("quota", "limit", "exceeded", "exhausted", "reset", "tokens")
    _has_quota_ctx = any(kw in body for kw in _quota_kw)
    if not has_response and not is_http_429:
        if _has_quota_ctx and ("weekly" in body or "week" in body):
            logger.warning(
                "[_cooldown_seconds_for_429] non-HTTP 24h cooldown (weekly hint) body=%r",
                body[:400],
            )
            return 24 * 3600.0
        if _has_quota_ctx and ("daily" in body or "session limit" in body):
            logger.warning(
                "[_cooldown_seconds_for_429] non-HTTP 1h cooldown (daily hint) body=%r",
                body[:400],
            )
            return 3600.0
        _reset_match = _re.search(r'(?:reset in|retry in|retry_after|remaining)[:\s]*(\d+)\s*(second|minute|hour|day|week)s?', body)
        if _reset_match:
            _amount = int(_reset_match.group(1))
            _unit = _reset_match.group(2)
            _multipliers = {"second": 1, "minute": 60, "hour": 3600}
            _raw = _amount * _multipliers.get(_unit, 3600)
            return _raw if _max <= 0 else min(_raw, _max)
        _hour_match = _re.search(r'(\d+)\s*-?\s*hour', body)
        if _hour_match:
            _raw = max(3600.0, int(_hour_match.group(1)) * 3600.0)
            return _raw if _max <= 0 else min(_raw, _max)
        if "hour" in body:
            return 3600.0
        return 600.0

    # 3. For real HTTP errors, parse body hints more aggressively.
    # Common formats: "reset in 5 days", "5 days remaining", "retry in 24 hours"
    _reset_match = _re.search(r'(?:reset in|retry in|retry_after|remaining)[:\s]*(\d+)\s*(second|minute|hour|day|week)s?', body)
    if _reset_match:
        _amount = int(_reset_match.group(1))
        _unit = _reset_match.group(2)
        _multipliers = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800}
        _raw = _amount * _multipliers.get(_unit, 3600)
        # Only cap hourly limits — weekly/daily limits should be respected.
        return _raw if _max <= 0 else min(_raw, _max)

    if _has_quota_ctx and ("weekly" in body or "week" in body):
        # Weekly limit — respect the full window, don't cap it.
        logger.warning(
            "[_cooldown_seconds_for_429] HTTP 7-day cooldown (weekly hint) body=%r",
            body[:400],
        )
        return 7 * 24 * 3600.0
    if _has_quota_ctx and ("daily" in body or "day" in body):
        # Daily limit — respect the full window, don't cap it.
        logger.warning(
            "[_cooldown_seconds_for_429] HTTP 24h cooldown (daily hint) body=%r",
            body[:400],
        )
        return 24 * 3600.0
    # "5 hour", "5-hour", "5hour" etc.
    _hour_match = _re.search(r'(\d+)\s*-?\s*hour', body)
    if _hour_match:
        _raw = max(3600.0, int(_hour_match.group(1)) * 3600.0)
        # Only cap hourly limits — they might be overly aggressive.
        return _raw if _max <= 0 else min(_raw, _max)
    if "hour" in body:
        return 3600.0
    # ChatGPT/Codex quota errors can say "resets 5am (UTC)" without a
    # numeric duration. Cool down until the next stated UTC reset time.
    _utc_reset = _re.search(r"resets?\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*\(?utc\)?", body)
    if _utc_reset:
        _hour = int(_utc_reset.group(1))
        _minute = int(_utc_reset.group(2) or "0")
        _ampm = _utc_reset.group(3)
        if _ampm == "pm" and _hour != 12:
            _hour += 12
        elif _ampm == "am" and _hour == 12:
            _hour = 0
        if 0 <= _hour <= 23 and 0 <= _minute <= 59:
            _now = _datetime.now(_timezone.utc)
            _reset_at = _now.replace(hour=_hour, minute=_minute, second=0, microsecond=0)
            if _reset_at <= _now:
                _reset_at += _timedelta(days=1)
            return max(60.0, (_reset_at - _now).total_seconds())

    # 4. Generic rate-limit — 10 minutes.
    return 600.0


def _exception_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    match = re.search(r"\bHTTP\s+(\d{3})\b|\bstatus(?:_code)?[=:]\s*(\d{3})\b", str(exc), re.IGNORECASE)
    if match:
        try:
            return int(match.group(1) or match.group(2))
        except Exception:
            return None
    return None


def _is_provider_exhaustion_error(exc: Exception) -> bool:
    """Return True for quota/session-limit errors that should fall through."""
    msg = str(exc or "").lower()
    status_code = _exception_status_code(exc)
    if status_code == 429:
        return True
    exhaustion_markers = (
        "session limit",
        "usage limit",
        "rate limit",
        "rate_limit",
        "quota",
        "quota_exceeded",
        "insufficient_quota",
        "resource_exhausted",
        "too many requests",
        "go_usagelimit",
        "limitname",
        "resets ",
        "retry-after",
        "retry after",
    )
    return any(marker in msg for marker in exhaustion_markers)


def _is_provider_level_quota_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return any(
        marker in msg
        for marker in (
            "weekly",
            "daily",
            "usage limit",
            "session limit",
            "go_usagelimit",
            "limitname",
            "practical_brown",
        )
    )


def _sanitize_passthrough_error_for_client(exc: Exception) -> str:
    if exc and (_is_provider_exhaustion_error(exc) or _is_provider_exhaustion_content(str(exc))):
        return "provider quota or session limit exhausted"
    return str(exc) if exc else ""


def _is_provider_exhaustion_content(content: Any) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    lower = text.lower()

    # Assistant content is much less structured than exceptions, so keep this
    # deliberately narrower than _is_provider_exhaustion_error(). Generic
    # markers like "resets" or "limit" can appear in valid task output.
    if "you've hit your session limit" in lower or "you have hit your session limit" in lower:
        return True
    if "session limit" in lower and re.search(r"\b(reset|resets|try again|available again)\b", lower):
        return True

    auth_markers = (
        "failed to authenticate",
        "invalid authentication credentials",
        "api error: 401",
        "error code: 401",
        "401 unauthorized",
        "401 invalid authentication",
        "authenticationerror",
        "unauthorized. api error",
        "not logged in",
        "please run /login",
        "oauth session expired",
        "session expired and could not be refreshed",
    )
    if any(marker in lower for marker in auth_markers):
        return True

    quota_markers = (
        "token_quota_exceeded",
        "quota_exceeded",
        "rate_limit_exceeded",
        "too many requests",
    )
    if any(marker in lower for marker in quota_markers):
        return True
    if "rate limit" in lower and re.search(r"\b(exceeded|reached|hit|retry|try again)\b", lower):
        return True
    if "quota" in lower and re.search(r"\b(exceeded|exhausted|reached|hit)\b", lower):
        return True

    return False


def _mark_hermes_code_provider_exhausted(
    *,
    provider_model: str,
    runtime_kwargs: Optional[Dict[str, Any]],
    exc: Exception,
    stream: bool,
) -> None:
    provider = provider_model.split("/", 1)[0] if "/" in provider_model else "openai"
    status_code = _exception_status_code(exc)
    cooldown_seconds = _cooldown_seconds_for_429(exc)
    cpool = runtime_kwargs.get("credential_pool") if isinstance(runtime_kwargs, dict) else None

    if cpool is not None:
        try:
            cpool.mark_exhausted_and_rotate(
                status_code=status_code,
                error_context={
                    "reason": "provider_exhausted",
                    "message": str(exc)[:500],
                },
            )
            logger.warning(
                "[hermes-code] %s: rotated credential pool after provider exhaustion (cooldown_hint=%.0fs)",
                provider_model,
                cooldown_seconds,
            )
        except Exception as pool_exc:
            logger.warning("[hermes-code] %s: credential pool rotate failed: %s", provider_model, pool_exc)
        _RUNTIME_KWARGS_CACHE.pop(provider, None)
        _RUNTIME_KWARGS_CACHE_AT.pop(provider, None)
        return

    reason = "hermes_code_stream_provider_exhausted" if stream else "hermes_code_passthrough_provider_exhausted"
    try:
        from agent.model_cooldown_db import mark_model_cooldown

        if _is_provider_level_quota_error(exc):
            for idx in range(1, 50):
                model = os.environ.get(f"HERMES_CODE_FALLBACK_{idx}", "")
                if model and model.startswith(provider + "/"):
                    mark_model_cooldown(
                        provider=provider,
                        model=model,
                        cooldown_seconds=cooldown_seconds,
                        reason=f"{reason}_provider_quota",
                    )
            logger.warning(
                "[hermes-code] provider-level exhaustion for %s; cooled %s fallbacks for %.0fs",
                provider_model,
                provider,
                cooldown_seconds,
            )
        else:
            mark_model_cooldown(
                provider=provider,
                model=provider_model,
                cooldown_seconds=cooldown_seconds,
                reason=reason,
            )
            logger.warning("[hermes-code] %s cooled down for %.0fs after provider exhaustion", provider_model, cooldown_seconds)
    except Exception:
        pass


def _skip_provider_exhaustion_content(
    *,
    provider_model: str,
    runtime_kwargs: Optional[Dict[str, Any]],
    content: Any,
    stream: bool,
) -> None:
    text = str(content or "").strip()
    logger.warning(
        "[hermes-code] %s _skip_provider_exhaustion_content: text_len=%d text=%.1000s",
        provider_model,
        len(text),
        text,
    )
    if not _is_provider_exhaustion_content(content):
        return
    exc = RuntimeError(text)
    logger.warning(
        "[hermes-code] %s matched provider-exhaustion pattern; suppressing content and trying next provider",
        provider_model,
    )
    _mark_hermes_code_provider_exhausted(
        provider_model=provider_model,
        runtime_kwargs=runtime_kwargs,
        exc=exc,
        stream=stream,
    )
    _invalidate_selectable_pool_cache()
    raise _CodexPassthroughSkip("provider_exhaustion_content")


def _invoke_passthrough_hooks(hook_name: str, **kwargs: Any) -> None:
    """Fire a plugin hook from the passthrough path. Always fail-open.

    Ensures plugin discovery has run once per process then delegates to
    ``invoke_hook``.  Importing and calling this is safe even when no plugins
    are loaded or the plugin system is unavailable.
    """
    try:
        from hermes_cli.plugins import invoke_hook, _ensure_plugins_discovered
        _ensure_plugins_discovered()
        invoke_hook(hook_name, **kwargs)
    except Exception:
        pass
# Process-level cache for _runtime_kwargs_for_model_id, keyed by provider prefix
# (e.g. "github-copilot").  Credential resolution for some providers takes
# 1-3s per call; the cache eliminates redundant resolves across the 4-6 calls
# made per request for different swarm-pool models that share credentials.
_RUNTIME_KWARGS_CACHE: Dict[str, Dict[str, Any]] = {}
_RUNTIME_KWARGS_CACHE_AT: Dict[str, float] = {}
_RUNTIME_KWARGS_CACHE_TTL = 86400.0
_RUNTIME_KWARGS_CACHE_EXCLUDED_PROVIDERS = {
    "github-copilot-enterprise",
    "minimax",
    "openai",
    "openai-codex",
    "opencode-go",
    "opencode-zen",
    "zai",
}

# Process-level cache for _build_env_fallback_chain, keyed by env prefix.
# The fallback chain is purely env-var-driven and never changes mid-process,
# but calling resolve_runtime_provider for each of 20+ fallback models takes ~6.7s.
_FALLBACK_CHAIN_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# TTL cache for _ordered_hermes_code_selectable_pool().
# Checking all 25+ models on every request (credential pool + cooldown DB lookups)
# is expensive and generates massive TRACE log spam.  60s TTL means most requests
# hit the cache; the fallback chain still validates cooldown per-attempt at runtime.
_SELECTABLE_POOL_CACHE: List[str] = []
_SELECTABLE_POOL_CACHE_AT: float = 0.0
_SELECTABLE_POOL_CACHE_TTL = 300.0  # 5 min; invalidated on provider failure, passthrough checks cooldown per-attempt


def _invalidate_selectable_pool_cache() -> None:
    """Force next call to _ordered_hermes_code_selectable_pool to re-evaluate all models."""
    global _SELECTABLE_POOL_CACHE  # noqa: PLW0603
    _SELECTABLE_POOL_CACHE = []

# Session-level sticky hermes-code routing. Once a session successfully picks a
# backend, keep using it for subsequent turns unless it becomes unavailable or
# the conversation grows beyond that model's context window.
_HERMES_CODE_SESSION_STICKY: Dict[str, Dict[str, Any]] = {}
_HERMES_CODE_SESSION_STICKY_TTL = 12 * 60 * 60.0
_HERMES_CODE_SESSION_STICKY_MAX = 4096
_HERMES_CODE_STICKY_DISABLED = bool(os.getenv("HERMES_DISABLE_SESSION_STICKY", "").strip())
# Session collision registry.
# Problem: two OMP clients with the same first message + same IP/UA get the
# same base session ID.  The registry resolves this without requiring any
# client-side header by fingerprinting the *full* message history:
#
#   base_key  = sha256(salt + system + first_msg)[:16]  -- stable across turns
#   hist_sig  = sha256(all_message_content)[:12]        -- unique per conversation
#
# On first turn the history is just [user_msg], so two colliding clients that
# happen to start with identical text still collide here.  But from turn 2
# onward their histories diverge, and the registry cleanly separates them.
# For turn-1 collisions the second client gets a '-v{n}' variant suffix.
_SESSION_REGISTRY_LOCK = threading.Lock()
# base_key -> {hist_sig -> session_id}
_SESSION_REGISTRY: Dict[str, Dict[str, str]] = {}
_SESSION_REGISTRY_TTL = 24 * 60 * 60.0
_SESSION_REGISTRY_MAX = 8192
# session_id -> last_seen timestamp (for TTL eviction)
_SESSION_REGISTRY_MTIME: Dict[str, float] = {}
# session_id -> (hist_sig, hist_raw) for prefix matching on multi-turn requests
_SESSION_REGISTRY_LASTHIST: Dict[str, Any] = {}


def _purge_session_registry(now: Optional[float] = None) -> None:
    now = now or time.time()
    with _SESSION_REGISTRY_LOCK:
        dead_sessions = {
            sid for sid, ts in _SESSION_REGISTRY_MTIME.items()
            if now - ts > _SESSION_REGISTRY_TTL
        }
        if not dead_sessions and len(_SESSION_REGISTRY_MTIME) <= _SESSION_REGISTRY_MAX:
            return
        # Evict by TTL
        for sid in dead_sessions:
            _SESSION_REGISTRY_MTIME.pop(sid, None)
            _SESSION_REGISTRY_LASTHIST.pop(sid, None)
        # Prune base_key -> hist_sig -> session_id
        for base_key in list(_SESSION_REGISTRY):
            hist = _SESSION_REGISTRY[base_key]
            for hs in list(hist):
                if hist[hs] in dead_sessions:
                    del hist[hs]
            if not hist:
                del _SESSION_REGISTRY[base_key]
        # Hard cap: evict oldest if still over limit
        if len(_SESSION_REGISTRY_MTIME) > _SESSION_REGISTRY_MAX:
            oldest = sorted(_SESSION_REGISTRY_MTIME, key=_SESSION_REGISTRY_MTIME.__getitem__)
            for sid in oldest[:len(_SESSION_REGISTRY_MTIME) - _SESSION_REGISTRY_MAX]:
                _SESSION_REGISTRY_MTIME.pop(sid, None)
                _SESSION_REGISTRY_LASTHIST.pop(sid, None)


def _resolve_session_id(
    salt: str,
    system_prompt: Optional[str],
    conversation_messages: list,
) -> str:
    """Return a stable, collision-free session ID.

    Uses two fingerprints:
      base_key  = sha256(salt + system + first_msg)[:16]  -- stable across turns
      hist_sig  = sha256(all_messages)[:12]               -- unique per conversation

    A continuing conversation (turn 2+) arrives with its full history, so
    hist_sig is unique and matches the registered entry exactly.

    A new conversation (turn 1) arrives with only one user message.  If
    base_key is already registered, this MUST be a different client — we
    mint a variant.  Two simultaneous turn-1 collisions also get variants.
    """
    # Stable key from first user message (unchanged across turns of same convo)
    first_user = ""
    for m in conversation_messages:
        if isinstance(m, dict) and m.get("role") == "user":
            first_user = _normalize_chat_content(m.get("content", ""))
            break
    base_seed = (salt or "") + "\n" + (system_prompt or "") + "\n" + first_user
    base_key = hashlib.sha256(base_seed.encode()).hexdigest()[:16]

    # History fingerprint: hash ALL message content (role + content of every turn).
    hist_parts = []
    for m in conversation_messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                (p.get("text") or "") if isinstance(p, dict) else str(p)
                for p in content
            )
        hist_parts.append(role + ":" + str(content or ""))
    hist_raw = "|||".join(hist_parts)
    hist_sig = hashlib.sha256(hist_raw.encode()).hexdigest()[:12]

    # Is this a turn-1 request? Only one user message, no assistant turns yet.
    user_turns = sum(1 for m in conversation_messages if isinstance(m, dict) and m.get("role") == "user")
    assistant_turns = sum(1 for m in conversation_messages if isinstance(m, dict) and m.get("role") == "assistant")
    is_turn_one = user_turns == 1 and assistant_turns == 0

    now = time.time()
    _purge_session_registry(now)

    with _SESSION_REGISTRY_LOCK:
        # _SESSION_REGISTRY: base_key -> {hist_sig -> session_id}
        # _SESSION_REGISTRY_LASTHIST: session_id -> last_hist_sig seen
        #
        # Strategy:
        #   Turn-1: always mint a new session. Two clients with the same
        #   first message cannot be distinguished, so we never reuse.
        #
        #   Turn-2+: the current hist_sig covers all prior messages including
        #   assistant responses, so it is unique per conversation. But we
        #   can't match it exactly to the turn-1 hist_sig stored in the bucket.
        #   Instead we check _SESSION_REGISTRY_LASTHIST: if any session's
        #   last-seen hist_sig is a prefix of the current hist_sig content,
        #   it is the parent session. We detect this by storing the raw
        #   history string length so we can check content inclusion cheaply.
        #
        #   Concretely: _SESSION_REGISTRY_LASTHIST[sid] = (hist_sig, hist_raw_len)
        #   A turn-2 request's hist_parts begins with the turn-1 hist_parts,
        #   so we check if the current hist_raw starts with any known session's
        #   last hist_raw prefix.

        bucket = _SESSION_REGISTRY.setdefault(base_key, {})

        if not is_turn_one:
            # Exact match: this precise history was seen before (e.g. retry)
            if hist_sig in bucket:
                sid = bucket[hist_sig]
                _SESSION_REGISTRY_MTIME[sid] = now
                _SESSION_REGISTRY_LASTHIST[sid] = (hist_sig, hist_raw)
                return sid
            # Prefix match: find a session whose last hist_raw is a prefix of ours
            best_sid = None
            best_len = 0
            for sid, (_, stored_raw) in list(_SESSION_REGISTRY_LASTHIST.items()):
                if sid in _SESSION_REGISTRY_MTIME and hist_raw.startswith(stored_raw) and len(stored_raw) > best_len:
                    best_sid = sid
                    best_len = len(stored_raw)
            if best_sid is not None:
                bucket[hist_sig] = best_sid
                _SESSION_REGISTRY_MTIME[best_sid] = now
                _SESSION_REGISTRY_LASTHIST[best_sid] = (hist_sig, hist_raw)
                return best_sid

        # Turn-1 or no prefix match found: mint a new session.
        if not bucket:
            canonical = "api-" + base_key
            sid = canonical
        else:
            sid = "api-" + base_key[:8] + "-" + os.urandom(4).hex()
            logger.info(
                "[api_server] session collision resolved: base=%s is_turn1=%s sid=%s bucket_size=%d",
                base_key, is_turn_one, sid, len(bucket),
            )
        bucket[hist_sig] = sid
        _SESSION_REGISTRY_MTIME[sid] = now
        _SESSION_REGISTRY_LASTHIST[sid] = (hist_sig, hist_raw)
        return sid

_HERMES_CODE_RR_LOCK = threading.Lock()
_HERMES_CODE_RR_INDEX = 0

# ---------------------------------------------------------------------------
# Reasoning-content extraction helper
# ---------------------------------------------------------------------------

def _extract_reasoning_content_from_msg(msg: Any) -> str:
    """Return reasoning/thinking text from a response message object.

    Handles three provider shapes:
    - Top-level attribute: ``msg.reasoning_content`` (DeepSeek flash/R1, Kimi, GLM)
    - Top-level attribute: ``msg.reasoning`` (some OpenRouter wrappers)
    - Typed content-block list (DeepSeek V4 Pro thinking mode):
        ``msg.content = [{"type": "thinking", "thinking": "..."}, ...]``

    The content-block shape is the one that previously caused the gateway to
    silently drop thinking text, leading to an HTTP 400 on the next turn
    ("content[].thinking must be passed back to the API").

    Returns ``""`` when no reasoning text is found.
    """
    for field in ("reasoning_content", "reasoning"):
        val = getattr(msg, field, None)
        if val and isinstance(val, str):
            return val
    # Typed-block list fallback (DeepSeek V4 Pro / compatible providers)
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = (block.get("thinking") or block.get("text") or "").strip()
                if text and text not in parts:
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Reranker round-robin router with dynamic cooldown
# ---------------------------------------------------------------------------

_RERANKER_COHERE_URL = "https://api.cohere.com/v2/rerank"
_RERANKER_VOYAGE_URL = "https://api.voyageai.com/v1/rerank"
_RERANKER_JINA_URL   = "https://api.jina.ai/v1/rerank"
_RERANKER_DEFAULT_COHERE_MODEL = "rerank-v3.5"
_RERANKER_DEFAULT_VOYAGE_MODEL = "rerank-2"
_RERANKER_DEFAULT_JINA_MODEL   = "jina-reranker-v2-base-multilingual"
_RERANKER_TIMEOUT = 30.0


class _RerankProvider:
    """Immutable descriptor for a reranker backend."""
    __slots__ = ("name", "url", "api_key", "model")

    def __init__(self, name: str, url: str, api_key: str, model: str) -> None:
        self.name = name
        self.url = url
        self.api_key = api_key
        self.model = model


class _RerankRouter:
    """Round-robin reranker router with per-provider dynamic cooldown.

    Providers rotate on each successful call.  On 429 the offending provider
    is cooled down using the shared model_cooldown_db, respecting Retry-After
    when present (defaults to 60 s — safe for Cohere's 10 req/min window).
    The router tries every available provider before surfacing an error.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cursor = 0
        self._providers: List["_RerankProvider"] = []
        self._built = False

    def _build(self) -> None:
        """Populate provider list from env (called once under lock)."""
        cohere_key = os.getenv("COHERE_API_KEY", "").strip()
        cohere_model = os.getenv("HERMES_RERANKER_COHERE_MODEL", _RERANKER_DEFAULT_COHERE_MODEL).strip()
        voyage_key = os.getenv("VOYAGE_API_KEY", "").strip()
        voyage_model = os.getenv("HERMES_RERANKER_VOYAGE_MODEL", _RERANKER_DEFAULT_VOYAGE_MODEL).strip()
        jina_key = os.getenv("JINA_API_KEY", "").strip()
        jina_model = os.getenv("HERMES_RERANKER_JINA_MODEL", _RERANKER_DEFAULT_JINA_MODEL).strip()
        providers = []
        if cohere_key:
            providers.append(_RerankProvider("cohere", _RERANKER_COHERE_URL, cohere_key, cohere_model))
        if voyage_key:
            providers.append(_RerankProvider("voyage", _RERANKER_VOYAGE_URL, voyage_key, voyage_model))
        if jina_key:
            providers.append(_RerankProvider("jina", _RERANKER_JINA_URL, jina_key, jina_model))
        self._providers = providers
        self._built = True
        logger.info(
            "[reranker] initialized with %d provider(s): %s",
            len(providers),
            [p.name for p in providers],
        )

    def _providers_snapshot(self) -> "List[_RerankProvider]":
        with self._lock:
            if not self._built:
                self._build()
            return list(self._providers)

    def _is_cooled_down(self, provider: "_RerankProvider") -> bool:
        try:
            from agent.model_cooldown_db import model_cooldown_remaining
            return model_cooldown_remaining(
                provider=provider.name,
                model=provider.model,
                base_url=provider.url,
            ) > 0.0
        except Exception:
            return False

    def _apply_cooldown(self, provider: "_RerankProvider", seconds: float, reason: str) -> None:
        try:
            from agent.model_cooldown_db import mark_model_cooldown
            mark_model_cooldown(
                provider=provider.name,
                model=provider.model,
                base_url=provider.url,
                reason=reason,
                cooldown_seconds=seconds,
            )
            logger.warning(
                "[reranker] provider=%s cooled down for %.0fs reason=%s",
                provider.name, seconds, reason,
            )
        except Exception as exc:
            logger.debug("[reranker] could not mark cooldown for %s: %s", provider.name, exc)

    def _cooldown_seconds_from_headers(self, headers: Dict[str, str]) -> float:
        """Parse Retry-After; fall back to 60 s (Cohere 10 req/min window)."""
        retry_after = headers.get("retry-after") or headers.get("Retry-After") or ""
        if retry_after:
            try:
                # Honor the actual value — no hard cap. Use
                # HERMES_MAX_CIRCUIT_BREAKER_COOLDOWN env var to cap if needed.
                import os as _os
                _max = float(_os.getenv("HERMES_MAX_CIRCUIT_BREAKER_COOLDOWN", "0") or "0")
                _val = max(1.0, float(retry_after))
                return _val if _max <= 0 else min(_val, _max)
            except ValueError:
                pass
        return 60.0

    def _advance_cursor(self) -> None:
        with self._lock:
            n = len(self._providers)
            if n:
                self._cursor = (self._cursor + 1) % n

    async def rerank(
        self,
        query: str,
        documents: List[Any],
        top_n: Optional[int],
        model_override: Optional[str],
    ) -> Dict[str, Any]:
        """Call reranker APIs in round-robin order, falling back on 429/error.

        Returns a Cohere-wire-format dict with ``results`` and ``model``.
        Raises RuntimeError when no provider succeeds.
        """
        import aiohttp as _aiohttp

        providers = self._providers_snapshot()
        if not providers:
            raise RuntimeError(
                "No reranker providers configured — set COHERE_API_KEY, VOYAGE_API_KEY, and/or JINA_API_KEY"
            )

        with self._lock:
            start = self._cursor % len(providers)
        ordered = providers[start:] + providers[:start]
        last_error: Optional[Exception] = None

        for provider in ordered:
            if self._is_cooled_down(provider):
                logger.debug("[reranker] skipping cooled-down provider=%s", provider.name)
                continue

            effective_model = model_override or provider.model
            body: Dict[str, Any] = {
                "model": effective_model,
                "query": query,
                "documents": documents,
                "return_documents": False,
            }
            if top_n is not None:
                body["top_n"] = top_n

            try:
                timeout = _aiohttp.ClientTimeout(total=_RERANKER_TIMEOUT)
                async with _aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        provider.url,
                        json=body,
                        headers={
                            "Authorization": f"Bearer {provider.api_key}",
                            "Content-Type": "application/json",
                        },
                    ) as resp:
                        resp_body = await resp.text()
                        if resp.status == 429:
                            cooldown = self._cooldown_seconds_from_headers(dict(resp.headers))
                            self._apply_cooldown(provider, cooldown, "rate_limit_429")
                            last_error = RuntimeError(
                                f"provider {provider.name} rate-limited (429)"
                            )
                            continue

                        if resp.status >= 400:
                            last_error = RuntimeError(
                                f"provider {provider.name} HTTP {resp.status}: {resp_body[:200]}"
                            )
                            self._apply_cooldown(provider, 10.0, f"http_{resp.status}")
                            continue

                        try:
                            result = json.loads(resp_body)
                        except json.JSONDecodeError:
                            last_error = RuntimeError(
                                f"provider {provider.name} non-JSON response: {resp_body[:200]}"
                            )
                            continue

                        result.setdefault("model", effective_model)
                        # Voyage uses "data" key; Cohere uses "results" — normalise to "results"
                        if "results" not in result and "data" in result:
                            result["results"] = result["data"]
                        result.setdefault("results", [])
                        self._advance_cursor()
                        logger.info(
                            "[reranker] provider=%s model=%s docs=%d top_result_score=%.4f",
                            provider.name, effective_model, len(documents),
                            result["results"][0].get("relevance_score", 0) if result["results"] else 0,
                        )
                        return result

            except Exception as exc:
                last_error = exc
                logger.warning("[reranker] provider=%s request failed: %s", provider.name, exc)
                continue

        raise RuntimeError(f"All reranker providers exhausted. Last error: {last_error}")


_rerank_router = _RerankRouter()


# Dynamic role aliases exposed by Hermes Gateway. These are virtual models
# from the client's point of view, but they route through hermes-swarm using
# role-specific routing hints instead of a fixed backend mapping.
ROLE_ALIAS_CONFIG = {
    "hermes-gateway/hermes-translator": {
        "mode": "swarm",
        "hint": {
            "role": "translator",
            "task_type": "translation",
            "recommended_tier": "premium",
        },
    },
    "hermes-gateway/hermes-triage": {
        "mode": "swarm",
        "hint": {
            "role": "triage",
            "task_type": "triage",
            "recommended_tier": "balanced",
        },
    },
    "hermes-gateway/hermes-duplicate-pr": {
        "mode": "swarm",
        "hint": {
            "role": "duplicate-pr",
            "task_type": "repo_review",
            "recommended_tier": "balanced",
        },
    },
    "hermes-gateway/hermes-fast": {
        "mode": "swarm",
        "hint": {
            "role": "fast",
            "task_type": "general",
            "recommended_tier": "cheap",
        },
    },
    "hermes-gateway/roo-architect": {
        "mode": "swarm",
        "hint": {
            "role": "roo-architect",
            "task_type": "planning",
            "recommended_tier": "cheap",
            "action_mode": "plan_only",
        },
    },
    "hermes-gateway/roo-ask": {
        "mode": "swarm",
        "hint": {
            "role": "roo-ask",
            "task_type": "qa",
            "recommended_tier": "cheap",
            "action_mode": "answer_only",
        },
    },
    "hermes-gateway/roo-debug": {
        "mode": "swarm",
        "hint": {
            "role": "roo-debug",
            "task_type": "debugging",
            "recommended_tier": "balanced",
        },
    },
    "hermes-gateway/hermes-balanced": {
        "mode": "swarm",
        "hint": {
            "role": "balanced",
            "task_type": "general",
            "recommended_tier": "balanced",
        },
    },
    "hermes-gateway/hermes-reflect": {
        "mode": "swarm",
        "hint": {
            "role": "reflect",
            "task_type": "synthesis",
            "recommended_tier": "cheap",
            "max_tokens": 4096,
        },
    },
}


def _get_role_alias_config(model: str) -> Optional[Dict[str, Any]]:
    raw = str(model or "").strip().lower()
    if not raw:
        return None
    for alias, cfg in ROLE_ALIAS_CONFIG.items():
        if alias.lower() == raw:
            return cfg
    return None


def _resolve_role_alias(model: str) -> str | None:
    """Legacy fixed mapping lookup.

    Dynamic aliases defined in ROLE_ALIAS_CONFIG are handled by swarm routing
    and intentionally do not resolve to a single concrete backend here.
    """
    if _get_role_alias_config(model):
        return None
    return None



_SWARM_PREMIUM_MODEL_HINTS = (
    # GHE Copilot Enterprise — strongest models, requires copilot-integration-id header
    # (automatically set by copilot_request_headers() for copilot-api.* base URLs).
    # Claude via /v1/messages, GPT-5 via /v1.responses, GPT-4o via /chat/completions.
    "github-copilot-enterprise/claude-sonnet-4.6",
    "github-copilot-enterprise/gpt-5.5",
    "github-copilot-enterprise/gpt-5.4",
    # zai/minimax fallbacks
    "zai/glm-4.7",
    "minimax/MiniMax-M2.7",
    # Xiaomi high-capacity / large-context models
    "xiaomi/mimo-v2.5-pro",
    "xiaomi/mimo-v2-pro",
)

_SWARM_BALANCED_MODEL_HINTS = (
    # GHE Copilot Enterprise — smaller GPT-5 models for balanced-tier tasks
    "github-copilot-enterprise/gpt-5-mini",
    "minimax/MiniMax-M2.7",
    "opencode-go/qwen3.6-plus",
    "opencode-go/minimax-m2.7",
    "ollama/glm-5.1",
    "ollama-mac/shirdel-coder-9b-claude-fable-5:latest",  # local M4 Max, 9B Nemotron coder, Q6_K, ~23 tok/s, free
    "mlx-mac/qwen3-coder-30b-a3b-instruct-4bit",  # local M4 Max, 30B-A3B MoE via mlx_lm.server, ~40 tok/s, free
    # Xiaomi balanced models: good all-rounders and multimodal omni
    "xiaomi/mimo-v2-omni",
    "xiaomi/mimo-v2.5",
)

_SWARM_CHEAP_MODEL_HINTS = (
    # GHE Copilot Enterprise — cheapest models for cheap-tier tasks
    "github-copilot-enterprise/gpt-4o-mini",
    "opencode-go/qwen3.5-plus",
    "opencode-go/deepseek-v4-flash",
    "opencode-zen/minimax-m2.5-free",
    "opencode-zen/hy3-preview-free",
    "ollama/qwen3-coder-next",
    "ollama-mac/shirdel-coder-9b-claude-fable-5:latest",  # local M4 Max, 9B, free
    # Xiaomi free tier (MiMo Auto) — zero cost
    "xiaomi/mimo-auto",
    # Xiaomi low-cost flash model (fast, lower-capacity)
    "xiaomi/mimo-v2-flash",
)

# Order matters: best quality FIRST, fallbacks last
# This is the priority order when primary model fails
# GHE Copilot Enterprise models are placed first/premium because they
# provide GPT-5.4 and Claude Opus/Sonnet via the copilot-api endpoint

_SUBSCRIPTION_PROVIDER_PREFIXES = frozenset({
    "openai",
    "openai-codex",
    "github-copilot",
    "github-copilot-enterprise",
    "opencode-go",
    "opencode-zen",
    "minimax",
    "zai",
    "xiaomi",
    "ollama",
    "ollama-mac",
    "mlx-mac",
    "synthetic",
    "arliai",
})

_PROVIDER_BILLING_MODE: Dict[str, str] = {
    "github-copilot": "subscription",
    "github-copilot-enterprise": "subscription",
    "opencode-go": "subscription",
    "opencode-zen": "subscription",
    "minimax": "subscription",
    "zai": "subscription",
    "xiaomi": "subscription",
    "ollama": "subscription",
    "ollama-mac": "subscription",
    "mlx-mac": "subscription",
    "synthetic": "subscription",
    "arliai": "subscription",
    "openai": "subscription",
    "openai-codex": "subscription",
    "anthropic": "pay_per_use",
    "google": "pay_per_use",
    "nvidia": "pay_per_use",
    "openrouter": "mixed_pay_per_use",
}

_OPENROUTER_FREE_CACHE: Dict[str, tuple[float, bool]] = {}
_OPENROUTER_FREE_CACHE_TTL_SECONDS = 300.0


def _openrouter_nonfree_blocked(model_id: str) -> bool:
    """Return True when paid OpenRouter routing should be refused."""
    raw = str(model_id or "").strip().lower()
    if not raw:
        return False
    if raw.startswith("openrouter/"):
        return ":free" not in raw
    return ":free" not in raw


def _openrouter_model_free_cached(model_id: str) -> bool:
    """Best-effort free-tier check for OpenRouter models.

    Fast path: models explicitly suffixed with ``:free`` are treated as free.
    Otherwise, query OpenRouter model metadata and only allow models whose
    prompt and completion pricing are both zero. Results are cached briefly so
    repeated swarm checks don't hammer the models endpoint.
    """
    raw = str(model_id or "").strip()
    if not raw:
        return False
    if ":free" in raw.lower():
        return True

    now = time.time()
    cached = _OPENROUTER_FREE_CACHE.get(raw)
    if cached and (now - cached[0]) < _OPENROUTER_FREE_CACHE_TTL_SECONDS:
        return bool(cached[1])

    is_free = False
    try:
        import httpx

        headers = {}
        token = os.getenv("OPENROUTER_API_KEY", "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = httpx.get(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/models",
            headers=headers,
            timeout=5.0,
        )
        if response.status_code == 200:
            payload = response.json()
            for item in payload.get("data", []):
                if str(item.get("id") or "").strip() != raw:
                    continue
                pricing = item.get("pricing") or {}

                def _as_float(value: Any) -> float:
                    try:
                        return float(value)
                    except Exception:
                        return 0.0

                prompt_cost = _as_float(pricing.get("prompt", 0))
                completion_cost = _as_float(pricing.get("completion", 0))
                is_free = (prompt_cost == 0.0 and completion_cost == 0.0)
                break
    except Exception:
        is_free = False

    _OPENROUTER_FREE_CACHE[raw] = (now, is_free)
    return is_free


# Default settings


def _resolve_swarm_model(pool, *, context_overflow: bool = False, estimated_tokens: int = 0):
    """Resolve a model from the swarm pool based on selection policy.

    When context_overflow is True and large_context_fallbacks are available,
    switch to the largest available model (no compression needed).

    When estimated_tokens > 0, models whose context window cannot safely
    hold the request are excluded from consideration, preventing 413 errors.
    """
    import random, os
    primary = pool.get("primary", "")
    fallbacks = pool.get("fallbacks", [])
    large_context_fallbacks = pool.get("large_context_fallbacks", [])
    policy = pool.get("selection_policy", "cost-balanced")
    # Use _swarm_model_is_available to check both credentials AND cooldown status
    available_fallbacks = [m for m in fallbacks if _swarm_model_is_available(m)]
    available_large_context = [m for m in large_context_fallbacks if _swarm_model_is_available(m)]
    primary_available = bool(primary) and _swarm_model_is_available(primary)

    def _model_safe_for_tokens(model: str, tokens: int) -> bool:
        """Return True if model's context window can safely hold `tokens` tokens."""
        if tokens <= 0:
            return True
        ctx_len = _model_context_length(model)
        if ctx_len <= 0:
            return True  # Unknown context — skip filtering
        # Use 85% of context window as safe limit (leaves room for output + buffer)
        safe_limit = int(ctx_len * 0.85)
        return tokens <= safe_limit

    # Filter candidates by context fit when we have an estimate
    if estimated_tokens > 0:
        _ctx_filtered_fallbacks = [m for m in available_fallbacks if _model_safe_for_tokens(m, estimated_tokens)]
        _ctx_filtered_large = [m for m in available_large_context if _model_safe_for_tokens(m, estimated_tokens)]
        _ctx_filtered_primary = _model_safe_for_tokens(primary, estimated_tokens)
        if _ctx_filtered_fallbacks:
            available_fallbacks = _ctx_filtered_fallbacks
        if _ctx_filtered_large:
            available_large_context = _ctx_filtered_large
        if not _ctx_filtered_primary:
            primary_available = False
        logger.info(
            "[api_server] context filter: estimated_tokens=%d, primary_fit=%s, "
            "fallbacks_context_fit=%s, large_context_fit=%s",
            estimated_tokens, _ctx_filtered_primary,
            [m for m in available_fallbacks], [m for m in available_large_context],
        )

    # Context overflow path: use a large-context model instead of compressing
    if context_overflow and available_large_context:
        from agent.model_metadata import get_model_context_length_quick
        # Sort large-context models by context length descending, pick the largest
        _sorted = sorted(
            available_large_context,
            key=lambda m: get_model_context_length_quick(m),
            reverse=True,
        )
        if _sorted:
            logger.info(f"[api_server] context overflow — switching to large-context model: {_sorted[0]}")
            return _sorted[0]
        # Fall through to normal selection if large-context list is empty
    
    if not fallbacks:
        return "openrouter/free"
    if primary and not primary_available:
        logger.warning("[api_server] primary swarm model unavailable (missing credentials or in cooldown): %s", primary)
    # IMPORTANT: when all available_fallbacks are cooldown'd, we must NOT fall
    # back to the raw fallbacks list — those are also cooldown'd and trying them
    # again wastes API calls and triggers more 429s.  Fall back to the primary
    # (which may itself be cooldown'd but we try anyway) or return the free
    # openrouter model as last resort.
    if not available_fallbacks:
        if primary and primary != "openrouter/free":
            logger.warning("[api_server] all fallbacks in cooldown — falling back to primary %s", primary)
            return primary
        logger.warning("[api_server] all models unavailable — falling back to openrouter/free")
        return "openrouter/free"
    candidates = available_fallbacks
    if policy == "complexity-aware":
        routing_hint = pool.get("routing_hint") or {}
        logger.info("[api_server] complexity-aware selection: estimated_tokens=%s routing_hint=%s", estimated_tokens, routing_hint)
        recommended_tier = str(routing_hint.get("recommended_tier") or "").strip().lower()
        task_type = str(routing_hint.get("task_type") or "").strip().lower()
        needs_instruction_following = bool(routing_hint.get("needs_instruction_following"))
        needs_repo_reasoning = bool(routing_hint.get("needs_repo_reasoning"))
        needs_bug_judgement = bool(routing_hint.get("needs_bug_judgement"))

        def _pick(preferred: tuple[str, ...]) -> Optional[str]:
            for preferred_model in preferred:
                for candidate in candidates:
                    if candidate == preferred_model:
                        return candidate
            return None

        def _pick_stronger_than_primary() -> Optional[str]:
            for preferred_model in (
                "github-copilot-enterprise/claude-sonnet-4.6",
                "github-copilot-enterprise/gpt-5.4",
                "zai/glm-4.7",
                "minimax/MiniMax-M2.7",
                "ollama/deepseek-v4-pro",
            ):
                for candidate in candidates:
                    if candidate == preferred_model and candidate != primary:
                        return candidate
            for candidate in candidates:
                if candidate != primary:
                    return candidate
            return None

        if recommended_tier == "premium":
            premium_choice = _pick(_SWARM_PREMIUM_MODEL_HINTS)
            if premium_choice:
                logger.info("[api_server] scout escalation → premium model: %s", premium_choice)
                return premium_choice
            stronger_choice = _pick_stronger_than_primary()
            if stronger_choice:
                logger.info("[api_server] premium requested but unavailable; using strongest available non-primary model: %s", stronger_choice)
                return stronger_choice
        elif recommended_tier == "balanced":
            balanced_choice = _pick(_SWARM_BALANCED_MODEL_HINTS)
            if balanced_choice:
                logger.info("[api_server] scout routing → balanced model: %s", balanced_choice)
                return balanced_choice
        elif recommended_tier == "cheap":
            cheap_choice = _pick(_SWARM_CHEAP_MODEL_HINTS)
            if cheap_choice:
                logger.info("[api_server] scout routing → cheap model: %s", cheap_choice)
                return cheap_choice

        if task_type in {"repo_review", "debugging", "implementation", "architecture"} or (
            needs_instruction_following and (needs_repo_reasoning or needs_bug_judgement)
        ):
            premium_choice = _pick(_SWARM_PREMIUM_MODEL_HINTS)
            if premium_choice:
                logger.info("[api_server] heuristic escalation → premium model: %s", premium_choice)
                return premium_choice

        if estimated_tokens > 8000:
            premium = [m for m in candidates if "gpt-5.3-codex" in m or "gpt-5.2-codex" in m]
            if premium:
                logger.info("[api_server] HIGH complexity (%s tokens) — using premium: %s", estimated_tokens, premium[0])
                return premium[0]
        if estimated_tokens > 2000:
            balanced = [m for m in candidates if m in {"minimax/MiniMax-M2.7", "opencode-go/qwen3.6-plus", "ollama/glm-5.1"}]
            if balanced:
                logger.info("[api_server] MEDIUM complexity (%s tokens) — using: %s", estimated_tokens, balanced[0])
                return balanced[0]
        if primary_available:
            logger.info("[api_server] SIMPLE task — using primary: %s", primary)
            return primary
        return candidates[0]
    if primary_available:
        return primary
    if policy == "round-robin":
        return candidates[0]
    elif policy == "cost-balanced":
        cheap = [m for m in candidates if any(x in m for x in ["gemma", "free", "nemotron"])]
        return random.choice(cheap) if cheap else candidates[0]
    return candidates[0]


# Pattern for detecting context overflow errors
_CONTEXT_OVERFLOW_PATTERNS = (
    "context length", "context size", "maximum context", "token limit",
    "too many tokens", "context window", "prompt is too long",
    "context length exceeded", "context overflow", "too many input tokens",
    "maximum tokens", "output tokens", "context_exceeded",
)


def _is_context_overflow_error(error_msg: str) -> bool:
    """Return True if error_msg indicates a context-length overflow."""
    if not error_msg:
        return False
    msg_lower = error_msg.lower()
    return any(pat in msg_lower for pat in _CONTEXT_OVERFLOW_PATTERNS)


_EXPLICIT_MODEL_PROVIDER_ALIASES = {
    "openrouter": "openrouter",
    "openai": "openai-codex",
    "openai-codex": "openai-codex",
    "codex": "openai-codex",
    "anthropic": "anthropic",
    "copilot": "copilot",
    "github-copilot": "copilot",
    "github-copilot-enterprise": "copilot",
    "copilot-acp": "copilot-acp",
    "opencode": "opencode-zen",
    "opencode-zen": "opencode-zen",
    "zen": "opencode-zen",
    "opencode-go": "opencode-go",
    "go": "opencode-go",
    "nous": "nous",
    "custom": "custom",
    "xai": "xai",
    "zai": "zai",
    "kimi-coding": "kimi-coding",
    "kimi-coding-cn": "kimi-coding-cn",
    "minimax": "minimax",
    "minimax-cn": "minimax-cn",
    "ai-gateway": "ai-gateway",
    "kilocode": "kilocode",
    "alibaba": "alibaba",
    "arliai": "arliai",
    "arcee": "arcee",
    "huggingface": "huggingface",
    "xiaomi": "xiaomi",
    "bedrock": "bedrock",
    "google": "google",
    "gemini": "google",
    "groq": "groq",
    "cohere": "cohere",
    "cerebras": "cerebras",
    "nvidia": "nvidia",
    "synthetic": "synthetic",
    "synthetic-anthropic": "synthetic-anthropic",
    "qwen-oauth": "qwen-oauth",
    "ollama-cloud": "ollama-cloud",
    "ollama": "ollama-cloud",
    "ollama-local": "ollama-local",
    "ollama-mac": "ollama-mac",
    "mlx-mac": "mlx-mac",
}


def _explicit_provider_from_model(model: str) -> str:
    raw = str(model or "").strip()
    if "/" not in raw:
        return ""
    prefix = raw.split("/", 1)[0].strip().lower()
    return _EXPLICIT_MODEL_PROVIDER_ALIASES.get(prefix, "")


def _model_provider_prefix(model: str) -> str:
    raw = str(model or "").strip()
    return raw.split("/", 1)[0].strip().lower() if "/" in raw else ""


def _provider_billing_mode(provider_prefix: str) -> str:
    return _PROVIDER_BILLING_MODE.get(str(provider_prefix or "").strip().lower(), "unknown")


def _is_subscription_model(model: str) -> bool:
    return _model_provider_prefix(model) in _SUBSCRIPTION_PROVIDER_PREFIXES


def _align_runtime_with_explicit_model(runtime_kwargs: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Honor provider-prefixed model IDs before the first upstream call.

    This prevents auto provider resolution from selecting an unrelated backend
    (for example Codex OAuth) when the model string itself already names the
    intended provider (for example ``opencode-zen/big-pickle``).
    """
    explicit_provider = _explicit_provider_from_model(model)
    if not explicit_provider:
        return runtime_kwargs

    current_provider = str(runtime_kwargs.get("provider") or "").strip().lower()
    raw_prefix = str(model or "").split("/", 1)[0].strip().lower() if "/" in str(model or "") else ""
    
    logger.debug(
        "[api_server] _align_runtime_with_explicit_model INPUT: model=%s, current_provider=%s, raw_prefix=%s, runtime_kwargs_keys=%s",
        model, current_provider, raw_prefix, list(runtime_kwargs.keys())
    )
    if runtime_kwargs.get("api_key"):
        logger.debug("[api_server] _align_runtime_with_explicit_model INPUT: runtime_kwargs[api_key][:30]=%s, runtime_kwargs[base_url]=%s",
                    runtime_kwargs["api_key"][:30], runtime_kwargs.get("base_url"))
    
    if (
        current_provider == explicit_provider
        and runtime_kwargs.get("api_key")
        and raw_prefix != "github-copilot-enterprise"
    ):
        logger.debug("[api_server] _align_runtime_with_explicit_model: early-return (current_provider matches, not GHE)")
        return runtime_kwargs

    try:
        explicit_runtime_kwargs, _normalized_model = _runtime_kwargs_for_model_id(model)
    except Exception:
        explicit_runtime_kwargs, _normalized_model = {}, str(model or "").strip()

    explicit_runtime_provider = str(explicit_runtime_kwargs.get("provider") or "").strip().lower()
    if explicit_runtime_provider == explicit_provider:
        merged = dict(runtime_kwargs)
        for key in ("api_key", "base_url", "provider", "api_mode", "credential_pool"):
            value = explicit_runtime_kwargs.get(key)
            if value is not None and value != "":
                merged[key] = value
        logger.info(
            "[api_server] aligned runtime provider to %s for explicit model %s via model-id runtime resolution",
            explicit_provider,
            model,
        )
        logger.debug(
            "[api_server] _align_runtime_with_explicit_model OUTPUT: merged[api_key][:30]=%s, merged[base_url]=%s, merged[provider]=%s",
            merged.get("api_key", "")[:30] if merged.get("api_key") else "N/A",
            merged.get("base_url"),
            merged.get("provider")
        )
        return merged

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        resolved = resolve_runtime_provider(requested=explicit_provider)
    except Exception as exc:
        logger.warning(
            "[api_server] failed to resolve provider %s for explicit model %s: %s",
            explicit_provider,
            model,
            exc,
        )
        return runtime_kwargs

    merged = dict(runtime_kwargs)
    for key in ("api_key", "base_url", "provider", "api_mode"):
        value = resolved.get(key)
        if value is not None:
            merged[key] = value

    logger.info(
        "[api_server] aligned runtime provider to %s for explicit model %s",
        explicit_provider,
        model,
    )
    return merged


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642

# Gateway states for which the /ready endpoint returns 200.
#
# "running"   — main loop is up, platforms connected, ready to serve traffic
# "degraded"  — main loop is up but some platforms failed to connect; the
#               gateway is still useful (cron jobs, reconnect watcher,
#               HTTP API all still work) so we keep the pod in the
#               Service endpoint set rather than dropping traffic.
#
# States that return 503 (pod is not yet ready / no longer ready):
#   "starting"        — fresh process, main loop not yet running
#   "startup_failed"  — fatal startup conflict (e.g. port already bound);
#                       the runner will exit, the pod will restart
#   "draining"        — graceful shutdown in progress; existing requests
#                       finish but no new traffic should be sent.  K8s
#                       removes the pod from Service endpoints when
#                       /ready returns 503, which is exactly what we
#                       want for a rolling update or pod deletion.
#   "stopped"         — clean shutdown completed
#
# Unknown / unset values also return 503 — defensive default in case a
# future gateway version introduces a new state we haven't mapped yet.
_READY_STATES = frozenset({"running", "degraded"})
MAX_STORED_RESPONSES = 100
MAX_REQUEST_BYTES = int(os.getenv("HERMES_API_MAX_REQUEST_BYTES", str(32 * 1024 * 1024)))
CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0
MAX_NORMALIZED_TEXT_LENGTH = 65_536  # 64 KB cap for normalized content parts
MAX_CONTENT_LIST_SIZE = 1_000  # Max items when content is an array
MAX_HISTORY_MESSAGES = 200
MAX_HISTORY_TEXT_LENGTH = 240_000


def _passthrough_request_timeout(provider_id: str, model: str | None = None) -> float:
    """Resolve hermes-code passthrough request timeout.

    Keep passthrough aligned with the primary AIAgent timeout policy:
    provider/model config wins, then an explicit passthrough env override,
    then the legacy HERMES_API_TIMEOUT default.
    """
    try:
        from hermes_cli.timeouts import get_provider_request_timeout
        configured = get_provider_request_timeout(provider_id, model)
        if configured is not None:
            return configured
    except Exception:
        pass

    raw = os.getenv("HERMES_CODE_PASSTHROUGH_TIMEOUT") or os.getenv("HERMES_API_TIMEOUT")
    if raw:
        try:
            timeout = float(raw)
            if timeout > 0:
                return timeout
        except (TypeError, ValueError):
            pass
    return 1800.0


def _transform_messages_to_anthropic(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]] | None]:
    """Transform OpenAI-style messages to Anthropic message format.

    Returns (messages, tools) tuple where tools is the transformed Anthropic tools list
    if any tools were provided, or None if no tools.
    """
    anthropic_messages = []
    anthropic_tools = None

    if tools:
        anthropic_tools = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}})
            })
    from agent.anthropic_adapter import _convert_content_part_to_anthropic

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "developer"):
            anthropic_messages.append({"role": "user", "content": str(content) if content else ""})
        elif role == "user":
            if isinstance(content, list):
                blocks = []
                for part in content:
                    converted = _convert_content_part_to_anthropic(part)
                    if converted is not None:
                        blocks.append(converted)
                anthropic_messages.append({"role": "user", "content": blocks or ""})
            else:
                anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and anthropic_tools:
                content_blocks = []
                if content:
                    if isinstance(content, list):
                        for part in content:
                            converted = _convert_content_part_to_anthropic(part)
                            if converted is not None:
                                content_blocks.append(converted)
                    else:
                        content_blocks.append({"type": "text", "text": str(content)})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    # Anthropic enforces a 200-char limit on tool_use.name.
                    # Truncate if needed so the request isn't rejected outright.
                    if len(tool_name) > 200:
                        tool_name = tool_name[:197] + "..."
                    try:
                        args_obj = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args_obj = {"raw": func.get("arguments", "")}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"call_{tool_name}"),
                        "name": tool_name,
                        "input": args_obj
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                if isinstance(content, list):
                    blocks = []
                    for part in content:
                        converted = _convert_content_part_to_anthropic(part)
                        if converted is not None:
                            blocks.append(converted)
                    anthropic_messages.append({"role": "assistant", "content": blocks or ""})
                else:
                    anthropic_messages.append({"role": "assistant", "content": str(content) if content else ""})
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = msg.get("content", "")
            anthropic_messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": str(tool_content) if tool_content else ""}]
            })
    return anthropic_messages, anthropic_tools


def _model_context_length(model_name: str) -> int:
    """Look up the context window size for a model, or 0 if unknown."""
    try:
        from agent.model_metadata import get_model_context_length
        return get_model_context_length(model_name) or 0
    except Exception:
        return 0


def _model_supports_vision(model_name: str) -> bool:
    raw = str(model_name or "").strip()
    if "/" not in raw:
        return False
    provider, _, model = raw.partition("/")
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(provider, model)
        return bool(caps and getattr(caps, "supports_vision", False))
    except Exception:
        return False


def _model_supports_audio_input(model_name: str) -> bool:
    raw = str(model_name or "").strip()
    if "/" not in raw:
        return False
    provider, _, model = raw.partition("/")
    try:
        from agent.models_dev import _get_provider_models, _find_model_entry  # type: ignore
        models = _get_provider_models(provider)
        if not models:
            return False
        entry = _find_model_entry(models, model)
        if not isinstance(entry, dict):
            return False
        modalities = entry.get("modalities", {})
        if isinstance(modalities, dict):
            inputs = modalities.get("input", [])
            return isinstance(inputs, list) and "audio" in inputs
    except Exception:
        pass
    return False


def _messages_have_image_parts(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "").strip().lower() in {"image_url", "input_image"}:
                return True
    return False


def _messages_have_audio_parts(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "").strip().lower() in {"input_audio", "audio_url"}:
                return True
    return False


def _messages_use_external_image_urls(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "").strip().lower() not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url", {})
            url = image_value.get("url", "") if isinstance(image_value, dict) else str(image_value or "")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return True
    return False


def _model_safe_for_tokens(model: str, tokens: int, margin_fraction: float = 0.85) -> bool:
    """Return True if model's context window can safely hold `tokens` tokens.

    This module-level helper mirrors the in-function implementation used by
    _resolve_swarm_model but ensures other code paths (like the swarm scout)
    can call it without raising NameError if the local inner function isn't in
    scope.
    """
    if tokens <= 0:
        return True
    ctx_len = _model_context_length(model)
    if ctx_len <= 0:
        return True  # Unknown context — skip filtering
    safe_limit = int(ctx_len * margin_fraction)
    return tokens <= safe_limit


def _messages_token_count(messages: List[Dict[str, Any]], system_prompt: str = "") -> int:
    """Estimate token count for messages + optional system prompt."""
    try:
        from agent.model_metadata import estimate_messages_tokens_rough
        total = estimate_messages_tokens_rough(messages)
        if system_prompt:
            total += estimate_messages_tokens_rough([{"role": "system", "content": system_prompt}])
        return total
    except Exception:
        # Fallback: rough char-based estimate
        total = len(system_prompt)
        for msg in messages:
            total += len(str(msg.get("content") or ""))
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    total += len(str(tc.get("function", {}).get("arguments") or ""))
        return total // 4  # ~4 chars per token


def _model_can_handle_context(model_name: str, estimated_tokens: int, margin_fraction: float = 0.85) -> bool:
    """Return True if the model's context window can safely hold the estimated tokens."""
    ctx_len = _model_context_length(model_name)
    if ctx_len <= 0:
        return True  # Unknown context — assume it's fine
    safe_limit = int(ctx_len * margin_fraction)
    return estimated_tokens <= safe_limit


def _compact_message_history(
    messages: List[Dict[str, Any]],
    session_id: str = "unknown",
    *,
    system_prompt: str = "",
    target_model: str = "",
) -> List[Dict[str, Any]]:
    """Keep recent history bounded so provider request bodies stay sane.

    Applies two filters:
    1. Hard cap: last MAX_HISTORY_MESSAGES messages.
    2. Token-aware cap: if target_model is known, truncate to its safe context limit.
       Falls back to MAX_HISTORY_TEXT_LENGTH chars when target_model is unknown.

    This prevents 413 errors by ensuring the final payload fits the model's window
    before it reaches the LLM API call.
    """
    import logging
    _logger = logging.getLogger(__name__)

    if not messages:
        return []

    original_count = len(messages)
    trimmed = list(messages[-MAX_HISTORY_MESSAGES:])
    trimmed_count = len(trimmed)

    # Determine the effective token budget from target model (or fallback to chars)
    estimated_tokens = _messages_token_count(trimmed, system_prompt)
    target_ctx = _model_context_length(target_model)
    if target_ctx > 0:
        # Use 85% of context window as the safe token budget (leaves room for output)
        effective_budget = int(target_ctx * 0.85)
        token_based = True
    else:
        # Fallback: use char-based budget scaled to ~4 chars/token
        effective_budget = MAX_HISTORY_TEXT_LENGTH // 4
        token_based = False

    def _msg_cost(msg: Dict[str, Any]) -> int:
        cost = len(str(msg.get("content") or ""))
        reasoning_content = msg.get("reasoning_content")
        if isinstance(reasoning_content, str):
            cost += len(reasoning_content)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        cost += len(str(fn.get("arguments") or ""))
        return cost // 4  # convert chars to rough token estimate

    total_cost = sum(_msg_cost(msg) for msg in trimmed)

    if original_count > MAX_HISTORY_MESSAGES:
        _logger.warning(
            "[CONTEXT] Session %s compacting: %d messages -> %d (limit: %d, total ~%d tokens)",
            session_id, original_count, trimmed_count, MAX_HISTORY_MESSAGES, total_cost,
        )

    if total_cost <= effective_budget:
        return trimmed

    kept: List[Dict[str, Any]] = []
    running = 0
    dropped_count = 0
    for msg in reversed(trimmed):
        cost = _msg_cost(msg)
        if kept and running + cost > effective_budget:
            dropped_count += 1
            continue
        kept.append(msg)
        running += cost
        if running >= effective_budget:
            break

    kept.reverse()

    if dropped_count > 0:
        budget_desc = f"~{effective_budget:,} tokens ({target_model})" if token_based else f"{MAX_HISTORY_TEXT_LENGTH:,} chars"
        _logger.warning(
            "[CONTEXT] Session %s TRUNCATED: dropped %d messages (budget: %s, ~%d total tokens). "
            "Original messages: %d, Kept: %d",
            session_id, dropped_count, budget_desc, running,
            original_count, len(kept),
        )

    return kept


def _extract_openai_tool_calls(raw_tool_calls: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI-style tool_calls from an incoming chat request."""
    normalized: List[Dict[str, Any]] = []
    if not isinstance(raw_tool_calls, list):
        return normalized
    for tc in raw_tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = str(arguments)
        call_id = tc.get("id") or tc.get("call_id")
        item: Dict[str, Any] = {
            "type": tc.get("type", "function"),
            "function": {
                "name": name.strip(),
                "arguments": arguments,
            },
        }
        if isinstance(call_id, str) and call_id.strip():
            item["id"] = call_id.strip()
            item["call_id"] = call_id.strip()
        # Preserve extra_content (contains google.thought_signature for Gemini)
        _ec = tc.get("extra_content")
        if _ec:
            item["extra_content"] = _ec
        normalized.append(item)
    return normalized


_CLIENT_TOOL_EQUIVALENTS: Dict[str, Tuple[str, ...]] = {
    # OMP renamed grep-style content search to `search`; older sessions and
    # model priors still commonly emit `grep`.
    "grep": ("search", "search_files"),
    "search_files": ("search", "grep"),
    "rg": ("search", "grep", "search_files"),
    # File discovery aliases.
    "glob": ("find", "search"),
    "ls": ("find", "list"),
    "list": ("find", "ls"),
    # Shell aliases.
    "terminal": ("bash", "shell"),
    "shell": ("bash", "terminal"),
}


def _advertised_client_tool_names(tools: Any) -> set[str]:
    """Return client-advertised function names from a passthrough tool list."""
    if not isinstance(tools, list):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            names.add(_normalize_external_tool_name(name.strip()))
    return names


def _resolve_client_tool_name(name: Any, advertised_names: Optional[set[str]] = None) -> Optional[str]:
    """Map a model-emitted external tool name to one the client advertised."""
    raw = str(name or "").strip()
    if not raw:
        return None
    normalized = _normalize_external_tool_name(raw)
    if not advertised_names:
        return normalized
    if normalized in advertised_names:
        return normalized
    for candidate in _CLIENT_TOOL_EQUIVALENTS.get(normalized, ()):
        candidate_norm = _normalize_external_tool_name(candidate)
        if candidate_norm in advertised_names:
            return candidate_norm
    return None


def _enrich_client_tool_call(tc: Dict[str, Any], advertised_names: Optional[set[str]] = None) -> Optional[Dict[str, Any]]:
    """Ensure client-visible tool calls satisfy strict downstream schemas."""
    if not isinstance(tc, dict):
        return tc
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return tc
    name = _resolve_client_tool_name(fn.get("name"), advertised_names)
    if not name:
        return None
    fn["name"] = name
    arguments = fn.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return tc
    elif isinstance(arguments, dict):
        parsed = dict(arguments)
    else:
        return tc

    _norm = _normalize_external_tool_name(name)
    if _norm in {"bash", "terminal", "read", "edit", "write", "search", "grep", "search_files", "glob"}:
        parsed = _external_tool_call_arguments(name, parsed)

    fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
    tc["function"] = fn
    return tc


class _CodexPassthroughSkip(Exception):
    """Raised inside the codex passthrough path when tools were provided but no
    tool calls were returned.  Caught by the passthrough model loop's exception
    handler — the provider is NOT penalised (the issue is codex backend model
    behaviour, not an API error), and the next model in the chain is tried."""


def _has_empty_bash_tool_call(tool_calls: list) -> bool:
    """Check if any bash tool call has an unusable command argument.

    Some models (notably gpt-5.4/gpt-5.5 via codex backend) return valid
    tool call JSON with ``{"command": ""}`` — a tool call that will always
    fail. Others return malformed or partial bash arguments with metadata
    fields like timeout/cwd/async but no command. Treat these the same as
    text-only responses and skip to the next provider.
    """
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if _normalize_external_tool_name(name) != "bash":
            continue
        args_raw = tc.get("function", {}).get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                return True
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return True
        if not isinstance(args, dict):
            return True
        cmd = args.get("command", "")
        if not cmd or not cmd.strip():
            return True
    return False


def _invalid_bash_tool_call_summary(tool_calls: list) -> Optional[str]:
    """Return a compact description of the first unusable bash tool call."""
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("function", {}).get("name", "")
        if _normalize_external_tool_name(name) != "bash":
            continue
        args_raw = tc.get("function", {}).get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                return f"bash arguments were not valid JSON: {args_raw[:300]}"
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            return f"bash arguments had unsupported type: {type(args_raw).__name__}"
        if not isinstance(args, dict):
            return f"bash arguments decoded to {type(args).__name__}, not an object"
        cmd = args.get("command", "")
        if not cmd or not str(cmd).strip():
            return f"bash arguments omitted a non-empty command field: {json.dumps(args, ensure_ascii=False)[:300]}"
    return None


_MINIMAX_M3_BASH_TOOL_PROMPT = (
    "Hermes tool-calling contract for MiniMax-M3:\n"
    "- When calling the bash tool, function.arguments MUST be valid JSON.\n"
    "- The JSON MUST include a non-empty string field named \"command\" containing the full shell command or pipeline.\n"
    "- cwd, timeout, and async are optional metadata only; never send them without command.\n"
    "- Correct example: {\"command\":\"cd /tmp && tail -40 app.log\",\"cwd\":\"/tmp\",\"timeout\":60}."
)


def _bash_tool_retry_prompt(tool_calls: list) -> str:
    summary = _invalid_bash_tool_call_summary(tool_calls) or "the previous bash tool call was invalid"
    return (
        "Hermes rejected your previous bash tool call because it cannot be executed.\n"
        f"Invalid call: {summary}\n"
        "Retry the same task now. If you call bash, function.arguments must be valid JSON "
        "with a non-empty string field named \"command\" containing the full shell command. "
        "Do not send cwd, timeout, or async without command."
    )


def _messages_with_retry_tool_prompt(
    messages: List[Dict[str, Any]],
    tool_calls: list,
) -> List[Dict[str, Any]]:
    prompt = _bash_tool_retry_prompt(tool_calls)
    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        patched = [dict(messages[0])]
        patched[0]["content"] = f"{messages[0].get('content')}\n\n{prompt}"
        patched.extend(messages[1:])
        return patched
    return [{"role": "system", "content": prompt}, *messages]


def _tool_schema_contains_bash(tools: Optional[list]) -> bool:
    if not tools:
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name", name)
        if _normalize_external_tool_name(str(name or "")) == "bash":
            return True
    return False


def _is_minimax_m3_model(provider_model: str, provider: str, resolved_model: str = "") -> bool:
    provider_model_l = (provider_model or "").lower()
    provider_l = (provider or "").lower()
    resolved_l = (resolved_model or "").lower()
    combined = f"{provider_model_l} {resolved_l}"
    return "minimax-m3" in combined and (provider_l == "minimax" or "minimax/" in provider_model_l)


def _messages_with_provider_tool_prompt(
    messages: List[Dict[str, Any]],
    *,
    provider_model: str,
    provider: str,
    resolved_model: str = "",
    tools: Optional[list] = None,
) -> List[Dict[str, Any]]:
    """Add narrow provider guidance for models with known tool-call quirks."""
    if not (_tool_schema_contains_bash(tools) and _is_minimax_m3_model(provider_model, provider, resolved_model)):
        return messages

    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        patched = [dict(messages[0])]
        patched[0]["content"] = f"{messages[0].get('content')}\n\n{_MINIMAX_M3_BASH_TOOL_PROMPT}"
        patched.extend(messages[1:])
        return patched

    return [{"role": "system", "content": _MINIMAX_M3_BASH_TOOL_PROMPT}, *messages]


def _call_codex_passthrough(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str,
    tools: List[Dict[str, Any]] = None,
    timeout: float = 300,
) -> Any:
    """Call chatgpt.com/backend-api/codex via the Responses API.

    Uses client.responses.create(stream=True) rather than client.responses.stream()
    because the latter's ResponseStream wrapper calls parse_response() on the final
    event, which crashes when response.output is None — a quirk of this endpoint.
    We iterate raw SSE events instead and collect content + function_call items.

    Returns an object with .choices[0].message and .usage compatible with the
    passthrough serialisation layer.
    """
    import types, platform
    from agent.auxiliary_client import _codex_cloudflare_headers
    from agent.codex_responses_adapter import (
        _chat_messages_to_responses_input,
        _responses_tools,
        _deterministic_call_id,
    )
    effective_base = (base_url or "https://chatgpt.com/backend-api/codex").rstrip("/")
    logger.info("[HTTP_LOG] Codex passthrough URL=%s model=%s auth_key_preview=%s key_len=%d", effective_base, model, api_key[:10], len(api_key))

    # Build a keepalive httpx client to prevent the SITA NGFW or upstream
    # from closing idle connections before the first SSE event arrives.
    import socket
    import httpx
    _sock_opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPIDLE"):
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60))
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15))
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4))
    elif hasattr(socket, "TCP_KEEPALIVE"):
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 60))
    _keepalive_httpx = httpx.Client(
        transport=httpx.HTTPTransport(socket_options=_sock_opts),
    )

    # Extract system prompt (instructions) from messages
    instructions = ""
    input_messages = messages
    if messages and messages[0].get("role") == "system":
        instructions = str(messages[0].get("content") or "").strip()
        input_messages = messages[1:]
    if not instructions:
        instructions = "You are a helpful assistant."

    headers = _codex_cloudflare_headers(api_key)
    # Manually add Authorization — the OpenAI SDK adds it via _build_request,
    # but we are going around the SDK, so it must be explicit.
    headers["Authorization"] = f"Bearer {api_key}"

    responses_input = _chat_messages_to_responses_input(input_messages)
    responses_tools = _responses_tools(tools) if tools else None
    # The OpenAI SDK's responses.create() appends "/responses" to base_url, but
    # chatgpt.com/backend-api/codex already includes the path segment, so the
    # SDK ends up calling /responses/responses → Cloudflare challenge HTML.
    # Use raw httpx streaming against the correct /responses URL instead.
    url = f"{effective_base}/responses"
    # The Codex endpoint rejects max_output_tokens and temperature — omit
    # both to avoid 400 errors.
    body = {
        "model": model,
        "instructions": instructions,
        "input": responses_input,
        "store": False,
        "stream": True,
    }
    if responses_tools:
        body["tools"] = responses_tools

    # Build kwargs dict for logging (avoid building twice)
    kwargs = body

    # Parse raw SSE events. We cannot use client.responses.stream() because its
    # ResponseStream wrapper calls parse_response() on response.completed, which
    # iterates response.output — but this endpoint sometimes sends output=None in
    # the final event, raising TypeError. client.responses.create(stream=True)
    # gives us the raw event iterator without that finalisation step.
    content_parts: List[str] = []
    # call_id -> {"name": str, "arguments": str}
    tc_map: Dict[str, Dict[str, str]] = {}
    tc_order: List[str] = []  # preserve insertion order
    usage_obj = None
    print(f"[HTTP_LOG] REQUEST codex passthrough model={model} url={url}", flush=True)
    with _keepalive_httpx.stream("POST", url, json=body, headers=headers, timeout=timeout) as resp:
        if resp.status_code >= 400:
            err_body = b"".join(resp.iter_bytes())[:500]
            raise RuntimeError(
                f"codex passthrough HTTP {resp.status_code}: {err_body!r}"
            )
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").strip() if isinstance(line, bytes) else line.strip()
            if not text.startswith("data: "):
                continue
            try:
                data = json.loads(text[6:])
            except json.JSONDecodeError:
                continue
            etype = data.get("type", "")
            if etype == "response.output_text.delta":
                delta = data.get("delta", "") or ""
                if delta:
                    content_parts.append(delta)
            elif etype == "response.output_item.added":
                item = data.get("item")
                if item and item.get("type") == "function_call":
                    call_id = str(item.get("call_id", "") or "").strip()
                    name = str(item.get("name", "") or "").strip()
                    if call_id and call_id not in tc_map:
                        tc_map[call_id] = {"name": name, "arguments": ""}
                        tc_order.append(call_id)
            elif etype == "response.function_call_arguments.done":
                call_id = str(data.get("call_id", "") or "").strip()
                args = str(data.get("arguments", "") or "")
                if call_id in tc_map:
                    tc_map[call_id]["arguments"] = args
            elif etype == "response.output_item.done":
                item = data.get("item")
                if item and item.get("type") == "function_call":
                    call_id = str(item.get("call_id", "") or "").strip()
                    name = str(item.get("name", "") or "").strip()
                    arguments = item.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    if call_id:
                        entry = tc_map.get(call_id)
                        if entry is None:
                            tc_map[call_id] = {"name": name, "arguments": arguments or ""}
                            tc_order.append(call_id)
                        else:
                            if name and not entry.get("name"):
                                entry["name"] = name
                            if arguments:
                                entry["arguments"] = arguments
            elif etype == "response.failed":
                resp_obj = data.get("response", {})
                err_obj = resp_obj.get("error", {}) or {}
                err_code = err_obj.get("code", "unknown")
                err_msg = err_obj.get("message", str(resp_obj)[:200])
                raise RuntimeError(
                    f"codex passthrough response.failed: {err_code} {err_msg}"
                )
            elif etype == "response.completed":
                resp_obj = data.get("response")
                if resp_obj is not None:
                    usage_obj = resp_obj.get("usage")
                    # Check if the response itself reports failure status
                    if resp_obj.get("status") == "failed":
                        err_obj = resp_obj.get("error", {}) or {}
                        err_code = err_obj.get("code", "unknown")
                        err_msg = err_obj.get("message", "no error details")
                        raise RuntimeError(
                            f"codex passthrough response completed with status=failed: {err_code} {err_msg}"
                        )
    content = "".join(content_parts)

    tool_calls_out: List[Any] = []
    for i, call_id in enumerate(tc_order):
        tc = tc_map[call_id]
        name = _normalize_external_tool_name(tc["name"])
        arguments = tc["arguments"] or "{}"
        if not call_id:
            call_id = _deterministic_call_id(name, arguments, i)
        tool_calls_out.append(types.SimpleNamespace(
            id=call_id,
            type="function",
            function=types.SimpleNamespace(name=name, arguments=arguments),
        ))

    finish_reason = "tool_calls" if tool_calls_out else "stop"
    usage = types.SimpleNamespace(
        prompt_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
        total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
    )
    msg = types.SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls_out if tool_calls_out else None,
        reasoning_content=None,
    )
    choice = types.SimpleNamespace(message=msg, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _enrich_client_tool_calls(tool_calls: Any, advertised_tools: Any = None) -> List[Dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    advertised_names = _advertised_client_tool_names(advertised_tools) if advertised_tools else None
    enriched: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        item = _enrich_client_tool_call(dict(tc), advertised_names)
        if item is not None:
            enriched.append(item)
        else:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            logger.warning(
                "[hermes-code] dropping unmapped client tool call name=%r advertised=%s",
                fn.get("name") if isinstance(fn, dict) else None,
                sorted(advertised_names or []),
            )
    return enriched


def _extract_text_tool_calls_for_passthrough(content: Any, advertised_tools: Any = None) -> tuple[List[Dict[str, Any]], Any]:
    """Convert XML/DSML text-formatted tool calls into OpenAI tool_calls.

    Some tool-capable models emit tool calls as assistant text instead of the
    structured ``tool_calls`` field. The normal chat transport handles this,
    but hermes-code passthrough serializes SSE directly, so it needs the same
    rescue path before it streams raw content to the client.
    """
    if not isinstance(content, str) or not content:
        return [], content
    try:
        from agent.mimocode_code_client import _clean_tool_text, _parse_tool_calls_xml

        parsed_calls = _parse_tool_calls_xml(content)
        if not parsed_calls:
            return [], content

        converted: List[Dict[str, Any]] = []
        for idx, parsed in enumerate(parsed_calls):
            if not isinstance(parsed, dict):
                continue
            function = parsed.get("function") or {}
            if not isinstance(function, dict):
                continue
            raw_name = function.get("name") or "unknown"
            name = _normalize_external_tool_name(raw_name)
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            call_id = parsed.get("id") or parsed.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                digest = hashlib.sha256(f"{name}:{arguments}:{idx}".encode()).hexdigest()[:16]
                call_id = f"call_text_{digest}"
            converted.append({
                "id": call_id.strip(),
                "call_id": call_id.strip(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments or "{}",
                },
            })

        if not converted:
            return [], content
        cleaned = _clean_tool_text(content).strip()
        return _enrich_client_tool_calls(converted, advertised_tools), cleaned
    except Exception as exc:
        logger.debug("text tool-call extraction failed: %s", exc)
        return [], content


def _normalize_external_tool_name(name: Any) -> str:
    """Map Hermes/internal tool names to client-exposed OpenCode tool names."""
    raw = str(name or "").strip()
    if raw == "terminal":
        return "bash"
    # claude-code-cli: mcp__hermes-tools__<name> → <name>
    if raw.startswith("mcp__hermes-tools__"):
        return raw[len("mcp__hermes-tools__"):]
    # mimocode-cli: mcp_<name> → <name>
    if raw.startswith("mcp_"):
        return raw[4:]
    return raw



# Regex for detecting MCP-prefixed tool names leaked as text content.
# Matches: "mcp__hermes-tools__bash: find /..." or "mcp_bash: find /..."
_RE_MCP_TEXT_LEAK = re.compile(
    r"^(?:mcp__hermes-tools__|mcp_)([A-Za-z0-9_-]+)\s*:\s*",
)


def _normalize_bridge_text(text: str) -> str:
    """Strip MCP-prefixed tool names that leaked into bridge text content.

    When Claude Code or mimocode-cli fall back to text output for a tool
    call, the text may start with ``mcp__hermes-tools__<name>: <args>``.
    This strips the prefix so clients never see internal provider naming.
    """
    m = _RE_MCP_TEXT_LEAK.match(text)
    if m:
        tool_name = _normalize_external_tool_name(m.group(1))
        remainder = text[m.end():]
        return f"{tool_name}: {remainder}"
    return text


def _extract_malformed_json_string_field(raw: Any, field: str) -> Optional[str]:
    """Extract a JSON string field when inner quotes made the object invalid.

    Some tool-capable models emit bash arguments like:
    {"command":"curl -H "Authorization: Bearer $TOKEN" ...","i":"..."}

    The shell command is usable, but the JSON wrapper is not.  Only recover
    when the closing quote is followed by a clear JSON field/object boundary.
    """
    if not isinstance(raw, str) or not field:
        return None
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', raw, re.DOTALL)
    if not match:
        return None
    start = match.end()
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch != '"':
            continue
        tail = raw[idx + 1 :].lstrip()
        if tail.startswith("}") or re.match(r',\s*"[^"]+"\s*:', tail, re.DOTALL):
            value = raw[start:idx]
            return value.replace('\\"', '"')
    return None


def _external_tool_call_arguments(name: Any, args: Any) -> Dict[str, Any]:
    """Return schema-safe arguments for a client-executed tool call."""
    tool_name = _normalize_external_tool_name(name)
    malformed_raw_args: Optional[str] = None
    if isinstance(args, dict):
        parsed = dict(args)
    elif isinstance(args, str) and args.strip():
        try:
            loaded = json.loads(args)
            parsed = dict(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            malformed_raw_args = args
            parsed = {}
    else:
        parsed = {}

    if tool_name == "bash":
        cmd = parsed.get("command")
        raw_value = parsed.get("raw")
        if not isinstance(cmd, str) and isinstance(raw_value, str):
            extracted = _extract_malformed_json_string_field(raw_value, "command")
            if extracted and extracted.strip():
                cmd = extracted
        if not isinstance(cmd, str):
            extracted = _extract_malformed_json_string_field(malformed_raw_args, "command")
            if extracted and extracted.strip():
                cmd = extracted
        if not isinstance(cmd, str) and isinstance(malformed_raw_args, str):
            raw = malformed_raw_args.strip()
            if raw and not raw.startswith(("{", "[")):
                cmd = raw
        if not isinstance(cmd, str):
            for key in ("cmd", "shell", "script", "bash", "input", "text"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    cmd = value
                    break
        if not isinstance(cmd, str):
            cmd = ""
        parsed["command"] = cmd
        if not isinstance(parsed.get("description"), str):
            parsed["description"] = f"Execute command: {cmd[:100]}"
    elif tool_name in {"read", "edit", "write"}:
        file_value = parsed.get("filePath")
        if not isinstance(file_value, str) or not file_value.strip():
            for key in ("file", "path", "filename", "target", "file_path", "filepath"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    file_value = value
                    break
        if isinstance(file_value, str):
            parsed["filePath"] = file_value
        if tool_name == "write" and "content" not in parsed:
            for key in ("text", "contents", "data"):
                value = parsed.get(key)
                if isinstance(value, str):
                    parsed["content"] = value
                    break
        if tool_name == "edit":
            if "oldString" not in parsed:
                for key in ("old", "old_string", "search", "find"):
                    value = parsed.get(key)
                    if isinstance(value, str):
                        parsed["oldString"] = value
                        break
            if "newString" not in parsed:
                for key in ("new", "new_string", "replace", "replacement"):
                    value = parsed.get(key)
                    if isinstance(value, str):
                        parsed["newString"] = value
                        break
    # Alias `regex` <-> `pattern` for search tools to preserve client parameter name.
    if tool_name in {"search", "grep", "search_files", "glob"}:
        if "regex" in parsed and "pattern" not in parsed:
            parsed["pattern"] = parsed["regex"]
        elif "pattern" in parsed and "regex" not in parsed:
            parsed["regex"] = parsed["pattern"]
    return parsed


def _external_tool_call_arguments_str(name: Any, args: Any) -> str:
    return json.dumps(_external_tool_call_arguments(name, args), ensure_ascii=False)


def _fallback_provider_for_model(model_id: str) -> tuple[str, str]:
    raw = str(model_id or "").strip()
    if not raw:
        return "", ""
    if "/" not in raw:
        return "openrouter", raw

    prefix, rest = raw.split("/", 1)
    prefix = prefix.strip().lower()
    rest = rest.strip()
    if prefix in ("openai", "openai-codex"):
        return "openai-codex", raw
    if prefix in {"github-copilot", "opencode-go", "opencode-zen", "zai", "minimax", "xiaomi", "ollama", "arliai"}:
        # CRITICAL FIX: Return bare model name (rest) without provider prefix for OpenCode providers
        # and direct API providers (zai, minimax, xiaomi, ollama, arliai). These APIs only accept bare model names
        # (e.g., "glm-4.7", not "zai/glm-4.7"; "GLM-4.6-Derestricted-v5", not "arliai/GLM-4.6-Derestricted-v5")
        return prefix, rest
    if prefix == "openrouter":
        return "openrouter", rest
    return "openrouter", raw


def _build_env_fallback_chain(prefix: str) -> List[Dict[str, Any]]:
    """Build fallback provider chain from HERMES_*_FALLBACK_{N} env vars.

    Result is cached by prefix since env vars don't change at runtime.
    Without caching, the ~6.7s spent in resolve_runtime_provider() per call
    (20+ fallback models × ~0.3s each) would repeat on every request.
    """
    if prefix in _FALLBACK_CHAIN_CACHE:
        logger.debug("[timing] _build_env_fallback_chain: CACHED for prefix=%s", prefix)
        return _FALLBACK_CHAIN_CACHE[prefix]

    from hermes_cli.runtime_provider import resolve_runtime_provider

    chain: List[Dict[str, Any]] = []
    # Auto-block mimocode-cli from env-driven fallback chains until timeout
    # issues are resolved. Users can re-enable by setting HERMES_SWARM_ALLOW_MIMOCODE=1.
    _allow_mimocode = os.getenv("HERMES_SWARM_ALLOW_MIMOCODE", "").strip().lower() in ("1", "true", "yes")
    for idx in range(1, 33):
        raw_model = os.getenv(f"{prefix}_{idx}", "").strip()
        if not raw_model:
            continue
        if not _allow_mimocode and raw_model.startswith("mimocode-cli"):
            logger.info("[swarm] blocking mimocode-cli from fallback chain (set HERMES_SWARM_ALLOW_MIMOCODE=1 to re-enable)")
            continue
        provider, resolved_model = _fallback_provider_for_model(raw_model)
        if not provider or not resolved_model:
            continue
        try:
            runtime = resolve_runtime_provider(requested=provider)
        except Exception:
            runtime = {}
        resolved_provider = runtime.get("provider") or ""
        normalized_provider = str(resolved_provider or runtime.get("requested_provider") or provider).strip()
        chain.append({
            "provider": normalized_provider,
            "model": resolved_model,
            "base_url": str(runtime.get("base_url") or "").strip(),
            "api_key": str(runtime.get("api_key") or "").strip(),
        })
    _FALLBACK_CHAIN_CACHE[prefix] = chain
    logger.debug("[timing] _build_env_fallback_chain: built %d entries for prefix=%s", len(chain), prefix)
    return chain


def _build_swarm_model_pool(*, estimated_tokens: int = 0, routing_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build hermes-swarm routing config from environment variables."""
    primary = os.getenv("HERMES_SWARM_PRIMARY_MODEL", "ollama/qwen3-coder-next").strip()
    fallback_chain = [primary] if primary else []
    for idx in range(1, 33):
        fb = os.getenv(f"HERMES_SWARM_FALLBACK_{idx}", "").strip()
        if fb:
            fallback_chain.append(fb)

    seen = set()
    deduped: List[str] = []
    for model in fallback_chain:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    fallback_chain = deduped
    blocked = {m.strip() for m in os.getenv("HERMES_SWARM_BLOCKED_MODELS", "").split(",") if m.strip()}
    # Auto-block mimocode-cli until timeout issues are resolved.
    # Users can re-enable by adding "mimocode-cli" to HERMES_SWARM_ALLOWED_BLOCKED_MODELS.
    if not os.getenv("HERMES_SWARM_ALLOW_MIMOCODE", "").strip().lower() in ("1", "true", "yes"):
        blocked.add("mimocode-cli")
        blocked.add("mimocode-cli/mimo-auto")
        blocked.add("mimocode-cli/mimo")
        blocked.add("mimocode-cli/mimo-auto")
    fallback_chain = [m for m in fallback_chain if m not in blocked and "\n" not in m and "HERMES_SWARM_" not in m] 

    large_context_fallbacks: List[str] = []
    for idx in range(1, 17):
        fb = os.getenv(f"HERMES_SWARM_LARGE_CONTEXT_FALLBACK_{idx}", "").strip()
        if fb and fb not in large_context_fallbacks:
            large_context_fallbacks.append(fb)

    if not large_context_fallbacks:
        try:
            from agent.model_metadata import get_model_context_length_quick
        except Exception:
            get_model_context_length_quick = _model_context_length
        primary_ctx = get_model_context_length_quick(primary) if primary else 0
        large_context_fallbacks = [
            model for model in fallback_chain
            if get_model_context_length_quick(model) > primary_ctx
        ]

    scout_fallbacks: List[str] = []
    for idx in range(1, 17):
        fb = os.getenv(f"HERMES_SWARM_SCOUT_FALLBACK_{idx}", "").strip()
        if fb and fb not in scout_fallbacks:
            scout_fallbacks.append(fb)

    if not scout_fallbacks:
        scout_fallbacks = [
            "minimax/MiniMax-M2.7",
            "opencode-go/qwen3.6-plus",
            "ollama/glm-5.1",
            "xiaomi/mimo-v2.5-pro",
            primary,
        ]

    scout_fallbacks = [m for m in scout_fallbacks if m and m not in blocked and "\n" not in m and "HERMES_SWARM_" not in m]

    return {
        "primary": primary,
        "fallbacks": fallback_chain,
        "selection_policy": os.getenv("HERMES_SWARM_SELECTION_POLICY", "cost-balanced"),
        "large_context_fallbacks": large_context_fallbacks,
        "scout_fallbacks": scout_fallbacks,
        "estimated_tokens": estimated_tokens,
        "routing_hint": dict(routing_hint or {}),
    }


_HERMES_CODE_DEFAULT_MAX_FALLBACKS = 64
_HERMES_CODE_LARGE_CONTEXT_MIN = 512_000
_HERMES_CODE_LARGE_CONTEXT_TRIGGER = 96_000
_HERMES_CODE_ADVERTISED_CONTEXT_LIMIT = 256_000


def _hermes_code_max_fallbacks() -> int:
    try:
        return max(1, int(os.getenv("HERMES_CODE_MAX_FALLBACKS", str(_HERMES_CODE_DEFAULT_MAX_FALLBACKS))))
    except Exception:
        return _HERMES_CODE_DEFAULT_MAX_FALLBACKS


def _provider_prefix_from_model(model: str) -> str:
    raw = str(model or "").strip().lower()
    if not raw:
        return ""
    if "/" in raw:
        return raw.split("/", 1)[0]
    return raw


def _passthrough_fallback_provider_excluded(model: str, *, privacy: bool = False) -> bool:
    """Return True when a model family is disabled for automatic fallback.

    Explicit model requests are still handled elsewhere in strict mode. This
    guard only prevents background fallback selection from drifting into auth-
    backed CLI providers that can sit silent or require fresh operator login.
    """
    prefix = _provider_prefix_from_model(model)
    if not prefix:
        return False
    env_name = "HERMES_PRIVACY_EXCLUDED_FALLBACK_PROVIDERS" if privacy else "HERMES_CODE_EXCLUDED_FALLBACK_PROVIDERS"
    default = "openai-codex,claude-code-cli,mimocode-cli" if privacy else "mimocode-cli"
    raw = os.getenv(env_name, default)
    excluded = {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}
    return prefix in excluded


def _is_prompt_too_long_error(error_text: str) -> bool:
    """Detect upstream "context/prompt too large" error strings."""
    txt = str(error_text or "").lower()
    return any(
        marker in txt
        for marker in (
            "model_max_prompt_tokens_exceeded",
            "prompt token count of",
            "max prompt tokens",
            "too many input tokens",
            "context length exceeded",
            "context overflow",
        )
    )


def _copilot_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Hermes/OpenAI tool-loop transcripts into Copilot-safe messages.

    Copilot's OpenAI-compat surface rejects bare ``role="tool"`` messages and
    is strict about the tool_calls / tool response pairing.  We collapse each
    ``assistant tool_calls + tool result`` round-trip into a single synthetic
    user message that preserves the call/result semantics, and drop dangling
    assistant ``tool_calls`` so we don't trigger Copilot's
    "tool_call_ids did not have response messages" 400.
    """
    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(msgs):
        m = msgs[i] if isinstance(msgs[i], dict) else {}
        role = m.get("role")
        if role == "assistant" and isinstance(m.get("tool_calls"), list) and m.get("tool_calls"):
            tool_calls = m.get("tool_calls") or []
            tool_results: List[Dict[str, Any]] = []
            j = i + 1
            while j < len(msgs):
                nxt = msgs[j] if isinstance(msgs[j], dict) else {}
                if nxt.get("role") != "tool":
                    break
                tool_results.append(nxt)
                j += 1
            parts: List[str] = []
            assistant_text = str(m.get("content") or "").strip()
            if assistant_text:
                parts.append(f"Assistant context before tool use:\n{assistant_text}")
            parts.append("Prior tool interaction summary:")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = str(tc.get("id") or tc.get("tool_call_id") or "")
                fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                fn_name = str(fn.get("name") or "")
                fn_args = str(fn.get("arguments") or "{}")
                matched = next(
                    (tr for tr in tool_results if str(tr.get("tool_call_id") or "") == tc_id),
                    None,
                )
                tr_content = str((matched or {}).get("content") or "")
                parts.append(
                    f"- tool={fn_name} tool_call_id={tc_id}\narguments={fn_args}\nresult={tr_content}"
                )
            result.append({"role": "user", "content": "\n\n".join(parts)})
            i = j
            continue
        if role == "tool":
            result.append({
                "role": "user",
                "content": f"Tool result:\n{m.get('content', '(tool result)')}",
            })
            i += 1
            continue
        if role == "assistant" and m.get("tool_calls"):
            clean = dict(m)
            clean.pop("tool_calls", None)
            result.append(clean)
            i += 1
            continue
        result.append(m)
        i += 1
    return result


def _hermes_code_skip_toxic_fallback(provider_model: str) -> bool:
    """Return True if a fallback entry has accumulated extreme recent failures.

    Some models (e.g. ``opencode-go/deepseek-v4-pro``) are repeatedly broken and
    re-entering the cooldown DB at 300s per trip would still let the swarm hit
    them every ~5 minutes.  After three recent failures we hard-skip them for
    the rest of the process lifetime.
    """
    raw = str(provider_model or "").strip().lower()
    if raw != "opencode-go/deepseek-v4-pro":
        return False
    try:
        from agent.model_cooldown_db import provider_failure_count
        return provider_failure_count("opencode-go", "opencode-go/deepseek-v4-pro") >= 3
    except Exception:
        return False


def _build_hermes_code_model_pool() -> List[str]:
    configured = os.getenv("HERMES_CODE_MODEL", "").strip()
    candidates: List[str] = [configured] if configured else []
    for idx in range(1, _hermes_code_max_fallbacks() + 1):
        fb = os.getenv(f"HERMES_CODE_FALLBACK_{idx}", "").strip()
        if fb:
            candidates.append(fb)


    seen: set[str] = set()
    ordered: List[str] = []
    for model in candidates:
        model = str(model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        if "\n" in model or "HERMES_CODE_" in model:
            logger.warning(
                "[api_server] ignoring malformed HERMES_CODE model candidate: %s",
                model,
            )
            continue
        ordered.append(model)
    return ordered


def _build_hermes_privacy_model_pool() -> List[str]:
    """Build the privacy model pool from HERMES_PRIVACY_MODEL and HERMES_PRIVACY_FALLBACK_{N}.

    Mirrors _build_hermes_code_model_pool() but reads the parallel HERMES_PRIVACY_*
    env-var prefix. Only models explicitly configured here are included — the privacy
    chain is fully operator-controlled, never inheriting from HERMES_CODE_*.
    """
    configured = os.getenv("HERMES_PRIVACY_MODEL", "").strip()
    candidates: List[str] = [configured] if configured else []
    for idx in range(1, _hermes_code_max_fallbacks() + 1):
        fb = os.getenv(f"HERMES_PRIVACY_FALLBACK_{idx}", "").strip()
        if fb:
            candidates.append(fb)

    seen: set[str] = set()
    ordered: List[str] = []
    for model in candidates:
        model = str(model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        if "\n" in model or "HERMES_PRIVACY_" in model:
            logger.warning(
                "[api_server] ignoring malformed HERMES_PRIVACY model candidate: %s",
                model,
            )
            continue
        ordered.append(model)
    return ordered

def _build_hermes_code_audio_pool() -> List[str]:
    """Build the audio-capable model pool from HERMES_CODE_AUDIO_MODEL and HERMES_CODE_AUDIO_FALLBACK_{N}."""
    configured = os.getenv("HERMES_CODE_AUDIO_MODEL", "").strip()
    candidates: List[str] = [configured] if configured else []
    for idx in range(1, _hermes_code_max_fallbacks() + 1):
        fb = os.getenv(f"HERMES_CODE_AUDIO_FALLBACK_{idx}", "").strip()
        if fb:
            candidates.append(fb)
    # Hardcoded default: Google Gemini via native API (supports audio via inlineData)
    if not candidates:
        candidates.append("google/gemini-2.5-flash")

    seen: set[str] = set()
    ordered: List[str] = []
    for model in candidates:
        model = str(model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        if "\n" in model or "HERMES_CODE_" in model:
            logger.warning(
                "[api_server] ignoring malformed HERMES_CODE_AUDIO model candidate: %s",
                model,
            )
            continue
        ordered.append(model)
    return ordered


# Ordered list of GitHub Copilot Enterprise models that are exposed as directly
# selectable passthrough entries in /v1/models.  Ordered by quality tier.
_GHE_COPILOT_PASSTHROUGH_MODELS: tuple[tuple[str, str], ...] = (
    # (bare_copilot_id, description_suffix)
    # Claude models — in catalog but /v1/messages endpoint returns HTML404 (routing bug)
    ("claude-sonnet-4.6",         "Anthropic Claude Sonnet 4.6 via GHE Copilot — vision, 200k context"),
    ("claude-opus-4.6",           "Anthropic Claude Opus 4.6 via GHE Copilot — vision, 200k context"),
    ("claude-sonnet-4.5",         "Anthropic Claude Sonnet 4.5 via GHE Copilot — vision"),
    ("claude-haiku-4.5",          "Anthropic Claude Haiku 4.5 via GHE Copilot — fast, vision"),
    # OpenAI models — confirmed working
    ("gpt-5.4",                   "OpenAI GPT-5.4 via GHE Copilot — vision, Responses API"),
    ("gpt-5.4-mini",              "OpenAI GPT-5.4-mini via GHE Copilot — vision, Responses API"),
    ("gpt-5.5",                   "OpenAI GPT-5.5 via GHE Copilot — vision, Responses API"),
    ("gpt-5-mini",                "OpenAI GPT-5-mini via GHE Copilot — Chat Completions API"),
    ("gpt-5.3-codex",             "OpenAI GPT-5.3-codex via GHE Copilot — agentic coding, Responses API"),
    ("gpt-4.1",                   "OpenAI GPT-4.1 via GHE Copilot"),
    ("gpt-4o-mini",               "OpenAI GPT-4o-mini via GHE Copilot — fast"),
    # Google models — confirmed working
    ("gemini-3.1-pro-preview",    "Google Gemini 3.1 Pro Preview via GHE Copilot — vision"),
    # Removed: claude-sonnet-4 (not in catalog), gpt-4o (not in catalog),
    # gpt-5.2-codex (not in catalog), gemini-3-pro-preview (not in catalog),
    # gemini-3-flash-preview (not in catalog), gemini-2.5-pro (blocked by org policy),
    # grok-code-fast-1 (not in catalog)
)

_GHE_PREFIX = "github-copilot-enterprise"
_GHE_VISION_PREFIXES = ("claude-", "gemini-", "gpt-4o", "gpt-4.1", "gpt-5.4", "gpt-5.5", "gpt-5-mini")
_GHE_CODEX_MARKERS = ("codex",)


def _ghe_model_supports_vision(bare_id: str) -> bool:
    b = bare_id.lower()
    if any(m in b for m in _GHE_CODEX_MARKERS):
        return False  # codex models are text-only
    return any(b.startswith(p) for p in _GHE_VISION_PREFIXES)


def _advertised_ghe_passthrough_models(now: int) -> list:
    """Return OpenAI-format model dicts for all selectable GHE Copilot passthrough models."""
    seen: set[str] = set()
    out: list = []

    # 1. Models explicitly configured in the hermes-code pool take priority order.
    for model in _build_hermes_code_model_pool():
        if not model.startswith(_GHE_PREFIX + "/"):
            continue
        bare = model[len(_GHE_PREFIX) + 1:]
        if bare in seen:
            continue
        seen.add(bare)
        vision = _ghe_model_supports_vision(bare)
        out.append(_ghe_model_entry(model, bare, now, vision))

    # 2. Full curated catalog — any not already emitted above.
    for bare, desc in _GHE_PASSTHROUGH_CATALOG:
        if bare in seen:
            continue
        seen.add(bare)
        model = f"{_GHE_PREFIX}/{bare}"
        vision = _ghe_model_supports_vision(bare)
        out.append(_ghe_model_entry(model, bare, now, vision, extra_desc=desc))

    return out


def _ghe_model_entry(model_id: str, bare_id: str, now: int, vision: bool, extra_desc: str = "") -> dict:
    desc = extra_desc or f"Direct passthrough to GitHub Copilot Enterprise — model: {bare_id}"
    caps = ["text"]
    if vision:
        caps.append("vision")
    return {
        "id": model_id,
        "object": "model",
        "created": now,
        "owned_by": "github-copilot-enterprise",
        "permission": [],
        "root": model_id,
        "parent": "hermes-code",
        "description": desc,
        "capabilities": {"input": caps},
    }


# Alias for use in _advertised_ghe_passthrough_models — refers to the curated list above.
_GHE_PASSTHROUGH_CATALOG = _GHE_COPILOT_PASSTHROUGH_MODELS

def _selectable_hermes_code_model_name(model: str) -> str:
    """Return the provider-normalized model id used for availability checks."""
    raw = str(model or "").strip()
    if "/" not in raw:
        return raw
    prefix, _, name = raw.partition("/")
    provider = _EXPLICIT_MODEL_PROVIDER_ALIASES.get(prefix.strip().lower(), prefix.strip().lower())
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider

        return normalize_model_for_provider(raw, provider)
    except Exception:
        return name.strip() or raw


def _purge_hermes_code_session_sticky(now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    stale = [
        session_id
        for session_id, entry in _HERMES_CODE_SESSION_STICKY.items()
        if now - float(entry.get("last_seen_at") or entry.get("selected_at") or 0.0) > _HERMES_CODE_SESSION_STICKY_TTL
    ]
    for session_id in stale:
        _HERMES_CODE_SESSION_STICKY.pop(session_id, None)
    if len(_HERMES_CODE_SESSION_STICKY) <= _HERMES_CODE_SESSION_STICKY_MAX:
        return
    overflow = sorted(
        _HERMES_CODE_SESSION_STICKY.items(),
        key=lambda item: float(item[1].get("last_seen_at") or item[1].get("selected_at") or 0.0),
    )
    for session_id, _ in overflow[: max(0, len(_HERMES_CODE_SESSION_STICKY) - _HERMES_CODE_SESSION_STICKY_MAX)]:
        _HERMES_CODE_SESSION_STICKY.pop(session_id, None)


def _remember_hermes_code_session_model(session_id: Optional[str], model: str) -> None:
    if _HERMES_CODE_STICKY_DISABLED:
        return
    session_key = str(session_id or "").strip()
    chosen_model = str(model or "").strip()
    if not session_key or not chosen_model:
        return
    now = time.time()
    _purge_hermes_code_session_sticky(now)
    _HERMES_CODE_SESSION_STICKY[session_key] = {
        "model": chosen_model,
        "selected_at": now,
        "last_seen_at": now,
    }


def _clear_hermes_code_session_model(session_id: Optional[str], reason: str = "") -> None:
    session_key = str(session_id or "").strip()
    if not session_key:
        return
    removed = _HERMES_CODE_SESSION_STICKY.pop(session_key, None)
    if removed and reason:
        logger.info(
            "[api_server] cleared sticky hermes-code model for session=%s model=%s reason=%s",
            session_key,
            removed.get("model"),
            reason,
        )


def _sticky_hermes_code_session_model(
    session_id: Optional[str], *, estimated_tokens: int = 0,
) -> Optional[str]:
    if _HERMES_CODE_STICKY_DISABLED:
        return None
    session_key = str(session_id or "").strip()
    if not session_key:
        return None
    now = time.time()
    _purge_hermes_code_session_sticky(now)
    entry = _HERMES_CODE_SESSION_STICKY.get(session_key)
    if not entry:
        return None
    model = str(entry.get("model") or "").strip()
    if not model:
        _HERMES_CODE_SESSION_STICKY.pop(session_key, None)
        return None
    if not _hermes_code_model_is_selectable(model):
        _clear_hermes_code_session_model(session_key, reason="model_unavailable")
        return None
    if estimated_tokens > 0 and not _model_can_handle_context(model, estimated_tokens):
        _clear_hermes_code_session_model(session_key, reason=f"context_mismatch:{estimated_tokens}")
        return None
    entry["last_seen_at"] = now
    logger.info(
        "[api_server] reusing sticky hermes-code model for session=%s model=%s est_tokens=%s",
        session_key,
        model,
        estimated_tokens,
    )
    return model


def _normalize_model_for_runtime_provider(model: str, runtime_provider: str) -> str:
    """Normalize explicit provider-prefixed models to provider-native ids.

    Some providers (notably ollama-cloud and opencode-go) advertise bare model ids
    on ``/models`` and reject requests that still include the gateway/provider
    prefix (e.g. ``opencode-go/deepseek-v4-pro`` instead of ``deepseek-v4-pro``).
    """
    raw = str(model or "").strip()
    provider = str(runtime_provider or "").strip().lower()
    if not raw or not provider:
        return raw
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider
        return normalize_model_for_provider(raw, provider)
    except Exception:
        explicit = _explicit_provider_from_model(raw)
        if explicit == provider and "/" in raw:
            return raw.partition("/")[2].strip() or raw
        return raw


def _hermes_code_selectable_pool(*, estimated_tokens: int = 0) -> List[str]:
    # Short-circuit: find the first working model instead of evaluating
    # every fallback.  Each _hermes_code_model_is_selectable call may
    # load credential pools and check cooldowns, so N fallbacks × N
    # credential pools × N selectability checks adds up fast.
    for model in _build_hermes_code_model_pool():
        if _passthrough_fallback_provider_excluded(model):
            continue
        if _hermes_code_model_is_selectable(model):
            selectable = [model]
            break
    else:
        return []

    if estimated_tokens > 0:
        if not _model_can_handle_context(model, estimated_tokens):
            # First available can't handle the context — scan for one that can.
            fitting = [
                m for m in _build_hermes_code_model_pool()
                if not _passthrough_fallback_provider_excluded(m)
                and _hermes_code_model_is_selectable(m)
                and _model_can_handle_context(m, estimated_tokens)
            ]
            if fitting:
                selectable = fitting[:1]
            # If nothing fits, return the first available anyway — caller
            # may truncate or the model may handle it.

        if estimated_tokens >= _HERMES_CODE_LARGE_CONTEXT_TRIGGER:
            large_context = [
                m for m in _build_hermes_code_model_pool()
                if not _passthrough_fallback_provider_excluded(m)
                and _hermes_code_model_is_selectable(m)
                and _model_context_length(m) >= _HERMES_CODE_LARGE_CONTEXT_MIN
            ]
            if large_context:
                selectable = large_context[:1]

    return selectable


def _ordered_hermes_code_selectable_pool(*, estimated_tokens: int = 0) -> List[str]:
    global _SELECTABLE_POOL_CACHE, _SELECTABLE_POOL_CACHE_AT
    now = time.time()
    if _SELECTABLE_POOL_CACHE and (now - _SELECTABLE_POOL_CACHE_AT) < _SELECTABLE_POOL_CACHE_TTL:
        selectable = _SELECTABLE_POOL_CACHE
    else:
        selectable = [
            m for m in _build_hermes_code_model_pool()
            if not _passthrough_fallback_provider_excluded(m)
            and _hermes_code_model_is_selectable(m)
        ]
        _SELECTABLE_POOL_CACHE = selectable
        _SELECTABLE_POOL_CACHE_AT = now

    if not selectable:
        return []

    if estimated_tokens > 0:
        fitting = [m for m in selectable if _model_can_handle_context(m, estimated_tokens)]
        if fitting:
            selectable = fitting
        if estimated_tokens >= _HERMES_CODE_LARGE_CONTEXT_TRIGGER:
            large_context = [m for m in selectable if _model_context_length(m) >= _HERMES_CODE_LARGE_CONTEXT_MIN]
            if large_context:
                selectable = large_context
    return selectable


def _next_hermes_code_rr_index(window: int) -> int:
    global _HERMES_CODE_RR_INDEX
    if window <= 1:
        return 0
    with _HERMES_CODE_RR_LOCK:
        idx = _HERMES_CODE_RR_INDEX % window
        _HERMES_CODE_RR_INDEX = (_HERMES_CODE_RR_INDEX + 1) % max(window, 1)
        return idx


def _select_hermes_code_model(
    *,
    estimated_tokens: int = 0,
    session_id: Optional[str] = None,
    require_vision: bool = False,
    require_audio: bool = False,
    avoid_external_image_incompatible: bool = False,
) -> str:
    sticky = _sticky_hermes_code_session_model(session_id, estimated_tokens=estimated_tokens)
    if sticky:
        if require_vision and not _model_supports_vision(sticky):
            sticky = None
        elif require_audio and not _model_supports_audio_input(sticky):
            sticky = None
        elif avoid_external_image_incompatible and sticky.startswith("github-copilot"):
            sticky = None
    if sticky:
        return sticky
    selectable = _ordered_hermes_code_selectable_pool(estimated_tokens=estimated_tokens)
    if require_vision:
        vision_models = [m for m in selectable if _model_supports_vision(m)]
        if vision_models:
            selectable = vision_models
    if require_audio:
        audio_pool = _build_hermes_code_audio_pool()
        audio_selectable = [m for m in audio_pool if _hermes_code_model_is_selectable(m)]
        if audio_selectable:
            selectable = audio_selectable
    if avoid_external_image_incompatible:
        compatible = [m for m in selectable if not m.startswith("github-copilot")]
        if compatible:
            selectable = compatible
    if selectable:
        try:
            rr_window = max(1, int(os.getenv("HERMES_CODE_ROUND_ROBIN_WINDOW", "3")))
        except Exception:
            rr_window = 3
        if rr_window > 1:
            return selectable[_next_hermes_code_rr_index(min(rr_window, len(selectable)))]
        return selectable[0]
    for model in _build_hermes_code_model_pool():
        if _passthrough_fallback_provider_excluded(model):
            continue
        if _hermes_code_model_is_selectable(model):
            return model
    return os.getenv("HERMES_CODE_MODEL", "minimax/MiniMax-M2.7")



def _hermes_privacy_model_is_selectable(model: str) -> bool:
    """Availability check for the privacy model chain. Reuses hermes-code machinery."""
    return _hermes_code_model_is_selectable(model)


def _hermes_privacy_selectable_pool(*, estimated_tokens: int = 0) -> List[str]:
    for model in _build_hermes_privacy_model_pool():
        if _passthrough_fallback_provider_excluded(model, privacy=True):
            continue
        if _hermes_privacy_model_is_selectable(model):
            return [model]
    return []


def _select_hermes_privacy_model(
    *,
    estimated_tokens: int = 0,
    session_id: Optional[str] = None,
    require_vision: bool = False,
    require_audio: bool = False,
    avoid_external_image_incompatible: bool = False,
) -> str:
    """Select a model from the privacy-restricted chain."""
    selectable = _hermes_privacy_selectable_pool(estimated_tokens=estimated_tokens)
    if require_vision:
        vision_models = [m for m in selectable if _model_supports_vision(m)]
        if vision_models:
            selectable = vision_models
    if avoid_external_image_incompatible:
        compatible = [m for m in selectable if not m.startswith("github-copilot")]
        if compatible:
            selectable = compatible
    if selectable:
        return selectable[0]
    for model in _build_hermes_privacy_model_pool():
        if _passthrough_fallback_provider_excluded(model, privacy=True):
            continue
        if _hermes_privacy_model_is_selectable(model):
            return model
    return os.getenv("HERMES_PRIVACY_MODEL", "")


def _hermes_privacy_advertised_context_length() -> int:
    selectable = _hermes_privacy_selectable_pool()
    candidates = selectable or _build_hermes_privacy_model_pool()
    lengths = [_model_context_length(model) for model in candidates]
    lengths = [length for length in lengths if length > 0]
    if lengths:
        return min(max(lengths), _HERMES_CODE_ADVERTISED_CONTEXT_LIMIT)
    selected = _select_hermes_privacy_model()
    return min(_model_context_length(selected) or 128_000, _HERMES_CODE_ADVERTISED_CONTEXT_LIMIT)


def _hermes_privacy_advertised_max_output_tokens() -> int:
    ctx = _hermes_privacy_advertised_context_length()
    if ctx >= 1_000_000:
        return 128_000
    if ctx >= 400_000:
        return 64_000
    return 16_384

def _hermes_code_advertised_context_length() -> int:
    selectable = _hermes_code_selectable_pool()
    candidates = selectable or _build_hermes_code_model_pool()
    lengths = [_model_context_length(model) for model in candidates]
    lengths = [length for length in lengths if length > 0]
    if lengths:
        return min(max(lengths), _HERMES_CODE_ADVERTISED_CONTEXT_LIMIT)
    selected = _select_hermes_code_model()
    return min(_model_context_length(selected) or 128_000, _HERMES_CODE_ADVERTISED_CONTEXT_LIMIT)


def _hermes_code_advertised_max_output_tokens() -> int:
    ctx = _hermes_code_advertised_context_length()
    if ctx >= 1_000_000:
        return 128_000
    if ctx >= 400_000:
        return 64_000
    return 16_384


def _hermes_code_model_is_selectable(model: str) -> bool:
    """Non-mutating availability check for the ordered hermes-code chain.

    The generic swarm availability path may resolve runtime providers through
    pool ``select()`` calls.  That is fine for execution, but it can rotate or
    otherwise perturb credential-pool state while merely choosing the first
    hermes-code candidate.  Keep this check read-only so the configured order is
    honored: best model first, then fall through only when credentials/cooldown
    make that model genuinely unavailable.
    """
    raw = str(model or "").strip()
    if not raw:
        return False
    prefix, _, model_name = raw.partition("/")
    prefix = prefix.lower().strip()
    model_name = _selectable_hermes_code_model_name(raw)

    if prefix == "openai":
        try:
            from agent.credential_pool import load_pool
            from agent.model_cooldown_db import model_cooldown_remaining

            pool = load_pool("openai-codex")
            if not pool.has_available():
                return False
            entries = [entry for entry in pool.entries() if getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")]
            if not entries:
                return False
            # Also check model-specific cooldown (e.g., empty bash blacklist for gpt-5.5).
            # This catches per-model cooling set when a model returns empty bash commands.
            _model_remaining = max(
                model_cooldown_remaining("openai", raw),
                model_cooldown_remaining("openai-codex", raw),
            )
            if _model_remaining and _model_remaining > 0:
                return False
            return True
        except Exception:
            return False

    if prefix == "github-copilot":
        try:
            from agent.credential_pool import load_pool
            from agent.model_cooldown_db import model_cooldown_remaining
            from hermes_cli.models import _copilot_catalog_ids

            pool = load_pool("copilot")
            if not pool.has_available():
                return False
            public_base = os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com").rstrip("/")
            entries = [
                entry for entry in pool.entries()
                if str(getattr(entry, "base_url", "") or "").rstrip("/") == public_base
                and (getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", ""))
            ]
            if not entries:
                try:
                    from hermes_cli.auth import read_credential_pool

                    entries = [
                        entry for entry in read_credential_pool("copilot")
                        if str(entry.get("base_url") or "").rstrip("/") == public_base
                        and (entry.get("access_token") or entry.get("runtime_api_key"))
                    ]
                except Exception:
                    pass
            if not entries:
                return False
            # Extract API key from the first available entry to fetch the catalog
            first_entry = entries[0]
            api_key = getattr(first_entry, "runtime_api_key", None) or getattr(first_entry, "access_token", None)
            if isinstance(first_entry, dict):
                api_key = first_entry.get("runtime_api_key") or first_entry.get("access_token")
            catalog_ids = _copilot_catalog_ids(api_key=api_key)
            if catalog_ids and model_name not in catalog_ids:
                # Model not in catalog - skip it
                return False
            # If catalog is empty but we have credentials, optimistically allow the model
            # (catalog fetch may fail due to permissions/network, but credentials exist)
            remaining = model_cooldown_remaining("copilot", model_name, base_url=public_base)
            return not (remaining and remaining > 0)
        except Exception:
            return False

    if prefix == "github-copilot-enterprise":
        try:
            from agent.credential_pool import load_pool
            from agent.model_cooldown_db import model_cooldown_remaining
            from hermes_cli.models import _copilot_catalog_ids

            pool = load_pool("copilot")
            enterprise_base = os.getenv("GITHUB_COPILOT_ENTERPRISE_BASE_URL", "").rstrip("/")
            if not enterprise_base:
                for entry in pool.entries():
                    candidate = str(getattr(entry, "base_url", "") or "").rstrip("/")
                    if "copilot-api." in candidate.lower():
                        enterprise_base = candidate
                        break
            if not enterprise_base:
                return False
            entries = [
                entry for entry in pool.entries()
                if str(getattr(entry, "base_url", "") or "").rstrip("/") == enterprise_base
                and (getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", ""))
            ]
            if not entries:
                try:
                    from hermes_cli.auth import read_credential_pool

                    entries = [
                        entry for entry in read_credential_pool("copilot")
                        if str(entry.get("base_url") or "").rstrip("/") == enterprise_base
                        and (entry.get("access_token") or entry.get("runtime_api_key"))
                    ]
                except Exception:
                    pass
            if not entries:
                return False
            first_entry = entries[0]
            api_key = getattr(first_entry, "runtime_api_key", None) or getattr(first_entry, "access_token", None)
            if isinstance(first_entry, dict):
                api_key = first_entry.get("runtime_api_key") or first_entry.get("access_token")
            catalog_ids = _copilot_catalog_ids(api_key=api_key, base_url=enterprise_base)
            if catalog_ids and model_name not in catalog_ids:
                return False
            remaining = model_cooldown_remaining("copilot", model_name, base_url=enterprise_base)
            return not (remaining and remaining > 0)
        except Exception:
            return False

    if prefix == "mimocode-cli":
        # Auto-disabled while subprocess timeouts are being investigated.
        # Re-enable with HERMES_SWARM_ALLOW_MIMOCODE=1.
        if not os.getenv("HERMES_SWARM_ALLOW_MIMOCODE", "").strip().lower() in ("1", "true", "yes"):
            return False
        return _swarm_model_is_available(raw)

    return _swarm_model_is_available(raw)


def _summarize_swarm_messages(
    *,
    system_prompt: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_message: str = "",
) -> str:
    history = conversation_history or []
    parts: List[str] = []
    if system_prompt:
        parts.append(f"SYSTEM:\n{system_prompt}")
    if history:
        trimmed = history[-8:]
        rendered = []
        for msg in trimmed:
            role = str(msg.get("role") or "user").lower().strip()
            if role != "user":
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            rendered.append(f"USER_HISTORY: {content[:1200]}")
        if rendered:
            parts.append("HISTORY:\n" + "\n".join(rendered))
    if user_message:
        parts.append(f"USER:\n{user_message[:2000]}")
    return "\n\n".join(parts)


def _heuristic_swarm_routing_hint(
    *,
    system_prompt: str = "",
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    user_message: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    estimated_tokens: int = 0,
) -> Dict[str, Any]:
    text = _summarize_swarm_messages(
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        user_message=user_message,
    ).lower()
    task_type = "general"
    recommended_tier = "primary"
    action_mode = "answer_only"
    needs_instruction_following = False
    needs_repo_reasoning = False
    needs_bug_judgement = False

    inline_instruction_context = any(
        k in text for k in (
            "repo instructions say:",
            "agents excerpt:",
            "agants excerpt:",
            "use only the provided context",
            "use only the context below",
        )
    )

    if any(k in text for k in ("agents.md", "workspace", "repo", "repository", "codebase", "readme", "package", "custom_components/")):
        needs_instruction_following = True
        needs_repo_reasoning = True
    if any(k in text for k in ("review", "analy", "find bug", "likely bug", "correctness", "why is", "root cause", "debug", "fix", "bug", "regression")):
        needs_bug_judgement = True
    if any(k in text for k in ("implement", "patch", "modify", "change code", "refactor")):
        task_type = "implementation"
        recommended_tier = "premium"
        action_mode = "execute_with_tools"
    elif any(k in text for k in ("review", "code review", "likely bug", "correctness")):
        task_type = "repo_review"
        recommended_tier = "premium"
        action_mode = "answer_only"
    elif any(k in text for k in ("debug", "root cause", "why is", "broken")):
        task_type = "debugging"
        recommended_tier = "premium"
        action_mode = "execute_with_tools" if tools else "answer_only"
    elif any(k in text for k in ("architecture", "design", "best approach", "tradeoff")):
        task_type = "architecture"
        recommended_tier = "premium"
        action_mode = "plan_only"

    if any(k in text for k in ("plan", "outline", "approach", "strategy")) and not any(k in text for k in ("implement", "modify", "edit", "apply", "deploy")):
        action_mode = "plan_only"
    if any(k in text for k in ("proceed", "continue", "fix", "deploy", "run", "test", "commit", "apply", "edit")):
        action_mode = "execute_with_tools" if tools else action_mode

    if tools:
        needs_repo_reasoning = True
        if recommended_tier == "primary":
            recommended_tier = "balanced"

    if estimated_tokens > 6000 and recommended_tier == "primary":
        recommended_tier = "balanced"

    if needs_instruction_following and needs_repo_reasoning and recommended_tier == "primary":
        recommended_tier = "balanced"
    if inline_instruction_context and needs_instruction_following:
        recommended_tier = "premium"
    if needs_bug_judgement and recommended_tier != "premium":
        recommended_tier = "premium"

    return {
        "task_type": task_type,
        "recommended_tier": recommended_tier,
        "action_mode": action_mode,
        "needs_instruction_following": needs_instruction_following,
        "needs_repo_reasoning": needs_repo_reasoning,
        "needs_bug_judgement": needs_bug_judgement,
        "provided_context_only": any(
            k in text for k in (
                "use only the context below",
                "use only the provided context",
                "provided snippets only",
                "do not assume file access",
            )
        ),
        "source": "heuristic",
        "confidence": 0.55,
    }


def _swarm_execution_system_prompt(routing_hint: Optional[Dict[str, Any]]) -> str:
    hint = routing_hint or {}
    parts = [
        "For hermes-swarm tasks: prioritize correctness over style, avoid hallucinating filesystem/tool access, and explicitly distinguish provided context from inferred assumptions.",
        "If AGENTS.md is not available on disk, treat that as missing optional repo guidance, not as a hard failure. Continue with standard assumptions unless the user explicitly required the physical file to be read.",
    ]
    action_mode = str(hint.get("action_mode") or "execute_with_tools").strip().lower()
    if action_mode == "plan_only":
        parts.append(
            "ACTION MODE: plan_only. Provide a concise plan/analysis only. Do not call tools and do not modify files unless the user explicitly changes the request to execute."
        )
    elif action_mode == "answer_only":
        parts.append(
            "ACTION MODE: answer_only. Answer directly from the prompt/context. Do not call tools unless essential to satisfy an explicit tool-use request."
        )
    else:
        parts.append(
            "ACTION MODE: execute_with_tools. Work autonomously, use available client tools when useful, and do not ask for step-by-step confirmations."
        )
    if hint.get("needs_instruction_following"):
        parts.append(
            "If the user supplies AGENTS.md or workflow instructions in the prompt, treat those quoted instructions as authoritative even if you cannot access the repo directly."
        )
        parts.append(
            "When AGENTS.md instructions are supplied inline, never answer that you cannot proceed just because the real file was not found. Use the supplied instructions."
        )
    if hint.get("provided_context_only"):
        parts.append(
            "Use only the context provided in the request. Do not claim files are missing, unreadable, or present unless the prompt itself states that."
        )
        parts.append(
            "Do not say that AGENTS.md, helper scripts, or repo files are missing when their relevant contents are already quoted in the prompt."
        )
    if hint.get("needs_bug_judgement"):
        parts.append(
            "Rank real correctness issues above style or micro-optimizations, and prefer saying 'only 2 high-confidence issues' over inventing a weak third issue."
        )
    return "\n".join(parts)


def _extract_agent_result_text(result: Any) -> str:
    """Best-effort extraction of assistant text from agent results."""
    if isinstance(result, str):
        return result.strip()
    if not isinstance(result, dict):
        return ""
    final_response = str(result.get("final_response") or "").strip()
    if final_response:
        return final_response
    for msg in reversed(list(result.get("messages", []))):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    error_text = str(result.get("error") or "").strip()
    return error_text


def _parse_loose_json_object(response_text: str) -> Dict[str, Any]:
    """Parse a JSON-ish object from model output."""
    text = str(response_text or "").strip()

    def _extract_first_balanced_object(raw: str) -> str:
        start = raw.find("{")
        if start < 0:
            return raw
        depth = 0
        in_string = False
        escape = False
        quote = ""
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue
            if ch in {'"', "'"}:
                in_string = True
                quote = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start : idx + 1]
        return raw[start:]

    candidates: List[str] = []
    fenced = re.findall(r"```(?:json|python)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(candidate.strip() for candidate in fenced if candidate.strip())
    balanced = _extract_first_balanced_object(text).strip()
    if balanced:
        candidates.append(balanced)
    if text:
        candidates.append(text)

    seen: set[str] = set()
    last_error: Optional[Exception] = None
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc

    raise ValueError(f"No JSON object found: {text[:200]}") from last_error


def _client_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if isinstance(func, dict):
            name = str(func.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def _agents_prefetch_tool_names(tool_names: set[str]) -> Optional[tuple[str, str]]:
    """Return the search/read tool pair available for AGENTS prefetch.

    Supports both Hermes-native API tool names (search_files/read_file) and the
    OpenCode client tool names exposed through hermes-swarm (glob/read).
    """
    if {"search_files", "read_file"}.issubset(tool_names):
        return ("search_files", "read_file")
    if {"glob", "read"}.issubset(tool_names):
        return ("glob", "read")
    return None


def _extract_tool_result_by_call_id(messages: List[Dict[str, Any]], call_id: str) -> Optional[str]:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and str(msg.get("tool_call_id") or "") == call_id:
            return str(msg.get("content") or "")
    return None


def _extract_first_path_from_search_result(raw: str) -> Optional[str]:
    try:
        data = _parse_loose_json_object(raw)
    except Exception:
        return None

    def _walk(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            for key in ("path", "file", "filepath"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            paths = value.get("paths")
            if isinstance(paths, list):
                for candidate in paths:
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
            for nested in value.values():
                found = _walk(nested)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = _walk(item)
                if found:
                    return found
        return None

    return _walk(data)


def _needs_agents_prefetch(user_message: str, system_prompt: Optional[str], tools: Any, messages: Optional[List[Dict[str, Any]]] = None) -> bool:
    tool_names = _client_tool_names(tools)
    if _agents_prefetch_tool_names(tool_names) is None:
        return False
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls", []) or []:
            if str(tc.get("id") or "").startswith("agents_prefetch_"):
                return True
        if str(msg.get("tool_call_id") or "").startswith("agents_prefetch_"):
            return True
    text = f"{system_prompt or ''}\n{user_message or ''}".lower()
    return any(marker in text for marker in (
        "agents.md",
        "read agents",
        "repo instructions",
        "workspace",
        "codebase",
    ))


def _build_agents_prefetch_tool_call(name: str, call_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _determine_agents_prefetch_action(messages: List[Dict[str, Any]], tools: Any = None) -> Dict[str, Any]:
    search_call_id = "agents_prefetch_search"
    read_call_id = "agents_prefetch_read"
    search_result = _extract_tool_result_by_call_id(messages, search_call_id)
    read_result = _extract_tool_result_by_call_id(messages, read_call_id)
    tool_names = _client_tool_names(tools)
    tool_pair = _agents_prefetch_tool_names(tool_names) or ("search_files", "read_file")
    search_tool, read_tool = tool_pair
    if read_result:
        return {"status": "done", "agents_text": read_result}
    if search_result:
        path = _extract_first_path_from_search_result(search_result)
        if path:
            read_args: Dict[str, Any] = {"offset": 1, "limit": 260}
            if read_tool == "read_file":
                read_args["path"] = path
            else:
                read_args["filePath"] = path
            return {
                "status": "need_read",
                "tool_call": _build_agents_prefetch_tool_call(
                    read_tool,
                    read_call_id,
                    read_args,
                ),
            }
        return {"status": "done", "agents_text": "No AGENTS.md found in client workspace; proceed with standard assumptions."}
    search_args: Dict[str, Any]
    if search_tool == "search_files":
        search_args = {"pattern": "AGENTS.md", "target": "files", "path": ".", "limit": 5}
    else:
        search_args = {"pattern": "**/AGENTS.md", "path": "."}
    return {
        "status": "need_search",
        "tool_call": _build_agents_prefetch_tool_call(
            search_tool,
            search_call_id,
            search_args,
        ),
    }


def _swarm_model_has_credentials(model: str) -> bool:
    raw = str(model or "").strip()
    if not raw:
        return False
    if "/" not in raw:
        return True
    prefix = raw.split("/", 1)[0].strip().lower()
    if prefix == "github-copilot":
        try:
            from agent.credential_pool import load_pool

            public_base = os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com").rstrip("/")
            if any(
                str(getattr(entry, "base_url", "") or "").rstrip("/") == public_base
                and bool(getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", ""))
                for entry in load_pool("copilot").entries()
            ):
                return True
        except Exception:
            pass
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token
            token, _source = resolve_copilot_token()
            return bool(token)
        except Exception:
            return bool(os.getenv("GITHUB_COPILOT_API_KEY", "").strip())
    if prefix == "github-copilot-enterprise":
        try:
            from agent.credential_pool import load_pool

            base_url = os.getenv("GITHUB_COPILOT_ENTERPRISE_BASE_URL", "").rstrip("/")
            entries = load_pool("copilot").entries()
            if not base_url:
                for entry in entries:
                    candidate = str(getattr(entry, "base_url", "") or "").rstrip("/")
                    if "copilot-api." in candidate.lower():
                        base_url = candidate
                        break
            return bool(base_url) and any(str(getattr(entry, "base_url", "") or "").rstrip("/") == base_url for entry in entries)
        except Exception:
            return False
    if prefix == "opencode-go":
        # Check credential pool for exhaustion status, not just env var
        try:
            from agent.credential_pool import load_pool
            pool = load_pool("opencode-go")
            if pool.has_available():
                return True
            # Pool has no available entries - model is exhausted
            return False
        except Exception:
            pass
        # Fall back to env var check
        return bool(os.getenv("OPENCODE_GO_API_KEY", "").strip())
    if prefix == "opencode-zen":
        return bool(os.getenv("OPENCODE_ZEN_API_KEY", "").strip())
    if prefix == "xiaomi":
        return bool(os.getenv("XIAOMI_API_KEY", "").strip())
    if prefix == "zai":
        return bool(os.getenv("ZAI_API_KEY", "").strip())
    if prefix == "minimax":
        return bool(os.getenv("MINIMAX_API_KEY", "").strip())
    if prefix == "synthetic":
        return bool(os.getenv("SYNTHETIC_API_KEY", "").strip())
    if prefix == "arliai":
        return bool(
            os.getenv("ARLIAI_API_KEY", "").strip()
            or os.getenv("ARLI_API_KEY", "").strip()
            or os.getenv("ARCEEAI_API_KEY", "").strip()
        )
    if prefix == "google":
        # Direct Google API key takes priority; otherwise we fall back to OpenRouter.
        # In containerized deployments keys may live in Hermes auth.json via pool
        # seeding from ~/.hermes/.env rather than process env, so consult the pool too.
        if (
            os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
        ):
            return True
        try:
            from agent.credential_pool import load_pool
            if load_pool("gemini").has_available():
                return True
        except Exception:
            pass
        # No direct key — will route via OpenRouter, so check for that key
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    if prefix == "nvidia":
        if os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVCLOUD_API_KEY", "").strip():
            return True
        return bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    if prefix == "local":
        return False
    if prefix == "ollama-mac":
        # Local LAN endpoint — always reachable if OLLAMA_MAC_BASE_URL is configured
        # Default base URL points to 10.0.0.139:11434; no API key required
        return True
    if prefix == "mlx-mac":
        # Local LAN endpoint — always reachable if MLX_MAC_BASE_URL is configured
        # Default base URL points to 10.0.0.139:11435; no API key required
        return True
    if prefix == "openai":
        # Check env vars first, then fall back to Hermes auth store for Codex OAuth tokens
        if (
            os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("OPENAI_CODEX_API_KEY", "").strip()
            or os.getenv("OPENAI_CODEX_TOKEN", "").strip()
            or os.getenv("CODEX_ACCESS_TOKEN", "").strip()
            or os.getenv("OPENAI_OAUTH_TOKEN", "").strip()
        ):
            return True
        # Check auth store for OpenAI Codex OAuth tokens
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            resolved = resolve_runtime_provider(requested="openai-codex")
            if resolved.get("api_key"):
                return True
        except Exception:
            pass
        try:
            from agent.credential_pool import load_pool

            return any(
                bool(getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", ""))
                for entry in load_pool("openai-codex").entries()
            )
        except Exception:
            pass
        return False
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def _runtime_kwargs_for_model_id(model: str) -> tuple[Dict[str, Any], str]:
    _t_rk = time.time()
    runtime_kwargs: Dict[str, Any] = {}
    provider_prefix = ""
    normalized_model = str(model or "").strip()

    if "/" in normalized_model:
        provider_prefix = normalized_model.split("/", 1)[0].strip().lower()
        # Process-level cache: same provider → same credentials
        # NOTE: github-copilot-enterprise is excluded because it uses pool-based
        # per-endpoint credential resolution that can return different api_modes
        # depending on the model (gpt-5.4 -> codex_responses, gpt-4o-mini -> chat_completions)
        if (
            provider_prefix in _RUNTIME_KWARGS_CACHE
            and provider_prefix not in _RUNTIME_KWARGS_CACHE_EXCLUDED_PROVIDERS
        ):
            cached_at = _RUNTIME_KWARGS_CACHE_AT.get(provider_prefix, 0)
            if time.time() - cached_at < _RUNTIME_KWARGS_CACHE_TTL:
                cached = _RUNTIME_KWARGS_CACHE[provider_prefix]
                result_model = normalized_model.split("/", 1)[1].strip()
                logging.getLogger(__name__).info(
                    "[timing] _runtime_kwargs_for_model_id: CACHED (%.3fs) for model=%s cached_provider=%s",
                    time.time() - _t_rk, model, cached.get("provider", "?"),
                )
                return dict(cached), result_model
        if provider_prefix == "opencode-zen":
            runtime_kwargs["base_url"] = os.getenv("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENCODE_ZEN_API_KEY", "")
            runtime_kwargs["provider"] = "opencode-zen"
        elif provider_prefix == "opencode-go":
            runtime_kwargs["base_url"] = os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENCODE_GO_API_KEY", "")
            runtime_kwargs["provider"] = "opencode-go"
        elif provider_prefix == "anthropic":
            # Anthropic models should use direct Anthropic API, not OpenRouter
            # Block if no ANTHROPIC_API_KEY is configured
            runtime_kwargs["base_url"] = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            runtime_kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
            runtime_kwargs["provider"] = "anthropic"
            runtime_kwargs["api_mode"] = "anthropic_messages"
            if not runtime_kwargs["api_key"]:
                raise RuntimeError(
                    f"Anthropic API key required for model {model}. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
        elif provider_prefix == "claude-code-cli":
            # External process provider — Claude Code CLI subprocess.
            try:
                from hermes_cli.auth import resolve_external_process_provider_credentials
                _cc_creds = resolve_external_process_provider_credentials("claude-code-cli")
                runtime_kwargs["base_url"] = _cc_creds.get("base_url", "claude://codex")
                runtime_kwargs["api_key"] = _cc_creds.get("api_key", "claude-code-cli")
                runtime_kwargs["provider"] = "claude-code-cli"
            except Exception as _cc_exc:
                logging.getLogger(__name__).warning(
                    "[hermes-code] claude-code-cli credential resolution failed: %s", _cc_exc,
                )
                runtime_kwargs["base_url"] = "claude://codex"
                runtime_kwargs["api_key"] = "claude-code-cli"
                runtime_kwargs["provider"] = "claude-code-cli"
        elif provider_prefix == "mimocode-cli":
            # External process provider — MiMoCode CLI subprocess.
            try:
                from hermes_cli.auth import resolve_external_process_provider_credentials
                _mc_creds = resolve_external_process_provider_credentials("mimocode-cli")
                runtime_kwargs["base_url"] = _mc_creds.get("base_url", "mimocode://codex")
                runtime_kwargs["api_key"] = _mc_creds.get("api_key", "mimocode-cli")
                runtime_kwargs["provider"] = "mimocode-cli"
            except Exception as _mc_exc:
                logging.getLogger(__name__).warning(
                    "[hermes-code] mimocode-cli credential resolution failed: %s", _mc_exc,
                )
                runtime_kwargs["base_url"] = "mimocode://codex"
                runtime_kwargs["api_key"] = "mimocode-cli"
                runtime_kwargs["provider"] = "mimocode-cli"
        elif provider_prefix in ("openai", "openai-codex"):
            explicit_codex = provider_prefix == "openai-codex"
            openai_api_key = "" if explicit_codex else os.getenv("OPENAI_API_KEY", "").strip()
            openai_base = "" if explicit_codex else os.getenv("OPENAI_BASE_URL", "").strip()
            codex_key = (
                os.getenv("OPENAI_CODEX_API_KEY", "").strip()
                or os.getenv("OPENAI_OAUTH_TOKEN", "").strip()
                or os.getenv("OPENAI_CODEX_TOKEN", "").strip()
                or os.getenv("CODEX_ACCESS_TOKEN", "").strip()
            )
            if openai_api_key:
                runtime_kwargs["base_url"] = openai_base or "https://api.openai.com/v1"
                runtime_kwargs["provider"] = "openai"
            elif openai_base:
                runtime_kwargs["base_url"] = openai_base
                runtime_kwargs["provider"] = "openai"
            else:
                # Try auth store for Codex OAuth tokens before defaulting
                _codex_resolved = False
                try:
                    from hermes_cli.runtime_provider import resolve_runtime_provider
                    resolved = resolve_runtime_provider(requested="openai-codex")
                    _codex_api_key = resolved.get("api_key", "")
                    _codex_base_url = resolved.get("base_url", "")
                    logger.debug("[hermes-code] resolve_runtime_provider: api_key=%s base_url=%s", 
                        "SET" if _codex_api_key else "EMPTY", _codex_base_url)
                    if _codex_api_key:
                        runtime_kwargs["base_url"] = _codex_base_url or os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
                        runtime_kwargs["api_key"] = _codex_api_key
                        runtime_kwargs["provider"] = "openai-codex"
                        runtime_kwargs["api_mode"] = "codex_responses"
                        _codex_resolved = True
                except Exception as exc:
                    logger.warning("[hermes-code] resolve_runtime_provider failed: %s", exc)
                    pass
                if not _codex_resolved:
                    try:
                        from agent.credential_pool import load_pool

                        _pool = load_pool("openai-codex")
                        _entry = _pool.peek()
                        logger.warning("[hermes-code] codex pool: entry=%s runtime_api_key=%s", 
                            getattr(_entry, 'label', 'unknown') if _entry else 'NONE',
                            "SET" if getattr(_entry, 'runtime_api_key', None) else "EMPTY")
                        _codex_api_key = getattr(_entry, "runtime_api_key", "") if _entry else ""
                        if _codex_api_key:
                            runtime_kwargs["base_url"] = getattr(_entry, "runtime_base_url", None) or getattr(_entry, "base_url", "") or os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
                            runtime_kwargs["api_key"] = _codex_api_key
                            runtime_kwargs["provider"] = "openai-codex"
                            runtime_kwargs["api_mode"] = "codex_responses"
                            runtime_kwargs["credential_pool"] = _pool
                            _codex_resolved = True
                    except Exception as exc:
                        logger.warning("[hermes-code] codex pool load failed: %s", exc)
                        pass
                if not _codex_resolved:
                    runtime_kwargs["base_url"] = os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
                    runtime_kwargs["provider"] = "openai-codex"
            if not runtime_kwargs.get("api_key"):
                runtime_kwargs["api_key"] = openai_api_key or codex_key
        elif provider_prefix == "github-copilot":
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider

                resolved = resolve_runtime_provider(requested="copilot")
                runtime_kwargs["base_url"] = resolved.get("base_url") or os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com")
                runtime_kwargs["api_key"] = resolved.get("api_key") or ""
                runtime_kwargs["provider"] = resolved.get("provider") or "copilot"
                runtime_kwargs["api_mode"] = resolved.get("api_mode")
                runtime_kwargs["credential_pool"] = resolved.get("credential_pool")
            except Exception as exc:
                logging.warning(f"[API_SERVER] Swarm: failed to resolve Copilot runtime provider: {exc}")
                _copilot_token = os.getenv("GITHUB_COPILOT_API_KEY", "")
                if not _copilot_token:
                    try:
                        from hermes_cli.copilot_auth import resolve_copilot_token
                        _copilot_token, _copilot_source = resolve_copilot_token()
                        if _copilot_token:
                            logging.warning(f"[API_SERVER] Swarm: resolved Copilot token from {_copilot_source}")
                    except Exception as inner_exc:
                        logging.warning(f"[API_SERVER] Swarm: failed to resolve Copilot token: {inner_exc}")
                runtime_kwargs["base_url"] = os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com")
                runtime_kwargs["api_key"] = _copilot_token
                runtime_kwargs["provider"] = "copilot"
            if not runtime_kwargs.get("api_key"):
                try:
                    from agent.credential_pool import load_pool

                    public_base = os.getenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com").rstrip("/")
                    for entry in load_pool("copilot").entries():
                        base = str(getattr(entry, "base_url", "") or "").rstrip("/")
                        token = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "") or ""
                        if base == public_base and token:
                            runtime_kwargs["base_url"] = public_base
                            runtime_kwargs["api_key"] = token
                            runtime_kwargs["provider"] = "copilot"
                            break
                except Exception:
                    pass
        elif provider_prefix == "github-copilot-enterprise":
            runtime_kwargs["base_url"] = os.getenv("GITHUB_COPILOT_ENTERPRISE_BASE_URL", "").rstrip("/")
            runtime_kwargs["base_url"] = os.getenv("GITHUB_COPILOT_ENTERPRISE_BASE_URL", "").rstrip("/")
            runtime_kwargs["api_key"] = ""
            try:
                from agent.credential_pool import load_pool
                from hermes_cli.models import copilot_model_api_mode

                pool_entries = load_pool("copilot").entries()
                if not runtime_kwargs["base_url"]:
                    for entry in pool_entries:
                        candidate = str(getattr(entry, "base_url", "") or "").rstrip("/")
                        if "copilot-api." in candidate.lower():
                            runtime_kwargs["base_url"] = candidate
                            break
                for entry in pool_entries:
                    base = str(getattr(entry, "base_url", "") or "").rstrip("/")
                    if base == runtime_kwargs["base_url"]:
                        raw_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "") or ""
                        # Re-exchange raw GitHub OAuth tokens (gho_*) for GHE Copilot
                        # Pool may have stored raw token if public exchange failed (401)
                        if raw_key.startswith("gho_"):
                            from hermes_cli.copilot_auth import get_copilot_api_token
                            runtime_kwargs["api_key"] = get_copilot_api_token(raw_key, base_url=runtime_kwargs["base_url"])
                        else:
                            runtime_kwargs["api_key"] = raw_key
                        break
                runtime_kwargs["api_mode"] = copilot_model_api_mode(
                    normalized_model,
                    api_key=runtime_kwargs.get("api_key") or None,
                    base_url=runtime_kwargs["base_url"] or None,
                )
            except Exception as exc:
                logging.warning(f"[API_SERVER] failed to resolve enterprise Copilot token: {exc}")
            runtime_kwargs["provider"] = "copilot"
            runtime_kwargs.setdefault("api_mode", "chat_completions")
        elif provider_prefix == "minimax":
            runtime_kwargs["base_url"] = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
            runtime_kwargs["api_key"] = os.getenv("MINIMAX_API_KEY", "")
            runtime_kwargs["provider"] = "minimax"
        elif provider_prefix == "synthetic":
            runtime_kwargs["base_url"] = os.getenv("SYNTHETIC_BASE_URL", "https://api.synthetic.new/openai/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("SYNTHETIC_API_KEY", "")
            runtime_kwargs["provider"] = "synthetic"
        elif provider_prefix == "synthetic-anthropic":
            runtime_kwargs["base_url"] = os.getenv("SYNTHETIC_BASE_URL", "https://api.synthetic.new/anthropic/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("SYNTHETIC_API_KEY", "")
            runtime_kwargs["provider"] = "synthetic-anthropic"
            runtime_kwargs["api_mode"] = "anthropic_messages"
        elif provider_prefix == "openrouter":
            runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "").strip()
            runtime_kwargs["provider"] = "openrouter"
        elif provider_prefix == "groq":
            runtime_kwargs["base_url"] = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("GROQ_API_KEY", "").strip()
            runtime_kwargs["provider"] = "groq"
        elif provider_prefix == "cohere":
            runtime_kwargs["base_url"] = os.getenv("COHERE_BASE_URL", "https://api.cohere.com/compatibility/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("COHERE_API_KEY", "").strip()
            runtime_kwargs["provider"] = "cohere"
        elif provider_prefix == "openai-codex":
            # Explicit handler for openai-codex models — prevents them falling
            # through to the OpenRouter last-resort handler below, which would
            # cause FORCE_FREE_OPENROUTER to block them incorrectly.
            try:
                from agent.credential_pool import load_pool
                _pool = load_pool("openai-codex")
                _entry = _pool.peek()
                _codex_api_key = getattr(_entry, "runtime_api_key", "") if _entry else ""
                if _codex_api_key:
                    runtime_kwargs["base_url"] = getattr(_entry, "runtime_base_url", None) or getattr(_entry, "base_url", "") or os.getenv("OPENAI_CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
                    runtime_kwargs["api_key"] = _codex_api_key
                    runtime_kwargs["provider"] = "openai-codex"
                    runtime_kwargs["api_mode"] = "codex_responses"
                    runtime_kwargs["credential_pool"] = _pool
            except Exception:
                pass
        elif provider_prefix == "cerebras":
            runtime_kwargs["base_url"] = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
            runtime_kwargs["api_key"] = os.getenv("CEREBRAS_API_KEY", "").strip()
            runtime_kwargs["provider"] = "cerebras"
        elif provider_prefix == "arliai":
            runtime_kwargs["base_url"] = (
                os.getenv("ARLIAI_BASE_URL", "").strip()
                or os.getenv("ARCEEAI_BASE_URL", "").strip()
                or "https://api.arliai.com/v1"
            ).rstrip("/")
            runtime_kwargs["api_key"] = (
                os.getenv("ARLIAI_API_KEY", "").strip()
                or os.getenv("ARLI_API_KEY", "").strip()
                or os.getenv("ARCEEAI_API_KEY", "").strip()
            )
            runtime_kwargs["provider"] = "arliai"
        elif provider_prefix == "google":
            _google_key = (
                os.getenv("GOOGLE_API_KEY", "").strip()
                or os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "").strip()
            )
            if _google_key:
                runtime_kwargs["base_url"] = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
                runtime_kwargs["api_key"] = _google_key
                runtime_kwargs["provider"] = "google"
            else:
                try:
                    from agent.credential_pool import load_pool

                    _gemini_pool = load_pool("gemini")
                    _gemini_entry = _gemini_pool.peek()
                    _gemini_key = getattr(_gemini_entry, "runtime_api_key", None) or getattr(_gemini_entry, "access_token", "") or ""
                    if _gemini_key:
                        runtime_kwargs["base_url"] = getattr(_gemini_entry, "runtime_base_url", None) or getattr(_gemini_entry, "base_url", "") or os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
                        runtime_kwargs["api_key"] = _gemini_key
                        runtime_kwargs["provider"] = "google"
                        runtime_kwargs["credential_pool"] = _gemini_pool
                except Exception:
                    pass
            if runtime_kwargs.get("provider") != "google":
                # No Google API key — enforce strict guard when forcing free OpenRouter
                if os.getenv("HERMES_SWARM_FORCE_FREE_OPENROUTER", "").strip().lower() in ("1", "true", "yes"):
                    runtime_kwargs["base_url"] = ""
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "blocked"
                else:
                    # Route to OpenRouter conservatively (requires :free if guard active)
                    runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    runtime_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "").strip()
                    runtime_kwargs["provider"] = "openrouter"
        elif provider_prefix == "nvidia":
            _nvidia_key = (
                os.getenv("NVIDIA_API_KEY", "").strip()
                or os.getenv("NVCLOUD_API_KEY", "").strip()
            )
            if _nvidia_key:
                runtime_kwargs["base_url"] = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
                runtime_kwargs["api_key"] = _nvidia_key
                runtime_kwargs["provider"] = "nvidia"
            else:
                # No NVIDIA API key — enforce openrouter free gate if configured
                if os.getenv("HERMES_SWARM_FORCE_FREE_OPENROUTER", "").strip().lower() in ("1","true","yes"):
                    runtime_kwargs["base_url"] = ""
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "blocked"
                else:
                    runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                    runtime_kwargs["api_key"] = ""
                    runtime_kwargs["provider"] = "openrouter"
        elif provider_prefix == "zai":
            runtime_kwargs["base_url"] = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
            runtime_kwargs["api_key"] = os.getenv("ZAI_API_KEY", "")
            runtime_kwargs["provider"] = "zai"
        elif provider_prefix == "xiaomi":
            # Xiaomi MiMo — API key auth (standard endpoint)
            # Free-tier bootstrap auth is available via mimo CLI but
            # the endpoint restricts direct API access by IP/region.
            base_url = os.getenv("XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
            runtime_kwargs["base_url"] = base_url
            runtime_kwargs["api_key"] = os.getenv("XIAOMI_API_KEY", "")
            runtime_kwargs["provider"] = "xiaomi"
        elif provider_prefix == "nous":
            # Nous API — routes nous/* models (e.g. nvidia/nemotron-3-ultra:free)
            # to the Nous inference endpoint, not OpenRouter.
            runtime_kwargs["base_url"] = os.getenv("NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1")
            runtime_kwargs["api_key"] = os.getenv("NOUS_API_KEY", "")
            runtime_kwargs["provider"] = "nous"
        elif provider_prefix == "qwen":
            runtime_kwargs["base_url"] = os.getenv("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENCODE_ZEN_API_KEY", "")
            runtime_kwargs["provider"] = "alibaba"
        elif provider_prefix in ("ollama-cloud", "ollama"):
            runtime_kwargs["base_url"] = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com/v1")
            runtime_kwargs["api_key"] = os.getenv("OLLAMA_API_KEY", "")
            runtime_kwargs["provider"] = "ollama-cloud"
        elif provider_prefix == "ollama-local":
            runtime_kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            runtime_kwargs["api_key"] = os.getenv("OLLAMA_LOCAL_API_KEY", os.getenv("OLLAMA_API_KEY", ""))
            runtime_kwargs["provider"] = "ollama-cloud"
        elif provider_prefix == "ollama-mac":
            runtime_kwargs["base_url"] = os.getenv("OLLAMA_MAC_BASE_URL", "http://10.0.0.139:11434/v1")
            runtime_kwargs["api_key"] = os.getenv("OLLAMA_MAC_API_KEY", "")
            runtime_kwargs["provider"] = "ollama-mac"
        elif provider_prefix == "mlx-mac":
            runtime_kwargs["base_url"] = os.getenv("MLX_MAC_BASE_URL", "http://10.0.0.139:11435/v1")
            runtime_kwargs["api_key"] = os.getenv("MLX_MAC_API_KEY", "")
            runtime_kwargs["provider"] = "mlx-mac"
        elif provider_prefix not in ("openrouter",):
            # Unknown non-openrouter prefix — still route via OpenRouter as last resort
            runtime_kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            runtime_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
            runtime_kwargs["provider"] = "openrouter"
        # NOTE: provider_prefix == "openrouter" is handled below
        else:
            runtime_kwargs["provider"] = provider_prefix
    else:
        # Special-case external-process providers whose bare model name does
        # not contain a slash and would otherwise be mis-routed by detect_vendor.
        if normalized_model == "claude-code-cli":
            try:
                from hermes_cli.auth import resolve_external_process_provider_credentials
                _cc_creds = resolve_external_process_provider_credentials("claude-code-cli")
                runtime_kwargs["base_url"] = _cc_creds.get("base_url", "claude://codex")
                runtime_kwargs["api_key"] = _cc_creds.get("api_key", "claude-code-cli")
            except Exception:
                runtime_kwargs["base_url"] = "claude://codex"
                runtime_kwargs["api_key"] = "claude-code-cli"
            runtime_kwargs["provider"] = "claude-code-cli"
        elif normalized_model == "mimocode-cli":
            try:
                from hermes_cli.auth import resolve_external_process_provider_credentials
                _mc_creds = resolve_external_process_provider_credentials("mimocode-cli")
                runtime_kwargs["base_url"] = _mc_creds.get("base_url", "mimocode://codex")
                runtime_kwargs["api_key"] = _mc_creds.get("api_key", "mimocode-cli")
            except Exception:
                runtime_kwargs["base_url"] = "mimocode://codex"
                runtime_kwargs["api_key"] = "mimocode-cli"
            runtime_kwargs["provider"] = "mimocode-cli"
        else:
            from hermes_cli.model_normalize import detect_vendor
            detected_provider = detect_vendor(normalized_model)
            if detected_provider:
                runtime_kwargs["provider"] = detected_provider

    # Cache the resolved credentials keyed by provider prefix so subsequent
    # calls for other models under the same provider (e.g. github-copilot/gpt-5.4
    # after github-copilot/gpt-5-mini) skip the expensive resolution.
    if (
        provider_prefix
        and runtime_kwargs.get("api_key")
        and provider_prefix not in _RUNTIME_KWARGS_CACHE_EXCLUDED_PROVIDERS
    ):
        _RUNTIME_KWARGS_CACHE[provider_prefix] = dict(runtime_kwargs)
        _RUNTIME_KWARGS_CACHE_AT[provider_prefix] = time.time()

    if "/" in normalized_model:
        normalized_model = normalized_model.split("/", 1)[1].strip()

    # ── Model-name normalization ───────────────────────────────────────────
    # Map known-bad or deprecated model names to their valid replacements.
    # This catches client-side requests (hermes-code passthrough) for models
    # that don't exist in the provider's API, avoiding spurious 404s.
    _MODEL_NAME_NORMALIZE: Dict[str, str] = {
        # Google Gemini: gemini-3.1-flash-preview does not exist in the API;
        # gemini-3.1-flash-lite-preview is the correct free-tier fast model.
        "gemini-3.1-flash-preview": "gemini-3.1-flash-lite-preview",
    }
    if normalized_model in _MODEL_NAME_NORMALIZE:
        normalized_model = _MODEL_NAME_NORMALIZE[normalized_model]

    logging.getLogger(__name__).info(
        "[timing] _runtime_kwargs_for_model_id: %.3fs for model=%s",
        time.time() - _t_rk, model,
    )
    return runtime_kwargs, normalized_model


def _swarm_model_is_available(model: str) -> bool:
    """Check if a model is available (has credentials AND is not in cooldown/sin-bin).

    This function extends _swarm_model_has_credentials to also check the cooldown DB,
    preventing the swarm from repeatedly selecting rate-limited providers.
    """
    raw = str(model or "").strip()
    if not raw:
        return False

    # Block google/nvidia without keys early.  Consult pool-backed credentials too,
    # because containerized deployments may seed these providers from ~/.hermes/.env
    # into auth.json rather than exporting process-level env vars.
    if raw.startswith("google/"):
        if not _swarm_model_has_credentials(raw):
            logger.info("[api_server] blocking google model %s due to missing provider keys", raw)
            return False
    if raw.startswith("nvidia/"):
        if not (os.getenv("NVIDIA_API_KEY", "").strip() or os.getenv("NVCLOUD_API_KEY", "").strip()):
            logger.info("[api_server] blocking nvidia model %s due to missing provider keys", raw)
            return False

    if not _swarm_model_has_credentials(raw):
        return False

    # ── OpenRouter cost guard ──────────────────────────────────────────────
    # When HERMES_SWARM_FORCE_FREE_OPENROUTER is enabled, reject any model
    # routed through OpenRouter that is not explicitly marked with the ":free"
    # suffix.  This prevents accidental spending on paid OpenRouter models
    # (e.g. google/gemini-2.5-flash costs $0.15/M completion tokens).
    runtime_kwargs, normalized_model_result = _runtime_kwargs_for_model_id(raw)
    provider = str(runtime_kwargs.get("provider") or "").strip().lower()
    if provider == "openrouter" and os.getenv("HERMES_SWARM_FORCE_FREE_OPENROUTER", "").strip().lower() in ("1", "true", "yes"):
        if ":free" not in raw.lower():
            logger.info(
                "[api_server] model %s blocked — FORCE_FREE_OPENROUTER is enabled and model is not :free",
                raw,
            )
            return False

    runtime_kwargs_out, model_name = runtime_kwargs, normalized_model_result
    provider = str(runtime_kwargs_out.get("provider") or "").strip().lower()
    base_url = str(runtime_kwargs_out.get("base_url") or "").strip()
    if not provider or not model_name:
        return True

    try:
        from agent.model_cooldown_db import model_cooldown_remaining
        remaining = model_cooldown_remaining(provider, model_name, base_url=base_url)
        if remaining and remaining > 0:
            logger.info(
                "[api_server] model %s in cooldown (%.0fs remaining) — skipping",
                raw, remaining,
            )
            return False
    except Exception as e:
        logger.warning("[api_server] cooldown DB check failed for %s: %s — assuming available", raw, e)

    if provider == "zai" and _zai_is_peak_hours():
        logger.info(
            "[api_server] model %s skipped during peak hours (14:00-18:00 UTC+8) — quota costs 3x",
            raw,
        )
        return False

    # ── Xiaomi MiMo time-of-use ────────────────────────────────────────────
    # Off-peak hours (16:00-24:00 UTC+8) offer 0.8x credit consumption.
    # Peak hours (00:00-16:00 UTC+8) cost 1.25x credits.
    # Prefer Xiaomi models during off-peak; deprioritise (but don't block) during peak.
    # TTS models are free for a limited time, so they are always available.
    if provider == "xiaomi":
        if _xiaomi_model_is_tts(model_name):
            # TTS models are free during the promotional period — always allow
            pass
        elif _xiaomi_is_peak_hours():
            logger.info(
                "[api_server] xiaomi model %s in peak hours (00:00-16:00 UTC+8) — "
                "credits cost 1.25x, deprioritising in favour of other providers",
                raw,
            )
            # Do not hard-block.  Xiaomi is already placed below cheaper or
            # stronger subscription routes in the hermes-code chain, so ranking
            # handles deprioritisation.  Returning False here would make MiMo
            # unreachable during peak hours even when all earlier models are
            # exhausted/unavailable.

    return True


def _zai_is_peak_hours() -> bool:
    """Check if current time is within ZAI peak hours (14:00-18:00 UTC+8).

    During peak hours, GLM-5.1 and GLM-5-Turbo consume quota at 3x normal rate.
    Off-peak usage is currently 1x through end of June (limited-time benefit).
    """
    import time
    utc_plus_8_offset = 8 * 3600
    utc_plus_8_seconds = time.time() + utc_plus_8_offset
    utc_plus_8_hour = (utc_plus_8_seconds % 86400) // 3600
    return 14 <= utc_plus_8_hour < 18


_XIAOMI_TTS_MODELS = frozenset({
    "mimo-v2-tts",
    "mimo-v2.5-tts",
    "mimo-v2.5-tts-voiceclone",
    "mimo-v2.5-tts-voicedesign",
})


def _xiaomi_model_is_tts(model_name: str) -> bool:
    """Return True if the Xiaomi model is a TTS variant (free during promotional period)."""
    return str(model_name or "").strip().lower() in _XIAOMI_TTS_MODELS


def _xiaomi_is_peak_hours() -> bool:
    """Check if current time is within Xiaomi MiMo peak hours.

    Xiaomi's token-plan pricing:
      - Off-peak (16:00-24:00 UTC+8 / 08:00-16:00 UTC): 0.8x credit consumption
      - Peak (00:00-16:00 UTC+8 / 16:00-00:00 UTC): 1.25x credit consumption

    Returns True during peak hours (00:00-16:00 UTC+8), when credits cost more.
    TTS models are exempt — they are free for a limited time.
    """
    import time
    utc_plus_8_offset = 8 * 3600
    utc_plus_8_seconds = time.time() + utc_plus_8_offset
    utc_plus_8_hour = (utc_plus_8_seconds % 86400) // 3600
    # Peak = 00:00 through 15:59 UTC+8  (0 ≤ hour < 16)
    return 0 <= utc_plus_8_hour < 16


def _is_opencode_user_agent(user_agent: str) -> bool:
    return isinstance(user_agent, str) and "opencode/" in user_agent.lower()


_COMPACTION_SUMMARY_MARKERS = (
    "[CONTEXT COMPACTION — REFERENCE ONLY]",
    "[CONTEXT SUMMARY]:",
)


def _sanitise_compaction_summary(content: str) -> str:
    """Strip or render pi/opencode internal XML annotations from a compaction summary.

    pi includes internal session-tracking blocks like ``<read-files>`` and
    ``<modified-files>`` in the compaction summary it sends to the API.  These
    are already tracked inside pi's own session state — they don't need to be
    forwarded to the model as raw XML.  We convert them to a compact, readable
    prose form instead so the model still has the context without seeing noisy
    markup.
    """
    import re as _re

    def _render_file_list(tag_label: str, body: str) -> str:
        files = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if not files:
            return ""
        listed = "\n".join(f"  - {f}" for f in files)
        return f"{tag_label}:\n{listed}"

    content = _re.sub(
        r"<read-files>([\s\S]*?)</read-files>",
        lambda m: _render_file_list("Files read", m.group(1)),
        content,
        flags=_re.IGNORECASE,
    )
    content = _re.sub(
        r"<modified-files>([\s\S]*?)</modified-files>",
        lambda m: _render_file_list("Files modified", m.group(1)),
        content,
        flags=_re.IGNORECASE,
    )
    return content


def _is_post_compaction_assistant_message(content: str) -> bool:
    """Return True if this assistant message is a context compaction summary.

    pi / opencode triggers compaction when the context window fills up.  The
    resulting summary is injected as an **assistant** message at the tail of the
    conversation.  When Hermes receives that as the last message it would
    silently set user_message="" and produce an empty response.
    """
    if not isinstance(content, str):
        return False
    for marker in _COMPACTION_SUMMARY_MARKERS:
        if marker in content:
            return True
    return False


def _infer_intent_from_compaction_summary(content: str) -> str:
    """Extract the outstanding task from a compaction summary."""
    import re as _re

    content = _sanitise_compaction_summary(content)

    m = _re.search(
        r"##\s*Active\s*Task[\s\S]*?\n([\s\S]+?)(?=\n##|\Z)",
        content,
        _re.IGNORECASE,
    )
    if m:
        task_text = m.group(1).strip()
        if task_text and task_text.lower() not in ("none.", "none", "n/a", ""):
            return (
                f"[Context was compacted. Resuming from the last active task below — "
                f"please continue without repeating completed work.]\n\n"
                f"{task_text}"
            )

    return (
        "[Context was compacted. Please review the summary above and continue "
        "with any outstanding tasks, picking up exactly where the work left off.]"
    )


def _find_last_nonempty_user_message(
    conversation_messages: list,
) -> Any:
    """Walk back through conversation_messages to find the last user message
    with non-blank content.

    pi/opencode sometimes sends an empty user message (content="") at the end
    of a tool-call cycle to signal "please continue".  In that case we fall
    back to the most recent non-empty user message so Hermes has something
    meaningful to act on.

    Returns the original content object (string or multimodal list) so callers
    can preserve image/audio parts in passthrough mode.
    """
    for msg in reversed(conversation_messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if _normalize_chat_content(content).strip():
                return content
    return ""


def _prior_assistant_has_pending_tool_calls(conversation_messages: list) -> bool:
    """Return True if the most recent assistant message in conversation_messages
    has tool_calls attached.

    When pi/opencode sends an empty user message at the end of a turn, there
    are two distinct cases:

    1. The prior assistant message had tool_calls → the agent is mid-loop and
       should continue processing those results without us injecting a stale
       user message as a prompt.
    2. The prior assistant message was a plain text reply → the empty user
       message is a stray "please continue" and we fall back to the last
       non-empty user message.
    """
    for msg in reversed(conversation_messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            return bool(isinstance(tool_calls, list) and tool_calls)
        if role == "user":
            # hit a user message before any assistant — no pending tool calls
            return False
    return False


_OMP_REMINDER_PREFIX = "<system-reminder>\nThe user sent the following message:\n"
_OMP_REMINDER_SUFFIX = "\n\nPlease address this message and continue with your tasks.\n</system-reminder>"


def _unwrap_omp_system_reminder(content: Any) -> Any:
    """Strip the oh-my-pi continuation wrapper from user message content.

    oh-my-pi wraps every user text part on step > 1 with:
        <system-reminder>
        The user sent the following message:
        {original}

        Please address this message and continue with your tasks.
        </system-reminder>

    Sending this verbatim to the provider on every turn causes the model to
    treat the same task as a fresh instruction each step, producing a loop.
    We unwrap it here so the provider sees the original user text.

    Handles both plain string content and OpenAI multimodal content arrays.
    Leaves anything that doesn't match the exact wrapper untouched.
    """
    if isinstance(content, str):
        s = content
        if s.startswith(_OMP_REMINDER_PREFIX) and s.endswith(_OMP_REMINDER_SUFFIX):
            return s[len(_OMP_REMINDER_PREFIX) : len(s) - len(_OMP_REMINDER_SUFFIX)]
        return content
    if isinstance(content, list):
        # Multimodal array — unwrap any text parts that carry the wrapper.
        # Non-text parts (images, audio) are left untouched.
        result = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                unwrapped = _unwrap_omp_system_reminder(text)
                if unwrapped is not text:
                    part = {**part, "text": unwrapped}
            result.append(part)
        return result
    return content

def _looks_like_polling_result(content: Any) -> bool:
    """Return True if a tool result looks like a pending-status poll response.

    Matches short outputs that indicate an async operation is still running:
    InProgress, Running, Pending, queued, etc.  Case-insensitive; strips
    whitespace and common OMP wall-time suffixes before matching.
    """
    if not isinstance(content, str):
        return False
    # Strip OMP wall-time footer ("Wall time: 1.23s") before checking
    text = content.strip()
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("wall time:"):
            text = text[: text.rfind(line)].strip()
            break
    text_lower = text.lower()
    # Short result that is just a status word
    if len(text) <= 60 and any(
        kw in text_lower
        for kw in ("inprogress", "in progress", "running", "pending", "queued", "waiting")
    ):
        return True
    return False


def _detect_and_nudge_tool_loop(
    passthrough_messages: List[Dict[str, Any]],
    threshold: int = 3,
) -> bool:
    """Detect a potential tool-call loop and inject a synthetic recovery
    message into ``passthrough_messages`` to break it.

    A loop is defined as the assistant invoking the **exact same tool call**
    (same function name AND same arguments) ``threshold`` or more consecutive
    times.  Each round-trip is:

        assistant (tool_calls=[...]) → one or more tool results

    We walk the message list backwards from the tail, collecting these
    round-trips until we hit a user message (new instruction resets the count).

    Only considers rounds where **all** tool calls in the round match the
    previous round's calls (same names and arguments).  Using the same tool
    with different arguments (e.g. different bash commands) is normal usage
    and does NOT trigger the detector.

    Returns True if a loop was detected AND a recovery message was injected.
    """
    rounds: List[tuple] = []
    recent_results: List[str] = []
    collecting_results = True
    i = len(passthrough_messages) - 1
    while i >= 0:
        msg = passthrough_messages[i]
        if not isinstance(msg, dict):
            i -= 1
            continue
        role = msg.get("role")
        if role == "user":
            break
        if role == "system":
            i -= 1
            continue
        if role == "tool":
            while i >= 0 and isinstance(passthrough_messages[i], dict) and passthrough_messages[i].get("role") == "tool":
                if collecting_results:
                    recent_results.append(passthrough_messages[i].get("content", ""))
                i -= 1
            collecting_results = False
            continue
        if role == "assistant":
            tcs = msg.get("tool_calls")
            if not isinstance(tcs, list) or not tcs:
                break
            # Build a signature of (function_name, arguments) tuples for the round.
            # This ensures that different arguments (e.g. different bash commands)
            # are NOT treated as identical — only truly identical tool calls loop.
            sig: tuple = tuple(
                (
                    tc.get("function", {}).get("name", "") or "",
                    tc.get("function", {}).get("arguments", "") or "",
                )
                for tc in tcs
                if isinstance(tc, dict)
            )
            if not sig:
                break
            rounds.append(sig)
            i -= 1
            continue
        i -= 1

    if len(rounds) < threshold:
        return False

    recent = rounds[:threshold]
    if len(set(recent)) != 1:
        return False

    # Extract the looped tool name(s) for the recovery message.
    sig = recent[0]
    fn_names = ", ".join(sorted({name for name, _ in sig})) or "<unknown>"
    # Show a brief preview of the repeated command.
    preview = ""
    for name, args in sig:
        if name == "bash" and args:
            # Try to extract the command from JSON args
            import re as _re
            cmd_match = _re.search(r'"command"\s*:\s*"([^"]{1,80})', args)
            if cmd_match:
                preview = f" — command: {cmd_match.group(1)}"
                break
    is_polling = any(_looks_like_polling_result(r) for r in recent_results)
    loop_type = "polling-loop" if is_polling else "stuck-loop"
    logger.warning(
        "[hermes-code] [ACTIVE] %s detected: tool(s) [%s]%s called %d+ times "
        "in a row (total_rounds_seen=%d, total_messages=%d) — injecting recovery message",
        loop_type, fn_names, preview, threshold, len(rounds), len(passthrough_messages),
    )
    # Inject a synthetic system message at the end of passthrough_messages
    # to break the loop.  The model will see this on the next provider call
    # and be forced to acknowledge it cannot re-issue the same tool call.
    recovery_msg = {
        "role": "system",
        "content": (
            f"[hermes-gateway] TOOL LOOP DETECTED: You have invoked the same tool call(s) "
            f"[{fn_names}]{preview} {len(rounds)} times consecutively with identical arguments. "
            f"The tool results are already in the conversation above — do NOT re-issue "
            f"the same tool call. Instead, analyze the results you already received and "
            f"provide a text summary or move on to the next step."
        ),
    }
    passthrough_messages.append(recovery_msg)
    return True


def _merge_session_history_with_request_delta(
    stored_history: List[Dict[str, Any]],
    request_messages: List[Dict[str, Any]],
    user_message: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Merge persisted session history with current request-body delta.

    For ``X-Hermes-Session-Id`` continuations, some clients send only the new
    turn delta (e.g. the latest user message or tool result), while others send
    the full transcript again.  The previous logic replaced request history with
    the DB copy wholesale, which dropped current-turn tool results and made
    server-side session continuity appear stateless.
    """
    stored = list(stored_history or [])
    current = list(request_messages or [])
    if not stored:
        return current[:-1] if current and current[-1].get("role") == "user" else current, user_message
    if not current:
        return stored, user_message

    # Full-history resend: request already contains the stored transcript.
    if len(current) >= len(stored) and current[: len(stored)] == stored:
        if current and current[-1].get("role") == "user":
            return current[:-1], current[-1].get("content", "") or user_message
        return current, user_message

    # Delta mode: append only the current request-body messages onto persisted history.
    if current and current[-1].get("role") == "user":
        return stored + current[:-1], current[-1].get("content", "") or user_message
    return stored + current, user_message


def _persist_passthrough_session_delta(
    db: Any,
    session_id: str,
    *,
    model_name: str,
    system_prompt: Optional[str],
    request_messages: List[Dict[str, Any]],
    assistant_content: Any = None,
    assistant_tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None,
    reasoning_content: Optional[str] = None,
) -> None:
    """Persist only the current request delta + assistant response for passthrough.

    Used only for explicit ``X-Hermes-Session-Id`` flows so server-side session
    continuation works even when the client sends only incremental turns.
    """
    if db is None or not session_id:
        return
    db.ensure_session(
        session_id,
        source="api_server",
        model=model_name,
        system_prompt=system_prompt or None,
    )
    if system_prompt:
        try:
            db.update_system_prompt(session_id, system_prompt)
        except Exception:
            pass

    for msg in request_messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        kwargs: Dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": msg.get("content"),
        }
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            kwargs["tool_calls"] = msg.get("tool_calls")
        if role == "tool" and msg.get("tool_call_id"):
            kwargs["tool_call_id"] = msg.get("tool_call_id")
        db.append_message(**kwargs)

    db.append_message(
        session_id=session_id,
        role="assistant",
        content=assistant_content,
        tool_calls=assistant_tool_calls,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
    )


def _looks_like_roo_condense_request(
    *,
    stream: bool,
    tools: Any,
    system_prompt: str,
    user_message: str,
) -> bool:
    """Return True for Roo's built-in context condensing prompt.

    Roo sends a large non-streaming, no-tools summarization request with stable
    marker text. Handling this as a direct single-shot summarization avoids the
    slower full agent path that can exceed client/proxy patience on huge
    histories.
    """
    if stream or tools:
        return False
    text = f"{system_prompt or ''}\n{user_message or ''}".lower()
    return (
        "this summarization request is a system operation" in text
        and "your task is to create a detailed summary of the conversation so far" in text
        and "please provide your summary based on the conversation so far" in text
    )


def _normalize_chat_content(
    content: Any, *, _max_depth: int = 10, _depth: int = 0,
) -> str:
    """Normalize OpenAI chat message content into a plain text string.

    Some clients (Open WebUI, LobeChat, etc.) send content as an array of
    typed parts instead of a plain string::

        [{"type": "text", "text": "hello"}, {"type": "input_text", "text": "..."}]

    This function flattens those into a single string so the agent pipeline
    (which expects strings) doesn't choke.

    Defensive limits prevent abuse: recursion depth, list size, and output
    length are all bounded.
    """
    if _depth > _max_depth:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:MAX_NORMALIZED_TEXT_LENGTH] if len(content) > MAX_NORMALIZED_TEXT_LENGTH else content

    if isinstance(content, list):
        parts: List[str] = []
        items = content[:MAX_CONTENT_LIST_SIZE] if len(content) > MAX_CONTENT_LIST_SIZE else content
        for item in items:
            if isinstance(item, str):
                if item:
                    parts.append(item[:MAX_NORMALIZED_TEXT_LENGTH])
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"text", "input_text", "output_text"}:
                    text = item.get("text", "")
                    if text:
                        try:
                            parts.append(str(text)[:MAX_NORMALIZED_TEXT_LENGTH])
                        except Exception:
                            pass
                # Silently skip image_url / other non-text parts
            elif isinstance(item, list):
                nested = _normalize_chat_content(item, _max_depth=_max_depth, _depth=_depth + 1)
                if nested:
                    parts.append(nested)
            # Check accumulated size
            if sum(len(p) for p in parts) >= MAX_NORMALIZED_TEXT_LENGTH:
                break
        result = "\n".join(parts)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result

    # Fallback for unexpected types (int, float, bool, etc.)
    try:
        result = str(content)
        return result[:MAX_NORMALIZED_TEXT_LENGTH] if len(result) > MAX_NORMALIZED_TEXT_LENGTH else result
    except Exception:
        return ""


def _preserve_multimodal_chat_content(content: Any) -> Any:
    """Return raw multimodal content when present, else normalized text.

    ``hermes-code`` passthrough should forward image/audio-bearing message parts
    unchanged to upstream providers. Non-list scalar content is normalized to a
    plain string for compatibility with the rest of the API server pipeline.
    """
    if isinstance(content, list):
        return content
    return _normalize_chat_content(content)


# Content types that are valid Anthropic blocks but unsupported/unknown to
# standard OpenAI-compat providers.  Strip these before forwarding so the
# provider doesn't reject the entire request with 400.
_OPENAI_UNSUPPORTED_CONTENT_TYPES: frozenset = frozenset({"document"})


def _strip_unsupported_content_for_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove content block types that OpenAI-compat providers reject.

    Anthropic-format ``document`` blocks (PDF, plain text files) are valid
    for Claude but cause 400 errors on OpenAI, Groq, Ollama, etc.  Strip them
    from the content list so the provider still sees the text portions of each
    message.  If removing all non-text blocks leaves a message with an empty
    content list, collapse it to a plain string so downstream providers don't
    choke on an empty array.

    Only touches messages with list-type content that actually contain
    unsupported blocks — everything else is returned unchanged (no copies).
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out.append(msg)
            continue
        has_unsupported = any(
            isinstance(p, dict) and p.get("type") in _OPENAI_UNSUPPORTED_CONTENT_TYPES
            for p in content
        )
        if not has_unsupported:
            out.append(msg)
            continue
        filtered = [p for p in content if not (isinstance(p, dict) and p.get("type") in _OPENAI_UNSUPPORTED_CONTENT_TYPES)]
        # Collapse to string if only one plain-text part remains
        if len(filtered) == 1 and isinstance(filtered[0], dict) and filtered[0].get("type") == "text":
            out.append({**msg, "content": filtered[0].get("text", "")})
        elif filtered:
            out.append({**msg, "content": filtered})
        else:
            # All content was stripped; keep the message with empty string so the
            # conversation structure is preserved (role ordering matters).
            out.append({**msg, "content": ""})
    return out
# ---------------------------------------------------------------------------
# Hermes-internal packed tool_call_id handling
# ---------------------------------------------------------------------------
# When Google's Gemini returns a tool_call with a ``thought_signature`` in
# ``extra_content``, we pack that signature into the tool_call id as
# ``<orig_id>:hermes_ts:<b64>`` so it survives the round-trip through clients
# that strip non-standard fields.  These helpers undo the packing — restoring
# the original id on assistant tool_calls AND the corresponding tool result
# message — so providers receive clean, un-suffixed ids.

_HERMES_TS_DELIMITER = ":hermes_ts:"


def _strip_hermes_ts_packed_ids(messages: List[Dict[str, Any]]) -> int:
    """Strip ``:hermes_ts:`` packed suffix from all tool_call ids.

    Walks both assistant tool_calls (``id`` / ``call_id``) and tool result
    messages (``tool_call_id``).  Returns the number of ids repaired.  Safe to
    run on any provider's input — the suffix is only present when a previous
    Google turn wrote it.
    """
    fixed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id") or tc.get("call_id") or ""
                if isinstance(tc_id, str) and _HERMES_TS_DELIMITER in tc_id:
                    original = tc_id.split(_HERMES_TS_DELIMITER, 1)[0]
                    tc["id"] = original
                    tc["call_id"] = original
                    fixed += 1
        elif msg.get("role") == "tool":
            tcid = msg.get("tool_call_id", "")
            if isinstance(tcid, str) and _HERMES_TS_DELIMITER in tcid:
                msg["tool_call_id"] = tcid.split(_HERMES_TS_DELIMITER, 1)[0]
                fixed += 1
    return fixed


def _unpack_hermes_ts_and_inject_signatures(messages: List[Dict[str, Any]]) -> tuple[int, int]:
    """Unpack ``:hermes_ts:`` packed ids AND restore Google thought_signature.

    For Google providers only: the packed base64 value encodes the
    ``thought_signature`` that Gemini 3.1+ requires on functionCall parts.
    This function strips the packed suffix, restores original ids on BOTH
    assistant tool_calls and tool result messages, AND injects the
    extracted signature back into ``extra_content.google.thought_signature``
    on the assistant tool_calls.

    Returns ``(injected, unpacked)`` counts.
    """
    import base64
    injected = 0
    unpacked = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                if not isinstance(tc, dict):
                    continue
                # Check if extra_content.google.thought_signature already exists
                _ec = tc.get("extra_content", {})
                _google = _ec.get("google", {}) if isinstance(_ec, dict) else {}
                _ts = _google.get("thought_signature") if isinstance(_google, dict) else None
                if not _ts:
                    # Unpack from packed call_id: "<id>:hermes_ts:<b64>"
                    tc_id = tc.get("id") or tc.get("call_id") or ""
                    if isinstance(tc_id, str) and _HERMES_TS_DELIMITER in tc_id:
                        parts = tc_id.split(_HERMES_TS_DELIMITER, 1)
                        if len(parts) == 2:
                            try:
                                b64 = parts[1]
                                pad = "=" * (-len(b64) % 4)
                                _ts = base64.urlsafe_b64decode(b64 + pad).decode("utf-8")
                                unpacked += 1
                                tc["id"] = parts[0]
                                tc["call_id"] = parts[0]
                            except Exception:
                                # Can't decode — just strip the suffix
                                tc["id"] = parts[0]
                                tc["call_id"] = parts[0]
                if _ts:
                    tc["extra_content"] = {"google": {"thought_signature": _ts}}
                    injected += 1
        elif msg.get("role") == "tool":
            tcid = msg.get("tool_call_id", "")
            if isinstance(tcid, str) and _HERMES_TS_DELIMITER in tcid:
                msg["tool_call_id"] = tcid.split(_HERMES_TS_DELIMITER, 1)[0]
    return injected, unpacked


def _strip_call_id_from_tool_calls(messages: List[Dict[str, Any]]) -> int:
    """Remove the ``call_id`` field from assistant tool_calls.

    Several OpenAI-compat providers (cerebras, groq, cohere, etc.) reject
    unknown fields on tool_calls with 400.  Stripping ``call_id`` before
    sending to providers that don't support it prevents a class of
    ``tool_call.id is invalid`` 400 errors.  Returns the number of fields
    removed.
    """
    removed = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict) and "call_id" in tc:
                    tc.pop("call_id", None)
                    removed += 1
    return removed

# Sentinel value that bypasses Google's per-functionCall signature validator.
# The native Gemini adapter uses this for non-Gemini-origin tool calls and
# the OpenAI-compat endpoint honours it the same way.  See
# ``agent/gemini_native_adapter.py::_translate_tool_call_to_gemini``.
_GOOGLE_THOUGHT_SIGNATURE_SENTINEL = "skip_thought_signature_validator"


def _count_unsigned_google_tool_calls(messages: List[Dict[str, Any]]) -> int:
    """Count assistant tool_calls that lack a Google thought_signature.

    Gemini 3.1+ rejects requests where any functionCall part is missing a
    ``thought_signature`` with HTTP 400.  When a session replays a
    conversation whose tool_calls came from non-Google providers (or whose
    signatures were stripped on the round-trip), the request will fail
    before producing a response.  Counting these lets the chain decide
    whether to inject a sentinel signature before calling Google.
    """
    count = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            ec = tc.get("extra_content")
            google = ec.get("google") if isinstance(ec, dict) else None
            ts = google.get("thought_signature") if isinstance(google, dict) else None
            if not ts:
                count += 1
    return count


# Backwards-compat alias — earlier callers used the predicate-style name.
_has_unsigned_google_tool_calls = _count_unsigned_google_tool_calls


def _inject_google_sentinel_signatures(messages: List[Dict[str, Any]]) -> int:
    """Inject a sentinel thought_signature on every unsigned tool_call.

    Walks assistant messages and sets
    ``extra_content.google.thought_signature = skip_thought_signature_validator``
    on any tool_call that lacks one.  This satisfies Google's per-functionCall
    signature requirement for tool calls that originated outside its own
    chain (e.g. from a non-Gemini provider, or whose signature was stripped
    by the client on the round-trip).

    Returns the number of tool_calls modified.  The function mutates
    ``messages`` in place — pass a deep copy if the caller needs the
    original.
    """
    modified = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            ec = tc.get("extra_content")
            google = ec.get("google") if isinstance(ec, dict) else None
            ts = google.get("thought_signature") if isinstance(google, dict) else None
            if ts:
                continue
            # Set/overwrite extra_content with the sentinel.  The OpenAI-compat
            # transport keeps extra_content for Gemini-family targets and
            # drops it for everyone else, so this is safe.
            tc["extra_content"] = {"google": {"thought_signature": _GOOGLE_THOUGHT_SIGNATURE_SENTINEL}}
            modified += 1
    return modified





def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """
    SQLite-backed LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.

    Persists across gateway restarts.  Falls back to in-memory SQLite
    if the on-disk path is unavailable.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
            except Exception:
                db_path = ":memory:"
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)
        ).fetchone()
        if row is None:
            return None
        import time
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id),
        )
        self._conn.commit()
        return json.loads(row[0])

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        import time
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            self._conn.execute(
                "DELETE FROM responses WHERE response_id IN "
                "(SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?)",
                (count - self._max_size,),
            )
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        cursor = self._conn.execute(
            "DELETE FROM responses WHERE response_id = ?", (response_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute(
            "SELECT response_id FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            (name, response_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

_CORS_HEADERS = {
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, Idempotency-Key",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def cors_middleware(request, handler):
        """Add CORS headers for explicitly allowed origins; handle OPTIONS preflight."""
        adapter = request.app.get("api_server_adapter")
        origin = request.headers.get("Origin", "")
        cors_headers = None
        if adapter is not None:
            if not adapter._origin_allowed(origin):
                return web.Response(status=403)
            cors_headers = adapter._cors_headers_for_origin(origin)

        if request.method == "OPTIONS":
            if cors_headers is None:
                return web.Response(status=403)
            return web.Response(status=200, headers=cors_headers)

        response = await handler(request)
        if cors_headers is not None:
            response.headers.update(cors_headers)
        return response
else:
    cors_middleware = None  # type: ignore[assignment]


def _openai_error(message: str, err_type: str = "invalid_request_error", param: str = None, code: str = None) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def body_limit_middleware(request, handler):
        """Reject overly large request bodies early based on Content-Length."""
        if request.method in ("POST", "PUT", "PATCH"):
            cl = request.headers.get("Content-Length")
            if cl is not None:
                try:
                    if int(cl) > MAX_REQUEST_BYTES:
                        return web.json_response(_openai_error("Request body too large.", code="body_too_large"), status=413)
                except ValueError:
                    return web.json_response(_openai_error("Invalid Content-Length header.", code="invalid_content_length"), status=400)
        return await handler(request)
else:
    body_limit_middleware = None  # type: ignore[assignment]

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def security_headers_middleware(request, handler):
        """Add security headers to all responses (including errors)."""
        response = await handler(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response
else:
    security_headers_middleware = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Request monitor middleware
# ---------------------------------------------------------------------------
# Logs every inbound POST/DELETE to a dedicated rotating file at
# /home/tusker/.hermes/logs/request_monitor.log (persistent across restarts).
# Captures: timestamp, method, path, status, latency, last-message role/content
# for the three agentic API paths (/v1/runs, /v1/responses, /v1/chat/completions).
# ---------------------------------------------------------------------------

import os as _os
import time as _time
import logging.handlers as _lh

_monitor_log_path = _os.path.join(
    _os.environ.get("HERMES_HOME", "/home/tusker/.hermes"),
    "logs",
    "request_monitor.log",
)
_monitor_logger = logging.getLogger("hermes.request_monitor")
if not _monitor_logger.handlers:
    try:
        _os.makedirs(_os.path.dirname(_monitor_log_path), exist_ok=True)
        _mh = _lh.RotatingFileHandler(
            _monitor_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        _mh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _monitor_logger.addHandler(_mh)
        _monitor_logger.addHandler(logging.StreamHandler())  # also to docker logs
        _monitor_logger.setLevel(logging.DEBUG)
        _monitor_logger.propagate = False
    except Exception:
        pass

_MONITOR_PATHS = {"/v1/runs", "/v1/responses", "/v1/chat/completions"}


if AIOHTTP_AVAILABLE:
    @web.middleware
    async def request_monitor_middleware(request, handler):
        """Log every request with method, path, status, latency, and last-message details."""
        import json as _json
        t0 = _time.monotonic()

        # Sniff last-message role/content for the three agentic paths
        last_role = "-"
        last_content_len = "-"
        last_content_preview = "-"
        is_agentic = request.method in ("POST",) and request.path in _MONITOR_PATHS

        if is_agentic:
            try:
                body_bytes = await request.read()
                body = _json.loads(body_bytes)
                msgs = body.get("messages") or body.get("input") or []
                if isinstance(msgs, list) and msgs:
                    lm = msgs[-1]
                    if isinstance(lm, dict):
                        last_role = lm.get("role", "?")
                        raw_content = lm.get("content") or ""
                        if isinstance(raw_content, list):
                            raw_content = " ".join(
                                part.get("text", "") for part in raw_content
                                if isinstance(part, dict)
                            )
                        last_content_len = str(len(raw_content))
                        last_content_preview = repr(raw_content[:120])
                elif isinstance(msgs, str):
                    last_role = "user(str)"
                    last_content_len = str(len(msgs))
                    last_content_preview = repr(msgs[:120])
            except Exception as _exc:
                last_role = f"parse_err:{_exc}"

        response = await handler(request)

        latency_ms = int((_time.monotonic() - t0) * 1000)
        _monitor_logger.info(
            "[MONITOR] %s %s → %s  %dms  last_role=%s content_len=%s preview=%s",
            request.method,
            request.path,
            response.status,
            latency_ms,
            last_role,
            last_content_len,
            last_content_preview,
        )
        return response
else:
    request_monitor_middleware = None  # type: ignore[assignment]


class _IdempotencyCache:
    """In-memory idempotency cache with TTL and basic LRU semantics."""
    def __init__(self, max_items: int = 1000, ttl_seconds: int = 300):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._ttl = ttl_seconds
        self._max = max_items

    def _purge(self):
        import time as _t
        now = _t.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def get_or_set(self, key: str, fingerprint: str, compute_coro):
        self._purge()
        item = self._store.get(key)
        if item and item["fp"] == fingerprint:
            return item["resp"]
        resp = await compute_coro()
        import time as _t
        self._store[key] = {"resp": resp, "fp": fingerprint, "ts": _t.time()}
        self._purge()
        return resp


_idem_cache = _IdempotencyCache()


def _make_request_fingerprint(body: Dict[str, Any], keys: List[str]) -> str:
    from hashlib import sha256
    subset = {k: body.get(k) for k in keys}
    return sha256(repr(subset).encode("utf-8")).hexdigest()


def _derive_chat_session_id(
    system_prompt: Optional[str],
    first_user_message: str,
    *,
    salt: Optional[str] = None,
) -> str:
    """Derive a stable session ID from the conversation's first user message.

    OpenAI-compatible frontends (Open WebUI, LibreChat, etc.) send the full
    conversation history with every request.  The system prompt and first user
    message are constant across all turns of the same conversation, so hashing
    them produces a deterministic session ID that lets the API server reuse
    the same Hermes session (and therefore the same Docker container sandbox
    directory) across turns.

    ``salt`` (typically the client IP) is included so that two different clients
    sending identical first messages do not collide into the same session, which
    could cause model-stickiness state and tool-call hub entries to bleed across
    users.
    """
    seed = f"{salt or ''}\n{system_prompt or ''}\n{first_user_message}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"api-{digest}"


class APIServerAdapter(BasePlatformAdapter):
    """
    OpenAI-compatible HTTP API server adapter.

    Runs an aiohttp web server that accepts OpenAI-format requests
    and routes them through hermes-agent's AIAgent.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        self._port: int = int(extra.get("port", os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))))
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity
        # In-memory gateway state, used by the /ready endpoint to gate
        # K8s Service endpoint membership.  Starts at "starting" so a
        # freshly-spawned pod returns 503 from /ready until the runner
        # explicitly transitions to "running" or "degraded".  See
        # set_gateway_state() and the /ready handler.  We hold this in
        # memory (not on the PVC-backed status file) so a brand-new
        # pod never inherits the previous pod's "running" state during
        # the window between process start and the first state write.
        self._gateway_state: str = "starting"

    def _create_listen_socket(self, backlog: int = 2048) -> Optional[_socket.socket]:
        """Create an explicit listening socket for concrete IP binds.

        aiohttp's ``TCPSite`` normally resolves/binds from a host string. In the
        Kubernetes deployment we want wildcard binds like ``0.0.0.0`` to result
        in a concrete non-loopback listening socket every time, so liveness
        probes that hit the pod IP succeed. For literal IPs we bind the socket
        ourselves and hand it to ``SockSite``; hostnames keep the default
        ``TCPSite`` path so normal name resolution still applies.
        """
        host = str(self._host or "").strip()
        if not host:
            return None
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            return None

        family = _socket.AF_INET6 if addr.version == 6 else _socket.AF_INET
        sock = _socket.socket(family, _socket.SOCK_STREAM)
        try:
            sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            if family == _socket.AF_INET6:
                try:
                    sock.setsockopt(_socket.IPPROTO_IPV6, _socket.IPV6_V6ONLY, 1)
                except (AttributeError, OSError):
                    pass
            sock.bind((host, self._port))
            sock.listen(backlog)
            sock.setblocking(False)
            return sock
        except Exception:
            sock.close()
            raise

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"
        """
        if explicit and explicit.strip():
            return explicit.strip()
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in ("default", "custom"):
                return profile
        except Exception:
            pass
        return "hermes-agent"

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed (only when API
        server is local).
        """
        if not self._api_key:
            return None  # No key configured — allow all (local-only use)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        if not auth_header:
            reason = "missing Authorization header"
        elif not auth_header.startswith("Bearer "):
            reason = f"malformed Authorization header (got '{auth_header[:20]}...')"
        else:
            reason = "wrong token"
        logger.warning(
            "Auth failure: %s from %s — %s %s",
            reason,
            request.remote,
            request.method,
            request.path,
        )
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _ensure_session_db(self):
        """Lazily initialise and return the shared SessionDB instance.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.
        """
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SessionDB unavailable for API server: %s", e)
        return self._session_db

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_gen_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        provider_mode: bool = False,
        swarm_mode: bool = False,
        swarm_model_pool: Optional[Dict[str, Any]] = None,
        estimated_tokens: int = 0,
        toolset_mode: str = "auto",
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        external_tool_mode: str = "none",
        user_model: Optional[str] = None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.
        """
        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model, _load_gateway_config
        from hermes_cli.tools_config import _get_platform_tools

        logging.debug(f"[API_SERVER] _create_agent called: swarm_mode={swarm_mode}, swarm_model_pool={swarm_model_pool}")

        # Swarm mode: select from free/cheap model pool FIRST
        # This must happen before _resolve_runtime_agent_kwargs() to avoid provider resolution errors
        if swarm_mode and swarm_model_pool:
            model = _resolve_swarm_model(
                swarm_model_pool,
                estimated_tokens=swarm_model_pool.get("estimated_tokens", 0),
            )
            logging.warning(f"[API_SERVER] Swarm mode: resolved model={model}")
            runtime_kwargs, model = self._runtime_kwargs_for_model(model)
            # Don't add model to runtime_kwargs since it's passed separately to AIAgent()
            logging.warning(f"[API_SERVER] Swarm: runtime_kwargs={runtime_kwargs}")

        # Only resolve credentials if NOT in swarm mode (swarm mode already has runtime_kwargs)
        # This prevents unnecessary Codex credential checks when using other providers
        if not swarm_mode:
            # Check if a specific provider is configured - if so, use it directly
            # without going through the default credential resolution (which checks Codex)
            requested_provider = os.getenv("HERMES_INFERENCE_PROVIDER")
            if not requested_provider:
                cfg = _load_gateway_config()
                model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
                if isinstance(model_cfg, dict):
                    requested_provider = str(model_cfg.get("provider") or "").strip() or None
            
            # When provider_mode=True (hermes-code), select the preferred runtime model.
            # Reuse a session-sticky choice when available; otherwise resolve from the
            # ordered hermes-code pool for this request.
            # When provider_mode=False and user provides an explicit model (e.g. anthropic/claude-sonnet-4.6),
            # use the user's model directly instead of HERMES_CODE_MODEL.
            if not provider_mode and user_model and user_model not in ("hermes-code", "hermes-swarm"):
                # User specified an explicit model - use it directly
                code_model = user_model
            elif provider_mode:
                code_model = _select_hermes_code_model(
                    estimated_tokens=estimated_tokens,
                    session_id=session_id,
                )
            else:
                code_model = os.getenv("HERMES_CODE_MODEL", "").strip()
            code_model_prefix = code_model.split("/")[0].lower() if code_model and "/" in code_model else ""
            
            if requested_provider and requested_provider.lower() in ("zai", "opencode-go", "opencode-zen"):
                # If provider_mode and model prefix doesn't match provider, resolve provider
                # from the model prefix instead of using HERMES_INFERENCE_PROVIDER.
                # e.g., HERMES_CODE_MODEL="openai/gpt-5.4" with HERMES_INFERENCE_PROVIDER="zai"
                # should route to openai-codex, not zai
                if provider_mode and code_model_prefix and code_model_prefix != requested_provider.lower():
                    # Resolve provider from model prefix, not from HERMES_INFERENCE_PROVIDER
                    # Map the prefix through _EXPLICIT_MODEL_PROVIDER_ALIASES (e.g., "openai" -> "openai-codex")
                    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
                    try:
                        # Map model prefix to canonical provider name
                        model_provider = _EXPLICIT_MODEL_PROVIDER_ALIASES.get(code_model_prefix, code_model_prefix)
                        runtime = resolve_runtime_provider(requested=model_provider)
                        runtime_kwargs = {
                            "api_key": runtime.get("api_key"),
                            "base_url": runtime.get("base_url"),
                            "provider": runtime.get("provider"),
                            "api_mode": runtime.get("api_mode"),
                            "command": runtime.get("command"),
                            "args": list(runtime.get("args") or []),
                            "credential_pool": runtime.get("credential_pool"),
                        }
                    except Exception as exc:
                        raise RuntimeError(format_runtime_provider_error(exc)) from exc
                elif code_model_prefix and code_model_prefix != requested_provider.lower() and not provider_mode:
                    # Non-provider_mode request with an explicit provider-prefixed model that differs
                    # from the configured provider. Seed runtime_kwargs from the configured provider
                    # first, then let _align_runtime_with_explicit_model() switch to the explicit
                    # model's runtime when appropriate. Without this seed, runtime_kwargs is never
                    # bound and explicit model requests can crash with UnboundLocalError.
                    runtime_kwargs = _resolve_runtime_agent_kwargs()
                elif requested_provider.lower() == "zai":
                    runtime_kwargs = {
                        "base_url": "https://api.z.ai/api/coding/paas/v4",
                        "api_key": os.getenv("ZAI_API_KEY", ""),
                        "provider": "zai",
                    }
                else:
                    runtime_kwargs = {
                        "base_url": os.getenv(f"{requested_provider.upper().replace('-', '_')}_BASE_URL", f"https://opencode.ai/zen/go/v1" if requested_provider.lower() == "opencode-go" else "https://opencode.ai/zen/v1"),
                        "api_key": os.getenv(f"{requested_provider.upper().replace('-', '_')}_API_KEY", ""),
                        "provider": requested_provider.lower(),
                    }
            elif code_model_prefix:
                # No configured provider (or provider not in zai/opencode-zen/opencode-go list),
                # but an explicit provider-prefixed model was requested.
                # Resolve credentials from the model prefix using _runtime_kwargs_for_model_id,
                # which reads env vars directly for direct-API providers like xiaomi, ollama,
                # minimax, etc. (these are NOT handled by resolve_runtime_provider).
                from hermes_cli.runtime_provider import format_runtime_provider_error
                try:
                    runtime_kwargs, _resolved_model = _runtime_kwargs_for_model_id(code_model)
                except Exception as exc:
                    raise RuntimeError(format_runtime_provider_error(exc)) from exc
            else:
                runtime_kwargs = _resolve_runtime_agent_kwargs()
        
        # Non-swarm path continues here - model resolution (swarm already has model at this point)
        if not swarm_mode:
            if provider_mode:
                # OpenCode routes hermes-code through provider mode. Keep the
                # requested hermes-code model stable even when there is no gateway
                # config model.default configured.
                model = code_model or _select_hermes_code_model(estimated_tokens=estimated_tokens)
            elif code_model:
                # An explicit provider-prefixed model was requested (e.g.
                # github-copilot-enterprise/gpt-5.4). Use it directly instead
                # of falling back to the gateway default model.
                model = code_model
            else:
                model = _resolve_gateway_model()

            runtime_kwargs = _align_runtime_with_explicit_model(runtime_kwargs, model)
            _runtime_provider_name = str(runtime_kwargs.get("provider") or "").strip().lower()
            _normalized_runtime_model = _normalize_model_for_runtime_provider(model, _runtime_provider_name)
            if provider_mode and "/" in str(model or "") and _normalized_runtime_model == model:
                _explicit = _explicit_provider_from_model(model)
                if _explicit and _runtime_provider_name and _explicit == _runtime_provider_name:
                    _normalized_runtime_model = str(model).partition("/")[2].strip() or model
            if _normalized_runtime_model and _normalized_runtime_model != model:
                logger.info(
                    "[api_server] normalized explicit model for provider %s: %s -> %s",
                    _runtime_provider_name or "unknown",
                    model,
                    _normalized_runtime_model,
                )
                model = _normalized_runtime_model
            # If the model had an explicit provider prefix (eg. "anthropic/..." or "ollama/...")
            # but the requested provider couldn't be resolved (missing creds), try to find a
            # viable provider for the bare model name. This prevents mismatched provider+model
            # calls (e.g., calling openai-codex with an anthropic/ model) which produce
            # non-retryable 400 errors. We only do this when the explicit provider was requested.
            try:
                from hermes_cli.models import detect_provider_for_model
                from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
                explicit_provider = _explicit_provider_from_model(model)
                if explicit_provider:
                    current_provider = str(runtime_kwargs.get("provider") or "").strip().lower()
                    # If runtime doesn't match explicit provider or lacks api_key, attempt resolution
                    if current_provider != explicit_provider or not runtime_kwargs.get("api_key"):
                        try:
                            # Try to resolve the explicit provider (may raise if creds missing)
                            resolved = resolve_runtime_provider(requested=explicit_provider)
                            runtime_kwargs = {
                                "api_key": resolved.get("api_key"),
                                "base_url": resolved.get("base_url"),
                                "provider": resolved.get("provider"),
                                "api_mode": resolved.get("api_mode"),
                                "command": resolved.get("command"),
                                "args": list(resolved.get("args") or []),
                                "credential_pool": resolved.get("credential_pool"),
                            }
                        except Exception:
                            # Couldn't resolve explicit provider (likely missing credentials).
                            # Try to detect an alternative provider for the bare model name.
                            try:
                                bare = model.split("/", 1)[1].strip() if "/" in model else model
                                detected = detect_provider_for_model(bare, current_provider)
                                if detected:
                                    alt_provider = detected[0]
                                    if alt_provider == "openrouter" and _openrouter_nonfree_blocked(model):
                                        logging.warning(
                                            "[API_SERVER] explicit provider %s unavailable; refusing paid OpenRouter fallback for model %s — cooling down provider",
                                            explicit_provider,
                                            model,
                                        )
                                        try:
                                            from agent.model_cooldown_db import mark_model_cooldown
                                            mark_model_cooldown(
                                                provider=explicit_provider,
                                                model=bare,
                                                reason="unavailable_provider",
                                                cooldown_seconds=3600,
                                            )
                                        except Exception:
                                            logging.debug("[API_SERVER] failed to mark cooldown for unavailable provider %s", explicit_provider)
                                        # Fall through: keep existing runtime_kwargs rather than hard-failing
                                    try:
                                        alt_runtime = resolve_runtime_provider(requested=alt_provider)
                                        runtime_kwargs = {
                                            "api_key": alt_runtime.get("api_key"),
                                            "base_url": alt_runtime.get("base_url"),
                                            "provider": alt_runtime.get("provider"),
                                            "api_mode": alt_runtime.get("api_mode"),
                                            "command": alt_runtime.get("command"),
                                            "args": list(alt_runtime.get("args") or []),
                                            "credential_pool": alt_runtime.get("credential_pool"),
                                        }
                                        logging.warning(
                                            "[API_SERVER] explicit provider %s unavailable; routed model %s to provider %s",
                                            explicit_provider,
                                            model,
                                            alt_provider,
                                        )
                                    except Exception:
                                        logging.warning(
                                            "[API_SERVER] explicit provider %s unavailable and alternative provider resolution failed; proceeding with default runtime kwargs",
                                            explicit_provider,
                                        )
                                else:
                                    logging.warning(
                                        "[API_SERVER] explicit provider %s unavailable and no alternative provider detected for model %s",
                                        explicit_provider,
                                        model,
                                    )
                            except RuntimeError:
                                raise
                            except Exception:
                                logging.exception("[API_SERVER] error while attempting alternative provider detection for model %s", model)
            except RuntimeError:
                raise
            except Exception:
                # If the import or detection fails, just continue with existing runtime_kwargs
                logging.debug("[API_SERVER] provider alignment/detection helper failed; continuing")

            if provider_mode:
                _resolved_provider = str(runtime_kwargs.get("provider") or "").strip() or "unknown"
                _resolved_base_url = str(runtime_kwargs.get("base_url") or "").strip() or "unknown"
                logger.info(
                    "[api_server] final hermes-code resolution: requested=%s resolved_provider=%s resolved_model=%s base_url=%s",
                    code_model or model,
                    _resolved_provider,
                    model,
                    _resolved_base_url,
                )
                _remember_hermes_code_session_model(session_id, code_model or model)

        _t_cfg = time.time()
        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))
        skip_memory = False
        skip_context_files = False
        if swarm_mode and swarm_model_pool:
            swarm_prompt = _swarm_execution_system_prompt(swarm_model_pool.get("routing_hint"))
            if swarm_prompt:
                ephemeral_system_prompt = (
                    f"{ephemeral_system_prompt}\n\n{swarm_prompt}" if ephemeral_system_prompt else swarm_prompt
                )
        if provider_mode:
            enabled_toolsets = []
            skip_memory = True
            skip_context_files = True
            if ephemeral_system_prompt is None:
                ephemeral_system_prompt = "You are a helpful AI assistant."
        elif swarm_mode and external_tool_mode in {"broker", "inband"}:
            enabled_toolsets = []
            skip_memory = True
            skip_context_files = True
            action_mode = str((swarm_model_pool or {}).get("routing_hint", {}).get("action_mode") or "execute_with_tools").strip().lower()
            if action_mode in {"plan_only", "answer_only"}:
                tools = []
                tool_choice = "none"
                external_tool_mode = "none"
        elif toolset_mode == "local":
            enabled_toolsets = sorted(
                [t for t in enabled_toolsets if t in ("terminal", "hermes-cli")]
            )
        elif toolset_mode == "remote":
            enabled_toolsets = sorted(
                [t for t in enabled_toolsets if t in ("web", "skills")]
            )
        # "full", "auto" or anything else uses all configured toolsets
        _t_cfg_done = time.time()
        logger.info("[timing] _create_agent config+tools: %.3fs", _t_cfg_done - _t_cfg)

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        from gateway.run import GatewayRunner
        _t_fb = time.time()
        fallback_model = GatewayRunner._load_fallback_model()
        if not fallback_model:
            if provider_mode:
                fallback_model = _build_env_fallback_chain("HERMES_AGENT_FALLBACK")
            elif swarm_mode:
                fallback_model = _build_env_fallback_chain("HERMES_SWARM_FALLBACK")
        _t_fb_done = time.time()
        logger.info("[timing] _create_agent fallback+env: %.3fs", _t_fb_done - _t_fb)

        _t_aiagent = time.time()
        logger.info(
            "[timing] _create_agent AIAgent: model=%s api_mode=%s provider=%s api_key=%s",
            model,
            runtime_kwargs.get("api_mode", "MISSING"),
            runtime_kwargs.get("provider", "MISSING"),
            "SET" if runtime_kwargs.get("api_key") else "EMPTY",
        )
        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_gen_callback=tool_gen_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            skip_memory=skip_memory,
            skip_context_files=skip_context_files,
        )
        _t_aiagent_done = time.time()
        logger.info("[timing] _create_agent AIAgent.__init__: %.3fs", _t_aiagent_done - _t_aiagent)
        try:
            agent._provider_mode = provider_mode
            agent._tools_from_request = bool(tools)
            agent._toolset_mode = toolset_mode
            agent._external_tool_mode = external_tool_mode
        except Exception:
            pass
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    def set_gateway_state(self, state: Optional[str]) -> None:
        """Update the in-memory gateway state used by the /ready endpoint.

        Called by ``GatewayRunner._update_runtime_status`` on every state
        transition.  Valid states are the same ones that get persisted to
        the runtime status file (``starting``, ``running``, ``degraded``,
        ``startup_failed``, ``draining``, ``stopped``).  ``None`` is a
        no-op so callers can forward ``gateway_state=None`` through
        without a special case.

        Thread-safety: this is intended to be called from the runner's
        event loop.  We do not add a lock because aiohttp's request
        handlers run on the same loop and the kubelet polls /ready at
        most a few times per second — even a torn read would just
        return 503 (state not in _READY_STATES) for one probe tick,
        which kubelet tolerates.  If the gateway later moves to a
        multi-threaded model, swap this for a ``threading.Lock`` or
        an ``asyncio.Event``-based signal.
        """
        if state is None:
            return
        # Coerce unknown values to "starting" defensively — _handle_ready
        # treats unknown states as not-ready, so we don't need to raise
        # here, but normalise common variations.
        if not isinstance(state, str) or not state:
            state = "starting"
        self._gateway_state = state

    def get_gateway_state(self) -> str:
        """Return the current in-memory gateway state (read-only view)."""
        return self._gateway_state

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple liveness check.

        Returns 200 as long as the HTTP server is responding.  Does NOT
        reflect gateway startup state — for that, see ``/ready``.  K8s
        liveness probes use this endpoint: a 200 here means "process
        alive, not deadlocked", which is the only thing the kubelet
        should use to decide whether to kill the container.
        """
        return web.json_response({"status": "ok", "platform": "hermes-agent"})

    async def _handle_ready(self, request: "web.Request") -> "web.Response":
        """GET /ready — Kubernetes readiness probe target.

        Returns 200 only when the gateway has fully initialized and can
        serve traffic.  Returns 503 during startup, in fatal-startup
        failure, or while draining/stopping.  The kubelet polls this
        endpoint and only adds the pod to the Service endpoint set when
        it returns 200, so traffic is never routed to a half-initialised
        pod (which would otherwise happen with a single-replica
        deployment using ``maxUnavailable: 0``).

        State is held in process memory — see ``set_gateway_state()``.
        The previous design read from a file on the PVC, which is
        racy: a fresh pod would briefly see ``gateway_state: running``
        written by the previous pod before the new pod overwrites it.
        In-memory state starts at ``"starting"`` and is only flipped to
        ``"running"`` (or ``"degraded"``) by the running gateway
        itself, so the kubelet never sees a stale "ready" signal.
        """
        state = self._gateway_state
        if state in _READY_STATES:
            return web.json_response(
                {"status": "ok", "gateway_state": state},
                status=200,
            )
        return web.json_response(
            {"status": "not_ready", "gateway_state": state},
            status=503,
        )

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  No authentication required.
        """
        from gateway.status import read_runtime_status

        runtime = read_runtime_status() or {}
        gateway_state = self._gateway_state
        runtime_state = runtime.get("gateway_state")
        if runtime_state and not (
            gateway_state in _READY_STATES
            and runtime_state in {"starting", "draining", "stopped", "startup_failed"}
        ):
            gateway_state = runtime_state
        return web.json_response({
            "status": "ok",
            "platform": "hermes-agent",
            "gateway_state": gateway_state,
            "platforms": runtime.get("platforms", {}),
            "active_agents": runtime.get("active_agents", 0),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        })

    async def _handle_stats(self, request: "web.Request") -> "web.Response":
        """GET /stats — return SmartRouter and Deduplicator statistics.
        
        Returns aggregated stats from:
        - SmartRouter: routing decisions, cost savings
        - Deduplicator: cache hits, dedup rate
        - Combined: estimated total savings
        """
        try:
            from agent.deduplicator import get_global_deduplicator
            from agent.smart_router import get_global_router
            
            dedup = get_global_deduplicator()
            router = get_global_router()
            
            dedup_stats = dedup.get_stats().to_dict() if hasattr(dedup, 'get_stats') else {}
            router_stats = router.get_stats().to_dict() if hasattr(router, 'get_stats') else {}
            
            # Calculate combined savings
            dedup_savings = dedup_stats.get('cache_hits', 0) * 0.5  # Rough estimate
            routing_savings = router_stats.get('cost_savings_cents', 0)
            total_savings = dedup_savings + routing_savings
            
            return web.json_response({
                "status": "ok",
                "deduplicator": dedup_stats,
                "smart_router": router_stats,
                "combined": {
                    "estimated_cost_savings_cents": round(total_savings, 2),
                    "dedup_rate_pct": dedup_stats.get('dedup_rate_pct', 0),
                    "cache_hit_rate_pct": dedup_stats.get('cache_hit_rate_pct', 0),
                    "routing_decisions": router_stats.get('total_requests', 0),
                    "simple_routed_to_cheap": router_stats.get('simple_routed_to_cheap', 0),
                }
            })
        except Exception as e:
            logger.warning("Failed to get stats: %s", e)
            return web.json_response({
                "status": "ok",
                "deduplicator": {},
                "smart_router": {},
                "combined": {},
                "error": str(e)
            })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — return hermes-agent as an available model.

        Auth is NOT required here: model names are not sensitive and IDE
        extensions (e.g. Zoo Code) call this endpoint before knowing a key.
        The actual chat/completion endpoint still enforces auth.
        """

        now = int(time.time())
        hermes_code_context = _hermes_code_advertised_context_length()
        hermes_code_max_output = _hermes_code_advertised_max_output_tokens()
        hermes_code_selected = _select_hermes_code_model()
        data = [
            {
                "id": self._model_name,
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": self._model_name,
                "parent": None,
            },
            {
                "id": "hermes-code",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-code",
                "parent": None,
                "description": f"Routes all tools to client (OpenCode local). Long-context provider alias; currently selects from Hermes premium coding pool (active candidate: {hermes_code_selected}).",
                "context_length": hermes_code_context,
                "max_completion_tokens": hermes_code_max_output,
                "context_window": {
                    "context_length": hermes_code_context,
                    "max_output_tokens": hermes_code_max_output,
                },
                "input": ["text", "image"],

                "metadata": {
                    "selected_model": hermes_code_selected,
                    "large_context_min": _HERMES_CODE_LARGE_CONTEXT_MIN,
                    "large_context_trigger": _HERMES_CODE_LARGE_CONTEXT_TRIGGER,
                },
            },
            {
                "id": "hermes-privacy",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-privacy",
                "parent": None,
                "description": f"Privacy-restricted coding pool - ZDR-compliant providers only. Currently selects: {_select_hermes_privacy_model()}.",
                "context_length": _hermes_privacy_advertised_context_length(),
                "max_completion_tokens": _hermes_privacy_advertised_max_output_tokens(),
                "context_window": {
                    "context_length": _hermes_privacy_advertised_context_length(),
                    "max_output_tokens": _hermes_privacy_advertised_max_output_tokens(),
                },
                "input": ["text", "image"],
                "metadata": {
                    "selected_model": _select_hermes_privacy_model(),
                },
            },
            {
                "id": "hermes-agentic-remote",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-agentic-remote",
                "parent": None,
            },
            {
                "id": "hermes-agentic-full",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-agentic-full",
                "parent": None,
            },
            {
                "id": "hermes-swarm",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-swarm",
                "parent": None,
            },
            {
                "id": "hermes-reranker",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "hermes-reranker",
                "parent": None,
                "description": "Round-robin reranker pool (Cohere + Voyage AI). POST /v1/rerank — Cohere-compatible request/response format.",
            },
            {
                "id": "mlx-mac/qwen3-coder-30b-a3b-instruct-4bit",
                "object": "model",
                "created": now,
                "owned_by": "hermes",
                "permission": [],
                "root": "mlx-mac/qwen3-coder-30b-a3b-instruct-4bit",
                "parent": None,
                "description": "Qwen3-Coder-30B-A3B-Instruct (MLX-4bit MoE) on local M4 Max via mlx_lm.server. ~40 tok/s, 30B total / 3.3B active. LAN-only, free.",
                "context_length": 32768,
            },
            *[
                {
                    "id": alias,
                    "object": "model",
                    "created": now,
                    "owned_by": "hermes",
                    "permission": [],
                    "root": alias,
                    "parent": "hermes-swarm",
                    "description": "Role alias managed by Hermes Gateway. Routed dynamically through hermes-swarm.",
                }
                for alias in ROLE_ALIAS_CONFIG
            ],
            *_advertised_ghe_passthrough_models(now),
        ]
        return web.json_response({
            "object": "list",
            "data": data,
        })


    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions — OpenAI Chat Completions format."""
        _t0 = time.time()
        _req_id = id(request)
        logger.debug("[%d] --> %s %s from %s", _req_id, request.method, request.path, request.remote or "?")
        auth_err = self._check_auth(request)
        if auth_err:
            logger.debug("[%d] <-- auth rejected (%d)", _req_id, auth_err.status)
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        # DEBUG: log the last message so we can understand what pi sends as a blank continuation
        _last_msg = messages[-1] if messages else None
        logger.info(
            "[api_server][chat] last_msg role=%s content_len=%s content_preview=%s",
            _last_msg.get("role") if isinstance(_last_msg, dict) else None,
            len(str(_last_msg.get("content", "") or "")) if isinstance(_last_msg, dict) else 0,
            repr((str(_last_msg.get("content", "") or ""))[:120]) if isinstance(_last_msg, dict) else None,
        )

        stream = body.get("stream", False)

        # Extract tools from request (passed by OpenCode client)
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        user_agent = request.headers.get("User-Agent", "")
        force_connection_close = _is_opencode_user_agent(user_agent)
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool["_from_client"] = True

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            content = _normalize_chat_content(raw_content)
            preserved_content = _preserve_multimodal_chat_content(raw_content)
            if role == "system":
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role == "assistant":
                assistant_entry: Dict[str, Any] = {"role": "assistant", "content": preserved_content}
                tool_calls = _extract_openai_tool_calls(msg.get("tool_calls"))
                if tool_calls:
                    assistant_entry["tool_calls"] = tool_calls
                # Preserve reasoning_content for providers that use it
                # (Moonshot/Kimi, GLM, Novita, OpenRouter) so multi-turn
                # conversations maintain reasoning context.
                reasoning_content = msg.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    assistant_entry["reasoning_content"] = reasoning_content
                conversation_messages.append(assistant_entry)
            elif role == "tool":
                tool_entry: Dict[str, Any] = {"role": "tool", "content": content}
                tool_call_id = msg.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id.strip():
                    tool_entry["tool_call_id"] = tool_call_id.strip()
                conversation_messages.append(tool_entry)
            elif role == "user":
                conversation_messages.append({"role": role, "content": preserved_content})

        user_message = ""
        history = []
        if conversation_messages:
            last_message = conversation_messages[-1]
            if last_message.get("role") == "user":
                raw_last_content = last_message.get("content", "") or ""
                user_message = _unwrap_omp_system_reminder(raw_last_content)
                if user_message is not raw_last_content:
                    logger.info(
                        "[api_server] stripped OMP system-reminder wrapper from user message; "
                        "original len=%d unwrapped len=%d preview=%r",
                        len(raw_last_content),
                        len(_normalize_chat_content(user_message)),
                        _normalize_chat_content(user_message)[:120],
                    )
                history = conversation_messages[:-1]
                # Empty user message at end of a tool cycle: pi/opencode sometimes
                # sends content="" as a "please continue" signal.  Walk back to
                # find the most recent non-blank user message so Hermes has
                # something meaningful to act on rather than producing the
                # "It looks like your message came through empty!" response.
                if not _normalize_chat_content(user_message).strip():
                    # If the prior assistant message had tool_calls, the agent
                    # is mid-loop — pass history as-is so it continues
                    # processing the tool results rather than replaying a stale
                    # user prompt.
                    if _prior_assistant_has_pending_tool_calls(conversation_messages[:-1]):
                        logger.info(
                            "[api_server] empty user message after assistant tool_calls — "
                            "continuing tool loop without injecting stale user message"
                        )
                        history = conversation_messages[:-1]
                        # user_message stays empty; the tool-result continuation
                        # path below will handle forwarding the history correctly
                    else:
                        user_message = _find_last_nonempty_user_message(conversation_messages[:-1])
                        if user_message:
                            logger.info(
                                "[api_server] empty user message at end of cycle — "
                                "using last non-empty user message as continuation: %s",
                                _normalize_chat_content(user_message)[:200],
                            )
            elif last_message.get("role") == "assistant" and _is_post_compaction_assistant_message(
                last_message.get("content", "")
            ):
                # Post-compaction continuation: the last assistant message is a
                # compaction summary injected by pi/opencode after the context
                # window filled up.  Extract the outstanding task so Hermes
                # resumes correctly instead of stalling with an empty user_message.
                user_message = _infer_intent_from_compaction_summary(
                    last_message.get("content", "")
                )
                history = conversation_messages  # keep the summary as context
                logger.info(
                    "[api_server] post-compaction continuation detected — inferred intent: %s",
                    user_message[:200],
                )
            else:
                # Last message is a tool result (or other non-user, non-assistant role).
                # The original user instruction is already visible in `history`,
                # so do NOT re-inject it as a new user_message — doing so makes
                # the provider see the same instruction twice and interpret it
                # as a new task, causing the model to repeat work.
                history = conversation_messages
        # NOTE: Message history is NOT pre-truncated here. The agent's own
        # context compressor (agent/context_compressor.py) handles context
        # overflow adaptively based on the actual model's context window.
        # Pre-truncation was destroying context for large conversations.
        # Dynamic model selection now picks a model with enough context
        # for the conversation size, so pre-truncation is unnecessary.

        # Tool loop prevention: allow legitimate OpenAI tool continuation
        # (assistant tool_calls -> tool results), but reject orphaned tool-only
        # continuations that have no preceding assistant tool call context.
        has_user_msg = bool(_normalize_chat_content(user_message).strip())
        is_tool_result_only = bool(
            conversation_messages and
            conversation_messages[-1].get("role") == "tool"
        )
        last_non_tool = None
        if is_tool_result_only:
            for msg in reversed(conversation_messages[:-1]):
                if isinstance(msg, dict) and msg.get("role") != "tool":
                    last_non_tool = msg
                    break
        has_assistant_tool_context = bool(
            isinstance(last_non_tool, dict)
            and last_non_tool.get("role") == "assistant"
            and isinstance(last_non_tool.get("tool_calls"), list)
            and last_non_tool.get("tool_calls")
        )
        if is_tool_result_only and not has_user_msg and not has_assistant_tool_context:
            return web.json_response(
                {"error": {"message": "Cannot continue with orphaned tool results. Include a user message, or send the preceding assistant tool_calls in the conversation.", "type": "invalid_request_error"}},
                status=400,
            )

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        #
        # Security: session continuation exposes conversation history, so it is
        # only allowed when the API key is configured and the request is
        # authenticated.  Without this gate, any unauthenticated client could
        # read arbitrary session history by guessing/enumerating session IDs.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        _request_conversation_messages = list(conversation_messages)
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity."
                )
                return web.json_response(
                    _openai_error(
                        "Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature."
                    ),
                    status=403,
                )
            # Sanitize: reject control characters that could enable header injection.
            if re.search(r'[\r\n\x00]', provided_session_id):
                return web.json_response(
                    {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                    status=400,
                )
            session_id = provided_session_id
            try:
                db = self._ensure_session_db()
                if db is not None:
                    _stored_history = db.get_messages_as_conversation(session_id)
                    history, user_message = _merge_session_history_with_request_delta(
                        _stored_history,
                        _request_conversation_messages,
                        user_message,
                    )
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Derive a collision-free session ID from the conversation fingerprint.
            # _resolve_session_id uses the full message history as a secondary
            # fingerprint so clients whose first messages collide get different
            # session IDs from turn 2 onward, and a random variant suffix for
            # simultaneous turn-1 collisions.
            omp_instance = request.headers.get("X-OMP-Instance", request.headers.get("User-Agent", ""))
            salt = f"{request.headers.get('X-Forwarded-For', request.remote or '').split(',')[0].strip()}|{omp_instance}"
            session_id = _resolve_session_id(salt, system_prompt, conversation_messages)

        # NOTE: Message history is NOT pre-truncated. Agent's context compressor
        # handles overflow based on actual model's context window.

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        # Short request ID for log correlation across provider attempts.
        # First 8 hex chars of the completion_id is unique enough to trace
        # a single request through fallback chain, quality scoring, etc.
        _req_id = completion_id.replace("chatcmpl-", "")[:8]
        _req_start = time.monotonic()
        model_name = body.get("model", self._model_name)
        # NOTE: TIMING START log is emitted AFTER passthrough_tools is assigned
        # (later in this function) so it can include has_tools. Don't reference
        # passthrough_tools here — it raises UnboundLocalError.
        role_cfg = _get_role_alias_config(model_name)
        role_hint = dict(role_cfg.get("hint") or {}) if role_cfg else None
        _toolset_mode = "auto"
        _provider_mode = False
        if model_name == "hermes-agentic-full":
            _toolset_mode = "full"
        elif model_name in ("hermes-code", "hermes-privacy", "claude-code-cli", "mimocode-cli") or (
            "/" in model_name and not role_cfg
        ):
            # Activate passthrough for hermes-code, claude-code-cli, mimocode-cli,
            # OR any model with / (provider/model format) — but NOT swarm role
            # aliases like hermes-gateway/hermes-reflect which have their own path.
            _provider_mode = True
        external_tool_mode = "none"
        if isinstance(tools, list) and tools:
            if model_name in ("hermes-code", "hermes-privacy"):
                external_tool_mode = "inband"
            else:
                external_tool_mode = "inband" if force_connection_close else "broker"
        logger.info(
            "[api_server][req=%s] chat request stream=%s tools=%s external_tool_mode=%s ua=%s model=%s",
            _req_id, stream, bool(tools), external_tool_mode, user_agent[:120], model_name,
        )

        _approx_tokens = 0
        if _provider_mode or model_name == "hermes-swarm" or (role_cfg and role_cfg.get("mode") == "swarm"):
            try:
                from agent.model_metadata import estimate_request_tokens_rough
                _approx_tokens = estimate_request_tokens_rough(
                    history or [],
                    system_prompt=system_prompt or "",
                    tools=tools,
                )
            except Exception:
                _approx_tokens = 0

        _needs_vision = _provider_mode and _messages_have_image_parts(conversation_messages)
        _needs_audio = _provider_mode and _messages_have_audio_parts(conversation_messages)
        _uses_external_image_urls = _provider_mode and _messages_use_external_image_urls(conversation_messages)

        if _provider_mode and _looks_like_roo_condense_request(
            stream=bool(stream),
            tools=tools,
            system_prompt=system_prompt or "",
            user_message=user_message or "",
        ):
            try:
                from agent.auxiliary_client import call_llm, extract_content_or_reasoning
                from agent.auxiliary_client import _resolve_task_provider_model

                aux_provider, aux_model, _aux_base_url, _aux_api_key, _aux_api_mode = _resolve_task_provider_model(
                    task="compression"
                )

                direct_messages: List[Dict[str, Any]] = []
                if system_prompt:
                    direct_messages.append({"role": "system", "content": system_prompt})
                direct_messages.extend(history)
                if user_message:
                    direct_messages.append({"role": "user", "content": user_message})

                condense_max_tokens = min(8192, _hermes_code_advertised_max_output_tokens())
                condense_timeout = float(os.getenv("HERMES_ROO_CONDENSE_TIMEOUT_SECONDS", "120"))
                logger.info(
                    "[api_server] Roo condense fast-path: aux_provider=%s aux_model=%s est_tokens=%s max_tokens=%s timeout=%ss",
                    aux_provider or "auto",
                    aux_model or "<provider-default>",
                    _approx_tokens,
                    condense_max_tokens,
                    condense_timeout,
                )
                response_obj = call_llm(
                    task="compression",
                    messages=direct_messages,
                    max_tokens=condense_max_tokens,
                    timeout=condense_timeout,
                )
                content = extract_content_or_reasoning(response_obj).strip()
                if content:
                    if provided_session_id:
                        try:
                            _persist_passthrough_session_delta(
                                self._ensure_session_db(),
                                session_id,
                                model_name=model_name,
                                system_prompt=system_prompt,
                                request_messages=_request_conversation_messages,
                                assistant_content=content,
                                finish_reason="stop",
                            )
                        except Exception as _persist_exc:
                            logger.warning("[api_server] failed to persist Roo condense session delta for %s: %s", session_id, _persist_exc)
                    usage_obj = getattr(response_obj, "usage", None)
                    response_data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                        },
                    }
                    headers = {"X-Hermes-Session-Id": session_id}
                    return web.json_response(response_data, headers=headers)
            except Exception as exc:
                logger.warning("[api_server] Roo condense fast-path failed; falling back to full agent path: %s", exc)

        # Hermes-code passthrough: client manages context, passthrough to LLM without AIAgent.
        # Works for both tool-less and tool-calling requests.
        # Falls back through HERMES_CODE_MODEL and HERMES_CODE_FALLBACK_* providers.
        if _provider_mode:
            from agent.auxiliary_client import call_llm, extract_content_or_reasoning

            # Build passthrough messages from history
            passthrough_messages: List[Dict[str, Any]] = []
            if system_prompt:
                passthrough_messages.append({"role": "system", "content": system_prompt})
            passthrough_messages.extend(history)
            if _normalize_chat_content(user_message).strip():
                passthrough_messages.append({"role": "user", "content": user_message})

            # reasoning_content handling: some providers (DeepSeek thinking mode,
            # Kimi, GLM) require reasoning_content to be echoed back in the
            # conversation history. Others (most OpenAI-compat providers, Groq,
            # Ollama) reject it with 400. Strip lazily per-provider inside the
            # loop so each provider gets the right view without mutating the shared
            # list permanently.
            _passthrough_has_reasoning = any(
                isinstance(m, dict) and m.get("role") == "assistant" and m.get("reasoning_content")
                for m in passthrough_messages
            )
            _has_rc_msgs = [
                {"role": m.get("role"), "rc_len": len(m.get("reasoning_content") or "")}
                for m in passthrough_messages
                if isinstance(m, dict) and m.get("reasoning_content")
            ]
            logger.debug(
                "[hermes-code] reasoning check: has_rc=%s rc_msgs=%s history_len=%d",
                _passthrough_has_reasoning, _has_rc_msgs, len(history),
            )

            def _strip_reasoning(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Return a shallow copy of msgs with reasoning_content removed from assistant turns."""
                out = []
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "assistant" and "reasoning_content" in m:
                        m = {k: v for k, v in m.items() if k != "reasoning_content"}
                    out.append(m)
                return out

            def _synthesize_reasoning_for_tool_calls(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Ensure every assistant message with tool_calls also has a reasoning_content
                field, defaulting to empty string if absent.

                DeepSeek (and proxies that forward to DeepSeek like opencode-zen/opencode-go)
                reject requests where an assistant message has tool_calls but is missing
                reasoning_content — the API requires the field to be present (even if empty)
                on every assistant turn in thinking mode. This synthesises an empty
                ``reasoning_content`` for any assistant turn that has tool_calls but no
                ``reasoning_content`` field, so the provider doesn't 400.
                """
                out = []
                for m in msgs:
                    if (
                        isinstance(m, dict)
                        and m.get("role") == "assistant"
                        and m.get("tool_calls")
                        and "reasoning_content" not in m
                    ):
                        m = {**m, "reasoning_content": ""}
                    out.append(m)
                return out

            def _fabricate_reasoning_for_consistency(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Add a placeholder reasoning_content to assistant messages that lack it.

                When a conversation has mixed history (some assistant messages have
                reasoning_content, some don't), models that require reasoning echo
                (DeepSeek, MiMo, etc.) are skipped.  This function adds a minimal
                placeholder to make the history consistent so these models can be used.
                The placeholder is stripped by _strip_reasoning for providers that reject it.
                """
                out = []
                for m in msgs:
                    if (
                        isinstance(m, dict)
                        and m.get("role") == "assistant"
                        and "reasoning_content" not in m
                    ):
                        m = {**m, "reasoning_content": " "}
                    out.append(m)
                return out

            # Fabricate reasoning_content for consistency when history is mixed.
            # This prevents models that require reasoning echo from being skipped
            # when some assistant messages have reasoning_content and some don't.
            _mixed_rc = _passthrough_has_reasoning and 0 < len(_has_rc_msgs) < sum(
                1 for m in passthrough_messages if isinstance(m, dict) and m.get("role") == "assistant"
            )
            if _mixed_rc:
                passthrough_messages = _fabricate_reasoning_for_consistency(passthrough_messages)
                _passthrough_has_reasoning = True  # now all have it

            def _truncate_messages_for_context(msgs: List[Dict[str, Any]], target_tokens: int) -> List[Dict[str, Any]]:
                """Truncate conversation to fit within target_tokens by removing middle messages.

                Keeps the head (system + first user-assistant exchange) and tail
                (recent messages), removing the middle.  Inserts a summary marker
                so the model understands context was compacted.
                """
                from agent.model_metadata import estimate_messages_tokens_rough
                current = estimate_messages_tokens_rough(msgs)
                if current <= target_tokens:
                    return msgs

                # Find head boundary: system prompt + first user message + first assistant response
                head_end = 0
                for i, m in enumerate(msgs):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        head_end = i + 1
                        break
                head_end = max(head_end, 2)  # at least system + first user

                # Find tail: work backwards from end, keeping messages until we hit target
                tail_start = len(msgs)
                tail_tokens = 0
                tail_budget = int(target_tokens * 0.4)  # 40% for tail
                for i in range(len(msgs) - 1, head_end - 1, -1):
                    msg_tokens = estimate_messages_tokens_rough([msgs[i]])
                    if tail_tokens + msg_tokens > tail_budget:
                        break
                    tail_tokens += msg_tokens
                    tail_start = i

                if tail_start <= head_end:
                    # Can't compress enough — return as-is and let provider fail
                    return msgs

                # Build compressed message list
                head = msgs[:head_end]
                tail = msgs[tail_start:]
                compressed = head + [
                    {"role": "system", "content": "[CONTEXT COMPACTED — middle messages removed to fit model context window. Continuing from recent conversation.]"}
                ] + tail

                new_tokens = estimate_messages_tokens_rough(compressed)
                logger.warning(
                    "[hermes-code] auto-truncated context: %d msgs (%d tokens) → %d msgs (%d tokens)",
                    len(msgs), current, len(compressed), new_tokens,
                )
                return compressed

            def _requires_reasoning_echo(model_id: str, provider: str = "", base_url: str = "") -> bool:
                """Return True for models that require reasoning_content echoed back.

                DeepSeek (all variants, not just thinking), Kimi/Moonshot, and Xiaomi
                MiMo all require ``reasoning_content`` on assistant messages in multi-turn
                requests. Most other providers (OpenAI, Groq, Ollama, Anthropic) reject
                it with 400. Match on model name, provider name, and base URL domain.
                """
                slug = model_id.lower()
                prov = provider.lower()
                url = base_url.lower() if base_url else ""

                # DeepSeek — all variants require echo, not just thinking/reasoner
                if "deepseek" in slug:
                    return True
                if prov == "deepseek" or "deepseek" in prov:
                    return True
                if "api.deepseek.com" in url:
                    return True

                # Kimi / Moonshot
                if "kimi" in slug or "moonshot" in slug:
                    return True
                if "kimi" in prov or "moonshot" in prov:
                    return True
                if "api.kimi.com" in url or "moonshot.ai" in url or "moonshot.cn" in url:
                    return True

                # Xiaomi MiMo
                if "xiaomi" in slug or "mimo" in slug:
                    return True
                if prov == "xiaomi":
                    return True
                if "xiaomimimo.com" in url:
                    return True

                # OpenCode Zen/Go aggregator proxies. These forward to multiple
                # backends including DeepSeek (which requires reasoning_content
                # echo). Since the proxy handles the routing server-side we
                # cannot inspect the effective downstream provider from the
                # client — preserve reasoning_content for all models behind
                # these aggregators so DeepSeek-tunneled models don't 400.
                if prov in ("opencode-zen", "opencode-go"):
                    return True

                return False

            def _strip_from_client_tools(tools: Any) -> Any:
                """Remove internal markers and unsupported fields from tool definitions.
                
                _from_client is added during request parsing for internal tracking.
                strict is an OpenAI 4.x feature not supported by Google and other providers.
                These MUST NOT be sent to external APIs.
                """
                if not isinstance(tools, list):
                    return tools
                result = []
                for tool in tools:
                    if isinstance(tool, dict):
                        cleaned = {
                            k: v for k, v in tool.items()
                            if k not in ("_from_client", "strict")
                        }
                        # Also strip strict from nested function.parameters
                        fn = cleaned.get("function")
                        if isinstance(fn, dict):
                            fn_cleaned = {k: v for k, v in fn.items() if k != "strict"}
                            cleaned["function"] = fn_cleaned
                        result.append(cleaned)
                    else:
                        result.append(tool)
                return result

            passthrough_tools = tools if tools else None
            # _from_client is added during request parsing (line 5113) for internal tracking.
            # External providers reject unknown fields — strip before sending to any API.
            if passthrough_tools:
                passthrough_tools = _strip_from_client_tools(passthrough_tools)
            # The CLI subprocesses (mimocode-cli, claude-code-cli) each prefix MCP
            # tool names differently in their agent/MCP config — and the agent.md
            # permission filter only admits the prefixed form, so client tool names
            # MUST be rewritten before the request reaches the CLI. The MCP bridge
            # strips the prefix back when executing.
            #   mimocode-cli:  mcp_<name>           (single underscore, mimo's MCP server)
            #   claude-code-cli: mcp__hermes-tools__<name>  (FastMCP convention)
            #   everything else: no rewrite needed
            if passthrough_tools and model_name in (
                "mimocode-cli", "mimocode-cli/mimocode-cli",
                "claude-code-cli", "claude-code-cli/claude-code-cli",
                "claude-code-cli/sonnet",  # legacy
            ):
                if model_name.startswith("claude-code-cli"):
                    _PREFIX = "mcp__hermes-tools__"
                else:
                    _PREFIX = "mcp_"
                def _mcp_prefix_tools(tools: Any) -> Any:
                    if not isinstance(tools, list):
                        return tools
                    out = []
                    for t in tools:
                        if isinstance(t, dict):
                            t2 = dict(t)
                            fn = t2.get("function")
                            if isinstance(fn, dict) and fn.get("name") and not fn["name"].startswith("mcp"):
                                fn2 = dict(fn)
                                fn2["name"] = _PREFIX + fn2["name"]
                                t2["function"] = fn2
                            out.append(t2)
                        else:
                            out.append(t)
                    return out
                passthrough_tools = _mcp_prefix_tools(passthrough_tools)
            # Emit the START timing log AFTER passthrough_tools is assigned
            # so the has_tools flag is accurate. Using _req_start above.
            logger.info("[TIMING][req=%s] chat_completions START model=%s has_tools=%s session_id=%s t_offset=%.3fs",
                        _req_id, model_name, bool(passthrough_tools), session_id,
                        time.monotonic() - _req_start)
            # ── Fallback tool set ──────────────────────────────────────────────
            # 24+ tool definitions overwhelm smaller/cheaper fallback models
            # (kimi-k2-thinking: 94% text-only, glm-5: 62%, etc.).  The primary
            # model gets the full tool set; fallback models get only essential tools.
            _fallback_essential_names = {
                n.strip().lower()
                for n in os.getenv("HERMES_FALLBACK_ESSENTIAL_TOOLS", "bash,read,edit,find,search,write,search_tool_bm25").split(",")
                if n.strip()
            }
            _fallback_tools = passthrough_tools
            if passthrough_tools and len(passthrough_tools) > len(_fallback_essential_names):
                _fallback_tools = [
                    t for t in passthrough_tools
                    if t.get("function", {}).get("name", "").lower() in _fallback_essential_names
                ]
                logger.warning(
                    "[hermes-code] fallback tool reduction: %d → %d tools (essential=%s)",
                    len(passthrough_tools), len(_fallback_tools), _fallback_essential_names,
                )
            # Preserve the full tool set so quality-aware logic can restore it
            # for models with low text-only rates.
            _passthrough_tools_full = list(passthrough_tools) if passthrough_tools else None
            # ── Tool-loop audit ───────────────────────────────────────────────
            # Observe-only: detects repeated identical tool calls and logs them.
            # Does NOT inject or mutate passthrough_messages.
            _loop_threshold = int(os.getenv("HERMES_TOOL_LOOP_THRESHOLD", "5"))
            _detect_and_nudge_tool_loop(passthrough_messages, threshold=_loop_threshold)

            # Log final message role sequence so any unexpected message can be
            # traced in production (WARNING level — visible alongside MONITOR).
            _pt_roles = ", ".join(
                m.get("role", "?") + ("+tc" if m.get("tool_calls") else "")
                for m in passthrough_messages
                if isinstance(m, dict)
            )
            logger.debug(
                "[hermes-code] passthrough → provider: %d msgs last_roles=[...%s] "
                "stream=%s tools=%d",
                len(passthrough_messages),
                _pt_roles[-240:],
                stream,
                len(passthrough_tools) if passthrough_tools else 0,
            )

            _passthrough_sse_response = None
            _passthrough_sse_completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            _passthrough_sse_created = int(time.time())

            async def _ensure_passthrough_sse_response():
                """Open the client SSE stream and send the role chunk once."""
                nonlocal _passthrough_sse_response
                if _passthrough_sse_response is not None:
                    return _passthrough_sse_response

                sse_headers = {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }
                if session_id:
                    sse_headers["X-Hermes-Session-Id"] = session_id

                response = web.StreamResponse(status=200, headers=sse_headers)
                await response.prepare(request)
                role_chunk = {
                    "id": _passthrough_sse_completion_id,
                    "object": "chat.completion.chunk",
                    "created": _passthrough_sse_created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
                _passthrough_sse_response = response
                logger.info("[hermes-code][req=%s] opened passthrough SSE heartbeat stream", _req_id)
                return response

            async def _emit_passthrough_thinking_sse(provider_model: str) -> None:
                """Emit an OpenAI-compatible no-op chunk to keep clients/proxies alive."""
                response = await _ensure_passthrough_sse_response()
                thinking_chunk = {
                    "id": _passthrough_sse_completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                }
                await response.write(b": thinking\n\n")
                await response.write(f"data: {json.dumps(thinking_chunk)}\n\n".encode())
                logger.debug("[hermes-code][req=%s] sent passthrough thinking heartbeat for %s", _req_id, provider_model)

            async def _await_passthrough_provider_call(awaitable, provider_model: str):
                """Wait for a blocking provider call, sending heartbeat chunks every 30s."""
                if not stream:
                    return await awaitable
                while True:
                    try:
                        return await asyncio.wait_for(asyncio.shield(awaitable), timeout=CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        await _emit_passthrough_thinking_sse(provider_model)

            async def _finish_passthrough_sse_error(message: str):
                response = await _ensure_passthrough_sse_response()
                error_chunk = {
                    "id": _passthrough_sse_completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": None}],
                }
                finish_chunk = {
                    "id": _passthrough_sse_completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                try:
                    await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
                    await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                except Exception:
                    # Client disconnected mid-stream; skip writing error to avoid ERROR traceback
                    pass
                return response

            # Build passthrough provider chain from HERMES_CODE_MODEL and HERMES_CODE_FALLBACK_*
            # Put user's requested model FIRST, then HERMES_CODE_MODEL as primary, then fallbacks
            _passthrough_models: List[str] = []
            _strict_mode = False  # when True, only the explicitly-requested model is tried
            # User's requested model goes first (unless already in chain)
            if model_name == "hermes-code":
                selected_code_model = _select_hermes_code_model(
                    estimated_tokens=_approx_tokens,
                    session_id=session_id,
                    require_vision=_needs_vision,
                    require_audio=_needs_audio,
                    avoid_external_image_incompatible=_uses_external_image_urls,
                )
                if selected_code_model and selected_code_model not in _passthrough_models:
                    _passthrough_models.append(selected_code_model)

            elif model_name == "hermes-privacy":
                selected_privacy_model = _select_hermes_privacy_model(
                    estimated_tokens=_approx_tokens,
                    session_id=session_id,
                    require_vision=_needs_vision,
                    require_audio=_needs_audio,
                    avoid_external_image_incompatible=_uses_external_image_urls,
                )
                if selected_privacy_model and selected_privacy_model not in _passthrough_models:
                    _passthrough_models.append(selected_privacy_model)
            elif model_name and model_name not in _passthrough_models:
                if "/" in model_name:
                    _passthrough_models.append(model_name)
                    _strict_mode = True  # explicit model — no silent fallback
                    logger.info(
                        "[hermes-code][req=%s] strict mode: explicit model %s requested, "
                        "fallback chain will be skipped",
                        _req_id, model_name,
                    )
                elif model_name in ("claude-code-cli", "mimocode-cli"):
                    # External-process providers: strict mode, pass through to provider-specific handler
                    _passthrough_models.append(model_name)
                    _strict_mode = True
            # Copernican copilot models go next (user's personal API)
            if model_name.startswith("github-copilot") or model_name.startswith("copilot-"):
                pass  # Already added above
            # Then HERMES_CODE_MODEL as the actual runtime model for hermes-code
            # (skipped in strict mode — the explicit user request is honored alone)
            primary = os.getenv("HERMES_PRIVACY_MODEL" if model_name == "hermes-privacy" else "HERMES_CODE_MODEL", "").strip()
            _is_privacy_passthrough = model_name == "hermes-privacy"
            if (
                not _strict_mode
                and primary
                and "/" in primary
                and primary not in _passthrough_models
                and not _passthrough_fallback_provider_excluded(primary, privacy=_is_privacy_passthrough)
            ):
                _passthrough_models.append(primary)
            if not _strict_mode:
                _cool = []
                try:
                    from agent.model_cooldown_db import model_cooldown_remaining
                    for idx in range(1, _hermes_code_max_fallbacks() + 1):
                        fb = os.getenv(f"HERMES_PRIVACY_FALLBACK_{idx}" if model_name == "hermes-privacy" else f"HERMES_CODE_FALLBACK_{idx}", "").strip()
                        if _passthrough_fallback_provider_excluded(fb, privacy=_is_privacy_passthrough):
                            continue
                        if fb and fb not in _passthrough_models:
                            _prov = fb.split("/")[0] if "/" in fb else ""
                            _rem = model_cooldown_remaining(_prov, fb) if _prov else 0
                            if _rem and _rem > 0:
                                logger.debug("[hermes-code] passthrough: skipping %s in cooldown (%.0fs)", fb, _rem)
                                continue
                            _passthrough_models.append(fb)
                except Exception:
                    # Fallback: add all models without checking (original behaviour)
                    for idx in range(1, _hermes_code_max_fallbacks() + 1):
                        fb = os.getenv(f"HERMES_PRIVACY_FALLBACK_{idx}" if model_name == "hermes-privacy" else f"HERMES_CODE_FALLBACK_{idx}", "").strip()
                        if _passthrough_fallback_provider_excluded(fb, privacy=_is_privacy_passthrough):
                            continue
                        if fb and fb not in _passthrough_models:
                            _passthrough_models.append(fb)
            # ── Quality-based reordering + quality floor ─────────────────────────
            # Sort the REST of the chain by live quality score (best first), then
            # drop any non-pinned model whose score is below the floor — they
            # belong in the dead-models cleanup script, not in a live fallback
            # chain.  Pin position 0 (user's requested model or HERMES_CODE_MODEL)
            # so the explicit request is always honoured first, even if its
            # recorded score is low (e.g. brand-new model, or model currently
            # being debugged).
            if not _strict_mode and len(_passthrough_models) > 1:
                try:
                    from agent.model_quality_db import get_quality_score
                    _pinned = _passthrough_models[0]  # user's request or HERMES_CODE_MODEL
                    _rest = _passthrough_models[1:]

                    def _score(m: str) -> float:
                        return get_quality_score(
                            m.split("/", 1)[0] if "/" in m else "",
                            m.split("/", 1)[1] if "/" in m else m,
                        )

                    _rest.sort(key=_score, reverse=True)
                    # Quality floor — drop chronically-bad models from the chain.
                    # Default 50.0 (a model with 50% success + 0/30 tool reliability
                    # + 0 latency bonus is at the floor; anything worse is dropped).
                    # Previously 60.0, but that drops too many primary models including
                    # the user's primary HERMES_CODE_MODEL (e.g. github-copilot/gpt-5-mini=58.0)
                    # and other useful chain models (xiaomi/mimo-v2.5-pro=59.4, etc.).
                    # Lowered to 50.0 so quality-floor acts as a sanity check on truly
                    # bad models (e.g. opencode-go/deepseek-v4-flash=45.4) without
                    # removing mid-quality options that the user explicitly picked.
                    # Set HERMES_CODE_QUALITY_FLOOR=0 to disable.
                    _floor = float(os.getenv("HERMES_PRIVACY_QUALITY_FLOOR" if model_name == "hermes-privacy" else "HERMES_CODE_QUALITY_FLOOR", "50.0") or 0.0)
                    _kept: List[str] = []
                    _dropped: List[Tuple[str, float]] = []
                    for m in _rest:
                        s = _score(m)
                        if s < _floor:
                            _dropped.append((m, s))
                        else:
                            _kept.append(m)
                    _passthrough_models = [_pinned] + _kept
                    if _dropped:
                        logger.info(
                            "[hermes-code][req=%s] quality floor %.1f dropped %d model(s): %s",
                            _req_id,
                            _floor,
                            len(_dropped),
                            ", ".join(f"{m}={s:.0f}" for m, s in _dropped),
                        )
                    logger.info(
                        "[hermes-code][req=%s] quality-sorted chain: pinned=%s rest_top5=%s",
                        _req_id,
                        _pinned,
                        [f"{m.split('/',1)[-1] if '/' in m else m}:{_score(m):.0f}" for m in _passthrough_models[1:6]],
                    )
                except Exception:
                    pass
            # ── End quality reordering ──────────────────────────────────────────
            logger.debug("[hermes-code] passthrough chain: first=%s models=%d", _passthrough_models[0] if _passthrough_models else "EMPTY", len(_passthrough_models))

            # ── Context overflow guard ──────────────────────────────────────────────
            # When ALL passthrough models have KNOWN context windows too small for the
            # estimated token count, OR all context-capable models are on cooldown,
            # auto-truncate the conversation to fit the largest available context.
            # If truncation still isn't enough, return 413 telling the client to compact.
            #
            # NOTE: Models with UNKNOWN context are NOT assumed to handle the request
            # — they may fail for other reasons. If all KNOWN models are too small
            # (or on cooldown), we try truncation first, then 413.
            _has_external_cli_passthrough = any(
                _pm == "claude-code-cli"
                or _pm.startswith("claude-code-cli/")
                or _pm == "mimocode-cli"
                or _pm.startswith("mimocode-cli/")
                for _pm in _passthrough_models
            )
            if _approx_tokens > 0 and len(_passthrough_models) > 0 and not _has_external_cli_passthrough:
                _max_known_ctx = 0
                _any_viable = False
                _known_models = []
                for _pm in _passthrough_models:
                    _ctx = _model_context_length(_pm)
                    if _ctx > 0:
                        _known_models.append((_pm, _ctx))
                        if _ctx > _max_known_ctx:
                            _max_known_ctx = _ctx
                        if _model_can_handle_context(_pm, _approx_tokens):
                            # Check this model's cooldown before declaring it viable
                            _prov = _pm.split("/")[0] if "/" in _pm else ""
                            _on_cooldown = False
                            if _prov:
                                try:
                                    from agent.model_cooldown_db import model_cooldown_remaining
                                    _rem = model_cooldown_remaining(_prov, _pm)
                                    if _rem and _rem > 0:
                                        _on_cooldown = True
                                except Exception:
                                    pass
                            if not _on_cooldown:
                                _any_viable = True
                if not _any_viable and _max_known_ctx < _approx_tokens:
                    # Context too large — auto-truncate to fit largest available model
                    _target_tokens = int(_max_known_ctx * 0.85)  # 85% margin
                    passthrough_messages = _truncate_messages_for_context(passthrough_messages, _target_tokens)
                    # Re-estimate tokens after truncation
                    from agent.model_metadata import estimate_messages_tokens_rough
                    _approx_tokens = estimate_messages_tokens_rough(passthrough_messages)
                    logger.info(
                        "[hermes-code] auto-truncated to ~%d tokens for max context %d",
                        _approx_tokens, _max_known_ctx,
                    )
                    # Re-check viability with truncated messages
                    _any_viable = False
                    for _pm in _passthrough_models:
                        _ctx = _model_context_length(_pm)
                        if _ctx > 0 and _model_can_handle_context(_pm, _approx_tokens):
                            _prov = _pm.split("/")[0] if "/" in _pm else ""
                            _on_cooldown = False
                            if _prov:
                                try:
                                    from agent.model_cooldown_db import model_cooldown_remaining
                                    _rem = model_cooldown_remaining(_prov, _pm)
                                    if _rem and _rem > 0:
                                        _on_cooldown = True
                                except Exception:
                                    pass
                            if not _on_cooldown:
                                _any_viable = True
                                break
                if not _any_viable:
                    _models_summary = ", ".join(
                        f"{m}({f'{c:,}'})" for m, c in _known_models[:5]
                    )
                    # Differentiate context-too-small from all-on-cooldown
                    if _max_known_ctx < _approx_tokens:
                        _detail = f"too many tokens: ~{_approx_tokens:,} exceeds max model context ({_max_known_ctx:,})"
                    else:
                        _detail = f"too many tokens: ~{_approx_tokens:,} and all context-capable models on cooldown"
                    logger.warning(
                        "[hermes-code] ALL %d known models not viable for ~%d tokens "
                        "(max: %s). Returning 413 — client should compact. Chain: %s",
                        len(_known_models), _approx_tokens,
                        f"{_max_known_ctx:,}" if _max_known_ctx > 0 else "unknown",
                        _models_summary,
                    )
                    return web.json_response(
                        _openai_error(
                            f"Context too large: {_detail}. "
                            f"Please compact the conversation history and retry.",
                            err_type="invalid_request_error",
                            code="context_too_large",
                        ),
                        status=413,
                    )
            # ── End context overflow guard ──────────────────────────────────────────

            passthrough_error = None
            _pt_call_count = [0]  # mutable counter: incremented per provider attempt

            logger.debug("[%d] streaming passthrough start: %d models", _req_id, len(_passthrough_models))
            # ── Max provider attempts cap ──────────────────────────────────────
            # Prevent runaway cascade where many providers return text-only
            # (no tool calls) and the loop tries every model in the chain.
            # Each provider attempt (API call) counts toward the limit.
            _max_provider_attempts = int(os.getenv("HERMES_CODE_MAX_PROVIDER_ATTEMPTS", "6") or "6")
            _provider_attempt_count = [0]  # mutable: incremented per actual attempt

            # Streaming passthrough: collect full response then stream as SSE
            if stream:
                for provider_model in _passthrough_models:
                    if "/" not in provider_model and provider_model not in ("claude-code-cli", "mimocode-cli"):
                        continue

                    # ── Tool set selection (quality-aware) ──
                    # Models with a high text-only rate (> 30%) get reduced essential tools
                    # because they frequently answer in plain text instead of emitting tool_calls.
                    # Models with low/unknown text-only rate get the full client tool set.
                    if passthrough_tools and _passthrough_tools_full:
                        from agent.model_quality_db import get_text_only_rate
                        _f_prov = provider_model.split("/")[0] if "/" in provider_model else ""
                        _f_rate = get_text_only_rate(_f_prov, provider_model)
                        if _f_rate > 0.30:
                            if passthrough_tools is not _fallback_tools:
                                passthrough_tools = _fallback_tools
                                logger.warning(
                                    "[hermes-code] %s: text-only rate %.0f%% → reduced to %d essential tools",
                                    provider_model, _f_rate * 100, len(_fallback_tools),
                                )
                        else:
                            if passthrough_tools is not _passthrough_tools_full:
                                passthrough_tools = _passthrough_tools_full
                                logger.debug(
                                    "[hermes-code] %s: text-only rate %.0f%% → restored full %d tools",
                                    provider_model, _f_rate * 100, len(passthrough_tools),
                                )

                    # Check cooldown before attempting this provider.
                    # Skip this check for external CLI providers — their cooldowns
                    # are about external subprocess/auth state, not about the
                    # passthrough fallback chain, and blocking them here causes
                    # the explicit "claude-code-cli" / "mimocode-cli" requests
                    # to fail with "no passthrough providers available" (503).
                    _is_external_cli = (
                        provider_model == "claude-code-cli"
                        or provider_model.startswith("claude-code-cli/")
                        or provider_model == "mimocode-cli"
                        or provider_model.startswith("mimocode-cli/")
                    )
                    _prov_prefix = provider_model.split("/")[0] if "/" in provider_model else ""
                    if _prov_prefix and not _is_external_cli:
                        try:
                            from agent.model_cooldown_db import model_cooldown_remaining
                            _cooldown_key = provider_model
                            _remaining = model_cooldown_remaining(_prov_prefix, _cooldown_key)
                            if _remaining and _remaining > 0:
                                logger.warning("[hermes-code] %s in cooldown (%.0fs remaining), skipping", _cooldown_key, _remaining)
                                continue
                        except Exception:
                            pass

                    # Skip toxic fallback models with extreme failure history
                    if _hermes_code_skip_toxic_fallback(provider_model):
                        logger.warning(
                            "[hermes-code] %s: hard-skipping toxic fallback (recent failure history)",
                            provider_model,
                        )
                        continue

                    # ── Max provider attempts cap (streaming) ──
                    _provider_attempt_count[0] += 1
                    if _max_provider_attempts > 0 and _provider_attempt_count[0] > _max_provider_attempts:
                        logger.warning(
                            "[hermes-code][req=%s] max provider attempts (%d) reached after %s, "
                            "stopping cascade to avoid amplification",
                            _req_id, _max_provider_attempts, provider_model,
                        )
                        break

                    # Google Gemini 3.1+ requires thought_signature on every
                    # functionCall part.  When the opencode/OMP client replays
                    # a conversation whose tool_calls came from a non-Gemini
                    # provider (or whose extra_content was stripped on the
                    # round-trip), the message history has unsigned function
                    # calls.  The cleanest fix is to inject a sentinel
                    # ``extra_content.google.thought_signature`` value on
                    # each unsigned call before sending to Google — Google's
                    # OpenAI-compat endpoint accepts the sentinel and skips
                    # the per-call signature validator.
                    if _prov_prefix == "google" and passthrough_messages:
                        _unsigned = _count_unsigned_google_tool_calls(passthrough_messages)
                        if _unsigned > 0:
                            logger.debug(
                                "[hermes-code] %s: %d assistant tool_call(s) lack thought_signature; sentinel will be injected in message build step",
                                provider_model, _unsigned,
                            )

                    # Skip models whose context window cannot safely hold the estimated
                    # request tokens. This prevents costly "prompt too long" round-trips
                    # that waste API quota and trigger circuit breakers unnecessarily.
                    if _approx_tokens > 0 and not _model_can_handle_context(provider_model, _approx_tokens):
                        _ctx_limit = _model_context_length(provider_model)
                        logger.warning(
                            "[hermes-code] %s context too small for ~%d tokens "
                            "(limit=%s), skipping",
                            provider_model, _approx_tokens,
                            f"{_ctx_limit:,}" if _ctx_limit > 0 else "unknown",
                        )
                        continue

                    # Skip models that require reasoning_content echo ONLY when the
                    # conversation has a MIXED history (some assistant messages have
                    # reasoning_content, some don't). When ALL lack reasoning_content
                    # (e.g. prior turns served by non-thinking models) there is nothing
                    # to echo — the request is perfectly valid as-is.
                    if "/" in provider_model:
                        _pp_prov, _pp_model = provider_model.split("/", 1)
                        if _requires_reasoning_echo(_pp_model, provider=_pp_prov):
                            _asst_msgs = [
                                m for m in passthrough_messages
                                if isinstance(m, dict) and m.get("role") == "assistant"
                            ]
                            _asst_count = len(_asst_msgs)
                            _rc_count = sum(1 for m in _asst_msgs if m.get("reasoning_content"))
                            # Only skip when there's a MIX: some have rc, some don't.
                            # If ALL have rc or NONE have rc, the request is consistent.
                            if _asst_count > 0 and 0 < _rc_count < _asst_count:
                                logger.warning(
                                    "[hermes-code] skipping %s: %d/%d assistant msgs have reasoning_content (mixed history, cannot satisfy echo requirement)",
                                    provider_model, _rc_count, _asst_count,
                                )
                                continue

                    # Fire pre_api_request hook for observability plugins (e.g. Langfuse).
                    _pt_call_count[0] += 1
                    _invoke_passthrough_hooks(
                        "pre_api_request",
                        task_id="", session_id=session_id or "", platform="api_server",
                        model=provider_model, provider=provider_model.split("/")[0],
                        base_url="", api_mode="",
                        api_call_count=_pt_call_count[0],
                        messages=passthrough_messages,
                        message_count=len(passthrough_messages),
                        tool_count=len(passthrough_tools) if passthrough_tools else 0,
                        approx_input_tokens=_approx_tokens,
                        max_tokens=16384,
                    )

                    if provider_model.startswith("github-copilot") or provider_model.startswith("copilot-"):
                        try:
                            runtime_kwargs, resolved_model = _runtime_kwargs_for_model_id(provider_model)
                            _copilot_provider = provider_model.split("/")[0]
                            api_key = runtime_kwargs.get("api_key", "")
                            base_url = runtime_kwargs.get("base_url", "") or None
                            api_mode = runtime_kwargs.get("api_mode", "anthropic_messages")
                            _copilot_timeout = _passthrough_request_timeout(_copilot_provider, resolved_model)

                            if not api_key:
                                logger.warning("[hermes-code] passthrough copilot %s: no API key, skipping", provider_model)
                                continue

                            _mapper = None  # no arliai sanitization needed for copilot

                            # ── Dispatch by API mode ──
                            if api_mode == "anthropic_messages":
                                # Anthropic Messages API (Claude models)
                                from agent.anthropic_adapter import build_anthropic_client
                                anthropic_client = build_anthropic_client(api_key, base_url, timeout=_copilot_timeout)
                                anthropic_messages, anthropic_tools = _transform_messages_to_anthropic(passthrough_messages, tools)

                                api_kwargs: Dict[str, Any] = {
                                    "model": resolved_model,
                                    "messages": anthropic_messages,
                                    "max_tokens": 16384,
                                }
                                if anthropic_tools:
                                    api_kwargs["tools"] = anthropic_tools

                                _s_loop = asyncio.get_running_loop()
                                response_obj = await _await_passthrough_provider_call(
                                    _s_loop.run_in_executor(
                                        None,
                                        lambda: anthropic_client.messages.create(**api_kwargs),
                                    ),
                                    provider_model,
                                )

                                # Parse Anthropic response
                                response_text = ""
                                content_out = ""
                                tool_calls_out = []
                                if hasattr(response_obj, 'content') and response_obj.content:
                                    for block in response_obj.content:
                                        if hasattr(block, 'text') and block.text:
                                            content_out = block.text
                                            response_text = content_out
                                        elif hasattr(block, 'type') and block.type == 'tool_use':
                                            tool_calls_out.append({
                                                "id": block.id,
                                                "type": "function",
                                                "function": {
                                                    "name": block.name,
                                                    "arguments": json.dumps(block.input)
                                                }
                                            })
                                tool_calls_out = _enrich_client_tool_calls(tool_calls_out, passthrough_tools)
                                usage_obj = getattr(response_obj, 'usage', None)
                                reasoning_content_out = None
                                finish_reason = "tool_calls" if tool_calls_out else "stop"

                            else:
                                # OpenAI-compatible API (chat_completions or codex_responses)
                                from openai import OpenAI
                                from hermes_cli.copilot_auth import copilot_request_headers

                                _s_loop = asyncio.get_running_loop()
                                # codex endpoints need the Cloudflare-safe originator/User-Agent
                                # (chatgpt.com whitelists only codex_cli_rs/codex_vscode/codex_sdk_ts).
                                if "chatgpt.com" in (base_url or "").lower() and api_mode == "codex_responses":
                                    from agent.auxiliary_client import _codex_cloudflare_headers
                                    headers = _codex_cloudflare_headers(api_key)
                                else:
                                    headers = copilot_request_headers(is_agent_turn=True, base_url=base_url)
                                client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers, max_retries=0)

                                if api_mode == "codex_responses":
                                    # Responses API (GPT-5.x): wrap in CodexAuxiliaryClient
                                    from agent.auxiliary_client import CodexAuxiliaryClient
                                    wrapped = CodexAuxiliaryClient(client, resolved_model)
                                    response_obj = await _await_passthrough_provider_call(
                                        _s_loop.run_in_executor(
                                            None,
                                            lambda: wrapped.chat.completions.create(
                                                messages=passthrough_messages,
                                                model=resolved_model,
                                                max_tokens=16384,
                                                tools=passthrough_tools,
                                                timeout=_copilot_timeout,
                                            ),
                                        ),
                                        provider_model,
                                    )
                                else:
                                    # Chat Completions API (GPT-5-mini, GPT-4o-mini, etc.)
                                    response_obj = await _await_passthrough_provider_call(
                                        _s_loop.run_in_executor(
                                            None,
                                            lambda: client.chat.completions.create(
                                                model=resolved_model,
                                                messages=_copilot_messages(passthrough_messages),
                                                max_tokens=16384,
                                                tools=passthrough_tools,
                                                timeout=_copilot_timeout,
                                            ),
                                        ),
                                        provider_model,
                                    )

                                # Parse OpenAI-style response
                                msg = response_obj.choices[0].message
                                content_out = extract_content_or_reasoning(response_obj).strip()
                                reasoning_content_out = _extract_reasoning_content_from_msg(msg)
                                tool_calls_raw = getattr(msg, "tool_calls", []) or []
                                tool_calls_out = []
                                for tc in tool_calls_raw:
                                    if hasattr(tc, "model_dump"):
                                        _tc_dict = tc.model_dump()
                                    elif hasattr(tc, "dict"):
                                        _tc_dict = tc.dict()
                                    elif isinstance(tc, dict):
                                        _tc_dict = tc
                                    else:
                                        _func = getattr(tc, "function", None)
                                        _tc_dict = {
                                            "id": str(getattr(tc, "id", "")),
                                            "type": "function",
                                            "function": {
                                                "name": str(getattr(_func, "name", getattr(tc, "name", ""))),
                                                "arguments": str(getattr(_func, "arguments", getattr(tc, "arguments", "{}"))),
                                            },
                                        }
                                    # Preserve extra_content (contains google.thought_signature for Gemini)
                                    _ec = getattr(tc, "extra_content", None) or (tc.get("extra_content") if isinstance(tc, dict) else None)
                                    if _ec:
                                        _tc_dict["extra_content"] = _ec
                                    # ── Google thought_signature: pack into call_id ────────
                                    # Standard OpenAI clients strip non-standard fields like
                                    # extra_content when re-sending history. Pack the signature
                                    # into the tool_call id as `<orig>:hermes_ts:<b64>` so it
                                    # survives the round-trip. The injection code unpacks it
                                    # back into extra_content on the next request.
                                    _google_ec = (_ec or {}).get("google") if isinstance(_ec, dict) else None
                                    _ts = _google_ec.get("thought_signature") if isinstance(_google_ec, dict) else None
                                    if _ts and isinstance(_tc_dict.get("id"), str) and ":hermes_ts:" not in _tc_dict["id"]:
                                        import base64
                                        _packed = base64.urlsafe_b64encode(_ts.encode("utf-8")).decode("ascii").rstrip("=")
                                        _packed_id = f"{_tc_dict['id']}:hermes_ts:{_packed}"
                                        _tc_dict["id"] = _packed_id
                                        _tc_dict["call_id"] = _packed_id
                                    tool_calls_out.append(_tc_dict)
                                tool_calls_out = _enrich_client_tool_calls(tool_calls_out, passthrough_tools)
                                if not tool_calls_out:
                                    _text_tool_calls, _cleaned_content = _extract_text_tool_calls_for_passthrough(content_out, passthrough_tools)
                                    if _text_tool_calls:
                                        logger.warning(
                                            "[hermes-code][req=%s] recovered %d text-formatted tool_call(s) from assistant content for %s",
                                            _req_id, len(_text_tool_calls), provider_model,
                                        )
                                        tool_calls_out = _text_tool_calls
                                        content_out = _cleaned_content if isinstance(_cleaned_content, str) else content_out
                                usage_obj = getattr(response_obj, "usage", None)
                                response_text = content_out
                                finish_reason = getattr(response_obj.choices[0], "finish_reason", "stop")
                                if tool_calls_out:
                                    finish_reason = "tool_calls"

                            # ── Common variables and success marking ──
                            _skip_provider_exhaustion_content(
                                provider_model=provider_model,
                                runtime_kwargs=runtime_kwargs,
                                content=content_out,
                                stream=True,
                            )
                            completion_id = _passthrough_sse_completion_id
                            created = _passthrough_sse_created
                            _args_preview = [
                                (tc.get("function", {}).get("name", ""),
                                 tc.get("function", {}).get("arguments", "")[:200])
                                for tc in tool_calls_out if isinstance(tc, dict)
                            ]
                            logger.warning(
                                "[hermes-code][req=%s] response: model=%s finish=%s tool_calls=%d args=%s content_len=%d",
                                _req_id,
                                provider_model,
                                finish_reason,
                                len(tool_calls_out),
                                _args_preview,
                                len(content_out) if content_out else 0,
                            )
                            if passthrough_tools and _has_empty_bash_tool_call(tool_calls_out):
                                logger.warning(
                                    "[hermes-code] %s returned bash with empty command — clearing tool_calls, returning text as-is (tools=%d, content_len=%d)",
                                    provider_model, len(passthrough_tools),
                                    len(content_out) if content_out else 0,
                                )
                                try:
                                    from agent.model_cooldown_db import mark_model_cooldown
                                    mark_model_cooldown(
                                        provider=provider_model.split("/")[0] if "/" in provider_model else "copilot",
                                        model=provider_model,
                                        cooldown_seconds=120.0,
                                        reason="empty_bash",
                                    )
                                except Exception:
                                    pass
                                try:
                                    from agent.model_quality_db import record_text_only
                                    record_text_only(provider_model.split("/")[0], provider_model, base_url=base_url or "")
                                except Exception:
                                    pass
                                tool_calls_out = []
                            if passthrough_tools and not tool_calls_out:
                                # Text-only response: return it to the client as-is.
                                # Previously this cascaded to the next provider via
                                # _CodexPassthroughSkip, but that burned through the
                                # entire fallback chain (4-150s each) only to return
                                # an error.  OMP/clients handle text-only responses
                                # fine — they can decide whether to re-prompt.
                                try:
                                    import pathlib
                                    _diag_file = pathlib.Path("/tmp/text_only_diag.log")
                                    _diag_file.parent.mkdir(parents=True, exist_ok=True)
                                    with open(_diag_file, "a") as _f:
                                        _f.write(f"--- {provider_model} (returned as-is) ---\n")
                                        _f.write(f"tools_sent: {len(passthrough_tools)}\n")
                                        _f.write(f"tool_names: {[t.get('function',{}).get('name','?') for t in passthrough_tools[:10]]}\n")
                                        _f.write(f"content: {(content_out or '(empty)')[:1000]}\n")
                                        _f.write(f"rc: {(reasoning_content_out or '(empty)')[:500]}\n")
                                        _f.write(f"last_role: {passthrough_messages[-1].get('role', '?') if passthrough_messages else '?'}\n")
                                        _f.write(f"msg_count: {len(passthrough_messages)}\n")
                                        _f.write(f"\n")
                                except Exception:
                                    pass
                                logger.warning(
                                    "[hermes-code] %s text-only (tools=%d, content_len=%d) — returning to client as-is",
                                    provider_model, len(passthrough_tools),
                                    len(content_out) if content_out else 0,
                                )
                                try:
                                    from agent.model_quality_db import record_text_only
                                    record_text_only(provider_model.split("/")[0], provider_model, base_url=base_url or "")
                                except Exception:
                                    pass
                            try:
                                from agent.model_cooldown_db import mark_provider_success
                                _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                                mark_provider_success(_cb_prov, provider_model, base_url=base_url or "")
                            except Exception:
                                pass
                            # Record quality metrics
                            try:
                                from agent.model_quality_db import record_success
                                record_success(_cb_prov, provider_model, base_url=base_url or "", latency_ms=0)
                            except Exception:
                                pass

                            if _mapper is not None and tool_calls_out:
                                tool_calls_out = _mapper.unsanitize_tool_calls(tool_calls_out)

                            # ── SSE serialisation (shared by all api_modes) ──
                            response = await _ensure_passthrough_sse_response()

                            # Stream reasoning_content deltas (OpenAI models only)
                            if reasoning_content_out:
                                for i in range(0, len(reasoning_content_out), 200):
                                    rc_chunk = reasoning_content_out[i:i+200]
                                    rc_chunk_data = {
                                        "id": completion_id, "object": "chat.completion.chunk",
                                        "created": created, "model": model_name,
                                        "choices": [{"index": 0, "delta": {"reasoning_content": rc_chunk}, "finish_reason": None}],
                                    }
                                    await response.write(f"data: {json.dumps(rc_chunk_data)}\n\n".encode())
                                    await asyncio.sleep(0.005)

                            if response_text:
                                for i in range(0, len(response_text), 100):
                                    chunk_text = response_text[i:i+100]
                                    text_chunk = {
                                        "id": completion_id, "object": "chat.completion.chunk",
                                        "created": created, "model": model_name,
                                        "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                                    }
                                    await response.write(f"data: {json.dumps(text_chunk)}\n\n".encode())
                                    await asyncio.sleep(0.01)

                            if tool_calls_out:
                                for i, tc in enumerate(tool_calls_out):
                                    tool_chunk = {
                                        "id": completion_id, "object": "chat.completion.chunk",
                                        "created": created, "model": model_name,
                                        "choices": [{"index": 0, "delta": {"tool_calls": [dict(tc, index=i)]}, "finish_reason": None}],
                                    }
                                    await response.write(f"data: {json.dumps(tool_chunk)}\n\n".encode())

                            # Normalise both Anthropic-style (input_tokens) and OpenAI-style (prompt_tokens) usage
                            _pt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
                            if not _pt:
                                _pt = int(getattr(usage_obj, "input_tokens", 0) or 0)
                            _ct = int(getattr(usage_obj, "completion_tokens", 0) or 0)
                            if not _ct:
                                _ct = int(getattr(usage_obj, "output_tokens", 0) or 0)
                            finish_chunk = {
                                "id": completion_id, "object": "chat.completion.chunk",
                                "created": created, "model": model_name,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                "usage": {
                                    "prompt_tokens": _pt,
                                    "completion_tokens": _ct,
                                    "total_tokens": _pt + _ct,
                                },
                            }
                            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
                            await response.write(b"data: [DONE]\n\n")
                            await response.write_eof()

                            if provided_session_id:
                                try:
                                    _persist_passthrough_session_delta(
                                        self._ensure_session_db(),
                                        session_id,
                                        model_name=model_name,
                                        system_prompt=system_prompt,
                                        request_messages=_request_conversation_messages,
                                        assistant_content=content_out,
                                        assistant_tool_calls=tool_calls_out,
                                        finish_reason=finish_reason,
                                        reasoning_content=reasoning_content_out,
                                    )
                                except Exception as _persist_exc:
                                    logger.warning("[api_server] failed to persist passthrough stream session delta for %s: %s", session_id, _persist_exc)
                            _invoke_passthrough_hooks(
                                "post_api_request",
                                task_id="", session_id=session_id or "", platform="api_server",
                                model=provider_model, provider=provider_model.split("/")[0],
                                base_url=base_url or "", api_mode=api_mode or "anthropic_messages",
                                api_call_count=_pt_call_count[0],
                                finish_reason=finish_reason,
                                assistant_content_chars=len(content_out) if content_out else 0,
                                assistant_tool_call_count=len(tool_calls_out),
                                usage={
                                    "input_tokens": _pt,
                                    "output_tokens": _ct,
                                },
                            )
                            return response

                        except Exception as exc:
                            # Client disconnect — check first so we don't penalise providers for OMP timeouts.
                            _exc_str = str(exc).lower()
                            _is_client_disconnect = (
                                isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError))
                                or "closing connection" in _exc_str or "closing transport" in _exc_str
                                or "connection reset" in _exc_str or "broken pipe" in _exc_str
                                or "client disconnected" in _exc_str
                            )
                            if _is_client_disconnect:
                                logger.info("[hermes-code] client disconnected mid-stream (copilot %s), not penalising provider", provider_model)
                                passthrough_error = exc
                                break
                            # Prompt too long — bail out of the chain. The client must
                            # compact its transcript; further attempts against larger-context
                            # models will fail the same way and just burn latency.
                            if _is_prompt_too_long_error(_exc_str):
                                logger.warning(
                                    "[hermes-code] prompt too long for %s (~%d tokens), stopping chain to avoid churn",
                                    provider_model, _approx_tokens,
                                )
                                passthrough_error = exc
                                break
                            # Don't penalise providers for context-overflow errors — these are
                            # routing/selection issues (resolved by the pre-filter above), not
                            # model or API failures that warrant circuit-breaking.
                            _is_ctx_overflow = _is_context_overflow_error(_exc_str) or _is_prompt_too_long_error(_exc_str)
                            if not _is_ctx_overflow:
                                try:
                                    from agent.model_cooldown_db import mark_provider_failure
                                    _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                                    mark_provider_failure(_cb_prov, provider_model, base_url=base_url or "", reason="passthrough_error")
                                except Exception:
                                    pass
                            # Record quality failure (unless context overflow — that's a routing issue)
                            if not _is_ctx_overflow:
                                try:
                                    from agent.model_quality_db import record_failure
                                    _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                                    record_failure(_cb_prov, provider_model, base_url=base_url or "", error_message=str(exc)[:200])
                                except Exception:
                                    pass
                            _invalidate_selectable_pool_cache()
                            # Check if this is a rate-limit (429) or auth (401) error; cooldown the provider.
                            _status_code = _exception_status_code(exc)
                            _is_rate_limit = _is_provider_exhaustion_error(exc)
                            _is_auth_error = _status_code == 401 or "401" in _exc_str or "unauthorized" in _exc_str or "authentication" in _exc_str
                            if _is_rate_limit:
                                _mark_hermes_code_provider_exhausted(
                                    provider_model=provider_model,
                                    runtime_kwargs=locals().get("runtime_kwargs"),
                                    exc=exc,
                                    stream=True,
                                )
                            elif _is_auth_error:
                                # Auth errors (401) — distinguish permanent from transient.
                                _is_token_invalidated = "token_invalidated" in _exc_str or "invalidated" in _exc_str
                                if _is_token_invalidated:
                                    # Permanent failure: token is dead and needs manual re-auth.
                                    # Retry in <24h won't help. Use 24h cooldown.
                                    try:
                                        from agent.model_cooldown_db import mark_model_cooldown
                                        mark_model_cooldown(
                                            provider=provider_model.split("/")[0] if "/" in provider_model else "copilot",
                                            model=provider_model,
                                            cooldown_seconds=86400.0,
                                            reason="hermes_code_stream_401_token_invalidated",
                                        )
                                        logger.warning("[hermes-code] stream %s cooled down for 24h after 401 (token invalidated — needs re-auth)", provider_model)
                                    except Exception:
                                        pass
                                else:
                                    # Transient auth error (e.g. rate limit on auth) — 120s cooldown.
                                    try:
                                        from agent.model_cooldown_db import mark_model_cooldown
                                        mark_model_cooldown(
                                            provider=provider_model.split("/")[0] if "/" in provider_model else "copilot",
                                            model=provider_model,
                                            cooldown_seconds=120.0,
                                            reason="hermes_code_stream_401",
                                        )
                                        logger.warning("[hermes-code] stream %s cooled down for 120s after 401", provider_model)
                                    except Exception:
                                        pass
                            elif _status_code == 400:
                                # 400 errors (bad request). Do not penalise prompt-too-long
                                # requests — those are routing/compaction issues, not provider
                                # health issues. Other 400s get a short cooldown.
                                if not _is_prompt_too_long_error(_exc_str):
                                    try:
                                        from agent.model_cooldown_db import mark_model_cooldown
                                        mark_model_cooldown(
                                            provider=provider_model.split("/")[0] if "/" in provider_model else "copilot",
                                            model=provider_model,
                                            cooldown_seconds=120.0,
                                            reason="hermes_code_stream_400",
                                        )
                                        logger.warning("[hermes-code] stream %s cooled down for 120s after 400", provider_model)
                                    except Exception:
                                        pass
                            elif _status_code == 500:
                                # 500 (server error) — provider is broken or degraded.
                                # Won't resolve in seconds. 5min cooldown.
                                try:
                                    from agent.model_cooldown_db import mark_model_cooldown
                                    mark_model_cooldown(
                                        provider=provider_model.split("/")[0] if "/" in provider_model else "copilot",
                                        model=provider_model,
                                        cooldown_seconds=300.0,
                                        reason="hermes_code_stream_500",
                                    )
                                    logger.warning("[hermes-code] stream %s cooled down for 5min after 500", provider_model)
                                except Exception:
                                    pass
                            logger.warning("[hermes-code] passthrough stream copilot %s failed: %s", provider_model, exc)
                            passthrough_error = exc
                            continue

                    # Non-copilot providers
                    try:
                        runtime_kwargs, resolved_model = _runtime_kwargs_for_model_id(provider_model)
                        prov = runtime_kwargs.get("provider", "")
                        api_key = runtime_kwargs.get("api_key", "")
                        base_url = runtime_kwargs.get("base_url", "") or None
                        _provider_timeout = _passthrough_request_timeout(prov, resolved_model)
                        _provider_messages = _messages_with_provider_tool_prompt(
                            passthrough_messages,
                            provider_model=provider_model,
                            provider=prov,
                            resolved_model=resolved_model,
                            tools=passthrough_tools,
                        )
                        if not api_key:
                            continue
                        _s_loop = asyncio.get_running_loop()
                        # Initialize tool_call_id mapper (always needed — referenced after dispatch)
                        _mapper = None
                        _raw_tool_calls = None
                        # Initialize _skip_normal_call so it's always defined regardless
                        # of which provider branch is taken (copilot/claude/gemini/else).
                        _skip_normal_call = False
                        # Initialize _msgs_to_send with a default provider-specific message list
                        # so it's always defined for the call_llm fallback path, even if
                        # the provider branch (openai-codex/claude-code-cli) doesn't set it.
                        _msgs_to_send = _provider_messages
                        # Initialize _acquired_stream for the same reason — the
                        # provider-specific branches (openai-codex/claude-code-cli) don't
                        # go through the parallel stream limiter, so this stays False.
                        _acquired_stream = False
                        if _needs_audio and prov == "google":
                            def _gemini_audio_call():
                                from agent.gemini_native_adapter import GeminiNativeClient
                                _gc = GeminiNativeClient(api_key=api_key, base_url=base_url)
                                return _gc._create_chat_completion(
                                    model=resolved_model,
                                    messages=_strip_reasoning(_provider_messages) if _passthrough_has_reasoning else _provider_messages,
                                    max_tokens=16384,
                                    tools=passthrough_tools,
                                    timeout=_provider_timeout,
                                )
                            response_obj = await _await_passthrough_provider_call(
                                _s_loop.run_in_executor(None, _gemini_audio_call),
                                provider_model,
                            )
                        elif prov == "openai-codex":
                            _skip_normal_call = True  # Uses _call_codex_passthrough, not call_llm
                            response_obj = await _await_passthrough_provider_call(
                                _s_loop.run_in_executor(
                                    None,
                                    lambda: _call_codex_passthrough(
                                        messages=_strip_unsupported_content_for_openai(
                                            _strip_reasoning(_provider_messages) if _passthrough_has_reasoning else _provider_messages
                                        ),
                                        model=resolved_model,
                                        api_key=api_key,
                                        base_url=base_url or "",
                                        tools=passthrough_tools,
                                        timeout=_provider_timeout,
                                    ),
                                ),
                                provider_model,
                            )
                        elif prov == "claude-code-cli":
                            # Claude Code CLI — MCP bridge mode.
                            # Uses run_with_tool_bridge generator to handle
                            # multi-turn tool calling via the MCP proxy.
                            logger.info("[hermes-code][req=%s] claude-code-cli bridge dispatch: prov=%s base_url=%s", _req_id, prov, base_url)
                            try:
                                from hermes_cli.auth import resolve_external_process_provider_credentials
                                _cc_creds = resolve_external_process_provider_credentials("claude-code-cli")
                                from agent.claude_code_client import ClaudeCodeClient
                                _cc_client = ClaudeCodeClient(
                                    api_key=_cc_creds.get("api_key", "claude-code-cli"),
                                    base_url=_cc_creds.get("base_url", "claude://codex"),
                                    command=_cc_creds.get("command"),
                                    args=_cc_creds.get("args"),
                                )
                            except Exception as _cc_exc:
                                logger.warning("[hermes-code][req=%s] claude-code-cli bridge credential resolution failed: %s", _req_id, _cc_exc)
                                _cc_client = None
                            if _cc_client is not None:
                                _skip_normal_call = True
                                logger.info("[hermes-code][req=%s] claude-code-cli bridge starting for model=%s", _req_id, resolved_model)
                                # ── Create SSE response early so bridge can stream directly ──
                                _bridge_completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
                                _bridge_sse_headers = {
                                    "Content-Type": "text/event-stream",
                                    "Cache-Control": "no-cache",
                                    "X-Accel-Buffering": "no",
                                }
                                if session_id:
                                    _bridge_sse_headers["X-Hermes-Session-Id"] = session_id
                                # Run the bridge generator in a thread, collect events via asyncio queue
                                _bridge_events: asyncio.Queue = asyncio.Queue()
                                _bridge_error: list[Exception] = []
                                def _run_bridge(_c=_cc_client, _m=resolved_model, _msgs=_provider_messages, _tools=passthrough_tools):
                                    try:
                                        gen = _c.run_with_tool_bridge(model=_m, messages=_msgs, tools=_tools)
                                        for event in gen:
                                            _bridge_events.put_nowait(event)
                                    except Exception as exc:
                                        _bridge_error.append(exc)
                                        _bridge_events.put_nowait({"type": "error", "message": str(exc)})
                                    finally:
                                        _bridge_events.put_nowait({"type": "_done"})
                                _bridge_thread = threading.Thread(target=_run_bridge, daemon=True)
                                _bridge_thread.start()
                                # Process events from the bridge — stream SSE chunks directly

                                _bridge_final_text = ""
                                _bridge_usage = {}
                                _bridge_model = resolved_model
                                _bridge_tool_calls: list[dict] = []
                                _bridge_error_msg: str | None = None  # track bridge error for content fallback
                                _bridge_resp_failed = False
                                _stream_response: web.StreamResponse | None = None
                                try:
                                    _stream_response = web.StreamResponse(status=200, headers=_bridge_sse_headers)
                                    await _stream_response.prepare(request)
                                except Exception as _resp_prep_exc:
                                    logger.warning(
                                        "[TIMING][req=%s][claude-stream] T+%.3fs claude-code-cli response.prepare failed: %s — falling back to plain response",
                                        _req_id, time.monotonic() - _req_start, _resp_prep_exc,
                                    )
                                    return web.json_response({
                                        "id": _bridge_completion_id,
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": _bridge_model,
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Stream preparation failed"}, "finish_reason": "error"}],
                                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                    })
                                while True:
                                    _be = await _bridge_events.get()
                                    if _be is None or _be.get("type") == "_done":
                                        break
                                    _bt = _be.get("type")
                                    if _bt == "tool_call":
                                        _call_id = _be.get("call_id", "")
                                        _tool_name = _be.get("name", "")
                                        # Strip MCP prefix on the way back to the client —
                                        # hermes is the boundary that translates between
                                        # client-friendly names and the CLI's mcp_-prefixed IDs.
                                        # claude-code uses mcp__hermes-tools__<name>;
                                        # mimocode uses mcp_<name>.
                                        if _tool_name.startswith("mcp__hermes-tools__"):
                                            _tool_name = _tool_name[len("mcp__hermes-tools__"):]
                                        elif _tool_name.startswith("mcp_"):
                                            _tool_name = _tool_name[len("mcp_"):]
                                        _tool_args = _be.get("arguments", {})
                                        _tc = {
                                            "id": _call_id,
                                            "type": "function",
                                            "function": {
                                                "name": _tool_name,
                                                "arguments": json.dumps(_tool_args),
                                            },
                                        }
                                        _bridge_tool_calls.append(_tc)
                                        _tc_chunk = {
                                            "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                            "created": int(time.time()), "model": model_name,
                                            "choices": [{"index": 0, "delta": {"tool_calls": [dict(_tc, index=0)]}, "finish_reason": None}],
                                        }
                                        # Stream the tool_call delta to the client in real time
                                        # (same pattern as the mimocode-cli streaming path).
                                        # Buffering until after the bridge finishes would deadlock
                                        # because the bridge itself is blocked waiting for the
                                        # client to POST the tool result back.
                                        try:
                                            await _stream_response.write(f"data: {json.dumps(_tc_chunk)}\n\n".encode())
                                        except Exception:
                                            _bridge_resp_failed = True
                                            _bridge_error_msg = "SSE write failed during tool_call"
                                            break
                                        # Emit a side-channel tool_call_request event so the
                                        # connected client (Mac OMP daemon) knows it needs to
                                        # execute the tool locally and POST the result back.
                                        # The hub registers a PendingCall keyed by
                                        # (session_id, call_id); the matching _handle_tool_responses
                                        # endpoint unblocks it via PendingCall.event.set().
                                        _tool_call_request_event = {
                                            "type": "tool_call_request",
                                            "call_id": _call_id,
                                            "name": _tool_name,
                                            "arguments": _tool_args,
                                        }
                                        _tcreq_chunk = {
                                            "id": _bridge_completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "data": _tool_call_request_event,
                                        }
                                        try:
                                            await _stream_response.write(f"data: {json.dumps(_tcreq_chunk)}\n\n".encode())
                                        except Exception:
                                            _bridge_resp_failed = True
                                            _bridge_error_msg = "SSE write failed during tool_call_request"
                                            break
                                        # Register with the hub for clients that support the
                                        # optional tool_call_request side-channel (e.g. the
                                        # OMP daemon polling /pending-tool-calls).
                                        # Do NOT block waiting for a hub result — the stream
                                        # must close immediately with finish_reason=tool_calls
                                        # so standard OpenAI clients receive the tool_calls
                                        # and can send a follow-up request with tool results.
                                        if session_id:
                                            try:
                                                from gateway.platforms import tool_call_hub
                                                tool_call_hub.register_call(
                                                    session_id, _call_id, tool_name=_tool_name,
                                                    arguments=_tool_args,
                                                )
                                                logger.info(
                                                    "[TIMING][req=%s][claude-stream] T+%.3fs registered tool_call_hub call_id=%s tool=%s",
                                                    _req_id, time.monotonic() - _req_start,
                                                    _call_id, _tool_name,
                                                )
                                            except Exception as _hub_exc:
                                                logger.warning("[hermes-code] bridge: tool_call_hub register failed: %s", _hub_exc)
                                        logger.info("[hermes-code] bridge: streamed tool_call %s (non-blocking)", _call_id)
                                    elif _bt in ("text", "assistant_text"):
                                        _text = _be.get("text", "")
                                        if _text:
                                            _text = _normalize_bridge_text(_text)
                                            _bridge_final_text += _text
                                            _text_chunk = {
                                                "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                                "created": int(time.time()), "model": model_name,
                                                "choices": [{"index": 0, "delta": {"content": _text}, "finish_reason": None}],
                                            }
                                            try:
                                                await _stream_response.write(f"data: {json.dumps(_text_chunk)}\n\n".encode())
                                            except Exception:
                                                _bridge_resp_failed = True
                                                _bridge_error_msg = "SSE write failed during text"
                                                break
                                    elif _bt == "final":
                                        _raw_final = _be.get("text", _bridge_final_text)
                                        _bridge_final_text = _normalize_bridge_text(_raw_final)
                                        _bridge_usage = _be.get("usage", {})
                                        _bridge_model = _be.get("model", _bridge_model)
                                    elif _bt == "error":
                                        _bridge_error_msg = _be.get("message", "")
                                        logger.warning("[hermes-code] claude-code-cli bridge error: %s", _bridge_error_msg)
                                # If the bridge reported an error and produced no real content, use the error
                                # message as the response text so downstream exhaustion/error checks catch it.
                                if _bridge_error_msg and not _bridge_final_text and not _bridge_tool_calls:
                                    _bridge_final_text = _bridge_error_msg
                                # Build response_obj and finalize SSE
                                _bridge_finish_reason = "tool_calls" if _bridge_tool_calls else "stop"
                                _bridge_usage_ns = SimpleNamespace(
                                    prompt_tokens=int(_bridge_usage.get("input_tokens", 0) or 0),
                                    completion_tokens=int(_bridge_usage.get("output_tokens", 0) or 0),
                                    total_tokens=int((_bridge_usage.get("input_tokens", 0) or 0) + (_bridge_usage.get("output_tokens", 0) or 0)),
                                )
                                _bridge_msg = SimpleNamespace(
                                    content=_bridge_final_text,
                                    tool_calls=_bridge_tool_calls if _bridge_tool_calls else None,
                                )
                                _bridge_choice = SimpleNamespace(
                                    message=_bridge_msg,
                                    finish_reason=_bridge_finish_reason,
                                )
                                response_obj = SimpleNamespace(
                                    choices=[_bridge_choice],
                                    usage=_bridge_usage_ns,
                                    model=_bridge_model,
                                )
                                if _bridge_final_text:
                                    logger.warning("[hermes-code] claude-code-cli bridge raw text: %.500s", _bridge_final_text)
                                _skip_provider_exhaustion_content(
                                    provider_model=provider_model,
                                    runtime_kwargs=runtime_kwargs,
                                    content=_bridge_final_text,
                                    stream=True,
                                )
                                # Write finish chunk and close the already-prepared SSE stream.
                                _bridge_finish_chunk = {
                                    "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": model_name,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": _bridge_finish_reason}],
                                    "usage": {
                                        "prompt_tokens": int(_bridge_usage_ns.prompt_tokens or 0),
                                        "completion_tokens": int(_bridge_usage_ns.completion_tokens or 0),
                                        "total_tokens": int(_bridge_usage_ns.total_tokens or 0),
                                    },
                                }
                                if not _bridge_resp_failed and _stream_response is not None:
                                    try:
                                        await _stream_response.write(f"data: {json.dumps(_bridge_finish_chunk)}\n\n".encode())
                                        await _stream_response.write(b"data: [DONE]\n\n")
                                        await _stream_response.write_eof()
                                    except Exception as _finish_write_exc:
                                        _bridge_resp_failed = True
                                        logger.warning("[hermes-code] claude-code-cli bridge: failed to write finish chunk: %s", _finish_write_exc)
                                logger.info("[hermes-code][req=%s] claude-code-cli bridge completed: text_len=%d tool_calls=%d", _req_id, len(_bridge_final_text), len(_bridge_tool_calls))
                                # Record success and hooks
                                try:
                                    from agent.model_cooldown_db import mark_provider_success
                                    mark_provider_success(prov, resolved_model, base_url=base_url or "")
                                except Exception:
                                    pass
                                try:
                                    from agent.model_quality_db import record_success
                                    record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                                except Exception:
                                    pass
                                _invoke_passthrough_hooks(
                                    "post_api_request",
                                    task_id="", session_id=session_id or "", platform="api_server",
                                    model=provider_model, provider=provider_model.split("/")[0],
                                    base_url=base_url or "", api_mode="",
                                    api_call_count=_pt_call_count[0],
                                    finish_reason=_bridge_finish_reason,
                                    assistant_content_chars=len(_bridge_final_text) if _bridge_final_text else 0,
                                    assistant_tool_call_count=len(_bridge_tool_calls),
                                    usage={
                                        "input_tokens": int(_bridge_usage_ns.prompt_tokens or 0),
                                        "output_tokens": int(_bridge_usage_ns.completion_tokens or 0),
                                    },
                                )
                                return _stream_response
                        elif prov == "mimocode-cli":
                            # MiMoCode CLI — simple mode.
                            # The mimo CLI handles tools internally via built-in
                            # tools (bash, read, etc.) — no MCP bridge needed.
                            logger.info("[hermes-code][req=%s] mimocode-cli dispatch: prov=%s base_url=%s tools=%d", _req_id, prov, base_url, len(passthrough_tools or []))
                            try:
                                from hermes_cli.auth import resolve_external_process_provider_credentials
                                _mc_creds = resolve_external_process_provider_credentials("mimocode-cli")
                                from agent.mimocode_code_client import MiMoCodeClient
                                _mc_client = MiMoCodeClient(
                                    api_key=_mc_creds.get("api_key", "mimocode-cli"),
                                    base_url=_mc_creds.get("base_url", "mimocode://codex"),
                                    command=_mc_creds.get("command"),
                                    args=_mc_creds.get("args"),
                                )
                            except Exception as _mc_exc:
                                logger.warning("[hermes-code][req=%s] mimocode-cli credential resolution failed: %s", _req_id, _mc_exc)
                                _mc_client = None
                            if _mc_client is not None:
                                _skip_normal_call = True
                                _bridge_completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
                                _bridge_sse_headers = {
                                    "Content-Type": "text/event-stream",
                                    "Cache-Control": "no-cache",
                                    "X-Accel-Buffering": "no",
                                }
                                if session_id:
                                    _bridge_sse_headers["X-Hermes-Session-Id"] = session_id
                                _bridge_events: asyncio.Queue = asyncio.Queue()
                                def _run_mc_bridge(_c=_mc_client, _m=resolved_model, _msgs=_provider_messages, _tools=passthrough_tools, _q=_bridge_events):
                                    try:
                                        gen = _c.stream_events(model=_m, messages=_msgs, tools=_tools)
                                        for event in gen:
                                            _q.put_nowait(event)
                                    except Exception as exc:
                                        _q.put_nowait({"type": "error", "message": str(exc)})
                                    finally:
                                        _q.put_nowait({"type": "_done"})
                                threading.Thread(target=_run_mc_bridge, daemon=True).start()
                                _bridge_final_text = ""
                                _bridge_usage = {}
                                _bridge_model = resolved_model
                                _bridge_tool_calls: list[dict] = []
                                _bridge_error_msg: str | None = None
                                _bridge_last_tool_result: str | None = None
                                _restart_count = 0
                                _MAX_RESTARTS = 3
                                _bridge_resp_failed = False
                                _stream_response: web.StreamResponse | None = None
                                try:
                                    _stream_response = web.StreamResponse(status=200, headers=_bridge_sse_headers)
                                    await _stream_response.prepare(request)
                                except Exception as _resp_prep_exc:
                                    logger.warning("[TIMING][req=%s][mimo-stream] T+%.3fs mimocode-cli response.prepare failed: %s — falling back to plain response",
                                        _req_id, time.monotonic() - _req_start, _resp_prep_exc)
                                    return web.json_response({
                                        "id": _bridge_completion_id,
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": _bridge_model,
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Stream preparation failed"}, "finish_reason": "error"}],
                                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                    })
                                while True:
                                    _be = await _bridge_events.get()
                                    if _be is None or _be.get("type") == "_done":
                                        break
                                    _bt = _be.get("type")
                                    if _bt == "tool_call":
                                        _call_id = _be.get("call_id", "")
                                        _tool_name = _be.get("name", "")
                                        # Strip MCP prefix on the way back to the client —
                                        # hermes is the boundary that translates between
                                        # client-friendly names and the CLI's mcp_-prefixed IDs.
                                        # claude-code uses mcp__hermes-tools__<name>;
                                        # mimocode uses mcp_<name>.
                                        if _tool_name.startswith("mcp__hermes-tools__"):
                                            _tool_name = _tool_name[len("mcp__hermes-tools__"):]
                                        elif _tool_name.startswith("mcp_"):
                                            _tool_name = _tool_name[len("mcp_"):]
                                        _tool_args = _be.get("arguments", {})
                                        _tc = {
                                            "id": _call_id,
                                            "type": "function",
                                            "function": {
                                                "name": _tool_name,
                                                "arguments": json.dumps(_tool_args),
                                            },
                                        }
                                        _bridge_tool_calls.append(_tc)
                                        _tc_chunk = {
                                            "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                            "created": int(time.time()), "model": model_name,
                                            "choices": [{"index": 0, "delta": {"tool_calls": [dict(_tc, index=0)]}, "finish_reason": None}],
                                        }
                                        try:
                                            await _stream_response.write(f"data: {json.dumps(_tc_chunk)}\n\n".encode())
                                        except Exception:
                                            _bridge_resp_failed = True
                                            _bridge_error_msg = "SSE write failed during tool_call"
                                            break
                                        # Emit side-channel tool_call_request so the connected
                                        # client (Mac OMP daemon) knows to execute the tool
                                        # locally and POST the result back via the hub.
                                        _tool_call_request_event = {
                                            "type": "tool_call_request",
                                            "call_id": _call_id,
                                            "name": _tool_name,
                                            "arguments": _tool_args,
                                        }
                                        _tcreq_chunk = {
                                            "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                            "created": int(time.time()), "model": model_name,
                                            "data": _tool_call_request_event,
                                        }
                                        try:
                                            await _stream_response.write(f"data: {json.dumps(_tcreq_chunk)}\n\n".encode())
                                        except Exception:
                                            _bridge_resp_failed = True
                                            _bridge_error_msg = "SSE write failed during tool_call_request"
                                            break
                                        # Register with the hub for clients that support the
                                        # optional tool_call_request side-channel.
                                        # Do NOT block waiting for a hub result — the stream
                                        # must close immediately with finish_reason=tool_calls
                                        # so standard OpenAI clients receive the tool_calls
                                        # and can send a follow-up request with tool results.
                                        if session_id:
                                            try:
                                                from gateway.platforms import tool_call_hub
                                                tool_call_hub.register_call(
                                                    session_id, _call_id, tool_name=_tool_name,
                                                    arguments=_tool_args,
                                                )
                                                logger.info(
                                                    "[TIMING][req=%s][mimo-stream] T+%.3fs registered tool_call_hub call_id=%s tool=%s",
                                                    _req_id, time.monotonic() - _req_start,
                                                    _call_id, _tool_name,
                                                )
                                            except Exception as _hub_exc:
                                                logger.warning("[hermes-code] mimocode-cli bridge: tool_call_hub register failed: %s", _hub_exc)
                                    elif _bt in ("text", "assistant_text"):
                                        _chunk_text = _be.get("text", "")
                                        _bridge_final_text += _chunk_text
                                        _tc_chunk = {
                                            "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                            "created": int(time.time()), "model": model_name,
                                            "choices": [{"index": 0, "delta": {"content": _chunk_text}, "finish_reason": None}],
                                        }
                                        try:
                                            await _stream_response.write(f"data: {json.dumps(_tc_chunk)}\n\n".encode())
                                        except Exception:
                                            _bridge_resp_failed = True
                                            _bridge_error_msg = "SSE write failed during text"
                                            break
                                    elif _bt == "error":
                                        _bridge_error_msg = _be.get("message", "")
                                        logger.warning("[hermes-code] mimocode-cli bridge error: %s", _bridge_error_msg)
                                    elif _bt == "final":
                                        _bridge_usage = _be.get("usage", {})
                                        _bridge_model = _be.get("model", resolved_model)
                                    # If we got a tool result back but the model never
                                    # produced a final text response, fall back to
                                    # using the tool result itself as the final text.
                                    # This matches OpenAI's chat completions flow where
                                    # the assistant often returns tool_calls without
                                    # content; the client (Mac daemon / OpenAI
                                    # library) is expected to POST tool results back
                                    # in a follow-up turn to get a final answer.
                                    # Since hermes handles the tool execution in a
                                    # side channel (not via the client), we synthesize
                                    # a final assistant turn that incorporates the
                                    # tool result content.
                                    if (
                                        _bt == "final"
                                        and _bridge_tool_calls
                                        and not _bridge_final_text
                                        and _bridge_last_tool_result
                                    ):
                                        _tool_summary = _bridge_last_tool_result
                                        if not isinstance(_tool_summary, str):
                                            _tool_summary = json.dumps(_tool_summary)
                                        # Truncate
                                        if len(_tool_summary) > 4000:
                                            _tool_summary = _tool_summary[:4000] + "\n... [truncated]"
                                        _bridge_final_text = (
                                            f"Tool execution result:\n```\n{_tool_summary}\n```"
                                        )
                                        logger.info(
                                            "[TIMING][req=%s][mimo-stream] T+%.3fs synthesized final text from tool result (len=%d)",
                                            _req_id, time.monotonic() - _req_start, len(_bridge_final_text),
                                        )
                                        # Emit as a final text chunk
                                        _synth_chunk = {
                                            "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                            "created": int(time.time()), "model": model_name,
                                            "choices": [{"index": 0, "delta": {"content": _bridge_final_text}, "finish_reason": None}],
                                        }
                                        try:
                                            await _stream_response.write(f"data: {json.dumps(_synth_chunk)}\n\n".encode())
                                        except Exception:
                                            _bridge_resp_failed = True
                                            _bridge_error_msg = "SSE write failed during synth chunk"
                                            break
                                        # Don't restart — the synthesized text IS the answer
                                        _bridge_finish_reason = "stop"
                                        break
                                    # === MULTI-TURN RESTART ===
                                    # mimo is a single-shot CLI — once it emits a tool_call
                                    # and gets the result, the subprocess often exits
                                    # without the model producing a final answer. We
                                    # detect that here and start a new mimo session
                                    # with the conversation extended to include the
                                    # tool_call + tool_result, so the model can produce
                                    # a final response. The SSE stays open the whole time.
                                    if (
                                        _bt == "final"
                                        and _bridge_tool_calls
                                        and not _bridge_final_text
                                        and passthrough_messages
                                        and _restart_count < _MAX_RESTARTS
                                    ):
                                        _restart_count += 1
                                        try:
                                            logger.info(
                                                "[TIMING][req=%s][mimo-stream] T+%.3fs starting multi-turn restart tool_calls=%d",
                                                _req_id, time.monotonic() - _req_start, len(_bridge_tool_calls),
                                            )
                                            # Build extended conversation:
                                            # original messages + assistant tool_calls + tool results
                                            _ext_messages = list(passthrough_messages)
                                            # Append assistant message with the tool_calls
                                            _asst_msg = {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": _bridge_tool_calls,
                                            }
                                            _ext_messages.append(_asst_msg)
                                            # Append role=tool messages for each tool_call
                                            # (the tool result was already written to the queue
                                            # but the subprocess exited before the model saw it)
                                            for _tc in _bridge_tool_calls:
                                                _tc_id = _tc.get("id", "")
                                                # Use the most recent tool result payload (one tool call per round-trip typically)
                                                _tool_content = _tool_result_payload.get("content", "")
                                                _ext_messages.append({
                                                    "role": "tool",
                                                    "tool_call_id": _tc_id,
                                                    "content": _tool_content if isinstance(_tool_content, str) else json.dumps(_tool_content),
                                                })
                                            # Append a USER message that tells the model
                                            # the tool result has been provided. mimo's
                                            # _build_prompt() only takes the LAST user
                                            # message, so we must include both the
                                            # original prompt and the tool result inline.
                                            _tool_summary = _tool_result_payload.get("content", "")
                                            if not isinstance(_tool_summary, str):
                                                _tool_summary = json.dumps(_tool_summary)
                                            _tool_summary_short = (
                                                _tool_summary[:4000] + "\n... [truncated]"
                                                if len(_tool_summary) > 4000
                                                else _tool_summary
                                            )
                                            # Find the original user prompt
                                            _original_user_msg = ""
                                            for _orig_msg in reversed(passthrough_messages):
                                                if _orig_msg.get("role") in ("user", "developer"):
                                                    _c = _orig_msg.get("content", "")
                                                    if isinstance(_c, str):
                                                        _original_user_msg = _c
                                                    elif isinstance(_c, list):
                                                        _original_user_msg = " ".join(
                                                            p.get("text", "") for p in _c
                                                            if isinstance(p, dict) and p.get("type") == "text"
                                                        )
                                                    break
                                            _ext_messages.append({
                                                "role": "user",
                                                "content": (
                                                    f"Original request:\n\n{_original_user_msg}\n\n"
                                                    f"---\n\n"
                                                    f"The previous tool returned:\n\n```\n{_tool_summary_short}\n```\n\n"
                                                    "Based on this result, please answer my "
                                                    "original request directly in plain text. "
                                                    "Do not call any more tools."
                                                ),
                                            })
                                            # Start new mimo session with extended conversation
                                            # Use stream_events() instead of _create_chat_completion()
                                            # so tool_calls go through the hub round-trip.
                                            def _run_mc_restart(_c=_mc_client, _m=resolved_model, _ext=_ext_messages, _q=_bridge_events):
                                                try:
                                                    for event in _c.stream_events(model=_m, messages=_ext, tools=passthrough_tools):
                                                        _q.put_nowait(event)
                                                except Exception as exc:
                                                    _q.put_nowait({"type": "error", "message": f"restart error: {exc}"})
                                                finally:
                                                    _q.put_nowait({"type": "_done"})
                                            threading.Thread(target=_run_mc_restart, daemon=True).start()
                                            logger.info(
                                                "[TIMING][req=%s][mimo-stream] T+%.3fs multi-turn restart thread spawned",
                                                _req_id, time.monotonic() - _req_start,
                                            )
                                            # DON'T break — let the main while loop continue
                                            # to drain events from the new thread
                                            _bridge_tool_calls = []  # reset to avoid emitting tool_calls chunk twice
                                            continue
                                        except Exception as _restart_exc:
                                            logger.warning("[hermes-code] mimocode-cli multi-turn restart failed: %s", _restart_exc)
                                _bridge_finish_reason = "tool_calls" if _bridge_tool_calls else "stop"
                                try:
                                    _bridge_usage_ns = SimpleNamespace(
                                        prompt_tokens=_bridge_usage.get("input_tokens", 0),
                                        completion_tokens=_bridge_usage.get("output_tokens", 0),
                                        total_tokens=_bridge_usage.get("total_tokens", 0),
                                    )
                                except Exception as _usage_exc:
                                    logger.warning("[hermes-code] mimocode-cli bridge: _bridge_usage_ns failed: %s usage=%r", _usage_exc, _bridge_usage)
                                    _bridge_usage_ns = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                                # If the bridge reported an error and produced no real content, use the error
                                # message as the response text so downstream exhaustion/error checks catch it.
                                if _bridge_error_msg and not _bridge_final_text and not _bridge_tool_calls:
                                    # If the CLI is busy (concurrency limit), cascade to the next
                                    # provider instead of returning the error text to the client.
                                    _busy_hint = _bridge_error_msg.lower()
                                    if "busy" in _busy_hint or "another instance running" in _busy_hint:
                                        logger.info("[hermes-code] mimocode-cli busy — cascading to next provider")
                                        raise _CodexPassthroughSkip("mimocode_cli_busy")
                                    _bridge_final_text = _bridge_error_msg
                                logger.info("[TIMING][req=%s][mimo-stream] T+%.3fs bridge loop exit — building SSE response text_len=%d tool_calls=%d",
                                    _req_id, time.monotonic() - _req_start, len(_bridge_final_text), len(_bridge_tool_calls))
                                _bridge_finish_chunk = {
                                    "id": _bridge_completion_id, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": _bridge_model,
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": _bridge_finish_reason}],
                                    "usage": {
                                        "prompt_tokens": int(_bridge_usage_ns.prompt_tokens or 0),
                                        "completion_tokens": int(_bridge_usage_ns.completion_tokens or 0),
                                        "total_tokens": int(_bridge_usage_ns.total_tokens or 0),
                                    },
                                }
                                _skip_provider_exhaustion_content(
                                    provider_model=provider_model,
                                    runtime_kwargs=runtime_kwargs,
                                    content=_bridge_final_text,
                                    stream=True,
                                )
                                try:
                                    if _bridge_resp_failed:
                                        raise ConnectionError("SSE stream previously failed")
                                    await _stream_response.write(f"data: {json.dumps(_bridge_finish_chunk)}\n\n".encode())
                                    await _stream_response.write(b"data: [DONE]\n\n")
                                    await _stream_response.write_eof()
                                    response = _stream_response
                                except Exception as _resp_exc:
                                    logger.warning("[TIMING][req=%s][mimo-stream] T+%.3fs mimocode-cli response.write failed: %s — falling back to plain response",
                                        _req_id, time.monotonic() - _req_start, _resp_exc)
                                    # Fall back to a plain JSON response (mimics what
                                    # the 503 path does, but with the synthesized
                                    # content rather than an error).
                                    return web.json_response({
                                        "id": _bridge_completion_id,
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": _bridge_model,
                                        "choices": [{
                                            "index": 0,
                                            "message": {"role": "assistant", "content": _bridge_final_text},
                                            "finish_reason": _bridge_finish_reason,
                                        }],
                                        "usage": {
                                            "prompt_tokens": int(_bridge_usage_ns.prompt_tokens or 0),
                                            "completion_tokens": int(_bridge_usage_ns.completion_tokens or 0),
                                            "total_tokens": int(_bridge_usage_ns.total_tokens or 0),
                                        },
                                    })
                                logger.info("[TIMING][req=%s][mimo-stream] T+%.3fs mimocode-cli completed: text_len=%d tool_calls=%d — RETURNING response",
                                    _req_id, time.monotonic() - _req_start, len(_bridge_final_text), len(_bridge_tool_calls))
                                try:
                                    from agent.model_cooldown_db import mark_provider_success
                                    mark_provider_success(prov, resolved_model, base_url=base_url or "")
                                except Exception:
                                    pass
                                try:
                                    from agent.model_quality_db import record_success
                                    record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                                except Exception:
                                    pass
                                return response
                        else:
                            _echo_rc = _passthrough_has_reasoning and _requires_reasoning_echo(resolved_model, provider=prov, base_url=base_url)
                            _msgs_to_send = (_provider_messages if _echo_rc else _strip_reasoning(_provider_messages)) if _passthrough_has_reasoning else _provider_messages
                            _msgs_to_send = _strip_unsupported_content_for_openai(_msgs_to_send)
                            # For providers that require reasoning_content echo (DeepSeek via
                            # opencode-zen/opencode-go), ensure every assistant turn that
                            # has tool_calls also has a reasoning_content field — even an
                            # empty one. The provider rejects the request if the field is
                            # missing on any assistant turn that emitted tool_calls in
                            # thinking mode. This must run AFTER _strip_reasoning so the
                            # synthesised field isn't removed by stripping.
                            if _requires_reasoning_echo(resolved_model, provider=prov, base_url=base_url):
                                _msgs_to_send = _synthesize_reasoning_for_tool_calls(_msgs_to_send)

                            # ── Packed tool_call_id cleanup ────────────────────────
                            # Hermes packs Google thought_signature into tool_call ids
                            # as ``<orig_id>:hermes_ts:<b64>`` so it survives the round-trip
                            # through clients that strip non-standard fields.  These packed
                            # IDs are garbage to every other provider, so we strip them for
                            # all providers; for Google specifically, we also extract the
                            # base64 signature and inject it into extra_content so the
                            # Gemini API accepts the request.
                            if "generativelanguage.googleapis.com" in (base_url or ""):
                                # 1) Strip any :hermes_ts: packed ids and restore signatures
                                _injected, _unpacked = _unpack_hermes_ts_and_inject_signatures(_msgs_to_send)
                                # 2) Inject sentinel thought_signature on unsigned tool_calls
                                #    (e.g. those originating from non-Google providers, or
                                #    whose extra_content was stripped on the round-trip).
                                _sentinel = _inject_google_sentinel_signatures(_msgs_to_send)
                                if _injected or _unpacked or _sentinel:
                                    logger.warning(
                                        "[hermes-code] Google thought_signature: injected=%d unpacked=%d sentinel=%d into %d messages for %s",
                                        _injected, _unpacked, _sentinel, len(_msgs_to_send), resolved_model,
                                    )
                            else:
                                _ts_fixed = _strip_hermes_ts_packed_ids(_msgs_to_send)
                                if _ts_fixed:
                                    logger.debug("[hermes-code] stripped %d hermes_ts packed ids for %s", _ts_fixed, provider_model)
                            # ── arliai tool_call_id sanitization ────────────────────
                            # arliai enforces ≤9-char tool_call_ids. Use a bidirectional
                            # mapper so the client sees the original IDs while upstream
                            # receives sanitized ones.
                            if base_url and "arliai" in (base_url or "").lower():
                                try:
                                    from agent._tool_id_sanitizer import ToolCallIdMapper
                                    _mapper = ToolCallIdMapper(max_length=9)
                                    _msgs_to_send = _mapper.sanitize_messages(_msgs_to_send)
                                    logger.debug("[hermes-code] arliai: sanitized tool_call_ids in %d messages", len(_msgs_to_send))
                                except Exception as _map_exc:
                                    logger.warning("[hermes-code] arliai: failed to init tool_id mapper: %s", _map_exc)

                            # ── cerebras call_id stripping ───────────────────────────
                            # cerebras does not support the `call_id` field in tool_calls.
                            # Strip it from all assistant tool_call messages before sending.
                            if prov == "cerebras":
                                _cc_removed = _strip_call_id_from_tool_calls(_msgs_to_send)
                                if _cc_removed:
                                    logger.debug("[hermes-code] cerebras: stripped call_id from %d tool_calls", _cc_removed)

                            logger.debug(
                                "[hermes-code] streaming call_llm: model=%s provider=%s has_rc=%s echo=%s msgs=%d",
                                resolved_model, prov, _passthrough_has_reasoning, _echo_rc, len(_msgs_to_send),
                            )
                            _skip_normal_call = False
                            if _passthrough_has_reasoning and _echo_rc:
                                _rc_in_msgs = [len(m.get("reasoning_content") or "") for m in _msgs_to_send if m.get("reasoning_content")]
                                logger.debug("[hermes-code] streaming call_llm: rc lengths in msgs=%s", _rc_in_msgs)
                                # Deep trace: show role+rc for each message going to DeepSeek provider
                                if prov and ("opencode" in prov.lower() or "deepseek" in resolved_model.lower()):
                                    _trace = [(m.get("role"), len(m.get("reasoning_content") or ""), len(m.get("content") or "")) for m in _msgs_to_send if m.get("role") in ("user", "assistant")]
                                    logger.debug("[hermes-code] DEEP TRACE → provider=%s model=%s msgs_sample=%s", prov, resolved_model, _trace[:5])
                                # Log what we're actually passing (just the assistant messages with rc)
                                _rc_assistants = [m for m in _msgs_to_send if m.get("role") == "assistant" and m.get("reasoning_content")]
                                if _rc_assistants:
                                    logger.debug("[hermes-code] PRE-CALL: sending %d assistant msgs with reasoning_content to provider=%s model=%s",
                                                   len(_rc_assistants), prov, resolved_model)
                                # For DeepSeek models via opencode-zen/opencode-go, make a direct HTTP call
                                # with full hex-dump logging to debug the exact request/response
                                if "deepseek" in resolved_model.lower() and prov in ("opencode-zen", "opencode-go"):
                                    try:
                                        import httpx
                                        _req_body = {
                                            "model": resolved_model,
                                            "messages": _msgs_to_send,
                                            "max_tokens": 16384,
                                            "stream": False,
                                        }
                                        if passthrough_tools:
                                            _req_body["tools"] = passthrough_tools
                                        _json_body = json.dumps(_req_body, ensure_ascii=False)
                                        _req_bytes = _json_body.encode("utf-8")
                                        logger.warning(
                                            "[hermes-code] DEEPHEX REQUEST provider=%s model=%s size=%d bytes",
                                            prov, resolved_model, len(_req_bytes),
                                        )
                                        # Log first 2000 chars of request body as text
                                        _text_preview = _json_body[:2000]
                                        logger.warning("[hermes-code] DEEPHEX REQUEST BODY START:\n%s", _text_preview)
                                        # Log summary of each message
                                        _msg_summary = [
                                            (m.get("role","?"), len(str(m.get("content",""))), bool(m.get("reasoning_content")), bool(m.get("tool_calls")))
                                            for m in _msgs_to_send
                                        ]
                                        logger.warning("[hermes-code] DEEPHEX MSG SUMMARY: %s", _msg_summary)
                                        # Log count of reasoning_content entries in request
                                        _rc_count = sum(1 for m in _msgs_to_send if m.get("reasoning_content"))
                                        logger.warning("[hermes-code] DEEPHEX: %d messages with reasoning_content in request", _rc_count)
                                        # Make direct httpx call
                                        async with httpx.AsyncClient(timeout=_provider_timeout) as _client:
                                            _headers = {
                                                "Authorization": f"Bearer {api_key}",
                                                "Content-Type": "application/json",
                                                "User-Agent": "python-httpx/0.28.1",
                                            }
                                            _resp = await _client.post(
                                                f"{base_url}/chat/completions",
                                                headers=_headers,
                                                content=_req_bytes,
                                            )
                                            _resp_body = _resp.read()
                                            logger.warning(
                                                "[hermes-code] DEEPHEX RESPONSE status=%d size=%d bytes",
                                                _resp.status_code, len(_resp_body),
                                            )
                                            _hex_resp = _resp_body[:1000].hex(" ", 1)
                                            logger.debug("[hermes-code] DEEPHEX RESPONSE HEX:\n%s", _hex_resp)
                                            if _resp.status_code != 200:
                                                _resp_text = _resp_body.decode("utf-8", errors="replace")
                                                logger.warning("[hermes-code] DEEPHEX ERROR BODY: %s", _resp_text[:2000])
                                                raise Exception(f"DeepSeek direct call failed: {_resp.status_code}")
                                            # Parse and wrap response like call_llm would
                                            import types
                                            _result_data = json.loads(_resp_body)
                                            # Build a mock response object that looks like OpenAI SDK response
                                            _mock_resp = types.SimpleNamespace()
                                            _mock_resp.choices = [types.SimpleNamespace()]
                                            _msg_data = _result_data["choices"][0]["message"]
                                            _mock_resp.choices[0].message = types.SimpleNamespace()
                                            _mock_resp.choices[0].message.content = _msg_data.get("content", "") or ""
                                            _mock_resp.choices[0].message.reasoning_content = _msg_data.get("reasoning_content") or ""
                                            _mock_resp.choices[0].message.tool_calls = None
                                            _mock_resp.choices[0].finish_reason = _result_data["choices"][0].get("finish_reason", "stop")
                                            _mock_resp.usage = types.SimpleNamespace()
                                            _u = _result_data.get("usage", {})
                                            _mock_resp.usage.prompt_tokens = _u.get("prompt_tokens", 0)
                                            _mock_resp.usage.completion_tokens = _u.get("completion_tokens", 0)
                                            _mock_resp.usage.total_tokens = _u.get("total_tokens", 0)
                                            response_obj = _mock_resp
                                            logger.debug("[hermes-code] DEEPHEX: SUCCESS via direct httpx, rc_len=%d", len(_mock_resp.choices[0].message.reasoning_content))
                                            # Skip the normal call_llm path
                                            _skip_normal_call = True
                                    except Exception as _dex:
                                        logger.debug("[hermes-code] DEEPHEX: direct call failed: %s, falling through to call_llm", _dex)
                                        _skip_normal_call = False
                        # ── Enforce parallel stream limit ─────────────────────────
                        _acquired_stream = False  # Defined before try so it's always accessible in exception handlers
                        _stream_start = _time.time()
                        logger.info("[hermes-code][req=%s] stream: attempting provider=%s model=%s base_url=%s", _req_id, prov, resolved_model, base_url)
                        try:
                            from agent.provider_parallel_limiter import acquire_stream, release_stream
                            logger.debug("[hermes-code] stream: acquiring parallel slot for %s (wait=30s)", prov)
                            _stream_acquired = acquire_stream(prov, wait=True, timeout=30.0)
                            _stream_acquire_time = _time.time() - _stream_start
                            logger.info("[hermes-code][req=%s] stream: acquire_stream=%s for %s in %.1fs", _req_id, _stream_acquired, prov, _stream_acquire_time)
                            if _stream_acquired:
                                _acquired_stream = True
                            else:
                                logger.warning(
                                    "[hermes-code] cannot acquire parallel stream slot for provider=%s model=%s, skipping",
                                    prov, resolved_model,
                                )
                                passthrough_error = Exception(f"{provider_model}: concurrent stream limit reached")
                                continue
                        except Exception as _stream_exc:
                            logger.warning("[hermes-code] stream: acquire_stream exception for %s: %s", prov, _stream_exc)
                        if not _skip_normal_call:
                            _call_start = _time.time()
                            _call_timeout = _provider_timeout
                            logger.info("[hermes-code][req=%s] stream: calling call_llm for %s timeout=%ss", _req_id, prov, _call_timeout)
                            try:
                                response_obj = await _await_passthrough_provider_call(
                                    _s_loop.run_in_executor(
                                        None,
                                        lambda: call_llm(
                                            task="chat",
                                            messages=_msgs_to_send,
                                            provider=prov,
                                            model=resolved_model,
                                            base_url=base_url,
                                            api_key=api_key,
                                            max_tokens=16384,
                                            timeout=_call_timeout,
                                            tools=passthrough_tools,
                                        ),
                                    ),
                                    provider_model,
                                )
                                _call_duration = _time.time() - _call_start
                                _total_duration = _time.time() - _stream_start
                                logger.info("[hermes-code][req=%s] stream: SUCCESS %s in %.1fs (total=%.1fs)", _req_id, prov, _call_duration, _total_duration)
                            except Exception as _call_exc:
                                _call_duration = _time.time() - _call_start
                                _total_duration = _time.time() - _stream_start
                                logger.warning("[hermes-code][req=%s] stream: FAILED %s after %.1fs (total=%.1fs): %s", _req_id, prov, _call_duration, _total_duration, _call_exc)
                                # Store latency on exception so outer handler can penalise slow providers
                                _call_exc._hermes_latency_ms = _call_duration * 1000
                                raise _call_exc
                        try:
                            from agent.model_cooldown_db import mark_provider_success
                            mark_provider_success(prov, resolved_model, base_url=base_url or "")
                        except Exception:
                            pass
                        try:
                            from agent.model_quality_db import record_success
                            record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                        except Exception:
                            pass
                        # Release parallel stream slot on success
                        if _acquired_stream:
                            try:
                                from agent.provider_parallel_limiter import release_stream
                                release_stream(prov)
                            except Exception:
                                pass

                        msg = response_obj.choices[0].message
                        content_out = extract_content_or_reasoning(response_obj).strip()
                        reasoning_content_out = _extract_reasoning_content_from_msg(msg)
                        _skip_provider_exhaustion_content(
                            provider_model=provider_model,
                            runtime_kwargs=runtime_kwargs,
                            content=content_out,
                            stream=True,
                        )
                        tool_calls_raw = getattr(msg, "tool_calls", []) or []
                        # Serialize tool_calls (may be Pydantic models from some providers)
                        tool_calls_out = []
                        for tc in tool_calls_raw:
                            if hasattr(tc, "model_dump"):
                                tool_calls_out.append(tc.model_dump())
                            elif hasattr(tc, "dict"):
                                tool_calls_out.append(tc.dict())
                            elif isinstance(tc, dict):
                                tool_calls_out.append(tc)
                            else:
                                _func = getattr(tc, "function", None)
                                _func_name = str(getattr(_func, "name", getattr(tc, "name", "")))
                                _func_args = str(getattr(_func, "arguments", getattr(tc, "arguments", "{}")))
                                tool_calls_out.append({"id": str(getattr(tc, "id", "")), "type": "function", "function": {"name": _func_name, "arguments": _func_args}})
                        tool_calls_out = _enrich_client_tool_calls(tool_calls_out, passthrough_tools)

                        # Restore original tool_call_ids for arliai responses.
                        if _mapper is not None and tool_calls_out:
                            tool_calls_out = _mapper.unsanitize_tool_calls(tool_calls_out)
                        if not tool_calls_out:
                            _text_tool_calls, _cleaned_content = _extract_text_tool_calls_for_passthrough(content_out, passthrough_tools)
                            if _text_tool_calls:
                                logger.warning(
                                    "[hermes-code][req=%s] recovered %d text-formatted tool_call(s) from assistant content for %s",
                                    _req_id, len(_text_tool_calls), provider_model,
                                )
                                tool_calls_out = _text_tool_calls
                                content_out = _cleaned_content if isinstance(_cleaned_content, str) else content_out

                        _bad_bash_summary = _invalid_bash_tool_call_summary(tool_calls_out)
                        if passthrough_tools and _bad_bash_summary and not _skip_normal_call:
                            logger.warning(
                                "[hermes-code][req=%s] %s returned invalid bash tool call; retrying once with corrective prompt: %s",
                                _req_id, provider_model, _bad_bash_summary,
                            )
                            _retry_msgs_to_send = _messages_with_retry_tool_prompt(_msgs_to_send, tool_calls_out)
                            _retry_start = _time.time()
                            response_obj = await _await_passthrough_provider_call(
                                _s_loop.run_in_executor(
                                    None,
                                    lambda: call_llm(
                                        task="chat",
                                        messages=_retry_msgs_to_send,
                                        provider=prov,
                                        model=resolved_model,
                                        base_url=base_url,
                                        api_key=api_key,
                                        max_tokens=16384,
                                        timeout=_call_timeout,
                                        tools=passthrough_tools,
                                    ),
                                ),
                                provider_model,
                            )
                            logger.info(
                                "[hermes-code][req=%s] stream: corrective retry completed for %s in %.1fs",
                                _req_id, provider_model, _time.time() - _retry_start,
                            )
                            msg = response_obj.choices[0].message
                            content_out = extract_content_or_reasoning(response_obj).strip()
                            reasoning_content_out = _extract_reasoning_content_from_msg(msg)
                            tool_calls_raw = getattr(msg, "tool_calls", []) or []
                            tool_calls_out = []
                            for tc in tool_calls_raw:
                                if hasattr(tc, "model_dump"):
                                    tool_calls_out.append(tc.model_dump())
                                elif hasattr(tc, "dict"):
                                    tool_calls_out.append(tc.dict())
                                elif isinstance(tc, dict):
                                    tool_calls_out.append(tc)
                                else:
                                    _func = getattr(tc, "function", None)
                                    _func_name = str(getattr(_func, "name", getattr(tc, "name", "")))
                                    _func_args = str(getattr(_func, "arguments", getattr(tc, "arguments", "{}")))
                                    tool_calls_out.append({"id": str(getattr(tc, "id", "")), "type": "function", "function": {"name": _func_name, "arguments": _func_args}})
                            tool_calls_out = _enrich_client_tool_calls(tool_calls_out, passthrough_tools)
                            if _mapper is not None and tool_calls_out:
                                tool_calls_out = _mapper.unsanitize_tool_calls(tool_calls_out)
                            if not tool_calls_out:
                                _text_tool_calls, _cleaned_content = _extract_text_tool_calls_for_passthrough(content_out, passthrough_tools)
                                if _text_tool_calls:
                                    logger.warning(
                                        "[hermes-code][req=%s] recovered %d text-formatted tool_call(s) from corrective retry content for %s",
                                        _req_id, len(_text_tool_calls), provider_model,
                                    )
                                    tool_calls_out = _text_tool_calls
                                    content_out = _cleaned_content if isinstance(_cleaned_content, str) else content_out

                        # If any provider returned no tool calls (or empty bash commands)
                        # despite having tools, skip to next. Apply a short cooldown so
                        # a model that repeatedly returns text-only (when tools are
                        # expected) doesn't keep burning through the entire fallback
                        # chain on every request. The cooldown is short enough that
                        # the model will be retried after ~2 min — long enough to
                        # skip past it on the current request and avoid the worst
                        # case where 30+ models each return text-only in sequence.
                        if passthrough_tools and (not tool_calls_out or _has_empty_bash_tool_call(tool_calls_out)):
                            # claude-code-cli and mimocode-cli are agents, not model APIs.
                            # The CLI itself executes tools (Bash, Read, etc.) and returns
                            # the final answer as text.  Don't penalise it for text-only —
                            # the "tool calls" happened inside the subprocess.
                            _is_claude_code = provider_model.startswith("claude-code-cli")
                            _is_mimocode = provider_model.startswith("mimocode-cli")
                            if not _is_claude_code and not _is_mimocode and not tool_calls_out:
                                # Text-only: return as-is to client (streaming SSE path).
                                # Previously cascaded via _CodexPassthroughSkip but that
                                # burned through the entire fallback chain only to return
                                # an error.  OMP/clients handle text-only fine.
                                logger.warning(
                                    "[hermes-code] %s text-only (tools=%d, content_len=%d) — returning to client as-is",
                                    provider_model, len(passthrough_tools),
                                    len(content_out) if content_out else 0,
                                )
                                try:
                                    from agent.model_cooldown_db import mark_model_cooldown
                                    mark_model_cooldown(
                                        provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                        model=provider_model,
                                        cooldown_seconds=120.0,
                                        reason="text_only_with_tools",
                                    )
                                except Exception:
                                    pass
                                try:
                                    from agent.model_quality_db import record_text_only
                                    record_text_only(provider_model.split("/")[0], provider_model, base_url=base_url or "")
                                except Exception:
                                    pass
                            elif _is_claude_code:
                                logger.info(
                                    "[hermes-code] claude-code-cli: text-only response accepted (CLI executes tools internally)",
                                    provider_model,
                                )
                            else:
                                # Empty bash tool call: clear it and return text as-is.
                                # Previously cascaded via _CodexPassthroughSkip which burned
                                # through the entire fallback chain.  The model returned
                                # content alongside the empty tool call — return the content.
                                logger.warning(
                                    "[hermes-code] %s returned bash with empty command — clearing tool_calls, returning text as-is (tools=%d, content_len=%d)",
                                    provider_model, len(passthrough_tools),
                                    len(content_out) if content_out else 0,
                                )
                                try:
                                    from agent.model_cooldown_db import mark_model_cooldown
                                    mark_model_cooldown(
                                        provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                        model=provider_model,
                                        cooldown_seconds=120.0,
                                        reason="empty_bash",
                                    )
                                except Exception:
                                    pass
                                try:
                                    from agent.model_quality_db import record_text_only
                                    record_text_only(provider_model.split("/")[0], provider_model, base_url=base_url or "")
                                except Exception:
                                    pass
                                tool_calls_out = []

                        # log tool_calls for debugging continuation issues
                        _raw_finish = getattr(response_obj.choices[0], "finish_reason", "stop")
                        finish_reason = "tool_calls" if tool_calls_out else _raw_finish
                        _args_preview = [
                            (tc.get("function", {}).get("name", ""),
                             tc.get("function", {}).get("arguments", "")[:200])
                            for tc in tool_calls_out if isinstance(tc, dict)
                        ]
                        logger.warning(
                            "[hermes-code][req=%s] response: model=%s finish=%s tool_calls=%d args=%s content_len=%d",
                            _req_id,
                            provider_model,
                            finish_reason,
                            len(tool_calls_out),
                            _args_preview,
                            len(content_out) if content_out else 0,
                        )
                        if passthrough_tools and not tool_calls_out:
                            _rc_snippet = reasoning_content_out[:300] if reasoning_content_out else ""
                            logger.warning(
                                "[hermes-code] DIAGNOSTIC %s text-only: tools=%d tool_calls=%d content_len=%d rc_len=%d content=%.200s rc=%.100s",
                                provider_model, len(passthrough_tools), len(tool_calls_out),
                                len(content_out) if content_out else 0,
                                len(reasoning_content_out) if reasoning_content_out else 0,
                                content_out[:200] if content_out else "(empty)",
                                _rc_snippet[:100],
                            )
                        try:
                            from agent.model_quality_db import record_success
                            _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                            record_success(_cb_prov, provider_model, base_url=base_url or "", latency_ms=0)
                        except Exception:
                            pass

                        completion_id = _passthrough_sse_completion_id
                        created = _passthrough_sse_created
                        response = await _ensure_passthrough_sse_response()

                        # Stream reasoning_content deltas first (for DeepSeek thinking mode etc.)
                        if reasoning_content_out:
                            logger.debug(
                                "[hermes-code] streaming reasoning_content to client: model=%s rc_len=%d",
                                provider_model, len(reasoning_content_out),
                            )
                            for i in range(0, len(reasoning_content_out), 200):
                                rc_chunk = reasoning_content_out[i:i+200]
                                rc_chunk_data = {
                                    "id": completion_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model_name,
                                    "choices": [{"index": 0, "delta": {"reasoning_content": rc_chunk}, "finish_reason": None}],
                                }
                                await response.write(f"data: {json.dumps(rc_chunk_data)}\n\n".encode())
                                await asyncio.sleep(0.005)
                        else:
                            logger.warning(
                                "[hermes-code] no reasoning_content to stream: model=%s content_len=%d",
                                provider_model, len(content_out) if content_out else 0,
                            )

                        if content_out:
                            for i in range(0, len(content_out), 100):
                                chunk_text = content_out[i:i+100]
                                text_chunk = {
                                    "id": completion_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model_name,
                                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                                }
                                await response.write(f"data: {json.dumps(text_chunk)}\n\n".encode())
                                await asyncio.sleep(0.01)

                        if tool_calls_out:
                            for i, tc in enumerate(tool_calls_out):
                                tool_chunk = {
                                    "id": completion_id, "object": "chat.completion.chunk",
                                    "created": created, "model": model_name,
                                    "choices": [{"index": 0, "delta": {"tool_calls": [dict(tc, index=i)]}, "finish_reason": None}],
                                }
                                await response.write(f"data: {json.dumps(tool_chunk)}\n\n".encode())

                        usage_obj = getattr(response_obj, "usage", None)
                        # Record GHE AIU spend and enforce monthly limit.
                        try:
                            from agent.copilot_spend_db import record_and_check
                            record_and_check(response_obj, provider_model=provider_model, base_url=base_url or "")
                        except Exception as _spend_exc:
                            logger.warning("[copilot_spend] record failed: %s", _spend_exc)
                        finish_chunk = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                            "usage": {
                                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                                "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                            },
                        }
                        await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
                        await response.write(b"data: [DONE]\n\n")
                        await response.write_eof()
                        if provided_session_id:
                            try:
                                _persist_passthrough_session_delta(
                                    self._ensure_session_db(),
                                    session_id,
                                    model_name=model_name,
                                    system_prompt=system_prompt,
                                    request_messages=_request_conversation_messages,
                                    assistant_content=content_out,
                                    assistant_tool_calls=tool_calls_out,
                                    finish_reason=finish_reason,
                                    reasoning_content=reasoning_content_out,
                                )
                            except Exception as _persist_exc:
                                logger.warning("[api_server] failed to persist passthrough stream session delta for %s: %s", session_id, _persist_exc)
                        _invoke_passthrough_hooks(
                            "post_api_request",
                            task_id="", session_id=session_id or "", platform="api_server",
                            model=provider_model, provider=provider_model.split("/")[0],
                            base_url=base_url or "", api_mode="",
                            api_call_count=_pt_call_count[0],
                            finish_reason=finish_reason,
                            assistant_content_chars=len(content_out) if content_out else 0,
                            assistant_tool_call_count=len(tool_calls_out),
                            usage={
                                "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                                "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                            },
                        )
                        return response

                    except _CodexPassthroughSkip as _skip_exc:
                        logger.warning(
                            "[hermes-code] %s skipped via _CodexPassthroughSkip, "
                            "trying next provider",
                            provider_model,
                        )
                        # Release parallel stream slot on skip
                        if _acquired_stream:
                            try:
                                from agent.provider_parallel_limiter import release_stream
                                release_stream(prov)
                            except Exception:
                                pass
                        passthrough_error = _skip_exc
                        continue

                    except Exception as exc:
                        # Client disconnect — check first so we don't penalise providers for OMP timeouts.
                        _exc_str = str(exc).lower()
                        _is_client_disconnect = (
                            isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError))
                            or "closing connection" in _exc_str or "closing transport" in _exc_str
                            or "connection reset" in _exc_str or "broken pipe" in _exc_str
                            or "client disconnected" in _exc_str
                        )
                        if _is_client_disconnect:
                            logger.info("[hermes-code] client disconnected mid-stream (%s), not penalising provider", provider_model)
                            passthrough_error = exc
                            # Release parallel stream slot on client disconnect
                            if _acquired_stream:
                                try:
                                    from agent.provider_parallel_limiter import release_stream
                                    release_stream(prov)
                                except Exception:
                                    pass
                            break
                        # Prompt too long — bail out of the chain. The client must
                        # compact its transcript; further attempts against larger-context
                        # models will fail the same way and just burn latency.
                        if _is_prompt_too_long_error(_exc_str):
                            logger.warning(
                                "[hermes-code] prompt too long for %s (~%d tokens), stopping chain",
                                provider_model, _approx_tokens,
                            )
                            passthrough_error = exc
                            if _acquired_stream:
                                try:
                                    from agent.provider_parallel_limiter import release_stream
                                    release_stream(prov)
                                except Exception:
                                    pass
                            break
                        # Release parallel stream slot on provider error
                        if _acquired_stream:
                            try:
                                from agent.provider_parallel_limiter import release_stream
                                release_stream(prov)
                            except Exception:
                                pass
                        # Don't penalise providers for context-overflow errors — these are
                        # routing/selection issues (resolved by the pre-filter above), not
                        # model or API failures that warrant circuit-breaking.
                        _is_ctx_overflow = _is_context_overflow_error(_exc_str)
                        _is_cli_busy = "busy" in _exc_str or "another instance running" in _exc_str
                        if not _is_ctx_overflow and not _is_cli_busy:
                            try:
                                from agent.model_cooldown_db import mark_provider_failure
                                _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "openai"
                                mark_provider_failure(_cb_prov, provider_model, base_url=base_url or "", reason="passthrough_error")
                            except Exception:
                                pass
                        # Record quality failure (unless context overflow — routing issue)
                        if not _is_ctx_overflow:
                            try:
                                from agent.model_quality_db import record_failure
                                _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "openai"
                                _latency_ms = getattr(exc, "_hermes_latency_ms", 0.0)
                                record_failure(_cb_prov, provider_model, base_url=base_url or "", latency_ms=_latency_ms, error_message=str(exc)[:200])
                            except Exception:
                                pass
                        _invalidate_selectable_pool_cache()
                        # Check if this is a rate-limit (429) or auth (401) error; cooldown the provider.
                        _status_code = _exception_status_code(exc)
                        _is_rate_limit = _is_provider_exhaustion_error(exc)
                        _is_auth_error = _status_code == 401 or "401" in _exc_str or "unauthorized" in _exc_str or "authentication" in _exc_str
                        if _is_rate_limit:
                            _mark_hermes_code_provider_exhausted(
                                provider_model=provider_model,
                                runtime_kwargs=runtime_kwargs,
                                exc=exc,
                                stream=True,
                            )
                        elif _status_code == 400:
                            # 400 errors (bad request). Skip cooldown for prompt-too-long
                            # errors — those are routing/compaction issues, not provider
                            # health. Other 400s get a short cooldown.
                            if not _is_prompt_too_long_error(_exc_str):
                                try:
                                    from agent.model_cooldown_db import mark_model_cooldown
                                    mark_model_cooldown(
                                        provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                        model=provider_model,
                                        cooldown_seconds=120.0,
                                        reason="hermes_code_stream_400",
                                    )
                                    logger.warning("[hermes-code] stream %s cooled down for 120s after 400", provider_model)
                                except Exception:
                                    pass
                        elif _is_auth_error:
                            # For pool-backed credentials (openai-codex), skip the model-level
                            # cooldown — the credential pool's mark_exhausted_and_rotate() below
                            # handles per-credential exhaustion and rotation.  A model-level
                            # cooldown would block ALL accounts for this model for 300s even
                            # when other credentials in the pool are perfectly valid.
                            _cpool = runtime_kwargs.get("credential_pool") if isinstance(runtime_kwargs, dict) else None
                            if _cpool is None:
                                try:
                                    from agent.model_cooldown_db import mark_model_cooldown
                                    mark_model_cooldown(
                                        provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                        model=provider_model,
                                        cooldown_seconds=120.0,
                                        reason="hermes_code_stream_401",
                                    )
                                    logger.warning("[hermes-code] stream %s cooled down for 120s after 401", provider_model)
                                except Exception:
                                    pass
                            # For pool-backed credentials (openai-codex), rotate to the next
                            # entry so subsequent requests don't reuse the invalidated token.
                            if _cpool is not None:
                                try:
                                    _cpool.mark_exhausted_and_rotate(
                                        status_code=401,
                                        error_context={"reason": "token_invalidated", "message": str(exc)[:200]},
                                    )
                                    logger.warning("[hermes-code] stream %s: rotated credential pool after 401", provider_model)
                                except Exception as _pool_exc:
                                    logger.warning("[hermes-code] stream %s: credential pool rotate failed: %s", provider_model, _pool_exc)
                                # Clear the cached runtime kwargs so the next
                                # passthrough attempt re-resolves credentials
                                # from the rotated pool instead of reusing the
                                # stale cached api_key (24h TTL otherwise).
                                _cb_prov = provider_model.split("/")[0] if "/" in provider_model else ""
                                if _cb_prov:
                                    _RUNTIME_KWARGS_CACHE.pop(_cb_prov, None)
                                    _RUNTIME_KWARGS_CACHE_AT.pop(_cb_prov, None)
                                    logger.info("[hermes-code] stream %s: cleared runtime kwargs cache for %s after 401", provider_model, _cb_prov)
                        logger.warning("[hermes-code] passthrough stream %s failed: %s", provider_model, exc)
                        passthrough_error = exc
                        logger.debug("[%d] !! stream provider error %s: %s", _req_id, type(exc).__name__, _exc_str[:200])
                        continue

                _err_msg = _sanitize_passthrough_error_for_client(passthrough_error) if passthrough_error else "all passthrough providers failed"
                if not _err_msg:
                    _err_msg = f"all providers failed ({type(passthrough_error).__name__})"
                    logger.warning(
                        "[hermes-code] passthrough stream exhausted: last error type=%s has empty message, "
                        "indicates silent provider failure — check provider logs above",
                        type(passthrough_error).__name__,
                    )
                else:
                    logger.warning("[hermes-code] passthrough stream exhausted providers: %s", _err_msg)
                if _passthrough_sse_response is not None:
                    return await _finish_passthrough_sse_error(
                        f"hermes-code passthrough exhausted all configured providers: {_err_msg}"
                    )
                return web.json_response(
                    _openai_error(
                        f"hermes-code passthrough exhausted all configured providers: {_err_msg}",
                        err_type="server_error",
                        code="passthrough_exhausted",
                    ),
                    status=503,
                )

            logger.debug("[%d] streaming passthrough exhausted all providers, trying non-streaming", _req_id)
            # Non-streaming passthrough
            _pt_call_count[0] = 0  # reset counter for non-streaming loop
            logger.warning("[hermes-code][req=%s] NS-ENTRY tools=%d tool_names=%s",
                _req_id,
                len(passthrough_tools) if passthrough_tools else 0,
                [t.get("function", {}).get("name", "?") for t in (passthrough_tools or [])[:5]],
            )
            for provider_model in _passthrough_models:
                if "/" not in provider_model and provider_model not in ("claude-code-cli", "mimocode-cli"):
                    continue

                # Check cooldown before attempting this provider.
                # Skip this check for external CLI providers — their cooldowns
                # are about external subprocess/auth state, not about the
                # passthrough fallback chain, and blocking them here causes
                # explicit "claude-code-cli" / "mimocode-cli" requests to fail
                # with "no passthrough providers available" (503).
                _is_external_cli = (
                    provider_model == "claude-code-cli"
                    or provider_model.startswith("claude-code-cli/")
                    or provider_model == "mimocode-cli"
                    or provider_model.startswith("mimocode-cli/")
                )
                _prov_prefix = provider_model.split("/")[0] if "/" in provider_model else ""
                if _prov_prefix and not _is_external_cli:
                    try:
                        from agent.model_cooldown_db import model_cooldown_remaining
                        _cooldown_key = provider_model
                        _remaining = model_cooldown_remaining(_prov_prefix, _cooldown_key)
                        if _remaining and _remaining > 0:
                            logger.warning("[hermes-code] %s in cooldown (%.0fs remaining), skipping", _cooldown_key, _remaining)
                            continue
                    except Exception:
                        pass

                # ── Max provider attempts cap (non-streaming) ──
                _provider_attempt_count[0] += 1
                if _max_provider_attempts > 0 and _provider_attempt_count[0] > _max_provider_attempts:
                    logger.warning(
                        "[hermes-code][req=%s] NS max provider attempts (%d) reached after %s, "
                        "stopping cascade",
                        _req_id, _max_provider_attempts, provider_model,
                    )
                    break

                # Google Gemini 3.1+ requires thought_signature on every
                # functionCall part.  The actual injection of a sentinel
                # ``extra_content.google.thought_signature`` value happens
                # later in the message-building step — here we just log
                # for observability.
                if _prov_prefix == "google" and passthrough_messages:
                    _unsigned = _count_unsigned_google_tool_calls(passthrough_messages)
                    if _unsigned > 0:
                        logger.debug(
                            "[hermes-code] ns %s: %d assistant tool_call(s) lack thought_signature; will inject sentinel",
                            provider_model, _unsigned,
                        )

                # Skip models whose context window cannot safely hold the estimated
                # request tokens. This prevents costly "prompt too long" round-trips
                # that waste API quota and trigger circuit breakers unnecessarily.
                if _approx_tokens > 0 and not _model_can_handle_context(provider_model, _approx_tokens):
                    _ctx_limit = _model_context_length(provider_model)
                    logger.warning(
                        "[hermes-code] ns-skip %s: context too small for ~%d tokens "
                        "(limit=%s), skipping",
                        provider_model, _approx_tokens,
                        f"{_ctx_limit:,}" if _ctx_limit > 0 else "unknown",
                    )
                    continue

                # Skip models that require reasoning_content echo ONLY when the
                # conversation has a MIXED history (some assistant messages have
                # reasoning_content, some don't).
                if "/" in provider_model:
                    _ns_prov, _ns_model = provider_model.split("/", 1)
                    if _requires_reasoning_echo(_ns_model, provider=_ns_prov):
                        _ns_asst_msgs = [
                            m for m in passthrough_messages
                            if isinstance(m, dict) and m.get("role") == "assistant"
                        ]
                        _ns_asst_count = len(_ns_asst_msgs)
                        _ns_rc_count = sum(1 for m in _ns_asst_msgs if m.get("reasoning_content"))
                        if _ns_asst_count > 0 and 0 < _ns_rc_count < _ns_asst_count:
                            logger.warning(
                                "[hermes-code] ns-skip %s: %d/%d assistant msgs have reasoning_content (mixed history, cannot satisfy echo requirement)",
                                provider_model, _ns_rc_count, _ns_asst_count,
                            )
                            continue

                    # Fire pre_api_request hook for observability plugins (e.g. Langfuse).
                    _pt_call_count[0] += 1
                    _invoke_passthrough_hooks(
                        "pre_api_request",
                        task_id="", session_id=session_id or "", platform="api_server",
                        model=provider_model, provider=provider_model.split("/")[0],
                        base_url="", api_mode="",
                        api_call_count=_pt_call_count[0],
                        messages=passthrough_messages,
                        message_count=len(passthrough_messages),
                        tool_count=len(passthrough_tools) if passthrough_tools else 0,
                        approx_input_tokens=_approx_tokens,
                        max_tokens=16384,
                    )

                if provider_model.startswith("github-copilot") or provider_model.startswith("copilot-"):
                    try:
                        runtime_kwargs, resolved_model = _runtime_kwargs_for_model_id(provider_model)
                        api_key = runtime_kwargs.get("api_key", "")
                        base_url = runtime_kwargs.get("base_url", "") or None
                        api_mode = runtime_kwargs.get("api_mode", "anthropic_messages")
                        _copilot_provider = provider_model.split("/")[0]
                        _copilot_timeout = _passthrough_request_timeout(_copilot_provider, resolved_model)

                        if not api_key:
                            logger.debug("hermes-code passthrough: %s has no API key, skipping", provider_model)
                            continue

                        # ── Dispatch by API mode ──
                        if api_mode == "anthropic_messages":
                            # Anthropic Messages API (Claude models)
                            from agent.anthropic_adapter import build_anthropic_client
                            anthropic_client = build_anthropic_client(api_key, base_url, timeout=_copilot_timeout)
                            anthropic_messages, anthropic_tools = _transform_messages_to_anthropic(passthrough_messages, tools)

                            api_kwargs: Dict[str, Any] = {
                                "model": resolved_model,
                                "messages": anthropic_messages,
                                "max_tokens": 16384,
                            }
                            if anthropic_tools:
                                api_kwargs["tools"] = anthropic_tools

                            _ns_loop = asyncio.get_running_loop()
                            response_obj = await _ns_loop.run_in_executor(
                                None,
                                lambda: anthropic_client.messages.create(**api_kwargs),
                            )

                            # Parse Anthropic response
                            response_text = ""
                            tool_calls = []
                            finish_reason = "stop"
                            if hasattr(response_obj, 'content') and response_obj.content:
                                for block in response_obj.content:
                                    if hasattr(block, 'text') and block.text:
                                        response_text = block.text
                                    elif hasattr(block, 'type') and block.type == 'tool_use':
                                        tool_calls.append({
                                            "id": block.id,
                                            "type": "function",
                                            "function": {
                                                "name": block.name,
                                                "arguments": json.dumps(block.input)
                                            }
                                        })
                            if hasattr(response_obj, 'stop_reason'):
                                if response_obj.stop_reason == 'tool_use':
                                    finish_reason = "tool_calls"
                                else:
                                    finish_reason = response_obj.stop_reason or "stop"
                            tool_calls = _enrich_client_tool_calls(tool_calls, passthrough_tools)
                            usage_obj = getattr(response_obj, 'usage', None)
                            reasoning_content = None

                        else:
                            # OpenAI-compatible API (chat_completions or codex_responses)
                            from openai import OpenAI
                            from hermes_cli.copilot_auth import copilot_request_headers

                            _ns_loop = asyncio.get_running_loop()
                            # codex endpoints need the Cloudflare-safe originator/User-Agent
                            # (chatgpt.com whitelists only codex_cli_rs/codex_vscode/codex_sdk_ts).
                            # The generic copilot headers trigger cf_chl_opt challenges.
                            if "chatgpt.com" in (base_url or "").lower() and api_mode == "codex_responses":
                                from agent.auxiliary_client import _codex_cloudflare_headers
                                headers = _codex_cloudflare_headers(api_key)
                            else:
                                headers = copilot_request_headers(is_agent_turn=True, base_url=base_url)
                            client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers, max_retries=0)

                            if api_mode == "codex_responses":
                                # Responses API (GPT-5.x): wrap in CodexAuxiliaryClient
                                from agent.auxiliary_client import CodexAuxiliaryClient
                                wrapped = CodexAuxiliaryClient(client, resolved_model)
                                response_obj = await _ns_loop.run_in_executor(
                                    None,
                                    lambda: wrapped.chat.completions.create(
                                        messages=passthrough_messages,
                                        model=resolved_model,
                                        max_tokens=16384,
                                        tools=passthrough_tools,
                                        timeout=_copilot_timeout,
                                    ),
                                )
                            else:
                                # Chat Completions API (GPT-5-mini, GPT-4o-mini, etc.)
                                response_obj = await _ns_loop.run_in_executor(
                                    None,
                                    lambda: client.chat.completions.create(
                                        model=resolved_model,
                                        messages=passthrough_messages,
                                        max_tokens=16384,
                                        tools=passthrough_tools,
                                        timeout=_copilot_timeout,
                                    ),
                                )

                            # Parse OpenAI-style response
                            msg = response_obj.choices[0].message
                            response_text = extract_content_or_reasoning(response_obj).strip()
                            reasoning_content = _extract_reasoning_content_from_msg(msg)
                            tool_calls_raw = getattr(msg, "tool_calls", []) or []
                            tool_calls = []
                            for tc in tool_calls_raw:
                                if hasattr(tc, "model_dump"):
                                    tool_calls.append(tc.model_dump())
                                elif hasattr(tc, "dict"):
                                    tool_calls.append(tc.dict())
                                elif isinstance(tc, dict):
                                    tool_calls.append(tc)
                                else:
                                    _func = getattr(tc, "function", None)
                                    tool_calls.append({
                                        "id": str(getattr(tc, "id", "")),
                                        "type": "function",
                                        "function": {
                                            "name": str(getattr(_func, "name", getattr(tc, "name", ""))),
                                            "arguments": str(getattr(_func, "arguments", getattr(tc, "arguments", "{}"))),
                                        },
                                    })
                            tool_calls = _enrich_client_tool_calls(tool_calls, passthrough_tools)
                            usage_obj = getattr(response_obj, "usage", None)
                            finish_reason = getattr(response_obj.choices[0], "finish_reason", "stop")
                            if tool_calls:
                                finish_reason = "tool_calls"

                        # ── Common success marking + JSON response ──
                        try:
                            from agent.model_cooldown_db import mark_provider_success
                            _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                            mark_provider_success(_cb_prov, provider_model, base_url=base_url or "")
                        except Exception:
                            pass
                        try:
                            from agent.model_quality_db import record_success
                            _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                            record_success(_cb_prov, provider_model, base_url=base_url or "", latency_ms=0)
                        except Exception:
                            pass

                        assistant_msg: Dict[str, Any] = {"role": "assistant"}
                        _skip_provider_exhaustion_content(
                            provider_model=provider_model,
                            runtime_kwargs=locals().get("runtime_kwargs"),
                            content=response_text,
                            stream=False,
                        )
                        if tool_calls:
                            assistant_msg["tool_calls"] = tool_calls
                        if response_text:
                            assistant_msg["content"] = response_text

                        # Record GHE AIU spend and enforce monthly limit.
                        try:
                            from agent.copilot_spend_db import record_and_check
                            record_and_check(response_obj, provider_model=provider_model, base_url=base_url or "")
                        except Exception as _spend_exc:
                            logger.warning("[copilot_spend] record failed: %s", _spend_exc)

                        # Normalise both Anthropic-style (input_tokens) and OpenAI-style (prompt_tokens) usage
                        _pt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
                        if not _pt:
                            _pt = int(getattr(usage_obj, "input_tokens", 0) or 0)
                        _ct = int(getattr(usage_obj, "completion_tokens", 0) or 0)
                        if not _ct:
                            _ct = int(getattr(usage_obj, "output_tokens", 0) or 0)

                        response_data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "message": assistant_msg,
                                "finish_reason": finish_reason,
                            }],
                            "usage": {
                                "prompt_tokens": _pt,
                                "completion_tokens": _ct,
                                "total_tokens": _pt + _ct,
                            },
                        }
                        if provided_session_id:
                            try:
                                _persist_passthrough_session_delta(
                                    self._ensure_session_db(),
                                    session_id,
                                    model_name=model_name,
                                    system_prompt=system_prompt,
                                    request_messages=_request_conversation_messages,
                                    assistant_content=response_text,
                                    assistant_tool_calls=tool_calls,
                                    finish_reason=finish_reason,
                                    reasoning_content=reasoning_content,
                                )
                            except Exception as _persist_exc:
                                logger.warning("[api_server] failed to persist passthrough session delta for %s: %s", session_id, _persist_exc)
                        headers = {"X-Hermes-Session-Id": session_id}
                        _invoke_passthrough_hooks(
                            "post_api_request",
                            task_id="", session_id=session_id or "", platform="api_server",
                            model=provider_model, provider=provider_model.split("/")[0],
                            base_url=base_url or "", api_mode=api_mode or "anthropic_messages",
                            api_call_count=_pt_call_count[0],
                            finish_reason=finish_reason,
                            assistant_content_chars=len(response_text) if response_text else 0,
                            assistant_tool_call_count=len(tool_calls),
                            usage={
                                "input_tokens": _pt,
                                "output_tokens": _ct,
                            },
                        )
                        return web.json_response(response_data, headers=headers)

                    except Exception as exc:
                        passthrough_error = exc
                        try:
                            from agent.model_cooldown_db import mark_provider_failure
                            _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                            mark_provider_failure(_cb_prov, provider_model, base_url=base_url or "", reason="passthrough_error")
                        except Exception:
                            pass
                        try:
                            from agent.model_quality_db import record_failure
                            _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "copilot"
                            record_failure(_cb_prov, provider_model, base_url=base_url or "", error_message=str(exc)[:200])
                        except Exception:
                            pass

                        _invalidate_selectable_pool_cache()
                        # Check if this is a rate-limit error; if so, cooldown the provider
                        _is_rate_limit = False
                        _status_code = _exception_status_code(exc)
                        if _status_code == 429:
                            _is_rate_limit = True
                        elif isinstance(exc, Exception):
                            _exc_str = str(exc).lower()
                            if _is_provider_exhaustion_error(exc):
                                _is_rate_limit = True
                        if _is_rate_limit:
                            _mark_hermes_code_provider_exhausted(
                                provider_model=provider_model,
                                runtime_kwargs=locals().get("runtime_kwargs"),
                                exc=exc,
                                stream=False,
                            )
                        logger.warning("[hermes-code] passthrough copilot %s failed: %s", provider_model, exc)
                        continue

                try:
                    runtime_kwargs, resolved_model = _runtime_kwargs_for_model_id(provider_model)
                    prov = runtime_kwargs.get("provider", "")
                    api_key = runtime_kwargs.get("api_key", "")
                    base_url = runtime_kwargs.get("base_url", "") or None
                    api_mode = runtime_kwargs.get("api_mode", "")
                    _provider_timeout_ns = _passthrough_request_timeout(prov, resolved_model)
                    _provider_messages = _messages_with_provider_tool_prompt(
                        passthrough_messages,
                        provider_model=provider_model,
                        provider=prov,
                        resolved_model=resolved_model,
                        tools=passthrough_tools,
                    )

                    logger.debug(
                        "[hermes-code] passthrough: trying provider=%s model=%s base_url=%s messages=%d",
                        prov, resolved_model, base_url, len(passthrough_messages),
                    )

                    if not api_key:
                        logger.warning("[hermes-code] passthrough: %s has no API key, skipping", provider_model)
                        continue
                    # ── Enforce parallel stream limit ─────────────────────────
                    _acquired_stream_ns = False
                    try:
                        from agent.provider_parallel_limiter import acquire_stream, release_stream
                        if acquire_stream(prov, wait=True, timeout=30.0):
                            _acquired_stream_ns = True
                        else:
                            logger.warning(
                                "[hermes-code] ns: cannot acquire parallel stream slot for provider=%s model=%s, skipping",
                                prov, resolved_model,
                            )
                            passthrough_error = Exception(f"{provider_model}: concurrent stream limit reached")
                            continue
                    except Exception:
                        pass


                    _ns_loop = asyncio.get_running_loop()
                    _mapper_ns = None
                    _msgs_to_send = None
                    if _needs_audio and prov == "google":
                        def _gemini_audio_call_ns():
                            from agent.gemini_native_adapter import GeminiNativeClient
                            _gc = GeminiNativeClient(api_key=api_key, base_url=base_url)
                            return _gc._create_chat_completion(
                                model=resolved_model,
                                messages=_strip_reasoning(_provider_messages) if _passthrough_has_reasoning else _provider_messages,
                                max_tokens=16384,
                                tools=passthrough_tools,
                                timeout=_provider_timeout_ns,
                            )
                        response_obj = await _ns_loop.run_in_executor(None, _gemini_audio_call_ns)
                    elif prov == "openai-codex":
                        response_obj = await _ns_loop.run_in_executor(
                            None,
                            lambda: _call_codex_passthrough(
                                messages=_strip_unsupported_content_for_openai(
                                    _strip_reasoning(_provider_messages) if _passthrough_has_reasoning else _provider_messages
                                ),
                                model=resolved_model,
                                api_key=api_key,
                                base_url=base_url or "",
                                tools=passthrough_tools,
                                timeout=_provider_timeout_ns,
                            ),
                        )
                        try:
                            from agent.model_cooldown_db import mark_provider_success
                            mark_provider_success(prov, resolved_model, base_url=base_url or "")
                        except Exception:
                            pass
                        try:
                            from agent.model_quality_db import record_success
                            record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                        except Exception:
                            pass
                    elif prov == "claude-code-cli":
                        # Claude Code CLI — MCP bridge mode (non-streaming).
                        logger.info("[hermes-code][req=%s] claude-code-cli ns bridge dispatch: prov=%s", _req_id, prov)
                        try:
                            from hermes_cli.auth import resolve_external_process_provider_credentials
                            _cc_creds_ns = resolve_external_process_provider_credentials("claude-code-cli")
                            from agent.claude_code_client import ClaudeCodeClient
                            _cc_client_ns = ClaudeCodeClient(
                                api_key=_cc_creds_ns.get("api_key", "claude-code-cli"),
                                base_url=_cc_creds_ns.get("base_url", "claude://codex"),
                                command=_cc_creds_ns.get("command"),
                                args=_cc_creds_ns.get("args"),
                            )
                        except Exception as _cc_exc:
                            logger.warning("[hermes-code][req=%s] claude-code-cli ns bridge credential resolution failed: %s", _req_id, _cc_exc)
                            _cc_client_ns = None
                        if _cc_client_ns is not None:
                            _skip_normal_call = True
                            _bridge_final_text_ns = ""
                            _bridge_usage_ns = {}
                            _bridge_model_ns = resolved_model
                            _bridge_tool_calls_ns: list[dict] = []
                            _bridge_error_msg_ns: str | None = None
                            _ns_events_q: asyncio.Queue = asyncio.Queue()
                            def _run_bridge_ns(_c=_cc_client_ns, _m=resolved_model, _msgs=_provider_messages, _tools=passthrough_tools):
                                try:
                                    for event in _c.run_with_tool_bridge(model=_m, messages=_msgs, tools=_tools):
                                        _ns_events_q.put_nowait(event)
                                except Exception as exc:
                                    _ns_events_q.put_nowait({"type": "error", "message": str(exc)})
                                finally:
                                    _ns_events_q.put_nowait({"type": "_done"})
                            _ns_bridge_thread = threading.Thread(target=_run_bridge_ns, daemon=True)
                            _ns_bridge_thread.start()
                            while True:
                                _be = await _ns_events_q.get()
                                if _be is None or _be.get("type") == "_done":
                                    break
                                _bt = _be.get("type")
                                if _bt == "tool_call":
                                    _call_id = _be.get("call_id", "")
                                    _tool_name = _be.get("name", "")
                                    # Strip MCP prefix on the way back to the client.
                                    # claude-code uses mcp__hermes-tools__<name>;
                                    # mimocode uses mcp_<name>.
                                    if _tool_name.startswith("mcp__hermes-tools__"):
                                        _tool_name = _tool_name[len("mcp__hermes-tools__"):]
                                    elif _tool_name.startswith("mcp_"):
                                        _tool_name = _tool_name[len("mcp_"):]
                                    _tool_args = _be.get("arguments", {})
                                    _tc = {
                                        "id": _call_id,
                                        "type": "function",
                                        "function": {
                                            "name": _tool_name,
                                            "arguments": json.dumps(_tool_args),
                                        },
                                    }
                                    _bridge_tool_calls_ns.append(_tc)
                                    # Register with the tool_call_hub and wait for the
                                    # connected client to POST the result. Log the
                                    # tool_call_request so the Mac daemon can also
                                    # pick it up via /v1/sessions/{session_id}/pending-tool-calls
                                    # or via log tailing.
                                    _pending_call_ns = None
                                    _tool_result_payload_ns: dict = {"content": "[Tool execution timed out — no response from connected client]"}
                                    if session_id:
                                        try:
                                            from gateway.platforms import tool_call_hub
                                            _pending_call_ns = tool_call_hub.register_call(
                                                session_id, _call_id, tool_name=_tool_name,
                                                arguments=_tool_args,
                                            )
                                            logger.info(
                                                "[hermes-code] ns claude bridge: tool_call_request call_id=%s tool=%s args=%s session=%s",
                                                _call_id, _tool_name, json.dumps(_tool_args)[:300], session_id,
                                            )
                                            def _wait_for_response_ns(_p=_pending_call_ns):
                                                _p.event.wait(timeout=300)
                                                return _p
                                            _pending_call_ns = await _s_loop.run_in_executor(None, _wait_for_response_ns)
                                            if _pending_call_ns.status == "ok":
                                                _tool_result_payload_ns = {"content": _pending_call_ns.result if _pending_call_ns.result is not None else ""}
                                            elif _pending_call_ns.status == "error":
                                                _tool_result_payload_ns = {"error": _pending_call_ns.result or "client-side tool error"}
                                            else:
                                                _tool_result_payload_ns = {"content": "[Tool execution timed out — no response from connected client]"}
                                        except Exception as _hub_exc:
                                            logger.warning("[hermes-code] ns claude bridge: tool_call_hub error: %s", _hub_exc)
                                            _tool_result_payload_ns = {"content": f"[Tool hub error: {_hub_exc}]"}
                                    if _cc_client_ns._queue_out_dir:
                                        _result_path = os.path.join(_cc_client_ns._queue_out_dir, f"{_call_id}.json")
                                        try:
                                            with open(_result_path, "w") as _rf:
                                                json.dump(_tool_result_payload_ns, _rf)
                                            logger.info("[hermes-code] ns claude bridge: wrote real result for %s (%d bytes)", _call_id, len(json.dumps(_tool_result_payload_ns)))
                                        except Exception as _re:
                                            logger.warning("[hermes-code] ns bridge: failed to write result: %s", _re)
                                elif _bt in ("text", "assistant_text"):
                                    _bridge_final_text_ns += _normalize_bridge_text(_be.get("text", ""))
                                elif _bt == "final":
                                    _bridge_final_text_ns = _normalize_bridge_text(_be.get("text", _bridge_final_text_ns))
                                    _bridge_usage_ns = _be.get("usage", {})
                                    _bridge_model_ns = _be.get("model", _bridge_model_ns)
                                elif _bt == "error":
                                    _bridge_error_msg_ns = _be.get("message", "")
                                    logger.warning("[hermes-code] claude-code-cli ns bridge error: %s", _bridge_error_msg_ns)
                            # If the bridge reported an error and produced no real content, use the error
                            # message as the response text so downstream exhaustion/error checks catch it.
                            if _bridge_error_msg_ns and not _bridge_final_text_ns and not _bridge_tool_calls_ns:
                                _bridge_final_text_ns = _bridge_error_msg_ns
                            _usage_ns_obj = SimpleNamespace(
                                prompt_tokens=int(_bridge_usage_ns.get("input_tokens", 0) or 0),
                                completion_tokens=int(_bridge_usage_ns.get("output_tokens", 0) or 0),
                                total_tokens=int((_bridge_usage_ns.get("input_tokens", 0) or 0) + (_bridge_usage_ns.get("output_tokens", 0) or 0)),
                            )
                            if _bridge_final_text_ns:
                                logger.warning("[hermes-code] claude-code-cli ns bridge raw text: %.500s", _bridge_final_text_ns)
                            _skip_provider_exhaustion_content(
                                provider_model=provider_model,
                                runtime_kwargs=runtime_kwargs,
                                content=_bridge_final_text_ns,
                                stream=False,
                            )
                            response_obj = SimpleNamespace(
                                choices=[SimpleNamespace(
                                    message=SimpleNamespace(content=_bridge_final_text_ns, tool_calls=_bridge_tool_calls_ns if _bridge_tool_calls_ns else None),
                                    finish_reason="tool_calls" if _bridge_tool_calls_ns else "stop",
                                )],
                                usage=_usage_ns_obj,
                                model=_bridge_model_ns,
                            )
                            logger.info("[hermes-code][req=%s] claude-code-cli ns bridge completed: text_len=%d tool_calls=%d", _req_id, len(_bridge_final_text_ns), len(_bridge_tool_calls_ns))
                            try:
                                from agent.model_cooldown_db import mark_provider_success
                                mark_provider_success(prov, resolved_model, base_url=base_url or "")
                            except Exception:
                                pass
                            try:
                                from agent.model_quality_db import record_success
                                record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                            except Exception:
                                pass
                        else:
                            continue
                    elif prov == "mimocode-cli":
                        # MiMoCode CLI — simple mode (non-streaming).
                        # The mimo CLI handles tools internally via built-in
                        # tools (bash, read, etc.) — no MCP bridge needed.
                        logger.info("[hermes-code][req=%s] mimocode-cli ns dispatch: prov=%s tools=%d", _req_id, prov, len(passthrough_tools or []))
                        try:
                            from hermes_cli.auth import resolve_external_process_provider_credentials
                            _mc_creds_ns = resolve_external_process_provider_credentials("mimocode-cli")
                            from agent.mimocode_code_client import MiMoCodeClient
                            _mc_client_ns = MiMoCodeClient(
                                api_key=_mc_creds_ns.get("api_key", "mimocode-cli"),
                                base_url=_mc_creds_ns.get("base_url", "mimocode://codex"),
                                command=_mc_creds_ns.get("command"),
                                args=_mc_creds_ns.get("args"),
                            )
                        except Exception as _mc_exc:
                            logger.warning("[hermes-code][req=%s] mimocode-cli credential resolution failed: %s", _req_id, _mc_exc)
                            _mc_client_ns = None
                        if _mc_client_ns is not None:
                            _skip_normal_call = True
                            _bridge_final_text_ns = ""
                            _bridge_usage_ns = {}
                            _bridge_model_ns = resolved_model
                            _bridge_tool_calls_ns: list[dict] = []
                            # Use stream_events() in a background thread so tool_calls
                            # can be proxied through the hub to the connected client.
                            _ns_events_q: asyncio.Queue = asyncio.Queue()
                            _bridge_error_msg_ns: str | None = None
                            def _run_mc_bridge_ns(_c=_mc_client_ns, _m=resolved_model, _msgs=_provider_messages, _tools=passthrough_tools, _q=_ns_events_q):
                                try:
                                    for event in _c.stream_events(model=_m, messages=_msgs, tools=_tools):
                                        _q.put_nowait(event)
                                except Exception as exc:
                                    _q.put_nowait({"type": "error", "message": str(exc)})
                                finally:
                                    _q.put_nowait({"type": "_done"})
                            threading.Thread(target=_run_mc_bridge_ns, daemon=True).start()
                            while True:
                                _be = await _ns_events_q.get()
                                if _be is None or _be.get("type") == "_done":
                                    break
                                _bt = _be.get("type")
                                if _bt == "tool_call":
                                    _call_id = _be.get("call_id", "")
                                    _tool_name = _be.get("name", "")
                                    # Strip MCP prefix on the way back to the client.
                                    if _tool_name.startswith("mcp__hermes-tools__"):
                                        _tool_name = _tool_name[len("mcp__hermes-tools__"):]
                                    elif _tool_name.startswith("mcp_"):
                                        _tool_name = _tool_name[len("mcp_"):]
                                    _tool_args = _be.get("arguments", {})
                                    _tc = {
                                        "id": _call_id,
                                        "type": "function",
                                        "function": {
                                            "name": _tool_name,
                                            "arguments": json.dumps(_tool_args),
                                        },
                                    }
                                    _bridge_tool_calls_ns.append(_tc)
                                    # Register with the tool_call_hub and wait for the
                                    # connected client to POST the result, then write
                                    # it to the MCP queue so the subprocess continues.
                                    _pending_call_ns = None
                                    _tool_result_payload_ns: dict = {"content": "[Tool execution timed out — no response from connected client]"}
                                    if session_id:
                                        try:
                                            from gateway.platforms import tool_call_hub
                                            _pending_call_ns = tool_call_hub.register_call(
                                                session_id, _call_id, tool_name=_tool_name,
                                                arguments=_tool_args,
                                            )
                                            logger.info(
                                                "[hermes-code] ns mimocode bridge: tool_call_request call_id=%s tool=%s args=%s session=%s",
                                                _call_id, _tool_name, json.dumps(_tool_args)[:300], session_id,
                                            )
                                            def _wait_for_response_ns(_p=_pending_call_ns):
                                                _p.event.wait(timeout=300)
                                                return _p
                                            _pending_call_ns = await _s_loop.run_in_executor(None, _wait_for_response_ns)
                                            if _pending_call_ns.status == "ok":
                                                _tool_result_payload_ns = {"content": _pending_call_ns.result if _pending_call_ns.result is not None else ""}
                                            elif _pending_call_ns.status == "error":
                                                _tool_result_payload_ns = {"error": _pending_call_ns.result or "client-side tool error"}
                                            else:
                                                _tool_result_payload_ns = {"content": "[Tool execution timed out — no response from connected client]"}
                                        except Exception as _hub_exc:
                                            logger.warning("[hermes-code] ns mimocode bridge: tool_call_hub error: %s", _hub_exc)
                                            _tool_result_payload_ns = {"content": f"[Tool hub error: {_hub_exc}]"}
                                    if _mc_client_ns._queue_out_dir:
                                        _result_path = os.path.join(_mc_client_ns._queue_out_dir, f"{_call_id}.json")
                                        try:
                                            os.makedirs(_mc_client_ns._queue_out_dir, exist_ok=True)
                                            with open(_result_path, "w") as _rf:
                                                json.dump(_tool_result_payload_ns, _rf)
                                            logger.info("[hermes-code] ns mimocode bridge: wrote result for %s (%d bytes)", _call_id, len(json.dumps(_tool_result_payload_ns)))
                                        except Exception as _re:
                                            logger.warning("[hermes-code] ns mimocode bridge: failed to write result: %s", _re)
                                elif _bt in ("text", "assistant_text"):
                                    _bridge_final_text_ns += _normalize_bridge_text(_be.get("text", ""))
                                elif _bt == "final":
                                    _bridge_final_text_ns = _normalize_bridge_text(_be.get("text", _bridge_final_text_ns))
                                    _bridge_usage_ns = _be.get("usage", {})
                                    _bridge_model_ns = _be.get("model", _bridge_model_ns)
                                elif _bt == "error":
                                    _bridge_error_msg_ns = _be.get("message", "")
                                    logger.warning("[hermes-code] mimocode-cli ns bridge error: %s", _bridge_error_msg_ns)
                            # If the bridge reported an error and produced no real content, use the error
                            # message as the response text so downstream exhaustion/error checks catch it.
                            if _bridge_error_msg_ns and not _bridge_final_text_ns and not _bridge_tool_calls_ns:
                                # If the CLI is busy (concurrency limit), cascade to the next
                                # provider instead of returning the error text to the client.
                                _busy_hint_ns = _bridge_error_msg_ns.lower()
                                if "busy" in _busy_hint_ns or "another instance running" in _busy_hint_ns:
                                    logger.info("[hermes-code] mimocode-cli ns busy — cascading to next provider")
                                    raise _CodexPassthroughSkip("mimocode_cli_busy")
                                _bridge_final_text_ns = _bridge_error_msg_ns
                            _usage_ns_obj = SimpleNamespace(
                                prompt_tokens=int(_bridge_usage_ns.get("input_tokens", 0) or 0),
                                completion_tokens=int(_bridge_usage_ns.get("output_tokens", 0) or 0),
                                total_tokens=int((_bridge_usage_ns.get("input_tokens", 0) or 0) + (_bridge_usage_ns.get("output_tokens", 0) or 0)),
                            )
                            _skip_provider_exhaustion_content(
                                provider_model=provider_model,
                                runtime_kwargs=runtime_kwargs,
                                content=_bridge_final_text_ns,
                                stream=False,
                            )
                            response_obj = SimpleNamespace(
                                choices=[SimpleNamespace(
                                    message=SimpleNamespace(content=_bridge_final_text_ns, tool_calls=_bridge_tool_calls_ns if _bridge_tool_calls_ns else None),
                                    finish_reason="tool_calls" if _bridge_tool_calls_ns else "stop",
                                )],
                                usage=_usage_ns_obj,
                                model=_bridge_model_ns,
                            )
                            logger.info("[hermes-code][req=%s] mimocode-cli ns completed: text_len=%d tool_calls=%d", _req_id, len(_bridge_final_text_ns), len(_bridge_tool_calls_ns))
                            try:
                                from agent.model_cooldown_db import mark_provider_success
                                mark_provider_success(prov, resolved_model, base_url=base_url or "")
                            except Exception:
                                pass
                            try:
                                from agent.model_quality_db import record_success
                                record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                            except Exception:
                                pass
                        else:
                            continue
                    else:
                        _echo_rc = _passthrough_has_reasoning and _requires_reasoning_echo(resolved_model, provider=prov, base_url=base_url)
                        _msgs_to_send = (_provider_messages if _echo_rc else _strip_reasoning(_provider_messages)) if _passthrough_has_reasoning else _provider_messages
                        _msgs_to_send = _strip_unsupported_content_for_openai(_msgs_to_send)
                        # For providers that require reasoning_content echo (DeepSeek via
                        # opencode-zen/opencode-go), ensure every assistant turn that
                        # has tool_calls also has a reasoning_content field — even an
                        # empty one. The provider rejects the request if the field is
                        # missing on any assistant turn that emitted tool_calls in
                        # thinking mode. This must run AFTER _strip_reasoning so the
                        # synthesised field isn't removed by stripping.
                        if _requires_reasoning_echo(resolved_model, provider=prov, base_url=base_url):
                            _msgs_to_send = _synthesize_reasoning_for_tool_calls(_msgs_to_send)

                        # ── Packed tool_call_id cleanup (non-streaming) ──────────
                        # Strip ``:hermes_ts:`` packed ids for all providers, and
                        # for Google also inject the thought_signature back into
                        # extra_content.  See the streaming path for the full
                        # rationale.
                        if "generativelanguage.googleapis.com" in (base_url or ""):
                            # 1) Strip any :hermes_ts: packed ids and restore signatures
                            _injected, _unpacked = _unpack_hermes_ts_and_inject_signatures(_msgs_to_send)
                            # 2) Inject sentinel thought_signature on unsigned tool_calls
                            _sentinel = _inject_google_sentinel_signatures(_msgs_to_send)
                            if _injected or _unpacked or _sentinel:
                                logger.warning(
                                    "[hermes-code] ns Google thought_signature: injected=%d unpacked=%d sentinel=%d into %d messages for %s",
                                    _injected, _unpacked, _sentinel, len(_msgs_to_send), resolved_model,
                                )
                        else:
                            _ts_fixed = _strip_hermes_ts_packed_ids(_msgs_to_send)
                            if _ts_fixed:
                                logger.debug("[hermes-code] ns: stripped %d hermes_ts packed ids for %s", _ts_fixed, provider_model)

                        # ── arliai tool_call_id sanitization ────────────────────
                        if base_url and "arliai" in (base_url or "").lower():
                            try:
                                from agent._tool_id_sanitizer import ToolCallIdMapper
                                _mapper_ns = ToolCallIdMapper(max_length=9)
                                _msgs_to_send = _mapper_ns.sanitize_messages(_msgs_to_send)
                                logger.debug("[hermes-code] arliai ns: sanitized tool_call_ids",)
                            except Exception as _map_exc:
                                logger.warning("[hermes-code] arliai ns: failed to init mapper: %s", _map_exc)

                        # ── cerebras call_id stripping (non-streaming) ───────────
                        if prov == "cerebras":
                            _cc_removed_ns = _strip_call_id_from_tool_calls(_msgs_to_send)
                            if _cc_removed_ns:
                                logger.debug("[hermes-code] cerebras ns: stripped call_id from %d tool_calls", _cc_removed_ns)

                        logger.debug(
                            "[hermes-code] non-streaming call_llm: model=%s provider=%s has_rc=%s echo=%s msgs=%d",
                            resolved_model, prov, _passthrough_has_reasoning, _echo_rc, len(_msgs_to_send),
                        )
                        if _passthrough_has_reasoning and _echo_rc:
                            _rc_in_msgs = [len(m.get("reasoning_content") or "") for m in _msgs_to_send if m.get("reasoning_content")]
                            logger.debug("[hermes-code] non-streaming call_llm: rc lengths in msgs=%s", _rc_in_msgs)
                        response_obj = await _ns_loop.run_in_executor(
                            None,
                            lambda: call_llm(
                                task="chat",
                                messages=_msgs_to_send,
                                provider=prov,
                                model=resolved_model,
                                base_url=base_url,
                                api_key=api_key,
                                max_tokens=16384,
                                timeout=_provider_timeout_ns,
                                tools=passthrough_tools,
                            ),
                        )
                        try:
                            from agent.model_cooldown_db import mark_provider_success
                            mark_provider_success(prov, resolved_model, base_url=base_url or "")
                        except Exception:
                            pass
                        try:
                            from agent.model_quality_db import record_success
                            record_success(prov, provider_model, base_url=base_url or "", latency_ms=0)
                        except Exception:
                            pass

                    msg = response_obj.choices[0].message
                    content = extract_content_or_reasoning(response_obj).strip()
                    reasoning_content = _extract_reasoning_content_from_msg(msg)
                    _skip_provider_exhaustion_content(
                        provider_model=provider_model,
                        runtime_kwargs=runtime_kwargs,
                        content=content,
                        stream=False,
                    )
                    if reasoning_content:
                        logger.debug(
                            "[hermes-code] non-streaming reasoning_content from provider: model=%s rc_len=%d",
                            provider_model, len(reasoning_content),
                        )
                    tool_calls_raw = getattr(msg, "tool_calls", []) or []
                    # Serialize tool_calls (may be Pydantic models from some providers)
                    tool_calls_out = []
                    for tc in tool_calls_raw:
                        if hasattr(tc, "model_dump"):
                            _tc_dict = tc.model_dump()
                        elif hasattr(tc, "dict"):
                            _tc_dict = tc.dict()
                        elif isinstance(tc, dict):
                            _tc_dict = tc
                        else:
                            _func = getattr(tc, "function", None)
                            _func_name = str(getattr(_func, "name", getattr(tc, "name", "")))
                            _func_args = str(getattr(_func, "arguments", getattr(tc, "arguments", "{}")))
                            _tc_dict = {"id": str(getattr(tc, "id", "")), "type": "function", "function": {"name": _func_name, "arguments": _func_args}}
                        # Preserve extra_content (contains google.thought_signature for Gemini)
                        _ec = getattr(tc, "extra_content", None) or (_tc_dict.get("extra_content") if isinstance(_tc_dict, dict) else None)
                        if _ec and isinstance(_tc_dict, dict):
                            _tc_dict["extra_content"] = _ec
                        # ── Google thought_signature: pack into call_id ────────
                        # Standard OpenAI clients strip non-standard fields like
                        # extra_content when re-sending history. Pack the signature
                        # into the tool_call id as `<orig>:hermes_ts:<b64>` so it
                        # survives the round-trip. The injection code unpacks it
                        # back into extra_content on the next request.
                        if isinstance(_tc_dict, dict):
                            _google_ec = (_ec or {}).get("google") if isinstance(_ec, dict) else None
                            _ts = _google_ec.get("thought_signature") if isinstance(_google_ec, dict) else None
                            if _ts and isinstance(_tc_dict.get("id"), str) and ":hermes_ts:" not in _tc_dict["id"]:
                                import base64
                                _packed = base64.urlsafe_b64encode(_ts.encode("utf-8")).decode("ascii").rstrip("=")
                                _packed_id = f"{_tc_dict['id']}:hermes_ts:{_packed}"
                                _tc_dict["id"] = _packed_id
                                _tc_dict["call_id"] = _packed_id
                        tool_calls_out.append(_tc_dict)
                    tool_calls_out = _enrich_client_tool_calls(tool_calls_out, passthrough_tools)

                    _bad_bash_summary_ns = _invalid_bash_tool_call_summary(tool_calls_out)
                    if passthrough_tools and _bad_bash_summary_ns and locals().get("_msgs_to_send") is not None:
                        logger.warning(
                            "[hermes-code][req=%s] %s returned invalid bash tool call; retrying once with corrective prompt: %s",
                            _req_id, provider_model, _bad_bash_summary_ns,
                        )
                        _retry_msgs_to_send_ns = _messages_with_retry_tool_prompt(_msgs_to_send, tool_calls_out)
                        response_obj = await _ns_loop.run_in_executor(
                            None,
                            lambda: call_llm(
                                task="chat",
                                messages=_retry_msgs_to_send_ns,
                                provider=prov,
                                model=resolved_model,
                                base_url=base_url,
                                api_key=api_key,
                                max_tokens=16384,
                                timeout=_provider_timeout_ns,
                                tools=passthrough_tools,
                            ),
                        )
                        msg = response_obj.choices[0].message
                        content = extract_content_or_reasoning(response_obj).strip()
                        reasoning_content = _extract_reasoning_content_from_msg(msg)
                        tool_calls_raw = getattr(msg, "tool_calls", []) or []
                        tool_calls_out = []
                        for tc in tool_calls_raw:
                            if hasattr(tc, "model_dump"):
                                _tc_dict = tc.model_dump()
                            elif hasattr(tc, "dict"):
                                _tc_dict = tc.dict()
                            elif isinstance(tc, dict):
                                _tc_dict = tc
                            else:
                                _func = getattr(tc, "function", None)
                                _tc_dict = {
                                    "id": str(getattr(tc, "id", "")),
                                    "type": "function",
                                    "function": {
                                        "name": str(getattr(_func, "name", getattr(tc, "name", ""))),
                                        "arguments": str(getattr(_func, "arguments", getattr(tc, "arguments", "{}"))),
                                    },
                                }
                            tool_calls_out.append(_tc_dict)
                        tool_calls_out = _enrich_client_tool_calls(tool_calls_out, passthrough_tools)
                        if _mapper_ns is not None and tool_calls_out:
                            tool_calls_out = _mapper_ns.unsanitize_tool_calls(tool_calls_out)

                    # If any provider returned no tool calls (or empty bash commands)
                    # despite having tools, skip to next. Apply a short cooldown so
                    # a model that repeatedly returns text-only (when tools are
                    # expected) doesn't keep burning through the entire fallback
                    # chain on every request. The cooldown is short enough that
                    # the model will be retried after ~2 min — long enough to
                    # skip past it on the current request and avoid the worst
                    # case where 30+ models each return text-only in sequence.
                    if passthrough_tools and (not tool_calls_out or _has_empty_bash_tool_call(tool_calls_out)):
                        # claude-code-cli is an agent, not a model API.
                        # The CLI itself executes tools (Bash, Read, etc.) and returns
                        # the final answer as text.  Don't penalise it for text-only —
                        # the "tool calls" happened inside the subprocess.
                        _is_claude_code_ns = provider_model.startswith("claude-code-cli")
                        _is_mimocode_ns = provider_model.startswith("mimocode-cli")
                        if _is_claude_code_ns or _is_mimocode_ns:
                            logger.info(
                                "[hermes-code] claude-code-cli: text-only response accepted (CLI executes tools internally)",
                                provider_model,
                            )
                        elif not tool_calls_out:
                            # Text-only response: return it to the client as-is.
                            # Previously this cascaded to the next provider, but
                            # that burned through the entire fallback chain only
                            # to return an error.  Clients handle text-only fine.
                            logger.warning(
                                "[hermes-code] %s text-only (tools=%d, content_len=%d) — returning to client as-is",
                                provider_model, len(passthrough_tools),
                                len(content) if content else 0,
                            )
                            try:
                                from agent.model_quality_db import record_text_only
                                record_text_only(provider_model.split("/")[0], provider_model, base_url=base_url or "")
                            except Exception:
                                pass
                        else:
                            # Empty bash tool call: clear it and return text as-is.
                            # Previously cascaded via _CodexPassthroughSkip which burned
                            # through the entire fallback chain and returned 503.
                            logger.warning(
                                "[hermes-code] %s returned bash with empty command — clearing tool_calls, returning text as-is (tools=%d, content_len=%d)",
                                provider_model, len(passthrough_tools),
                                len(content) if content else 0,
                            )
                            try:
                                from agent.model_cooldown_db import mark_model_cooldown
                                mark_model_cooldown(
                                    provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                    model=provider_model,
                                    cooldown_seconds=120.0,
                                    reason="empty_bash",
                                )
                            except Exception:
                                pass
                            try:
                                from agent.model_quality_db import record_text_only
                                record_text_only(provider_model.split("/")[0], provider_model, base_url=base_url or "")
                            except Exception:
                                pass
                            tool_calls_out = []

                    usage_obj = getattr(response_obj, "usage", None)
                    # Record GHE AIU spend and enforce monthly limit.
                    try:
                        from agent.copilot_spend_db import record_and_check
                        record_and_check(response_obj, provider_model=provider_model, base_url=base_url or "")
                    except Exception as _spend_exc:
                        logger.warning("[copilot_spend] record failed: %s", _spend_exc)

                    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
                    if reasoning_content and reasoning_content.strip():
                        assistant_msg["reasoning_content"] = reasoning_content.strip()
                    if tool_calls_out:
                        assistant_msg["tool_calls"] = tool_calls_out

                    response_data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "message": assistant_msg,
                            "finish_reason": "tool_calls" if tool_calls_out else "stop",
                        }],
                        "usage": {
                            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                        },
                    }
                    if provided_session_id:
                        try:
                            _persist_passthrough_session_delta(
                                self._ensure_session_db(),
                                session_id,
                                model_name=model_name,
                                system_prompt=system_prompt,
                                request_messages=_request_conversation_messages,
                                assistant_content=content,
                                assistant_tool_calls=tool_calls_out,
                                finish_reason="tool_calls" if tool_calls_out else "stop",
                                reasoning_content=reasoning_content,
                            )
                        except Exception as _persist_exc:
                            logger.warning("[api_server] failed to persist passthrough session delta for %s: %s", session_id, _persist_exc)
                    headers = {"X-Hermes-Session-Id": session_id}
                    _invoke_passthrough_hooks(
                        "post_api_request",
                        task_id="", session_id=session_id or "", platform="api_server",
                        model=provider_model, provider=provider_model.split("/")[0],
                        base_url=base_url or "", api_mode="",
                        api_call_count=_pt_call_count[0],
                        finish_reason="tool_calls" if tool_calls_out else "stop",
                        assistant_content_chars=len(content) if content else 0,
                        assistant_tool_call_count=len(tool_calls_out),
                        usage={
                            "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                            "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                        },
                    )
                    # Release parallel stream slot on success
                    if _acquired_stream_ns:
                        try:
                            from agent.provider_parallel_limiter import release_stream
                            release_stream(prov)
                        except Exception:
                            pass
                    return web.json_response(response_data, headers=headers)

                except _CodexPassthroughSkip as _skip_exc:
                    logger.warning(
                        "[hermes-code] %s skipped via _CodexPassthroughSkip, "
                        "trying next provider",
                        provider_model,
                    )
                    # Release parallel stream slot on skip
                    if _acquired_stream_ns:
                        try:
                            from agent.provider_parallel_limiter import release_stream
                            release_stream(prov)
                        except Exception:
                            pass
                    passthrough_error = _skip_exc
                    continue
                except Exception as exc:
                    # Release parallel stream slot on provider error
                    if _acquired_stream_ns:
                        try:
                            from agent.provider_parallel_limiter import release_stream
                            release_stream(prov)
                        except Exception:
                            pass
                    passthrough_error = exc
                    # If the CLI is busy (concurrency limit), cascade without
                    # penalising the provider — it's not a real failure.
                    _exc_busy = "busy" in str(exc).lower() or "another instance running" in str(exc).lower()
                    if not _exc_busy:
                        try:
                            from agent.model_cooldown_db import mark_provider_failure
                            _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "openai"
                            mark_provider_failure(_cb_prov, provider_model, base_url=base_url or "", reason="passthrough_error")
                        except Exception:
                            pass
                    try:
                        from agent.model_quality_db import record_failure
                        _cb_prov = provider_model.split("/")[0] if "/" in provider_model else "openai"
                        record_failure(_cb_prov, provider_model, base_url=base_url or "", error_message=str(exc)[:200])
                    except Exception:
                        pass
                    _invalidate_selectable_pool_cache()
                    # Check if this is a rate-limit error; if so, cooldown the provider
                    _is_rate_limit = False
                    _status_code = _exception_status_code(exc)
                    if _status_code == 429:
                        _is_rate_limit = True
                    elif isinstance(exc, Exception):
                        _exc_str = str(exc).lower()
                        if _is_provider_exhaustion_error(exc):
                            _is_rate_limit = True
                    if _is_rate_limit:
                        _mark_hermes_code_provider_exhausted(
                            provider_model=provider_model,
                            runtime_kwargs=runtime_kwargs,
                            exc=exc,
                            stream=False,
                        )
                    elif _status_code == 400:
                        # 400 errors (bad request) — 120s cooldown.
                        try:
                            from agent.model_cooldown_db import mark_model_cooldown
                            mark_model_cooldown(
                                provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                model=provider_model,
                                cooldown_seconds=120.0,
                                reason="hermes_code_passthrough_400",
                            )
                            logger.warning("[hermes-code] %s cooled down for 120s after 400", provider_model)
                        except Exception:
                            pass
                    elif _status_code == 401:
                        # 401 auth errors — distinguish permanent from transient.
                        _exc_str = str(exc).lower()
                        _is_token_invalidated = "token_invalidated" in _exc_str or "invalidated" in _exc_str
                        if _is_token_invalidated:
                            # Permanent failure: token is dead, needs re-auth. 24h cooldown.
                            try:
                                from agent.model_cooldown_db import mark_model_cooldown
                                mark_model_cooldown(
                                    provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                    model=provider_model,
                                    cooldown_seconds=86400.0,
                                    reason="hermes_code_passthrough_401_token_invalidated",
                                )
                                logger.warning("[hermes-code] %s cooled down for 24h after 401 (token invalidated)", provider_model)
                            except Exception:
                                pass
                        else:
                            # Transient auth error — 120s cooldown.
                            try:
                                from agent.model_cooldown_db import mark_model_cooldown
                                mark_model_cooldown(
                                    provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                    model=provider_model,
                                    cooldown_seconds=120.0,
                                    reason="hermes_code_passthrough_401",
                                )
                                logger.warning("[hermes-code] %s cooled down for 120s after 401", provider_model)
                            except Exception:
                                pass
                    elif _status_code == 500:
                        # 500 (server error) — provider is broken. 5min cooldown.
                        try:
                            from agent.model_cooldown_db import mark_model_cooldown
                            mark_model_cooldown(
                                provider=provider_model.split("/")[0] if "/" in provider_model else "openai",
                                model=provider_model,
                                cooldown_seconds=300.0,
                                reason="hermes_code_passthrough_500",
                            )
                            logger.warning("[hermes-code] %s cooled down for 5min after 500", provider_model)
                        except Exception:
                            pass
                    logger.warning("[hermes-code] passthrough %s failed: %s", provider_model, exc)
                    continue

            if passthrough_error or not _pt_call_count[0]:
                if not passthrough_error:
                    passthrough_error = Exception("no passthrough providers available (all skipped or missing credentials)")
                _err_msg = _sanitize_passthrough_error_for_client(passthrough_error)
                if _is_prompt_too_long_error(_err_msg):
                    logger.warning(
                        "[api_server] hermes-code passthrough stopped: prompt too long (~%d tokens). Returning 413.",
                        _approx_tokens,
                    )
                    return web.json_response(
                        _openai_error(
                            f"Context too large: too many tokens (~{_approx_tokens:,}) exceeds provider prompt limit. "
                            "Please compact the conversation history and retry.",
                            err_type="invalid_request_error",
                            code="context_too_large",
                        ),
                        status=413,
                    )
                if not _err_msg:
                    _err_msg = f"all providers failed ({type(passthrough_error).__name__})"
                    logger.warning(
                        "[api_server] hermes-code passthrough exhausted: last error type=%s has empty message, "
                        "indicates silent provider failure — check provider logs above",
                        type(passthrough_error).__name__,
                    )
                else:
                    logger.warning("[api_server] hermes-code passthrough exhausted providers: %s", _err_msg)
                return web.json_response(
                    _openai_error(
                        f"hermes-code passthrough exhausted all configured providers: {_err_msg}",
                        err_type="server_error",
                        code="passthrough_exhausted",
                    ),
                    status=503,
                )

        # Hermes-swarm: select free/cheap model pool
        swarm_mode = False
        swarm_model_pool = None
        if model_name == "hermes-swarm" or (role_cfg and role_cfg.get("mode") == "swarm"):
            swarm_mode = True
            swarm_model_pool = await self._prepare_swarm_model_pool(
                system_prompt=system_prompt or "",
                conversation_history=history,
                user_message=user_message,
                tools=tools,
                estimated_tokens=_approx_tokens,
                routing_hint=role_hint,
            )
            logger.info(
                "[api_server] swarm pool: primary=%s, fallbacks=%d models, "
                "large-context options: %s routing_hint=%s",
                swarm_model_pool["primary"], len(swarm_model_pool["fallbacks"]),
                swarm_model_pool["large_context_fallbacks"],
                swarm_model_pool.get("routing_hint"),
            )

            # Token-aware pre-truncation: ensure history fits the primary model's context.
            # Uses 85% of context window as safe budget. This prevents 413 errors before
            # they reach the LLM API. _resolve_swarm_model already filters candidate lists.
            _primary_model = swarm_model_pool.get("primary", "")
            if _primary_model and history:
                _history_tokens = _messages_token_count(history, system_prompt or "")
                _ctx_len = _model_context_length(_primary_model)
                if _ctx_len > 0 and _history_tokens > int(_ctx_len * 0.85):
                    history = _compact_message_history(
                        history,
                        session_id,
                        system_prompt=system_prompt or "",
                        target_model=_primary_model,
                    )
                    logger.info(
                        "[api_server] history pre-truncated: ~%d tokens -> ~%d, model=%s",
                        _history_tokens, _messages_token_count(history, system_prompt or ""), _primary_model,
                    )

        agents_prefetch_text = ""
        if swarm_mode and _needs_agents_prefetch(user_message, system_prompt, tools, conversation_messages):
            prefetch = _determine_agents_prefetch_action(conversation_messages, tools)
            status = prefetch.get("status")
            if status in {"need_search", "need_read"}:
                prefetch_call = prefetch.get("tool_call")
                response_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": _enrich_client_tool_calls([prefetch_call]) if prefetch_call else [],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                headers = {"X-Hermes-Session-Id": session_id}
                if not stream:
                    return web.json_response(response_data, headers=headers)
                sse_headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
                response = web.StreamResponse(status=200, headers=sse_headers)
                await response.prepare(request)
                chunk = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": response_data["choices"][0]["message"]["tool_calls"]},
                        "finish_reason": None,
                    }],
                }
                finish = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await response.write(f"data: {json.dumps(finish)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
                return response
            agents_prefetch_text = str(prefetch.get("agents_text") or "").strip()

        if agents_prefetch_text:
            system_prompt = (
                f"{system_prompt}\n\n[Deterministic AGENTS preflight]\n{agents_prefetch_text}"
                if system_prompt else f"[Deterministic AGENTS preflight]\n{agents_prefetch_text}"
            )

        created = int(time.time())
        logger.info("[timing] _handle_chat_completions pre-stream parse+swarm: %.3fs", time.time() - _t0)

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # Filter out None — the agent fires stream_delta_callback(None)
                # to signal the CLI display to close its response box before
                # tool execution, but the SSE writer uses None as end-of-stream
                # sentinel.  Forwarding it would prematurely close the HTTP
                # response, causing Open WebUI (and similar frontends) to miss
                # the final answer after tool calls.  The SSE loop detects
                # completion via agent_task.done() instead.
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Send tool progress as a separate SSE event.

                Previously, progress markers like ``⏰ list`` were injected
                directly into ``delta.content``.  OpenAI-compatible frontends
                (Open WebUI, LobeChat, …) store ``delta.content`` verbatim as
                the assistant message and send it back on subsequent requests.
                After enough turns the model learns to *emit* the markers as
                plain text instead of issuing real tool calls — silently
                hallucinating tool results.  See #6972.

                The fix: push a tagged tuple ``("__tool_progress__", payload)``
                onto the stream queue.  The SSE writer emits it as a custom
                ``event: hermes.tool.progress`` line that compliant frontends
                can render for UX but will *not* persist into conversation
                history.  Clients that don't understand the custom event type
                silently ignore it per the SSE specification.
                """
                if event_type != "tool.started":
                    return
                if name.startswith("_"):
                    return
                from agent.display import get_tool_emoji
                emoji = get_tool_emoji(name)
                label = preview or name
                _stream_q.put(("__tool_progress__", {
                    "tool": name,
                    "emoji": emoji,
                    "label": label,
                }))

            def _on_tool_gen(tool_name: str, call_id: Optional[str] = None, arguments: str = ""):
                """Emit function_call chunks when the model decides to use a tool."""
                # run_agent fires this callback as soon as a streamed response
                # starts *writing* a tool call, before argument JSON is complete.
                # Do not convert that progress notification into an OpenAI
                # tool_call chunk: OpenCode treats parsable `{}` / repaired
                # placeholders as complete tool input and executes them, causing
                # empty bash/read/write calls.  The completed call is emitted
                # after agent_task returns with tool_calls_pending.
                if (not isinstance(call_id, str) or not call_id.strip()) and not str(arguments or "").strip():
                    _stream_q.put(("__tool_progress__", {
                        "tool": _normalize_external_tool_name(tool_name),
                        "label": f"Writing {_normalize_external_tool_name(tool_name)} input...",
                    }))
                    return
                if not isinstance(call_id, str) or not call_id.strip():
                    basis = f"{session_id}:{completion_id}:{tool_name}:{arguments or ''}"
                    call_id = f"call_{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:24]}"
                if external_tool_mode == "broker":
                    try:
                        from gateway.platforms import tool_call_hub
                        tool_call_hub.register_call(session_id, call_id, tool_name, arguments)
                        logger.info(
                            "[api_server] registered external tool call session=%s call_id=%s tool=%s",
                            session_id, call_id, tool_name,
                        )
                    except Exception as e:
                        logger.debug("tool_call_hub.register_call failed: %s", e)
                safe_tool_name = _normalize_external_tool_name(tool_name)
                safe_arguments = _external_tool_call_arguments_str(safe_tool_name, arguments)
                tool_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": safe_tool_name,
                                    "arguments": safe_arguments,
                                },
                            }],
                        },
                        "finish_reason": None,
                    }],
                }
                _stream_q.put(("__tool_call_start__", {
                    "session_id": session_id,
                    "call_id": call_id,
                    "tool_name": safe_tool_name,
                    "register_with_hub": external_tool_mode == "broker",
                    "chunk": tool_chunk,
                }))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_gen_callback=_on_tool_gen,
                agent_ref=agent_ref,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                estimated_tokens=_approx_tokens,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
                user_model=model_name,
            ))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
                force_connection_close=force_connection_close,
                swarm_model_pool=swarm_model_pool,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            # Smart routing: check dedup cache and route based on complexity
            from agent.deduplicator import get_global_deduplicator
            from agent.smart_router import get_global_router
            
            # Build the full prompt for routing analysis
            routing_prompt = user_message
            if system_prompt:
                routing_prompt = f"{system_prompt}\n\n{routing_prompt}"
            
            # Check dedup cache first
            dedup = get_global_deduplicator()
            dedup_cache_key = dedup.compute_key(routing_prompt, model_name) if hasattr(dedup, 'compute_key') else None
            
            if dedup_cache_key:
                cached_response, found = dedup.get(routing_prompt, model_name)
                if found and cached_response:
                    logger.info("[smart_router] dedup cache hit for model=%s", model_name)
                    # Return cached response - but we still need to run agent for usage tracking
                    # For now, just log and continue to actual execution
            
            # Route based on complexity if smart routing is enabled
            router = get_global_router()
            routing_result = router.route(routing_prompt)
            
            if routing_result.get("complexity") == "simple":
                logger.info(
                    "[smart_router] routing simple query to %s (tier: %s, savings: %.4f)",
                    routing_result.get("model"),
                    routing_result.get("tier"),
                    routing_result.get("savings_vs_primary", 0),
                )
            
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                estimated_tokens=_approx_tokens,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
                user_model=model_name,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                _timing_compute = time.time()
                result, usage = await _compute_completion()
                logger.info("[timing] _compute_completion total: %.3fs (wall since request start: %.3fs)",
                    time.time() - _timing_compute, time.time() - _t0)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        logger.info("[timing] _handle_chat_completions agent returned, elapsed: %.3fs", time.time() - _t0)
        final_response = result.get("final_response", "")
        if result.get("tool_calls_pending"):
            last_assistant = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
                    last_assistant = msg
                    break
            response_data = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            # For pending tool calls, return tool_calls only.
                            # Some clients mis-handle mixed assistant text +
                            # tool_calls and render them out of order.
                            "content": "",
                            "tool_calls": _enrich_client_tool_calls((last_assistant or {}).get("tool_calls", [])) if last_assistant else [],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            headers = {"X-Hermes-Session-Id": session_id}
            if force_connection_close:
                headers["Connection"] = "close"
            return web.json_response(response_data, headers=headers)
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        # Extract reasoning_content from the last assistant message if available
        # This is needed for Kimi/Moonshot/GLM reasoning models
        reasoning_content = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                reasoning_content = msg.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    break

        message_data = {
            "role": "assistant",
            "content": final_response,
        }
        # Include reasoning_content if present and non-empty
        if reasoning_content:
            message_data["reasoning_content"] = reasoning_content

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message_data,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        headers = {"X-Hermes-Session-Id": session_id}
        if force_connection_close:
            headers["Connection"] = "close"
        return web.json_response(response_data, headers=headers)

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
        force_connection_close: bool = False, swarm_model_pool: dict = None,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue.

        If the client disconnects mid-stream (network drop, browser tab close),
        the agent is interrupted via ``agent.interrupt()`` so it stops making
        LLM API calls, and the asyncio task wrapper is cancelled.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        # CORS middleware can't inject headers into StreamResponse after
        # prepare() flushes them, so resolve CORS headers up front.
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        if force_connection_close:
            sse_headers["Connection"] = "close"
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            last_activity = time.monotonic()

            # Role chunk
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
            last_activity = time.monotonic()

            buffered_text_deltas: List[str] = []
            tool_call_started = False
            last_noop_chunk_at = 0.0

            async def _emit_noop_chunk() -> float:
                """Emit a standard OpenAI no-op chunk for liveness/progress.

                Roo/OpenAI-compatible clients appear happiest when every SSE
                frame is a normal ``data: {...}`` chat chunk. Use an empty
                delta chunk as a transport-safe heartbeat/progress tick.
                """
                nonlocal last_noop_chunk_at
                now = time.monotonic()
                if now - last_noop_chunk_at < 0.25:
                    return now
                keepalive_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                }
                await response.write(f"data: {json.dumps(keepalive_chunk)}\n\n".encode())
                last_noop_chunk_at = now
                return now

            # Helper — route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples ``("__tool_progress__", payload)`` are sent
                as a custom ``event: hermes.tool.progress`` SSE event so
                frontends can display them without storing the markers in
                conversation history.  See #6972.
                """
                nonlocal tool_call_started
                if isinstance(item, tuple) and len(item) == 2:
                    tag, payload = item
                    if tag == "__tool_progress__":
                        # Convert internal progress into a transport-safe no-op
                        # OpenAI chunk instead of SSE comments/custom events.
                        # This preserves liveness during tool execution or
                        # upstream model failover without mutating assistant
                        # text or violating Roo's stream expectations.
                        return await _emit_noop_chunk()
                    if tag == "__tool_call_start__":
                        tool_call_started = True
                        buffered_text_deltas.clear()
                        try:
                            logger.info(
                                "[api_server] emitting tool call SSE session=%s call_id=%s tool=%s",
                                payload.get("session_id", session_id), payload.get("call_id"), payload.get("tool_name"),
                            )
                            if payload.get("session_id") and payload.get("register_with_hub"):
                                from gateway.platforms import tool_call_hub
                                tool_call_hub.register_call(
                                    payload.get("session_id", session_id), payload.get("call_id"), payload.get("tool_name"),
                                    payload.get("arguments"),
                                )
                        except Exception:
                            pass
                        chunk = payload.get("chunk")
                        if isinstance(chunk, dict):
                            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            return time.monotonic()
                content_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                }
                if tool_call_started:
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                else:
                    buffered_text_deltas.append(item)
                return time.monotonic()

            # Stream content chunks as they arrive from the agent
            loop = asyncio.get_running_loop()
            while True:
                try:
                    delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain any remaining items
                        while True:
                            try:
                                delta = stream_q.get_nowait()
                                if delta is None:
                                    break
                                last_activity = await _emit(delta)
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        last_activity = await _emit_noop_chunk()
                    continue

                if delta is None:  # End of stream sentinel
                    break

                last_activity = await _emit(delta)

            # Get usage from completed agent
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            finish_reason = "stop"
            final_tool_calls: List[Dict[str, Any]] = []
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                if isinstance(result, dict) and result.get("tool_calls_pending"):
                    finish_reason = "tool_calls"
                    for msg in reversed(result.get("messages", [])):
                        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
                            final_tool_calls = _enrich_client_tool_calls(msg.get("tool_calls", []))
                            break
            except Exception:
                pass

            if finish_reason == "stop" and buffered_text_deltas:
                for item in buffered_text_deltas:
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())

            if finish_reason == "tool_calls" and final_tool_calls:
                tool_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"tool_calls": [dict(tc, index=i) for i, tc in enumerate(final_tool_calls)]}, "finish_reason": None}],
                }
                await response.write(f"data: {json.dumps(tool_chunk)}\n\n".encode())

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected mid-stream.  Interrupt the agent so it
            # stops making LLM API calls at the next loop iteration, then
            # cancel the asyncio task wrapper.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)

        return response

    async def _write_sse_responses(
        self,
        request: "web.Request",
        response_id: str,
        model: str,
        created_at: int,
        stream_q,
        agent_task,
        agent_ref,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        instructions: Optional[str],
        conversation: Optional[str],
        store: bool,
        session_id: str,
    ) -> "web.StreamResponse":
        """Write an SSE stream for POST /v1/responses (OpenAI Responses API).

        Emits spec-compliant event types as the agent runs:

        - ``response.created`` — initial envelope (status=in_progress)
        - ``response.output_text.delta`` / ``response.output_text.done`` —
          streamed assistant text
        - ``response.output_item.added`` / ``response.output_item.done``
          with ``item.type == "function_call"`` — when the agent invokes a
          tool (both events fire; the ``done`` event carries the finalized
          ``arguments`` string)
        - ``response.output_item.added`` with
          ``item.type == "function_call_output"`` — tool result with
          ``{call_id, output, status}``
        - ``response.completed`` — terminal event carrying the full
          response object with all output items + usage (same payload
          shape as the non-streaming path for parity)
        - ``response.failed`` — terminal event on agent error

        If the client disconnects mid-stream, ``agent.interrupt()`` is
        called so the agent stops issuing upstream LLM calls, then the
        asyncio task is cancelled.  When ``store=True`` the full response
        is persisted to the ResponseStore in a ``finally`` block so GET
        /v1/responses/{id} and ``previous_response_id`` chaining work the
        same as the batch path.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        # State accumulated during the stream
        final_text_parts: List[str] = []
        # Track open function_call items by name so we can emit a matching
        # ``done`` event when the tool completes.  Order preserved.
        pending_tool_calls: List[Dict[str, Any]] = []
        # Output items we've emitted so far (used to build the terminal
        # response.completed payload).  Kept in the order they appeared.
        emitted_items: List[Dict[str, Any]] = []
        # Monotonic counter for output_index (spec requires it).
        output_index = 0
        # Monotonic counter for call_id generation if the agent doesn't
        # provide one (it doesn't, from tool_progress_callback).
        call_counter = 0
        # Canonical Responses SSE events include a monotonically increasing
        # sequence_number. Add it server-side for every emitted event so
        # clients that validate the OpenAI event schema can parse our stream.
        sequence_number = 0
        # Track the assistant message item id + content index for text
        # delta events — the spec ties deltas to a specific item.
        message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        message_output_index: Optional[int] = None
        message_opened = False

        async def _write_event(event_type: str, data: Dict[str, Any]) -> None:
            nonlocal sequence_number
            if "sequence_number" not in data:
                data["sequence_number"] = sequence_number
            sequence_number += 1
            payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            await response.write(payload.encode())

        def _envelope(status: str) -> Dict[str, Any]:
            env: Dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "status": status,
                "created_at": created_at,
                "model": model,
            }
            return env

        final_response_text = ""
        agent_error: Optional[str] = None
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            # response.created — initial envelope, status=in_progress
            created_env = _envelope("in_progress")
            created_env["output"] = []
            await _write_event("response.created", {
                "type": "response.created",
                "response": created_env,
            })
            last_activity = time.monotonic()

            async def _open_message_item() -> None:
                """Emit response.output_item.added for the assistant message
                the first time any text delta arrives."""
                nonlocal message_opened, message_output_index, output_index
                if message_opened:
                    return
                message_opened = True
                message_output_index = output_index
                output_index += 1
                item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": message_output_index,
                    "item": item,
                })

            async def _emit_text_delta(delta_text: str) -> None:
                await _open_message_item()
                final_text_parts.append(delta_text)
                await _write_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": delta_text,
                    "logprobs": [],
                })

            async def _emit_tool_started(payload: Dict[str, Any]) -> str:
                """Emit response.output_item.added for a function_call.

                Returns the call_id so the matching completion event can
                reference it.  Prefer the real ``tool_call_id`` from the
                agent when available; fall back to a generated call id for
                safety in tests or older code paths.
                """
                nonlocal output_index, call_counter
                call_counter += 1
                call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{call_counter}"
                name = _normalize_external_tool_name(payload.get("name", ""))
                arguments_str = _external_tool_call_arguments_str(name, payload.get("arguments", {}))
                item = {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "in_progress",
                    "name": name,
                    "call_id": call_id,
                    "arguments": arguments_str,
                }
                idx = output_index
                output_index += 1
                pending_tool_calls.append({
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments_str,
                    "item_id": item["id"],
                    "output_index": idx,
                })
                emitted_items.append({
                    "type": "function_call",
                    "name": name,
                    "arguments": arguments_str,
                    "call_id": call_id,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": item,
                })
                return call_id

            async def _emit_tool_completed(payload: Dict[str, Any]) -> None:
                """Emit response.output_item.done (function_call) followed
                by response.output_item.added (function_call_output)."""
                nonlocal output_index
                call_id = payload.get("tool_call_id")
                result = payload.get("result", "")
                pending = None
                if call_id:
                    for i, p in enumerate(pending_tool_calls):
                        if p["call_id"] == call_id:
                            pending = pending_tool_calls.pop(i)
                            break
                if pending is None:
                    # Completion without a matching start — skip to avoid
                    # emitting orphaned done events.
                    return

                # function_call done
                done_item = {
                    "id": pending["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "name": pending["name"],
                    "call_id": pending["call_id"],
                    "arguments": pending["arguments"],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": pending["output_index"],
                    "item": done_item,
                })

                # function_call_output added (result)
                result_str = result if isinstance(result, str) else json.dumps(result)
                output_parts = [{"type": "input_text", "text": result_str}]
                output_item = {
                    "id": f"fco_{uuid.uuid4().hex[:24]}",
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                    "status": "completed",
                }
                idx = output_index
                output_index += 1
                emitted_items.append({
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": output_parts,
                })
                await _write_event("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": output_item,
                })
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": output_item,
                })

            # Main drain loop — thread-safe queue fed by agent callbacks.
            async def _dispatch(it) -> None:
                """Route a queue item to the correct SSE emitter.

                Plain strings are text deltas.  Tagged tuples with
                ``__tool_started__`` / ``__tool_completed__`` prefixes
                are tool lifecycle events.
                """
                if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                    tag, payload = it
                    if tag == "__tool_started__":
                        await _emit_tool_started(payload)
                    elif tag == "__tool_completed__":
                        await _emit_tool_completed(payload)
                    # Unknown tags are silently ignored (forward-compat).
                elif isinstance(it, str):
                    await _emit_text_delta(it)
                # Other types (non-string, non-tuple) are silently dropped.

            loop = asyncio.get_running_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain remaining
                        while True:
                            try:
                                item = stream_q.get_nowait()
                                if item is None:
                                    break
                                await _dispatch(item)
                                last_activity = time.monotonic()
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if item is None:  # EOS sentinel
                    break

                await _dispatch(item)
                last_activity = time.monotonic()

            # Pick up agent result + usage from the completed task
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
                # If the agent produced a final_response but no text
                # deltas were streamed (e.g. some providers only emit
                # the full response at the end), emit a single fallback
                # delta so Responses clients still receive a live text part.
                agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
                if agent_final and not final_text_parts:
                    await _emit_text_delta(agent_final)
                if agent_final and not final_response_text:
                    final_response_text = agent_final
                if isinstance(result, dict) and result.get("error") and not final_response_text:
                    agent_error = result["error"]
            except Exception as e:  # noqa: BLE001
                logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
                agent_error = str(e)

            # Close the message item if it was opened
            final_response_text = "".join(final_text_parts) or final_response_text
            if message_opened:
                await _write_event("response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": final_response_text,
                    "logprobs": [],
                })
                msg_done_item = {
                    "id": message_item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_response_text}
                    ],
                }
                await _write_event("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": message_output_index,
                    "item": msg_done_item,
                })

            # Always append a final message item in the completed
            # response envelope so clients that only parse the terminal
            # payload still see the assistant text.  This mirrors the
            # shape produced by _extract_output_items in the batch path.
            final_items: List[Dict[str, Any]] = list(emitted_items)
            final_items.append({
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_response_text or (agent_error or "")}
                ],
            })

            if agent_error:
                failed_env = _envelope("failed")
                failed_env["output"] = final_items
                failed_env["error"] = {"message": agent_error, "type": "server_error"}
                failed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_env,
                })
            else:
                completed_env = _envelope("completed")
                completed_env["output"] = final_items
                completed_env["usage"] = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                await _write_event("response.completed", {
                    "type": "response.completed",
                    "response": completed_env,
                })

                # Persist for future chaining / GET retrieval, mirroring
                # the batch path behavior.
                if store:
                    full_history = list(conversation_history)
                    full_history.append({"role": "user", "content": user_message})
                    if isinstance(result, dict) and result.get("messages"):
                        full_history.extend(result["messages"])
                    else:
                        full_history.append({"role": "assistant", "content": final_response_text})
                    self._response_store.put(response_id, {
                        "response": completed_env,
                        "conversation_history": full_history,
                        "instructions": instructions,
                        "session_id": session_id,
                    })
                    if conversation:
                        self._response_store.set_conversation(conversation, response_id)

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected — interrupt the agent so it stops
            # making upstream LLM calls, then cancel the task.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", response_id)

        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses — OpenAI Responses API format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                status=400,
            )

        raw_input = body.get("input")
        if raw_input is None:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        conversation = body.get("conversation")
        store = body.get("store", True)
        tool_choice = body.get("tool_choice")

        # Extract tools from request and mark them as from client
        tools = body.get("tools")
        if tools:
            for tool in tools:
                tool["_from_client"] = True

        # conversation and previous_response_id are mutually exclusive
        if conversation and previous_response_id:
            return web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

        # Resolve conversation name to latest response_id
        if conversation:
            previous_response_id = self._response_store.get_conversation(conversation)
            # No error if conversation doesn't exist yet — it's a new conversation

        # Normalize input to message list
        input_messages: List[Dict[str, str]] = []
        if isinstance(raw_input, str):
            input_messages = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            for item in raw_input:
                if isinstance(item, str):
                    input_messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = _normalize_chat_content(item.get("content", ""))
                    input_messages.append({"role": role, "content": content})
        else:
            return web.json_response(_openai_error("'input' must be a string or array"), status=400)

        # Accept explicit conversation_history from the request body.
        # This lets stateless clients supply their own history instead of
        # relying on server-side response chaining via previous_response_id.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, str]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored is None:
                return web.json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            # If no instructions provided, carry forward from previous
            if instructions is None:
                instructions = stored.get("instructions")

        # Append new input messages to history (all but the last become history)
        for msg in input_messages[:-1]:
            conversation_history.append(msg)

        # Last input message is the user_message
        last_input_msg = input_messages[-1] if input_messages else None
        user_message = (last_input_msg.get("content", "") or "") if last_input_msg else ""

        # Post-compaction continuation: pi/opencode sends a compaction summary
        # as the last assistant message with no following user message.
        if not user_message and last_input_msg and last_input_msg.get("role") == "assistant" and _is_post_compaction_assistant_message(last_input_msg.get("content", "")):
            user_message = _infer_intent_from_compaction_summary(last_input_msg.get("content", ""))
            conversation_history.append(last_input_msg)
            logger.info(
                "[api_server][responses] post-compaction continuation — inferred intent: %s",
                user_message[:200],
            )
        elif not user_message.strip() and last_input_msg and last_input_msg.get("role") == "user":
            # Empty user message at end of a tool cycle (pi/opencode "please continue" signal).
            # If the prior assistant message had tool_calls, the agent is mid-loop — let
            # it continue processing tool results rather than injecting a stale user prompt.
            if _prior_assistant_has_pending_tool_calls(input_messages[:-1]):
                logger.info(
                    "[api_server][responses] empty user message after assistant tool_calls — "
                    "continuing tool loop without injecting stale user message"
                )
                # user_message stays empty; tool-result continuation path handles it
            else:
                user_message = _find_last_nonempty_user_message(input_messages[:-1])
                if user_message:
                    logger.info(
                        "[api_server][responses] empty user message at end of cycle — "
                        "using last non-empty user message as continuation: %s",
                        user_message[:200],
                    )
                else:
                    return web.json_response(_openai_error("No user message found in input"), status=400)
        elif not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        # Truncation support
        if body.get("truncation") == "auto" and len(conversation_history) > 100:
            conversation_history = conversation_history[-100:]

        # NOTE: Message history is NOT pre-truncated. Agent's context compressor
        # handles overflow based on actual model's context window.

        # Reuse session from previous_response_id chain so the dashboard
        # groups the entire conversation under one session entry.
        session_id = stored_session_id or str(uuid.uuid4())

        model_name = body.get("model", self._model_name)
        role_cfg = _get_role_alias_config(model_name)
        role_hint = dict(role_cfg.get("hint") or {}) if role_cfg else None
        _toolset_mode = "auto"
        _provider_mode = False
        if model_name == "hermes-agentic-full":
            _toolset_mode = "full"
        elif model_name == "hermes-agentic-remote":
            _toolset_mode = "remote"
        elif model_name in ("hermes-code", "hermes-privacy", "claude-code-cli", "mimocode-cli") or (
            "/" in model_name and not role_cfg
        ):
            _provider_mode = True

        external_tool_mode = "none"
        if isinstance(tools, list) and tools:
            if model_name in ("hermes-code", "hermes-privacy"):
                external_tool_mode = "inband"
            else:
                external_tool_mode = "broker"

        # Extract model_name FIRST - before any agent creation  
        _model_name = body.get("model", self._model_name)
        
        # Handle hermes-swarm mode - use _model_name from request body
        swarm_mode = False
        swarm_model_pool = None
        _approx_tokens = 0
        if _provider_mode or _model_name == "hermes-swarm" or (role_cfg and role_cfg.get("mode") == "swarm"):
            try:
                from agent.model_metadata import estimate_request_tokens_rough
                _approx_tokens = estimate_request_tokens_rough(
                    conversation_history or [],
                    system_prompt=instructions or "",
                    tools=tools,
                )
            except Exception:
                _approx_tokens = 0
        if _model_name == "hermes-swarm" or (role_cfg and role_cfg.get("mode") == "swarm"):
            swarm_mode = True
            swarm_model_pool = await self._prepare_swarm_model_pool(
                system_prompt=instructions or "",
                conversation_history=conversation_history,
                user_message=user_message,
                tools=tools,
                estimated_tokens=_approx_tokens,
                routing_hint=role_hint,
            )

        stream = bool(body.get("stream", False))
        if stream:
            # Streaming branch — emit OpenAI Responses SSE events as the
            # agent runs so frontends can render text deltas and tool
            # calls in real time.  See _write_sse_responses for details.
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # None from the agent is a CLI box-close signal, not EOS.
                # Forwarding would kill the SSE stream prematurely; the
                # SSE writer detects completion via agent_task.done().
                if delta is not None:
                    _stream_q.put(delta)

            def _on_tool_progress(event_type, name, preview, args, **kwargs):
                """Queue non-start tool progress events if needed in future.

                The structured Responses stream uses ``tool_start_callback``
                and ``tool_complete_callback`` for exact call-id correlation,
                so progress events are currently ignored here.
                """
                return

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Queue a started tool for live function_call streaming."""
                _stream_q.put(("__tool_started__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Queue a completed tool result for live function_call_output streaming."""
                _stream_q.put(("__tool_completed__", {
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "arguments": function_args or {},
                    "result": function_result,
                }))

            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                estimated_tokens=_approx_tokens,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
                user_model=None,
            ))

            response_id = f"resp_{uuid.uuid4().hex[:28]}"
            created_at = int(time.time())

            return await self._write_sse_responses(
                request=request,
                response_id=response_id,
                model=_model_name,
                created_at=created_at,
                stream_q=_stream_q,
                agent_task=agent_task,
                agent_ref=agent_ref,
                conversation_history=conversation_history,
                user_message=user_message,
                instructions=instructions,
                conversation=conversation,
                store=store,
                session_id=session_id,
            )

        async def _compute_response():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                ephemeral_system_prompt=instructions,
                session_id=session_id,
                toolset_mode=_toolset_mode,
                provider_mode=_provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                estimated_tokens=_approx_tokens,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
                user_model=None,
                request_model=model_name,
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(
                body,
                keys=["input", "instructions", "previous_response_id", "conversation", "model", "tools"],
            )
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_response)
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_response()
            except Exception as e:
                logger.error("Error running agent for responses: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        created_at = int(time.time())

        if result.get("tool_calls_pending"):
            last_assistant = None
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
                    last_assistant = msg
                    break

            output_items: List[Dict[str, Any]] = []
            for tc in _enrich_client_tool_calls((last_assistant or {}).get("tool_calls", [])):
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                output_items.append({
                    "type": "function_call",
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                    "call_id": tc.get("id") or tc.get("call_id", ""),
                })

            response_data = {
                "id": response_id,
                "object": "response",
                "status": "completed",
                "created_at": created_at,
                "model": body.get("model", self._model_name),
                "output": output_items,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

            if store:
                full_history = list(conversation_history)
                full_history.append({"role": "user", "content": user_message})
                full_history.extend(result.get("messages", []))
                self._response_store.put(response_id, {
                    "response": response_data,
                    "conversation_history": full_history,
                    "instructions": instructions,
                    "session_id": session_id,
                })
                if conversation:
                    self._response_store.set_conversation(conversation, response_id)

            return web.json_response(response_data)

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        # Build the full conversation history for storage
        # (includes tool calls from the agent run)
        full_history = list(conversation_history)
        full_history.append({"role": "user", "content": user_message})
        # Add agent's internal messages if available
        agent_messages = result.get("messages", [])
        if agent_messages:
            full_history.extend(agent_messages)
        else:
            full_history.append({"role": "assistant", "content": final_response})

        # Build output items (includes tool calls + final message)
        output_items = self._extract_output_items(result)

        response_data = {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "created_at": created_at,
            "model": body.get("model", self._model_name),
            "output": output_items,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        # Store the complete response object for future chaining / GET retrieval
        if store:
            self._response_store.put(response_id, {
                "response": response_data,
                "conversation_history": full_history,
                "instructions": instructions,
                "session_id": session_id,
            })
            # Update conversation mapping so the next request with the same
            # conversation name automatically chains to this response
            if conversation:
                self._response_store.set_conversation(conversation, response_id)

        return web.json_response(response_data)

    # ------------------------------------------------------------------
    # GET / DELETE response endpoints
    # ------------------------------------------------------------------

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} — retrieve a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    # Check cron module availability once (not per-request)
    _CRON_AVAILABLE = False
    try:
        from cron.jobs import (
            list_jobs as _cron_list,
            get_job as _cron_get,
            create_job as _cron_create,
            update_job as _cron_update,
            remove_job as _cron_remove,
            pause_job as _cron_pause,
            resume_job as _cron_resume,
            trigger_job as _cron_trigger,
        )
        # Wrap as staticmethod to prevent descriptor binding — these are plain
        # module functions, not instance methods.  Without this, self._cron_*()
        # injects ``self`` as the first positional argument and every call
        # raises TypeError.
        _cron_list = staticmethod(_cron_list)
        _cron_get = staticmethod(_cron_get)
        _cron_create = staticmethod(_cron_create)
        _cron_update = staticmethod(_cron_update)
        _cron_remove = staticmethod(_cron_remove)
        _cron_pause = staticmethod(_cron_pause)
        _cron_resume = staticmethod(_cron_resume)
        _cron_trigger = staticmethod(_cron_trigger)
        _CRON_AVAILABLE = True
    except ImportError:
        pass

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update — prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    def _check_jobs_available(self) -> Optional["web.Response"]:
        """Return error response if cron module isn't available."""
        if not self._CRON_AVAILABLE:
            return web.json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: "web.Request") -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            return job_id, web.json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs — list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in ("true", "1")
            jobs = self._cron_list(include_disabled=include_disabled)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs — create a new cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            deliver = body.get("deliver", "local")
            skills = body.get("skills")
            repeat = body.get("repeat")

            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
            }
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = self._cron_create(**kwargs)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} — get a single cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_get(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} — update a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return web.json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            job = self._cron_update(job_id, sanitized)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} — delete a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            success = self._cron_remove(job_id)
            if not success:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause — pause a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_pause(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume — resume a paused cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_resume(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run — trigger immediate execution."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = self._cron_trigger(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the full output item array from the agent's messages.

        Walks *result["messages"]* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    # Normalise MCP-prefixed tool names (mcp__hermes-tools__read → read)
                    # so clients never see internal provider naming conventions.
                    _raw_name = func.get("name", "")
                    _norm_name = _normalize_external_tool_name(_raw_name)
                    items.append({
                        "type": "function_call",
                        "name": _norm_name,
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = result.get("error", "(No response generated)")

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_gen_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        toolset_mode: str = "auto",
        provider_mode: bool = False,
        swarm_mode: bool = False,
        swarm_model_pool = None,
        estimated_tokens: int = 0,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        external_tool_mode: str = "none",
        user_model: Optional[str] = None,
        request_model: str = "",
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.
        """
        loop = asyncio.get_running_loop()

        logging.debug(f"[API_SERVER] _run_agent called: swarm_mode={swarm_mode}, swarm_model_pool={swarm_model_pool}")

        def _run():
            generated_tool_calls: List[Dict[str, Any]] = []

            def _wrapped_tool_gen_callback(tool_name: str, call_id: Optional[str] = None, arguments: str = ""):
                if (not isinstance(call_id, str) or not call_id.strip()) and not str(arguments or "").strip():
                    if tool_gen_callback:
                        try:
                            tool_gen_callback(tool_name, call_id=None, arguments="")
                        except Exception:
                            pass
                    return
                safe_tool_name = _normalize_external_tool_name(tool_name)
                generated_tool_calls.append(_enrich_client_tool_call({
                    "id": call_id or "",
                    "type": "function",
                    "function": {
                        "name": safe_tool_name,
                        "arguments": _external_tool_call_arguments_str(safe_tool_name, arguments),
                    },
                }))
                if tool_gen_callback:
                    try:
                        tool_gen_callback(safe_tool_name, call_id=call_id, arguments=_external_tool_call_arguments_str(safe_tool_name, arguments))
                    except Exception:
                        pass

            _t_create = time.time()
            # MiMoCode CLI dispatch — intercept before standard agent creation
            if provider_mode and request_model in ("mimocode-cli",):
                try:
                    from hermes_cli.auth import resolve_external_process_provider_credentials
                    _mc_creds = resolve_external_process_provider_credentials("mimocode-cli")
                    from agent.mimocode_code_client import MiMoCodeClient
                    _mc_client = MiMoCodeClient(
                        api_key=_mc_creds.get("api_key", "mimocode-cli"),
                        base_url=_mc_creds.get("base_url", "mimocode://codex"),
                        command=_mc_creds.get("command"),
                        args=_mc_creds.get("args"),
                    )
                    messages = [{"role": "user", "content": user_message}]
                    if conversation_history:
                        for msg in conversation_history:
                            messages.insert(0, msg)
                    if ephemeral_system_prompt:
                        messages.insert(0, {"role": "system", "content": ephemeral_system_prompt})
                    result_obj = _mc_client._create_chat_completion(
                        model=request_model,
                        messages=messages,
                        tools=tools,
                    )
                    result = {
                        "final_response": result_obj.choices[0].message.content or "",
                        "tool_calls": [
                            {"name": _normalize_external_tool_name(tc["function"]["name"]), "arguments": tc["function"]["arguments"]}
                            for tc in (result_obj.choices[0].message.tool_calls or [])
                        ],
                        "messages": [],
                    }
                    usage = {
                        "input_tokens": getattr(result_obj.usage, "prompt_tokens", 0),
                        "output_tokens": getattr(result_obj.usage, "completion_tokens", 0),
                        "total_tokens": getattr(result_obj.usage, "total_tokens", 0),
                    }
                    logger.info("[hermes-code] mimocode-cli completed: text_len=%d", len(result.get("final_response", "")))
                    return result, usage
                except Exception as exc:
                    logger.warning("[hermes-code] mimocode-cli error: %s", exc)
                    return {"final_response": "", "tool_calls": [], "messages": []}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
                tool_gen_callback=_wrapped_tool_gen_callback,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
                toolset_mode=toolset_mode,
                provider_mode=provider_mode,
                swarm_mode=swarm_mode,
                swarm_model_pool=swarm_model_pool,
                estimated_tokens=estimated_tokens,
                tools=tools,
                tool_choice=tool_choice,
                external_tool_mode=external_tool_mode,
                user_model=user_model,
            )
            _t_created = time.time()
            logger.info("[timing] _create_agent: %.3fs", _t_created - _t_create)
            if agent_ref is not None:
                agent_ref[0] = agent
            _t_conv = time.time()
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                task_id="default",
            )
            _t_done = time.time()
            logger.info("[timing] run_conversation: %.3fs (agent_create: %.3fs, total _run: %.3fs)",
                _t_done - _t_conv, _t_created - _t_create, _t_done - _t_create)
            if (
                swarm_mode
                and swarm_model_pool
                and not stream_delta_callback
                and isinstance(result, dict)
                and str(result.get("final_response") or "").strip()
            ):
                routing_hint = swarm_model_pool.get("routing_hint") or {}
                should_verify = os.getenv("HERMES_SWARM_ENABLE_VERIFIER", "true").strip().lower() not in {"0", "false", "no"}
                should_verify = should_verify and str(routing_hint.get("recommended_tier") or "") in {"balanced", "premium"}
                if should_verify:
                    try:
                        verification = self._run_swarm_verifier_sync(
                            system_prompt=ephemeral_system_prompt or "",
                            conversation_history=conversation_history,
                            user_message=user_message,
                            candidate_response=str(result.get("final_response") or ""),
                            swarm_model_pool=swarm_model_pool,
                        )
                        if str(verification.get("verdict") or "").strip().lower() == "revise":
                            revised = str(verification.get("revised_response") or "").strip()
                            if revised:
                                logger.info("[api_server] swarm verifier revised final response")
                                if not isinstance(result, dict):
                                    result = {"final_response": revised, "messages": []}
                                else:
                                    result["final_response"] = revised
                                if isinstance(result.get("messages"), list):
                                    for msg in reversed(result["messages"]):
                                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                                            msg["content"] = revised
                                            break
                                meta = result.get("meta")
                                if not isinstance(meta, dict):
                                    meta = {}
                                    result["meta"] = meta
                                meta["swarm_verifier"] = verification
                        else:
                            logger.info("[api_server] swarm verifier accepted final response")
                    except Exception as exc:
                        logger.warning("[api_server] swarm verifier failed: %s", exc)
            if (
                isinstance(result, dict)
                and generated_tool_calls
                and getattr(agent, "_external_tool_mode", "none") in ("broker", "inband")
            ):
                messages = list(result.get("messages", []))
                has_tool_calls = any(
                    isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls")
                    for msg in messages
                )
                if not has_tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": result.get("final_response", "") or "",
                        "tool_calls": _enrich_client_tool_calls(generated_tool_calls),
                    })
                    result["messages"] = messages
                result["tool_calls_pending"] = True
                result["finish_reason"] = "tool_calls"
                result["completed"] = False
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    def _runtime_kwargs_for_model(self, model: str) -> tuple[Dict[str, Any], str]:
        return _runtime_kwargs_for_model_id(model)

    def _run_swarm_scout_sync(
        self,
        *,
        system_prompt: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        swarm_model_pool: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pool = swarm_model_pool or {}
        # Scout model selection: prefer cheapest model with credentials that can
        # safely handle the estimated token count. When context is small, gpt-5-mini
        # (cheap/Copilot) is selected. When context is large, MiniMax-M2.7 (200K)
        # passes the context filter and is selected instead.
        estimated_tokens = swarm_model_pool.get("estimated_tokens", 0)
        preferred_models = [os.getenv("HERMES_SWARM_SCOUT_MODEL", "").strip()]
        preferred_models.extend(pool.get("scout_fallbacks", []))
        scout_model = ""
        for candidate in preferred_models:
            if candidate and _swarm_model_is_available(candidate):
                # Skip models that can't hold the estimated context (prevents 413 on scout itself)
                if not _model_safe_for_tokens(candidate, estimated_tokens):
                    logger.info("[api_server] scout skipping %s (context %d > safe limit)", candidate, estimated_tokens)
                    continue
                scout_model = candidate
                break
        if not scout_model:
            raise RuntimeError("No available scout model with credentials")

        logger.warning(
            "[API_SERVER] swarm scout selected model=%s primary=%s scout_fallbacks=%s estimated_tokens=%s",
            scout_model,
            pool.get("primary", ""),
            pool.get("scout_fallbacks", []),
            estimated_tokens,
        )

        runtime_kwargs, scout_model_name = self._runtime_kwargs_for_model(scout_model)
        from run_agent import AIAgent

        scout_prompt = (
            "Classify this task for routing. Return ONLY compact JSON with keys: "
            "task_type, recommended_tier, needs_instruction_following, needs_repo_reasoning, "
            "needs_bug_judgement, action_mode, confidence, reason. "
            "recommended_tier must be one of cheap, primary, balanced, premium. "
            "action_mode must be one of answer_only, plan_only, execute_with_tools. "
            "Use plan_only when the user asks to compare, plan, discuss, or recommend without explicitly asking to change files. "
            "Use answer_only for simple factual/meta answers. Use execute_with_tools only when the user asks to implement, continue, test, deploy, commit, inspect files, or otherwise perform work. "
            "Use premium for repo review, debugging, implementation, or architectural tasks. "
            "Use balanced for instruction-sensitive workspace analysis. "
            "If AGENTS.md instructions are quoted inline, treat them as authoritative and do not say the file is missing. "
            "Do not solve the task itself. Do not call tools.\n\n"
            + _summarize_swarm_messages(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                user_message=user_message,
            )
        )
        if tools:
            scout_prompt += f"\n\nTOOLS_PRESENT: {len(tools)}"

        agent = AIAgent(
            model=scout_model_name,
            **runtime_kwargs,
            max_iterations=2,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt="You are a routing classifier.",
            enabled_toolsets=[],
            session_id=f"swarm-scout-{uuid.uuid4().hex[:8]}",
            platform="api_server",
            session_db=None,
            skip_memory=True,
            skip_context_files=True,
            tools=[],
        )
        result = agent.run_conversation(
            user_message=scout_prompt,
            conversation_history=[],
            task_id="swarm_scout",
        )
        response_text = _extract_agent_result_text(result)
        if not response_text:
            raise RuntimeError("Empty scout response")
        logger.info("[api_server] swarm scout raw response: %s", response_text[:400])
        parsed = _parse_loose_json_object(response_text)
        parsed["source"] = "scout"
        parsed["model"] = scout_model
        return parsed

    def _run_swarm_verifier_sync(
        self,
        *,
        system_prompt: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        candidate_response: str = "",
        swarm_model_pool: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pool = swarm_model_pool or {}
        preferred_models = [
            os.getenv("HERMES_SWARM_VERIFY_MODEL", "").strip(),
            "openai/gpt-5.5",
            "openai/gpt-5.3-codex",
            "google/gemini-2.5-flash",
            "openai/gpt-5-mini",
        ]
        preferred_models.extend(pool.get("fallbacks", []))
        verify_model = ""
        primary = str(pool.get("primary") or "").strip()
        for candidate in preferred_models:
            if candidate and candidate != primary and _swarm_model_is_available(candidate):
                verify_model = candidate
                break
        if not verify_model:
            raise RuntimeError("No available verifier model with credentials")

        runtime_kwargs, verify_model_name = self._runtime_kwargs_for_model(verify_model)
        from run_agent import AIAgent

        verifier_prompt = (
            "Review the candidate answer for correctness and grounding. Return ONLY compact JSON with keys: "
            "verdict, issues, revised_response. verdict must be one of ok or revise. "
            "Revise if the answer hallucinates missing files/access, ignores supplied AGENTS instructions, "
            "or misses an obvious higher-severity issue that is visible from the provided context. "
            "Treat missing AGENTS.md on disk as non-fatal when equivalent instructions are quoted inline. "
            "issues must be an array of short strings. If verdict is ok, revised_response should be empty.\n\n"
            + _summarize_swarm_messages(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                user_message=user_message,
            )
            + f"\n\nCANDIDATE_RESPONSE:\n{candidate_response[:12000]}"
        )

        agent = AIAgent(
            model=verify_model_name,
            **runtime_kwargs,
            max_iterations=2,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt="You are a strict answer verifier.",
            enabled_toolsets=[],
            session_id=f"swarm-verify-{uuid.uuid4().hex[:8]}",
            platform="api_server",
            session_db=None,
            skip_memory=True,
            skip_context_files=True,
            tools=[],
        )
        result = agent.run_conversation(
            user_message=verifier_prompt,
            conversation_history=[],
            task_id="swarm_verify",
        )
        response_text = _extract_agent_result_text(result)
        if not response_text:
            raise RuntimeError("Empty verifier response")
        logger.info("[api_server] swarm verifier raw response: %s", response_text[:400])
        if response_text.lower().startswith("invalid api response after"):
            return {"verdict": "ok", "issues": [response_text[:120]], "revised_response": "", "model": verify_model}
        parsed = _parse_loose_json_object(response_text)
        parsed["model"] = verify_model
        return parsed

    async def _prepare_swarm_model_pool(
        self,
        *,
        system_prompt: str = "",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_message: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        estimated_tokens: int = 0,
        routing_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pool = _build_swarm_model_pool(estimated_tokens=estimated_tokens, routing_hint=routing_hint)
        heuristic_hint = _heuristic_swarm_routing_hint(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            user_message=user_message,
            tools=tools,
            estimated_tokens=estimated_tokens,
        )
        merged_hint = dict(heuristic_hint)
        if routing_hint:
            merged_hint.update(routing_hint)
        pool["routing_hint"] = merged_hint

        should_scout = os.getenv("HERMES_SWARM_ENABLE_SCOUT", "true").strip().lower() not in {"0", "false", "no"}
        action_mode = str(merged_hint.get("action_mode") or "").strip().lower()
        # Explicit Roo/role aliases such as roo-architect and roo-ask already
        # constrain the action mode and tier.  Do not spend an extra model call
        # scouting those plan-only / answer-only turns; the whole point is to
        # make those stages cheap and predictable.
        if action_mode in {"plan_only", "answer_only"} and merged_hint.get("recommended_tier") == "cheap":
            should_scout = False
        if should_scout and merged_hint.get("recommended_tier") in {"balanced", "premium"}:
            try:
                loop = asyncio.get_running_loop()
                scout_hint = await loop.run_in_executor(
                    None,
                    lambda: self._run_swarm_scout_sync(
                        system_prompt=system_prompt,
                        conversation_history=conversation_history,
                        user_message=user_message,
                        tools=tools,
                        swarm_model_pool=pool,
                    ),
                )
                if isinstance(scout_hint, dict):
                    scout_merged_hint = dict(pool.get("routing_hint") or {})
                    scout_merged_hint.update(scout_hint)
                    if str(scout_merged_hint.get("action_mode") or "") not in {"answer_only", "plan_only", "execute_with_tools"}:
                        scout_merged_hint["action_mode"] = merged_hint.get("action_mode", "answer_only")
                    if scout_hint.get("recommended_tier") == "cheap" and merged_hint.get("recommended_tier") == "premium":
                        # The scout may correctly identify a meta/classification
                        # request as cheap, but it must not downshift tasks that
                        # heuristics identified as repo/debug/implementation.
                        scout_merged_hint["recommended_tier"] = "premium"
                    pool["routing_hint"] = scout_merged_hint
            except Exception as exc:
                logger.warning("[api_server] swarm scout failed, falling back to heuristics: %s", exc)
        return pool

    # ------------------------------------------------------------------
    # /v1/runs — structured event streaming
    # ------------------------------------------------------------------

    _MAX_CONCURRENT_RUNS = 10  # Prevent unbounded resource allocation
    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            q = self._run_streams.get(run_id)
            if q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

        def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            ts = time.time()
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "preview": preview,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                })
            # _thinking and subagent_progress are intentionally not forwarded

        return _callback

    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs — start an agent run, return run_id immediately."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Enforce concurrency limit
        if len(self._run_streams) >= self._MAX_CONCURRENT_RUNS:
            return web.json_response(
                _openai_error(f"Too many concurrent runs (max {self._MAX_CONCURRENT_RUNS})", code="rate_limit_exceeded"),
                status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)

        raw_input = body.get("input")
        if not raw_input:
            return web.json_response(_openai_error("Missing 'input' field"), status=400)

        # DEBUG: log what pi sends so we can understand blank continuation messages
        _last_run_msg_dbg = raw_input[-1] if isinstance(raw_input, list) else raw_input
        logger.info(
            "[api_server][runs] last_msg role=%s content_len=%s content_preview=%s",
            _last_run_msg_dbg.get("role") if isinstance(_last_run_msg_dbg, dict) else "(str)",
            len(str(_last_run_msg_dbg.get("content", "") or "")) if isinstance(_last_run_msg_dbg, dict) else len(str(_last_run_msg_dbg)),
            repr((str(_last_run_msg_dbg.get("content", "") or ""))[:200]) if isinstance(_last_run_msg_dbg, dict) else repr(str(_last_run_msg_dbg)[:200]),
        )

        last_run_msg = raw_input[-1] if isinstance(raw_input, list) else None
        user_message = raw_input if isinstance(raw_input, str) else ((last_run_msg.get("content", "") or "") if isinstance(last_run_msg, dict) else "")

        # Post-compaction continuation (runs API path)
        if not user_message and isinstance(last_run_msg, dict) and last_run_msg.get("role") == "assistant" and _is_post_compaction_assistant_message(last_run_msg.get("content", "")):
            user_message = _infer_intent_from_compaction_summary(last_run_msg.get("content", ""))
            logger.info(
                "[api_server][runs] post-compaction continuation — inferred intent: %s",
                user_message[:200],
            )
        elif not user_message.strip() and isinstance(last_run_msg, dict) and last_run_msg.get("role") == "user":
            # Empty user message at end of a tool cycle (pi/opencode "please continue" signal).
            # If the prior assistant message had tool_calls, the agent is mid-loop — let
            # it continue processing tool results rather than injecting a stale user prompt.
            prior_msgs = raw_input[:-1] if isinstance(raw_input, list) else []
            if _prior_assistant_has_pending_tool_calls(prior_msgs):
                logger.info(
                    "[api_server][runs] empty user message after assistant tool_calls — "
                    "continuing tool loop without injecting stale user message"
                )
                # user_message stays empty; tool-result continuation path handles it
            else:
                user_message = _find_last_nonempty_user_message(prior_msgs)
                if user_message:
                    logger.info(
                        "[api_server][runs] empty user message at end of cycle — "
                        "using last non-empty user message as continuation: %s",
                        user_message[:200],
                    )
                else:
                    return web.json_response(_openai_error("No user message found in input"), status=400)
        elif not user_message:
            return web.json_response(_openai_error("No user message found in input"), status=400)

        run_id = f"run_{uuid.uuid4().hex}"
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
        self._run_streams[run_id] = q
        self._run_streams_created[run_id] = time.time()

        event_cb = self._make_run_event_callback(run_id, loop)

        # Also wire stream_delta_callback so message.delta events flow through
        def _text_cb(delta: Optional[str]) -> None:
            if delta is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, {
                    "event": "message.delta",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "delta": delta,
                })
            except Exception:
                pass

        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")

        # Accept explicit conversation_history from the request body.
        # Precedence: explicit conversation_history > previous_response_id.
        conversation_history: List[Dict[str, str]] = []
        raw_history = body.get("conversation_history")
        if raw_history:
            if not isinstance(raw_history, list):
                return web.json_response(
                    _openai_error("'conversation_history' must be an array of message objects"),
                    status=400,
                )
            for i, entry in enumerate(raw_history):
                if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                    return web.json_response(
                        _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                        status=400,
                    )
                conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
            if previous_response_id:
                logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

        stored_session_id = None
        if not conversation_history and previous_response_id:
            stored = self._response_store.get(previous_response_id)
            if stored:
                conversation_history = list(stored.get("conversation_history", []))
                stored_session_id = stored.get("session_id")
                if instructions is None:
                    instructions = stored.get("instructions")

        # When input is a multi-message array, extract all but the last
        # message as conversation history (the last becomes user_message).
        # Only fires when no explicit history was provided.
        if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
            for msg in raw_input[:-1]:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    content = msg["content"]
                    if isinstance(content, list):
                        # Flatten multi-part content blocks to text
                        content = " ".join(
                            part.get("text", "") for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        )
                    conversation_history.append({"role": msg["role"], "content": str(content)})

        session_id = body.get("session_id") or stored_session_id or run_id
        ephemeral_system_prompt = instructions

        async def _run_and_close():
            try:
                agent = self._create_agent(
                    ephemeral_system_prompt=ephemeral_system_prompt,
                    session_id=session_id,
                    stream_delta_callback=_text_cb,
                    tool_progress_callback=event_cb,
                    provider_mode=provider_mode,
                    swarm_mode=swarm_mode,
                    swarm_model_pool=swarm_model_pool,
                )
                def _run_sync():
                    r = agent.run_conversation(
                        user_message=user_message,
                        conversation_history=conversation_history,
                        task_id="default",
                    )
                    u = {
                        "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                        "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                    }
                    return r, u

                result, usage = await asyncio.get_running_loop().run_in_executor(None, _run_sync)
                final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                q.put_nowait({
                    "event": "run.completed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "output": final_response,
                    "usage": usage,
                })
            except Exception as exc:
                logger.exception("[api_server] run %s failed", run_id)
                try:
                    q.put_nowait({
                        "event": "run.failed",
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "error": str(exc),
                    })
                except Exception:
                    pass
            finally:
                # Sentinel: signal SSE stream to close
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

        task = asyncio.create_task(_run_and_close())
        try:
            self._background_tasks.add(task)
        except TypeError:
            pass
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)

        return web.json_response({"run_id": run_id, "status": "started"}, status=202)

    async def _handle_run_events(self, request: "web.Request") -> "web.StreamResponse":
        """GET /v1/runs/{run_id}/events — SSE stream of structured agent lifecycle events."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        run_id = request.match_info["run_id"]

        # Allow subscribing slightly before the run is registered (race condition window)
        for _ in range(20):
            if run_id in self._run_streams:
                break
            await asyncio.sleep(0.05)
        else:
            return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

        q = self._run_streams[run_id]

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if event is None:
                    # Run finished — send final SSE comment and close
                    await response.write(b": stream closed\n\n")
                    break
                payload = f"data: {json.dumps(event)}\n\n"
                await response.write(payload.encode())
        except Exception as exc:
            logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
        finally:
            self._run_streams.pop(run_id, None)
            self._run_streams_created.pop(run_id, None)

        return response

    async def _sweep_orphaned_runs(self) -> None:
        """Periodically clean up run streams that were never consumed."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = [
                run_id
                for run_id, created_at in list(self._run_streams_created.items())
                if now - created_at > self._RUN_STREAM_TTL
            ]
            for run_id in stale:
                logger.debug("[api_server] sweeping orphaned run %s", run_id)
                self._run_streams.pop(run_id, None)
                self._run_streams_created.pop(run_id, None)

    async def _handle_rerank(self, request: "web.Request") -> "web.Response":
        """POST /v1/rerank — Cohere-compatible reranking via round-robin provider pool.

        Accepts:
            {
                "model": str (optional — overrides per-provider default),
                "query": str,
                "documents": [str | {"text": str}],
                "top_n": int (optional)
            }

        Returns Cohere-wire-format response:
            {"model": str, "results": [{"index": int, "relevance_score": float}]}

        Providers (Cohere, Voyage AI) round-robin; 429 triggers dynamic per-provider
        cooldown honouring Retry-After (defaults to 60 s for rate-limit windows).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(
                _openai_error("Invalid JSON in request body"), status=400
            )

        query = body.get("query")
        documents = body.get("documents")
        top_n = body.get("top_n")
        model_override = body.get("model") or None

        if not isinstance(query, str) or not query.strip():
            return web.json_response(
                _openai_error("'query' is required and must be a non-empty string"), status=400
            )
        if not isinstance(documents, list) or not documents:
            return web.json_response(
                _openai_error("'documents' is required and must be a non-empty list"), status=400
            )
        if top_n is not None:
            try:
                top_n = int(top_n)
            except (TypeError, ValueError):
                return web.json_response(
                    _openai_error("'top_n' must be an integer"), status=400
                )

        # Accept plain strings and {"text": str} dicts
        normalised_docs: List[Any] = []
        for doc in documents:
            if isinstance(doc, str):
                normalised_docs.append(doc)
            elif isinstance(doc, dict) and isinstance(doc.get("text"), str):
                normalised_docs.append(doc["text"])
            else:
                return web.json_response(
                    _openai_error('Each document must be a string or {"text": str} object'),
                    status=400,
                )

        try:
            result = await _rerank_router.rerank(
                query=query,
                documents=normalised_docs,
                top_n=top_n,
                model_override=model_override,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "No reranker providers configured" in msg:
                return web.json_response(
                    _openai_error(msg, err_type="configuration_error", code="no_reranker_providers"),
                    status=503,
                )
            logger.error("[api_server] rerank failed: %s", exc)
            return web.json_response(
                _openai_error(f"Reranking failed: {exc}", err_type="server_error", code="reranker_error"),
                status=502,
            )
        except Exception as exc:
            logger.exception("[api_server] unexpected rerank error")
            return web.json_response(
                _openai_error("Unexpected reranker error", err_type="server_error"),
                status=500,
            )

        return web.json_response(result)

    async def _handle_tool_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/sessions/{session_id}/tool_responses — ingest client tool results."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = (request.match_info.get("session_id") or "").strip()
        if not session_id:
            return web.json_response({"error": "Missing session_id in path"}, status=400)

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        call_id = body.get("call_id")
        status = body.get("status")
        result = body.get("result")

        if not isinstance(call_id, str) or not call_id.strip():
            return web.json_response({"error": "Missing or invalid 'call_id'"}, status=400)
        if status not in ("ok", "error"):
            return web.json_response({"error": "'status' must be 'ok' or 'error'"}, status=400)

        try:
            from gateway.platforms import tool_call_hub
            tool_call_hub.set_response(session_id, call_id, status, result)
            logger.info(
                "[TIMING] T+%.3fs tool_response POST received session=%s call_id=%s status=%s result_len=%d",
                time.monotonic() - _req_start_global, session_id, call_id, status,
                len(str(result or "")),
            )
        except Exception as e:
            logger.error("Failed to set tool response: %s", e, exc_info=True)
            return web.json_response({"error": "Internal server error"}, status=500)

        return web.json_response({"ok": True}, status=200)

    async def _handle_pending_tool_calls(self, request: "web.Request") -> "web.Response":
        """GET /v1/sessions/{session_id}/pending-tool-calls — long-poll for pending tool calls.

        The connected Mac daemon polls this endpoint to discover when the model
        has emitted a tool_call. It should execute the tool locally and POST the
        result to /v1/sessions/{session_id}/tool_responses.

        Long-polls up to `wait` seconds (default 25) for at least one pending
        call, OR returns immediately if one is already pending. Returns:
            {"tool_calls": [{"call_id": "...", "name": "...", "arguments": {...}, "tool_name": "..."}, ...]}
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = (request.match_info.get("session_id") or "").strip()
        if not session_id:
            return web.json_response({"error": "Missing session_id in path"}, status=400)

        try:
            wait_s = float(request.query.get("wait", "25"))
        except ValueError:
            wait_s = 25.0
        wait_s = min(max(wait_s, 0.0), 60.0)

        from gateway.platforms import tool_call_hub

        deadline = time.monotonic() + wait_s
        _poll_start = time.monotonic()
        _poll_iterations = 0
        while time.monotonic() < deadline:
            _poll_iterations += 1
            from gateway.platforms.tool_call_hub import _hub as _tool_hub
            with _tool_hub._lock:
                ses = _tool_hub._pending.get(session_id, {})
                pending = [
                    {"call_id": cid, "name": p.tool_name, "tool_name": p.tool_name, "arguments": dict(p.arguments or {})}
                    for cid, p in list(ses.items())
                ]
            if pending:
                logger.info(
                    "[TIMING] T+%.3fs pending-tool-calls RETURNED session=%s found=%d iters=%d elapsed=%.3fs",
                    time.monotonic() - _req_start_global, session_id, len(pending), _poll_iterations,
                    time.monotonic() - _poll_start,
                )
                return web.json_response({"tool_calls": pending}, status=200)
            await asyncio.sleep(0.1)

        logger.info(
            "[TIMING] T+%.3fs pending-tool-calls TIMEOUT session=%s iters=%d waited=%.1fs",
            time.monotonic() - _req_start_global, session_id, _poll_iterations, wait_s,
        )
        return web.json_response({"tool_calls": []}, status=200)

        return web.json_response({"ok": True}, status=200)


    async def _handle_audio_speech(self, request: "web.Request") -> "web.Response":
        """POST /v1/audio/speech — OpenAI-compatible TTS.

        Accepts:
            { "model": "tts-1|mimo-v2-tts|...", "input": "text", "voice": "alloy",
              "response_format": "mp3|wav|opus|aac|flac|pcm", "speed": 1.0 }

        Provider selection order:
          1. Xiaomi MiMo (XIAOMI_API_KEY) — free during promotional period
          2. OpenAI (OPENAI_API_KEY) — standard TTS-1 / TTS-1-HD

        Returns raw audio bytes with the appropriate Content-Type header.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status=400
            )

        text_input = body.get("input", "")
        if not isinstance(text_input, str) or not text_input.strip():
            return web.json_response(
                {"error": {"message": "'input' is required", "type": "invalid_request_error"}}, status=400
            )

        model_req = str(body.get("model") or "tts-1").strip()
        voice = str(body.get("voice") or "alloy").strip()
        response_format = str(body.get("response_format") or "mp3").strip().lower()
        speed = float(body.get("speed") or 1.0)

        _AUDIO_MIME = {
            "mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus",
            "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/pcm",
        }
        content_type = _AUDIO_MIME.get(response_format, "audio/mpeg")

        loop = asyncio.get_running_loop()

        def _tts_call() -> bytes:
            import urllib.request as _ur
            import urllib.error as _ue

            # ── Try Xiaomi MiMo TTS ─────────────────────────────────────────
            xiaomi_key = os.getenv("XIAOMI_API_KEY", "").strip()
            xiaomi_base = os.getenv("XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1").rstrip("/")
            if xiaomi_key:
                # MiMo TTS model name: prefer explicit mimo model, else default
                tts_model = model_req if _xiaomi_model_is_tts(model_req.split("/")[-1]) else "mimo-v2-tts"
                payload = json.dumps({
                    "model": tts_model,
                    "input": text_input,
                    "voice": voice,
                    "response_format": response_format,
                    "speed": speed,
                }).encode()
                req = _ur.Request(
                    f"{xiaomi_base}/audio/speech",
                    data=payload,
                    headers={"Authorization": f"Bearer {xiaomi_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with _ur.urlopen(req, timeout=60) as resp:
                        return resp.read()
                except _ue.HTTPError as exc:
                    logger.warning("[audio/speech] Xiaomi TTS failed: HTTP %d %s", exc.code, exc.read()[:200])
                except Exception as exc:
                    logger.warning("[audio/speech] Xiaomi TTS error: %s", exc)

            # ── Try OpenAI TTS ──────────────────────────────────────────────
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if openai_key:
                oai_model = model_req if model_req in ("tts-1", "tts-1-hd", "gpt-4o-mini-tts") else "tts-1"
                payload = json.dumps({
                    "model": oai_model,
                    "input": text_input,
                    "voice": voice,
                    "response_format": response_format,
                    "speed": speed,
                }).encode()
                req = _ur.Request(
                    f"{openai_base}/audio/speech",
                    data=payload,
                    headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with _ur.urlopen(req, timeout=60) as resp:
                        return resp.read()
                except _ue.HTTPError as exc:
                    logger.warning("[audio/speech] OpenAI TTS failed: HTTP %d %s", exc.code, exc.read()[:200])
                except Exception as exc:
                    logger.warning("[audio/speech] OpenAI TTS error: %s", exc)

            raise RuntimeError("No TTS provider available. Set XIAOMI_API_KEY or OPENAI_API_KEY.")

        try:
            audio_bytes = await loop.run_in_executor(None, _tts_call)
        except RuntimeError as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "service_unavailable"}}, status=503
            )
        except Exception as exc:
            logger.error("[audio/speech] unexpected error: %s", exc, exc_info=True)
            return web.json_response(
                {"error": {"message": "TTS generation failed", "type": "internal_server_error"}}, status=500
            )

        return web.Response(
            body=audio_bytes,
            content_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="speech.{response_format}"'},
        )

    async def _handle_audio_transcriptions(self, request: "web.Request") -> "web.Response":
        """POST /v1/audio/transcriptions — OpenAI Whisper-compatible STT.

        Accepts multipart/form-data with:
            file      — audio file (required)
            model     — whisper model (default: whisper-large-v3)
            language  — ISO-639-1 code (optional)
            prompt    — context hint (optional)
            response_format — json|text|srt|vtt|verbose_json (default: json)
            temperature — float 0-1 (optional)

        Provider selection order:
          1. Groq (GROQ_API_KEY) — fastest free Whisper
          2. OpenAI (OPENAI_API_KEY) — standard

        Returns transcription JSON: {"text": "..."} (or plain text for response_format=text).
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        content_type_hdr = request.content_type or ""
        if "multipart" not in content_type_hdr:
            return web.json_response(
                {"error": {"message": "Content-Type must be multipart/form-data", "type": "invalid_request_error"}},
                status=400,
            )

        try:
            reader = await request.multipart()
        except Exception as exc:
            return web.json_response(
                {"error": {"message": f"Failed to parse multipart: {exc}", "type": "invalid_request_error"}},
                status=400,
            )

        file_data: Optional[bytes] = None
        file_name: str = "audio.wav"
        file_ct: str = "audio/wav"
        form_fields: Dict[str, str] = {}

        async for part in reader:
            if part.name == "file":
                file_name = part.filename or "audio.wav"
                file_ct = part.headers.get("Content-Type", "audio/wav")
                file_data = await part.read()
            else:
                val = await part.read()
                form_fields[part.name] = val.decode(errors="replace") if isinstance(val, bytes) else str(val)

        if not file_data:
            return web.json_response(
                {"error": {"message": "'file' part is required", "type": "invalid_request_error"}}, status=400
            )

        model_req = form_fields.get("model", "whisper-large-v3")
        language = form_fields.get("language", "")
        prompt = form_fields.get("prompt", "")
        response_format = form_fields.get("response_format", "json")
        temperature = form_fields.get("temperature", "")
        loop = asyncio.get_running_loop()
        def _stt_call() -> bytes:
            import urllib.request as _ur
            import urllib.error as _ue
            def _build_multipart(boundary: bytes, fields: dict, file_bytes: bytes, fname: str, fct: str) -> bytes:
                body_parts = []
                for k, v in fields.items():
                    if v:
                        body_parts.append(
                            b"--" + boundary + b"\r\n"
                            b'Content-Disposition: form-data; name="' + k.encode() + b'"\r\n\r\n'
                            + v.encode() + b"\r\n"
                        )
                body_parts.append(
                    b"--" + boundary + b"\r\n"
                    b'Content-Disposition: form-data; name="file"; filename="' + fname.encode() + b'"\r\n'
                    b"Content-Type: " + fct.encode() + b"\r\n\r\n"
                    + file_bytes + b"\r\n"
                )
                body_parts.append(b"--" + boundary + b"--\r\n")
                return b"".join(body_parts)
            fields = {"model": model_req}
            if language:
                fields["language"] = language
            if prompt:
                fields["prompt"] = prompt
            if response_format and response_format != "json":
                fields["response_format"] = response_format
            if temperature:
                fields["temperature"] = temperature
            boundary = b"HERMESSTTBOUNDARY"
            body = _build_multipart(boundary, fields, file_data, file_name, file_ct)
            ct_hdr = f"multipart/form-data; boundary={boundary.decode()}"
            # ── Try Groq Whisper ────────────────────────────────────────────
            groq_key = os.getenv("GROQ_API_KEY", "").strip()
            groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
            if groq_key:
                groq_model = model_req if "whisper" in model_req.lower() else "whisper-large-v3"
                groq_fields = dict(fields, model=groq_model)
                groq_body = _build_multipart(boundary, groq_fields, file_data, file_name, file_ct)
                req = _ur.Request(
                    f"{groq_base}/audio/transcriptions",
                    data=groq_body,
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": ct_hdr},
                    method="POST",
                )
                try:
                    with _ur.urlopen(req, timeout=60) as resp:
                        return resp.read()
                except _ue.HTTPError as exc:
                    logger.warning("[audio/transcriptions] Groq STT failed: HTTP %d %s", exc.code, exc.read()[:200])
                except Exception as exc:
                    logger.warning("[audio/transcriptions] Groq STT error: %s", exc)
            # ── Try OpenAI Whisper ──────────────────────────────────────────
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if openai_key:
                oai_model = model_req if model_req.startswith("whisper") else "whisper-1"
                oai_fields = dict(fields, model=oai_model)
                oai_body = _build_multipart(boundary, oai_fields, file_data, file_name, file_ct)
                req = _ur.Request(
                    f"{openai_base}/audio/transcriptions",
                    data=oai_body,
                    headers={"Authorization": f"Bearer {openai_key}", "Content-Type": ct_hdr},
                    method="POST",
                )
                try:
                    with _ur.urlopen(req, timeout=60) as resp:
                        return resp.read()
                except _ue.HTTPError as exc:
                    logger.warning("[audio/transcriptions] OpenAI STT failed: HTTP %d %s", exc.code, exc.read()[:200])
                except Exception as exc:
                    logger.warning("[audio/transcriptions] OpenAI STT error: %s", exc)
            raise RuntimeError("No STT provider available. Set GROQ_API_KEY or OPENAI_API_KEY.")
        try:
            result_bytes = await loop.run_in_executor(None, _stt_call)
        except RuntimeError as exc:
            return web.json_response(
                {"error": {"message": str(exc), "type": "service_unavailable"}}, status=503
            )
        except Exception as exc:
            logger.error("[audio/transcriptions] unexpected error: %s", exc, exc_info=True)
            return web.json_response(
                {"error": {"message": "Transcription failed", "type": "internal_server_error"}}, status=500
            )
        if response_format == "text":
            return web.Response(body=result_bytes, content_type="text/plain")
        return web.Response(body=result_bytes, content_type="application/json")

    # ------------------------------------------------------------------
    # BasePlatformAdapter interface
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware, request_monitor_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws, client_max_size=MAX_REQUEST_BYTES)
            self._app["api_server_adapter"] = self
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/ready", self._handle_ready)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/stats", self._handle_stats)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_post("/v1/rerank", self._handle_rerank)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
            # Cron jobs management API
            self._app.router.add_get("/api/jobs", self._handle_list_jobs)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            self._app.router.add_post("/v1/sessions/{session_id}/tool_responses", self._handle_tool_responses)
            self._app.router.add_get("/v1/sessions/{session_id}/pending-tool-calls", self._handle_pending_tool_calls)
            # Audio — TTS and STT (OpenAI-compatible)
            self._app.router.add_post("/v1/audio/speech", self._handle_audio_speech)
            self._app.router.add_post("/v1/audio/transcriptions", self._handle_audio_transcriptions)
            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)
            # Start background sweeper for Codex pool token refresh
            try:
                from agent.codex_token_sweeper import _codex_token_sweeper_task
                codex_sweep_task = asyncio.create_task(_codex_token_sweeper_task())
                try:
                    self._background_tasks.add(codex_sweep_task)
                except TypeError:
                    pass
                if hasattr(codex_sweep_task, "add_done_callback"):
                    codex_sweep_task.add_done_callback(self._background_tasks.discard)
            except Exception:
                pass

            # Refuse to start network-accessible without authentication
            if is_network_accessible(self._host) and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: binding to %s requires API_SERVER_KEY. "
                    "Set API_SERVER_KEY or use the default 127.0.0.1.",
                    self.name, self._host,
                )
                return False

            # Refuse to start network-accessible with a placeholder key.
            # Ported from openclaw/openclaw#64586.
            if is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=8):
                        logger.error(
                            "[%s] Refusing to start: API_SERVER_KEY is set to a "
                            "placeholder value. Generate a real secret "
                            "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                            "before exposing the API server on %s.",
                            self.name, self._host,
                        )
                        return False
                except ImportError:
                    pass

            # Port conflict detection — fail fast if port is already in use
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(('127.0.0.1', self._port))
                logger.error('[%s] Port %d already in use. Set a different port in config.yaml: platforms.api_server.port', self.name, self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            self._runner = web.AppRunner(
                self._app,
                keepalive_timeout=120,       # keep idle connections alive longer
                shutdown_timeout=30,         # faster graceful shutdown
            )
            await self._runner.setup()
            listen_sock = self._create_listen_socket(backlog=2048)
            if listen_sock is not None:
                self._site = web.SockSite(self._runner, listen_sock, backlog=2048)
            else:
                self._site = web.TCPSite(self._runner, self._host, self._port, backlog=2048)
            await self._site.start()

            # Eager warm-up: pre-resolve credentials for slow OAuth-based
            # providers (Copilot) in a background thread so the process-level
            # cache is populated before the first user request.  This saves
            # ~11s on the first request by avoiding 4-6 redundant ~2-3s
            # credential resolution calls.  Only one model per provider is
            # needed — the cache is keyed on provider prefix, so resolving
            # a model pre-warms the credential cache for all models under
            # the same provider.  We include both minimax and GHE Copilot
            # enterprise so that both credential paths are pre-warmed.
            # Also pre-build the GHE model catalog/API-mode cache and the
            # fallback-chain cache so the first real user turn avoids extra
            # catalog fetch / endpoint-selection latency.
            loop = asyncio.get_running_loop()
            def _warmup():
                ghe_runtime = None
                try:
                    ghe_runtime, _ = _runtime_kwargs_for_model_id("github-copilot-enterprise/claude-sonnet-4.6")
                except Exception:
                    pass
                try:
                    _runtime_kwargs_for_model_id("github-copilot-enterprise/gpt-5.4")
                except Exception:
                    pass
                try:
                    _runtime_kwargs_for_model_id("minimax/MiniMax-M2.7")
                except Exception:
                    pass
                if ghe_runtime:
                    try:
                        from hermes_cli.models import fetch_github_model_catalog, copilot_model_api_mode
                        _catalog = fetch_github_model_catalog(
                            api_key=ghe_runtime.get("api_key"),
                            base_url=ghe_runtime.get("base_url"),
                        )
                        for _mid in (
                            "github-copilot-enterprise/claude-sonnet-4.6",
                            "github-copilot-enterprise/gpt-5.4",
                        ):
                            try:
                                copilot_model_api_mode(
                                    _mid,
                                    catalog=_catalog,
                                    api_key=ghe_runtime.get("api_key"),
                                    base_url=ghe_runtime.get("base_url"),
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                try:
                    _build_env_fallback_chain("HERMES_SWARM_FALLBACK")
                except Exception:
                    pass
            loop.run_in_executor(None, _warmup)

            self._mark_connected()
            if not self._api_key:
                logger.warning(
                    "[%s] ⚠️  No API key configured (API_SERVER_KEY / platforms.api_server.key). "
                    "All requests will be accepted without authentication. "
                    "Set an API key for production deployments to prevent "
                    "unauthorized access to sessions, responses, and cron jobs.",
                    self.name,
                )
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server."""
        self._mark_disconnected()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
