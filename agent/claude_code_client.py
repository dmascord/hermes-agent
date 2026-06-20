"""OpenAI-compatible shim that forwards Hermes requests to `claude -p`.

This adapter lets Hermes treat Claude Code CLI as a chat-style backend.
Each request runs Claude in print mode with a formatted conversation,
collects the result, and converts it back into the minimal shape
Hermes expects from an OpenAI client.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shlex
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from datetime import datetime, timezone
_logger = logging.getLogger(__name__)

# Anthropic OAuth constants (decoded from base64: Claude Code clientId)
_CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_CLAUDE_TOKEN_URLS = (
    "https://api.anthropic.com/v1/oauth/token",
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
)

_DEFAULT_TIMEOUT_SECONDS = 300.0  # 5 minutes for print mode

CLAUDE_CODE_BASE_URL = "claude://codex"

MODEL_MAP = {
    # Use Claude CLI's own model aliases; these work reliably in print mode.
    "sonnet": "sonnet",
    "opus": "opus",
    "haiku": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "claude-opus-4.6": "opus",
    "claude-haiku-4-5-20250520": "haiku",
}


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    """Default CLI args. Override via HERMES_CLAUDE_CODE_ARGS env var.

    Claude's CLI requires --verbose when using --output-format=stream-json.
    """
    raw = os.getenv("HERMES_CLAUDE_CODE_ARGS", "").strip()
    if not raw:
        return ["-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions", "--bare", "--add-dir", "/tmp", "/opt", "/home", "/root"]
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for child Claude Code processes.

    Prefer Hermes' per-profile subprocess HOME. The container entrypoint now
    restores Claude credentials into that location so the dropped `hermes`
    user can read them.
    """
    try:
        from hermes_constants import get_subprocess_home

        profile_home = get_subprocess_home()
        if profile_home:
            return profile_home
    except Exception:
        pass

    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()
        if resolved:
            return resolved
    except Exception:
        pass

    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    # Ensure ANTHROPIC_API_KEY is passed through if set
    if "ANTHROPIC_API_KEY" in os.environ:
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    return env


# Lock to serialise concurrent refresh attempts inside one process.
_claude_oauth_refresh_lock = threading.Lock()


def _claude_oauth_needs_refresh(creds: dict, *, skew_seconds: int = 300) -> bool:
    """Return True if the Claude OAuth access token is expired or about to expire.

    ``skew_seconds`` (default 5 min) is the safety margin — refresh proactively
    so a request in flight doesn't hit an expired token.
    """
    try:
        expires_at = int(creds.get("expiresAt") or 0)
    except Exception:
        return True
    if not expires_at:
        return True
    return expires_at <= int(time.time() * 1000) + skew_seconds * 1000


def _claude_oauth_refresh_token(refresh_token: str, *, timeout: float = 15.0) -> dict:
    """Call Anthropic's OAuth token endpoint with a refresh token.

    Prefer the shared Anthropic adapter implementation so the Claude CLI path
    uses the same known-good token endpoints and fallback behavior as the
    native Anthropic OAuth path.  A small local fallback remains for unusual
    import-time contexts.

    Returns a dict with at least ``access_token``, ``refresh_token``,
    ``expires_in`` seconds.  Raises on HTTP failure.
    """
    try:
        from agent.anthropic_adapter import refresh_anthropic_oauth_pure

        refreshed = refresh_anthropic_oauth_pure(refresh_token, use_json=False)
        expires_at_ms = int(refreshed.get("expires_at_ms") or 0)
        expires_in = max(1, int((expires_at_ms - int(time.time() * 1000)) / 1000))
        return {
            "access_token": refreshed["access_token"],
            "refresh_token": refreshed.get("refresh_token") or refresh_token,
            "expires_in": expires_in,
            "refresh_token_expires_in": refreshed.get("refresh_token_expires_in"),
        }
    except ImportError:
        pass

    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": _CLAUDE_CLIENT_ID,
        "refresh_token": refresh_token,
    }).encode("utf-8")
    last_error: Exception | None = None
    for token_url in _CLAUDE_TOKEN_URLS:
        req = urllib.request.Request(
            token_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "claude-cli/hermes (external, cli)",
                "Accept": "application/json",
                "anthropic-beta": "oauth-2025-04-20",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            last_error = exc
            continue
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Anthropic OAuth refresh returned non-object: {raw[:200]}")
        access = data.get("access_token")
        new_refresh = data.get("refresh_token") or refresh_token
        expires_in = int(data.get("expires_in") or 3600)
        if not access:
            raise RuntimeError(f"Anthropic OAuth refresh returned no access_token: {raw[:200]}")
        return {
            "access_token": access,
            "refresh_token": new_refresh,
            "expires_in": expires_in,
            "refresh_token_expires_in": data.get("refresh_token_expires_in"),
        }
    if last_error is not None:
        raise last_error
    raise RuntimeError("Anthropic OAuth refresh failed")


def _write_claude_credentials_file(creds_path: Path, data: dict[str, Any]) -> None:
    """Atomically write Claude credentials with owner-only permissions."""
    os.makedirs(creds_path.parent, exist_ok=True)
    tmp_path = creds_path.with_name(f"{creds_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, creds_path)
        os.chmod(creds_path, 0o600)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _credential_expires_at_ms(auth: dict[str, Any]) -> int:
    try:
        return int(auth.get("expiresAt") or 0)
    except Exception:
        return 0


def _credential_file_expires_at_ms(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        auth = data.get("claudeAiOauth", {}) if isinstance(data, dict) else {}
        return _credential_expires_at_ms(auth)
    except Exception:
        return 0


def _restore_claude_credentials_from_backup(creds_path: Path) -> bool:
    """Restore refreshable Claude credentials from the PVC backup if present."""
    backup_roots = []
    home_env = os.environ.get("HERMES_HOME", "").strip()
    if home_env:
        backup_roots.append(Path(home_env))
    try:
        resolved_home = Path(_resolve_home_dir())
        backup_roots.extend([resolved_home, resolved_home.parent])
    except Exception:
        pass

    seen: set[Path] = set()
    for root in backup_roots:
        pvc_backup = root / ".claude_backup" / ".credentials.json"
        if pvc_backup in seen:
            continue
        seen.add(pvc_backup)
        if not pvc_backup.exists():
            continue
        try:
            data = json.loads(pvc_backup.read_text(encoding="utf-8"))
            auth = data.get("claudeAiOauth", {}) if isinstance(data, dict) else {}
            if not auth.get("accessToken") or not auth.get("refreshToken"):
                _logger.debug("claude_oauth: backup %s is not refreshable", pvc_backup)
                continue
            backup_expires_ms = _credential_expires_at_ms(auth)
            active_expires_ms = _credential_file_expires_at_ms(creds_path) if creds_path.exists() else 0
            _auth_json_creds, auth_json_expires_ms = _auth_json_claude_credentials()
            if active_expires_ms and active_expires_ms > backup_expires_ms:
                _logger.info(
                    "claude_oauth: refusing to restore older PVC backup over active credentials "
                    "(backup=%s, active=%s)",
                    backup_expires_ms,
                    active_expires_ms,
                )
                continue
            if auth_json_expires_ms and auth_json_expires_ms > backup_expires_ms:
                _logger.info(
                    "claude_oauth: refusing to restore older PVC backup over auth.json credentials "
                    "(backup=%s, auth_json=%s)",
                    backup_expires_ms,
                    auth_json_expires_ms,
                )
                continue
            _write_claude_credentials_file(creds_path, data)
            _logger.info("claude_oauth: restored refreshable credentials from PVC backup %s", pvc_backup)
            return True
        except Exception as exc:
            _logger.debug("claude_oauth: PVC backup restore failed from %s: %s", pvc_backup, exc)
    return False

def _persist_claude_credentials_to_auth_json(
    full_credentials: dict[str, Any],
    expires_at_ms: int,
) -> None:
    """Persist the full Claude credentials dict to hermes auth.json.

    The Claude CLI requires several fields in .credentials.json beyond just
    the access/refresh tokens — ``scopes``, ``subscriptionType``,
    ``rateLimitTier`` are all needed for the CLI to consider the session
    valid.  We persist the full claudeAiOauth dict so we can rebuild the
    .credentials.json file byte-for-byte on recovery.

    The Claude CLI deletes .credentials.json on 401 — auth.json (managed by
    hermes_cli.auth) is never touched by the CLI, so it's always safe as
    a recovery source.
    """
    try:
        from hermes_cli.auth import (
            _load_auth_store,
            _save_provider_state,
            _save_auth_store,
            _auth_store_lock,
        )
        with _auth_store_lock():
            auth_store = _load_auth_store()
            _save_provider_state(auth_store, "claude-code-cli", {
                "full_credentials": full_credentials,
                "expires_at_ms": expires_at_ms,
                "auth_mode": "oauth",
            })
            _save_auth_store(auth_store)
        _logger.debug("claude_oauth: persisted full credentials to auth.json")
    except Exception as exc:
        _logger.debug("claude_oauth: failed to persist to auth.json: %s", exc)


def _recover_claude_tokens_from_auth_json() -> bool:
    """Try to recover Claude credentials from auth.json when .credentials.json is missing.

    The CLI deletes .credentials.json on 401.  If auth.json has a saved
    full credential dict (including scopes, subscriptionType, etc.),
    write it to .credentials.json so the next CLI call succeeds without
    requiring manual re-auth.
    """
    home = _resolve_home_dir()
    creds_path = Path(home) / ".claude" / ".credentials.json"
    try:
        from hermes_cli.auth import (
            _load_auth_store,
            _load_provider_state,
            _auth_store_lock,
        )
        with _auth_store_lock():
            auth_store = _load_auth_store()
            state = _load_provider_state(auth_store, "claude-code-cli") or {}

        # Try the new full_credentials format first
        full_creds = state.get("full_credentials", {})
        if full_creds.get("accessToken") and full_creds.get("refreshToken"):
            data = {"claudeAiOauth": full_creds}
            _write_claude_credentials_file(creds_path, data)
            expires_ms = state.get("expires_at_ms", 0)
            remaining_h = max(0, (expires_ms / 1000 - time.time()) / 3600)
            _logger.info(
                "claude_oauth: recovered full credentials from auth.json to %s "
                "(access_token=%s..., expires_in=%.1fh, scopes=%d)",
                creds_path,
                full_creds["accessToken"][:25],
                remaining_h,
                len(full_creds.get("scopes", [])),
            )
            return True

        # Fall back to legacy tokens format (3-field only, for compatibility)
        tokens = state.get("tokens", {})
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        if not refresh_token:
            _logger.debug("claude_oauth: no credentials in auth.json for recovery")
            return False
        data = {
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "expiresAt": state.get("expires_at_ms", 0),
            }
        }
        _write_claude_credentials_file(creds_path, data)
        expires_ms = state.get("expires_at_ms", 0)
        remaining_h = max(0, (expires_ms / 1000 - time.time()) / 3600)
        _logger.info(
            "claude_oauth: recovered legacy tokens from auth.json to %s "
            "(access_token=%s..., expires_in=%.1fh, WARNING: missing scopes)",
            creds_path,
            access_token[:25],
            remaining_h,
        )
        return True
    except Exception as exc:
        _logger.debug("claude_oauth: failed to recover from auth.json: %s", exc)
        return False


def _auth_json_claude_credentials() -> tuple[dict[str, Any], int]:
    """Return Claude Code full credentials from auth.json, plus expiry ms."""
    try:
        from hermes_cli.auth import (
            _load_auth_store,
            _load_provider_state,
            _auth_store_lock,
        )
        with _auth_store_lock():
            auth_store = _load_auth_store()
            state = _load_provider_state(auth_store, "claude-code-cli") or {}
        full_creds = state.get("full_credentials", {})
        if not isinstance(full_creds, dict):
            return {}, 0
        if not full_creds.get("accessToken") or not full_creds.get("refreshToken"):
            return {}, 0
        try:
            expires_ms = int(state.get("expires_at_ms") or full_creds.get("expiresAt") or 0)
        except Exception:
            expires_ms = 0
        return full_creds, expires_ms
    except Exception as exc:
        _logger.debug("claude_oauth: failed to inspect auth.json credentials: %s", exc)
        return {}, 0


def _recover_newer_claude_tokens_from_auth_json(current_expires_ms: int = 0) -> bool:
    """Recover auth.json credentials when they are newer than active disk creds."""
    full_creds, expires_ms = _auth_json_claude_credentials()
    if not full_creds:
        return False
    if expires_ms and current_expires_ms and expires_ms <= current_expires_ms:
        return False
    return _recover_claude_tokens_from_auth_json()


def _maybe_refresh_claude_oauth() -> bool:
    """Refresh the Claude OAuth token if it's expired.

    The Claude Code CLI does NOT auto-refresh expired access tokens — when the
    stored ``accessToken`` in ``.credentials.json`` is past its ``expiresAt``,
    the CLI returns "Not logged in · Please run /login" and fails every request.
    To keep the subprocess working, we proactively refresh using the stored
    ``refreshToken`` and write the new ``accessToken`` / ``expiresAt`` back to
    ``.credentials.json`` in place before each Claude CLI invocation.

    The refresh token is also persisted to ``auth.json`` (source of truth)
    so that even if the CLI deletes ``.credentials.json`` on a 401, we can
    always recover from auth.json on the next attempt.

    Returns True if a refresh was actually performed, False otherwise.
    """
    home = _resolve_home_dir()
    creds_path = Path(home) / ".claude" / ".credentials.json"

    # If credentials file is missing, try to recover from:
    # 1. PVC backup location (entrypoint restores from here on startup)
    # 2. auth.json (source of truth)
    restored_credentials = False
    if not creds_path.exists():
        _logger.info("claude_oauth: .credentials.json missing — checking auth.json, then PVC backup")

        # auth.json is Hermes' source of truth and may contain fresher tokens
        # than the pod-start PVC backup, which can lag after rotations.
        restored_credentials = _recover_claude_tokens_from_auth_json()

        # Fall back to PVC backup only when auth.json has no usable state.
        if not restored_credentials:
            restored_credentials = _restore_claude_credentials_from_backup(creds_path)
        if not restored_credentials:
            return False

    with _claude_oauth_refresh_lock:
        try:
            data = json.loads(creds_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _logger.debug("claude_oauth: failed to read %s: %s", creds_path, exc)
            return False
        auth = data.get("claudeAiOauth", {})
        access_token = auth.get("accessToken", "")
        refresh_token = auth.get("refreshToken", "")
        if not refresh_token:
            _logger.info("claude_oauth: credentials have no refresh token — checking auth.json, then backup")
            if _recover_claude_tokens_from_auth_json() or _restore_claude_credentials_from_backup(creds_path):
                restored_credentials = True
                try:
                    data = json.loads(creds_path.read_text(encoding="utf-8"))
                    auth = data.get("claudeAiOauth", {})
                    access_token = auth.get("accessToken", "")
                    refresh_token = auth.get("refreshToken", "")
                except Exception as exc:
                    _logger.debug("claude_oauth: failed to read restored credentials: %s", exc)
                    return False
            if not refresh_token:
                _logger.debug("claude_oauth: no refresh token — cannot refresh")
                return False
        now_ms = int(time.time() * 1000)
        expires_at = _credential_expires_at_ms(auth)
        remaining_ms = expires_at - now_ms
        if _recover_newer_claude_tokens_from_auth_json(expires_at):
            restored_credentials = True
            try:
                data = json.loads(creds_path.read_text(encoding="utf-8"))
                auth = data.get("claudeAiOauth", {})
                refresh_token = auth.get("refreshToken", "")
                expires_at = _credential_expires_at_ms(auth)
                remaining_ms = expires_at - now_ms
                _logger.info(
                    "claude_oauth: replaced stale active credentials with newer auth.json credentials "
                    "(expires_in=%.0fs)",
                    remaining_ms / 1000,
                )
            except Exception as exc:
                _logger.debug("claude_oauth: failed to read newer auth.json credentials: %s", exc)
                return False
        # Only refresh if within 5 minutes of expiry (or already expired)
        if remaining_ms > 5 * 60 * 1000:
            _logger.debug("claude_oauth: token still valid for %.0fs — no refresh needed", remaining_ms / 1000)
            return restored_credentials
        try:
            result = _claude_oauth_refresh_token(refresh_token)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            _logger.warning(
                "claude_oauth: refresh HTTP %d: %s", exc.code, error_body[:200],
            )
            # Record the error in the credentials file for operator visibility
            data.setdefault("claudeAiOauth", {})["last_refresh_error"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "http_code": exc.code,
                "error": error_body[:200],
            }
            try:
                _write_claude_credentials_file(creds_path, data)
            except Exception:
                pass
            # If the refresh token is dead (invalid_grant), write a tombstone
            # to auth.json to prevent repeated failed attempts
            if exc.code in (400, 401):
                _persist_claude_credentials_to_auth_json(
                    full_credentials={}, expires_at_ms=0,
                )
            return False
        except Exception as exc:
            _logger.warning("claude_oauth: refresh failed: %s", exc)
            return False
        _logger.info(
            "claude_oauth: refreshed access token, expires in %s seconds",
            result["expires_in"],
        )
        # Persist the new tokens back to .credentials.json so the next
        # subprocess invocation uses the fresh access token instead of
        # the expired one.
        auth["accessToken"] = result["access_token"]
        auth["refreshToken"] = result["refresh_token"]
        auth["expiresAt"] = int(time.time() * 1000) + result["expires_in"] * 1000
        if result.get("refresh_token_expires_in"):
            auth["refreshTokenExpiresAt"] = int(time.time() * 1000) + int(result["refresh_token_expires_in"]) * 1000
        auth.pop("last_refresh_error", None)
        # Guard against clobbering with stale backups or race conditions:
        # only persist if we are writing a genuinely newer token window.
        try:
            _disk_creds = json.loads(creds_path.read_text(encoding="utf-8"))
            _disk_auth = _disk_creds.get("claudeAiOauth", {})
            if _credential_expires_at_ms(_disk_auth) > _credential_expires_at_ms(auth):
                _logger.warning(
                    "claude_oauth: skipping write — disk has newer tokens (disk=%s, refresh=%s). "
                    "This may indicate a stale backup restoration.",
                    _disk_auth.get("expiresAt"), auth["expiresAt"]
                )
                return True
        except Exception as _exc:
            _logger.debug("claude_oauth: could not verify disk age: %s", _exc)

        try:
            _write_claude_credentials_file(creds_path, data)
        except Exception as exc:
            _logger.warning(
                "claude_oauth: failed to persist refreshed tokens to %s: %s",
                creds_path, exc,
            )
        # Also persist to auth.json (source of truth) — survives CLI 401 deletions
        _persist_claude_credentials_to_auth_json(
            full_credentials=data.get("claudeAiOauth", auth),
            expires_at_ms=auth["expiresAt"],
        )
        return True


def _format_messages_as_ndjson(
    messages: list[dict[str, Any]],
    model: str | None = None,
) -> str:
    """Convert OpenAI-style messages to Claude CLI NDJSON streaming input.

    The CLI accepts one JSON object per line on stdin when invoked with
    `--input-format stream-json`. Each line is a single `user` turn,
    using the same shape as the Agent SDK streaming input:

        {"type": "user", "message": {"role": "user", "content": "..."}}

    We coalesce the OpenAI-style transcript (system, user, assistant,
    tool) into a single user turn for print mode, since the CLI in
    print mode runs one conversation and exits.
    """
    system_parts: list[str] = []
    user_parts: list[dict[str, Any]] = []
    history_parts: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        content = message.get("content")
        # Tool-call messages (assistant with tool_calls) must NOT be skipped even
        # if they have no text content — the tool result that follows needs the
        # assistant message in the history for the model to understand context.
        _has_tool_calls = isinstance(message.get("tool_calls"), list) and message.get("tool_calls")
        if (content is None or content == "") and not _has_tool_calls:
            continue
        if role == "system":
            system_parts.append(str(content or ""))
            continue
        if role == "tool":
            history_parts.append(f"[Tool result]\n{content}")
            continue
        if role == "user":
            # Promote the last user message into the live user turn; keep
            # earlier ones in the history so the model sees full context.
            user_parts = [{"type": "text", "text": str(content or "")}]
            continue
        if role == "assistant":
            history_parts.append(f"[Assistant]\n{content or ''}")
            continue
        history_parts.append(f"[{role.title()}]\n{content or ''}")

    if system_parts or history_parts:
        prefix = []
        if system_parts:
            prefix.append("\n\n".join(system_parts))
        if history_parts:
            prefix.append("Conversation so far:\n\n" + "\n\n".join(history_parts))
        if user_parts:
            user_parts.insert(0, {"type": "text", "text": "\n\n".join(prefix)})
        else:
            user_parts = [{"type": "text", "text": "\n\n".join(prefix)}]

    if not user_parts:
        user_parts = [{"type": "text", "text": "Hello"}]

    payload = {
        "type": "user",
        "message": {
            "role": "user",
            "content": user_parts if len(user_parts) > 1 else user_parts[0]["text"],
        },
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _parse_stream_json_output(output: str) -> dict[str, Any]:
    """Parse Claude Code stream-json output.

    The CLI emits newline-delimited JSON objects. We care about the final
    `result` object, which contains the text result and usage information.
    """
    final: dict[str, Any] = {}
    assistant_text = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        if typ == "error":
            # CLI returned an explicit error — surface it so the caller
            # gets a meaningful exception instead of a silent empty result.
            err_msg = obj.get("error", {}) or {}
            if isinstance(err_msg, dict):
                err_text = err_msg.get("message") or err_msg.get("type") or str(obj)
            else:
                err_text = str(err_msg)
            raise RuntimeError(f"Claude Code CLI error: {err_text}")
        if typ == "assistant":
            msg = obj.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = str(part.get("text", ""))
                            if text:
                                assistant_text.append(text)
        elif typ == "result":
            final = obj
    if assistant_text and "assistant_text" not in final:
        final["assistant_text"] = "".join(assistant_text)
    return final


class _ClaudeCodeChatCompletions:
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCodeChatNamespace:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _ClaudeCodeChatCompletions(client)


class ClaudeCodeClient:
    """Minimal OpenAI-client-compatible facade for Claude Code CLI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        claude_command: str | None = None,
        claude_args: list[str] | None = None,
        claude_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-code"
        self.base_url = base_url or CLAUDE_CODE_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._claude_command = claude_command or command or _resolve_command()
        self._claude_args = list(claude_args or args or _resolve_args())
        self._claude_cwd = str(Path(claude_cwd or os.getcwd()).resolve())
        self.chat = _ClaudeCodeChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._tools_manifest_path: str | None = None
        self._mcp_config_path: str | None = None
        self._session_id: str = os.environ.get("HERMES_REQUEST_ID", uuid.uuid4().hex[:12])
        self._queue_in_path: str | None = None
        self._queue_out_dir: str | None = None

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _write_tools_manifest(self, tools: list[dict[str, Any]] | None) -> str | None:
        """Write the OMP client's tool definitions to a temp file for the MCP proxy.

        Returns the path to the manifest file, or None if no tools provided.
        """
        if not tools:
            return None
        manifest = []
        for t in tools:
            if isinstance(t, dict) and "function" in t:
                fn = t.get("function", {})
                if isinstance(fn, dict) and fn.get("name"):
                    manifest.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    })
        if not manifest:
            return None
        fd, path = tempfile.mkstemp(suffix=".json", prefix="hermes_tools_")
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f)
        self._tools_manifest_path = path
        return path

    def _build_mcp_config(self, tools_manifest_path: str | None) -> str | None:
        """Build an MCP config file pointing to the bridge proxy.

        Returns the path to the config file, or None.
        """
        if not tools_manifest_path:
            return None
        # The MCP bridge script is co-located with this module.
        bridge_script = os.path.join(os.path.dirname(__file__), "claude_mcp_bridge.py")
        if not os.path.exists(bridge_script):
            import logging
            logging.getLogger(__name__).warning(
                "[claude-code-client] MCP bridge script not found at %s", bridge_script
            )
            return None
        # Create session-specific queue directory
        self._queue_in_path = f"/tmp/hermes_queue_{self._session_id}.in"
        self._queue_out_dir = f"/tmp/hermes_result_{self._session_id}"
        os.makedirs(self._queue_out_dir, exist_ok=True)
        config = {
            "mcpServers": {
                "hermes-tools": {
                    "type": "stdio",
                    "command": "python3",
                    "args": [bridge_script],
                    "env": {
                        "HERMES_TOOLS_FILE": tools_manifest_path,
                        "HERMES_QUEUE_IN": self._queue_in_path,
                        "HERMES_QUEUE_OUT_DIR": self._queue_out_dir,
                    },
                },
            }
        }
        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="hermes_mcp_config_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f)
        self._mcp_config_path = config_path
        return config_path

    def _allowed_tool_names(self, tools: list[dict[str, Any]] | None) -> list[str]:
        if not tools:
            return []
        names: list[str] = []
        for tool in tools:
            if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
                continue
            name = tool["function"].get("name")
            if name:
                names.append(f"mcp__hermes-tools__{name}")
        return names

    def run_with_tool_bridge(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float = 300.0,
    ):
        """Run the Claude CLI subprocess with MCP tool bridging.

        Yields a sequence of dicts:
          {"type": "tool_call", "call_id": ..., "name": ..., "arguments": ...}
          {"type": "tool_result", "result": ...}   (caller must .send(result) back)
          {"type": "assistant_text", "text": ...}  (text deltas from CLI)
          {"type": "final", "text": ..., "usage": ...}
          {"type": "error", "message": ...}

        The caller must call .send(result_str) on tool_call events to provide
        the result, which is written to the bridge file for the MCP proxy.
        """
        _maybe_refresh_claude_oauth()
        model_flag = MODEL_MAP.get(model or "sonnet", "claude-sonnet-4-6")
        ndjson_payload = _format_messages_as_ndjson(messages or [], model=model)

        cmd_args = [
            self._claude_command, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", model_flag,
        ]
        if tools:
            cmd_args.extend(["--max-turns", "10"])
            manifest_path = self._write_tools_manifest(tools)
            mcp_config = self._build_mcp_config(manifest_path)
            if mcp_config:
                cmd_args.extend(["--mcp-config", mcp_config])
                cmd_args.append("--strict-mcp-config")
            tool_names = self._allowed_tool_names(tools)
            if tool_names:
                cmd_args.extend(["--allowedTools", ",".join(tool_names)])

        _logger.info("[claude-code-client] bridge cmd_args=%s", cmd_args)

        # Truncate the queue file (start fresh)
        if self._queue_in_path:
            try:
                if os.path.exists(self._queue_in_path):
                    os.unlink(self._queue_in_path)
            except Exception:
                pass

        try:
            proc = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._claude_cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude Code command '{self._claude_command}'. "
                "Install Claude Code CLI: npm install -g @anthropic-ai/claude-code"
            ) from exc

        with self._active_process_lock:
            self._active_process = proc

        mcp_events_lock = threading.Lock()
        mcp_events: list[dict[str, Any]] = []
        mcp_seen_call_ids: set[str] = set()
        mcp_stop = threading.Event()

        def _drain_mcp_events() -> list[dict[str, Any]]:
            with mcp_events_lock:
                drained = list(mcp_events)
                mcp_events.clear()
            return drained

        def _watch_mcp_queue() -> None:
            if not self._queue_in_path or not self._queue_out_dir:
                return
            offset = 0
            while not mcp_stop.is_set():
                try:
                    if not os.path.exists(self._queue_in_path):
                        time.sleep(0.05)
                        continue
                    with open(self._queue_in_path, "r", encoding="utf-8") as f:
                        f.seek(offset)
                        lines = f.readlines()
                        offset = f.tell()
                    for raw in lines:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            call = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        call_id = str(call.get("call_id") or "")
                        if not call_id or call_id in mcp_seen_call_ids:
                            continue
                        mcp_seen_call_ids.add(call_id)
                        event = {
                            "type": "tool_call",
                            "call_id": call_id,
                            "name": str(call.get("tool") or ""),
                            "arguments": call.get("arguments") or {},
                        }
                        with mcp_events_lock:
                            mcp_events.append(event)
                        result_path = os.path.join(
                            self._queue_out_dir, f"{call_id}.json"
                        )
                        if not os.path.exists(result_path):
                            with open(result_path, "w", encoding="utf-8") as f:
                                json.dump({"content": ""}, f)
                except Exception as exc:
                    _logger.warning("bridge: MCP queue monitor error: %s", exc)
                time.sleep(0.05)

        mcp_thread: threading.Thread | None = None
        if tools and self._queue_in_path and self._queue_out_dir:
            mcp_thread = threading.Thread(target=_watch_mcp_queue, daemon=True)
            mcp_thread.start()

        try:
            if proc.stdin:
                if ndjson_payload:
                    try:
                        proc.stdin.write(ndjson_payload)
                    except BrokenPipeError:
                        pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            # Read stdout line by line, yielding events
            pending_tool_calls: dict[str, str] = {}  # call_id -> result_str
            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = obj.get("type")
                if typ == "assistant":
                    for event in _drain_mcp_events():
                        yield event
                    msg = obj.get("message", {})
                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        ptype = part.get("type")
                        if ptype == "tool_use":
                            call_id = part.get("id", uuid.uuid4().hex)
                            tool_name = part.get("name", "")
                            tool_input = part.get("input", {})
                            if (
                                tool_name == "ToolSearch"
                                or tool_name.startswith("mcp__hermes-tools__")
                            ):
                                continue
                            # Yield the tool_call, caller will .send(result)
                            result = yield {
                                "type": "tool_call",
                                "call_id": call_id,
                                "name": tool_name,
                                "arguments": tool_input,
                            }
                            # Caller provided result — write to bridge file
                            if self._queue_out_dir:
                                result_path = os.path.join(
                                    self._queue_out_dir, f"{call_id}.json"
                                )
                                try:
                                    if isinstance(result, dict):
                                        payload = result
                                    else:
                                        payload = {"content": str(result)}
                                    with open(result_path, "w") as f:
                                        json.dump(payload, f)
                                except Exception as exc:
                                    _logger.error(
                                        "bridge: failed to write result for %s: %s",
                                        call_id, exc,
                                    )
                        elif ptype == "text":
                            text = part.get("text", "")
                            if text:
                                yield {"type": "assistant_text", "text": text}
                elif typ == "result":
                    for event in _drain_mcp_events():
                        yield event
                    result_text = str(obj.get("result") or "")
                    usage = obj.get("usage", {})
                    yield {
                        "type": "final",
                        "text": result_text,
                        "usage": usage,
                        "model": obj.get("model") or model or "claude-code",
                    }
                elif typ == "error":
                    err_msg = obj.get("error", {}) or {}
                    if isinstance(err_msg, dict):
                        err_text = err_msg.get("message") or err_msg.get("type") or str(obj)
                    else:
                        err_text = str(err_msg)
                    yield {"type": "error", "message": err_text}

            proc.wait(timeout=timeout_seconds)
            for event in _drain_mcp_events():
                yield event
            if proc.returncode != 0 and proc.stderr:
                err_text = proc.stderr.read()[:500]
                if err_text:
                    yield {"type": "error", "message": f"exit {proc.returncode}: {err_text}"}

        finally:
            mcp_stop.set()
            if mcp_thread is not None:
                mcp_thread.join(timeout=1.0)
            self.close()
            for _temp_path in (self._tools_manifest_path, self._mcp_config_path,
                                self._queue_in_path, self._queue_out_dir):
                if _temp_path and os.path.exists(_temp_path):
                    try:
                        if os.path.isdir(_temp_path):
                            import shutil
                            shutil.rmtree(_temp_path)
                        else:
                            os.unlink(_temp_path)
                    except Exception:
                        pass
            self._tools_manifest_path = None
            self._mcp_config_path = None

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        # Proactively refresh the Claude OAuth access token before launching
        # the subprocess. The CLI itself does NOT refresh expired tokens —
        # when the stored accessToken has passed its expiresAt, the CLI just
        # returns "Not logged in · Please run /login" and fails every request.
        # Best-effort: a refresh failure is logged but does not abort the call.
        _maybe_refresh_claude_oauth()
        # Map model hint to Claude Code model
        model_flag = MODEL_MAP.get(model or "sonnet", "claude-sonnet-4-6")

        # Build NDJSON payload for --input-format stream-json
        ndjson_payload = _format_messages_as_ndjson(messages or [], model=model)

        # Build command args in the exact order verified to work manually:
        #   claude -p --input-format stream-json --output-format stream-json
        #          --verbose --model sonnet
        cmd_args = [
            self._claude_command,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", model_flag,
        ]

        # Add max turns if tools are provided
        if tools:
            cmd_args.extend(["--max-turns", "10"])

        # Integrate MCP tools if provided
        manifest_path = self._write_tools_manifest(tools)
        mcp_config = self._build_mcp_config(manifest_path)
        if mcp_config:
            cmd_args.extend(["--mcp-config", mcp_config])
            cmd_args.append("--strict-mcp-config")

        # Add allowed tools (legacy internal tools, should still be allowed)
        if tools:
            tool_names = self._allowed_tool_names(tools)
            if tool_names:
                cmd_args.extend(["--allowedTools", ",".join(tool_names)])

        # Timeout handling
        effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        if max_tokens:
            # Rough estimate: 4 tokens per token, plus overhead
            effective_timeout = max(effective_timeout, max_tokens / 2.0)

        # DEBUG: keep an exact copy of the command line in logs when needed.
        # (Safe: args contain no secrets; auth is via restored CLI credentials.)
        try:
            import logging
            logging.getLogger(__name__).info("[claude-code-client] cmd_args=%s", cmd_args)
        except Exception:
            pass

        response_text = self._run_prompt(
            cmd_args,
            stdin_payload=ndjson_payload,
            timeout_seconds=effective_timeout,
        )

        parsed = _parse_stream_json_output(response_text)
        result_text = str(parsed.get("result") or parsed.get("assistant_text") or "")
        reasoning_text = str(parsed.get("reasoning") or "")

        # Extract tool calls if any (Claude Code uses JSON in stream output)
        tool_calls = []
        try:
            # If the final result text contains embedded tool_call JSON snippets,
            # preserve them for the gateway's tool-call handling.
            if "tool_call" in result_text.lower() or "<tool_call>" in result_text:
                tool_call_re = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
                for m in tool_call_re.finditer(result_text):
                    try:
                        obj = json.loads(m.group(1))
                        fn = obj.get("function", {})
                        if fn:
                            tool_calls.append(
                                SimpleNamespace(
                                    id=obj.get("id", f"call_{len(tool_calls)+1}"),
                                    type="function",
                                    function=SimpleNamespace(
                                        name=fn.get("name", ""),
                                        arguments=json.dumps(fn.get("arguments", {})),
                                    ),
                                )
                            )
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        usage_obj = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        usage = SimpleNamespace(
            prompt_tokens=int(usage_obj.get("input_tokens", 0) or 0),
            completion_tokens=int(usage_obj.get("output_tokens", 0) or 0),
            total_tokens=int((usage_obj.get("input_tokens", 0) or 0) + (usage_obj.get("output_tokens", 0) or 0)),
        )

        assistant_message = SimpleNamespace(
            content=result_text or response_text,
            tool_calls=tool_calls if tool_calls else None,
            reasoning=reasoning_text or None,
        )

        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)

        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=(parsed.get("model") if isinstance(parsed, dict) and parsed.get("model") else model or "claude-code"),
        )

    def _run_prompt(
        self,
        cmd_args: list[str],
        *,
        stdin_payload: str = "",
        timeout_seconds: float,
    ) -> str:
        """Run Claude Code with the given args and return the output."""
        try:
            proc = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._claude_cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude Code command '{self._claude_command}'. "
                "Install Claude Code CLI: npm install -g @anthropic-ai/claude-code"
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Claude Code process did not expose stdin/stdout pipes.")

        with self._active_process_lock:
            self._active_process = proc

        try:
            # Write the NDJSON payload to stdin and close it so the CLI
            # starts processing. We send all bytes up front, then close
            # — the CLI in print mode will run one conversation and exit.
            if proc.stdin:
                if stdin_payload:
                    try:
                        proc.stdin.write(stdin_payload)
                    except BrokenPipeError:
                        pass
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            # Read output with timeout
            output_lines: list[str] = []
            error_lines: list[str] = []

            # Use threads to read stdout and stderr
            output_queue: queue.Queue[tuple[str, str]] = queue.Queue()

            def _read_stdout() -> None:
                if proc.stdout:
                    for line in proc.stdout:
                        output_queue.put(("stdout", line))

            def _read_stderr() -> None:
                if proc.stderr:
                    for line in proc.stderr:
                        output_queue.put(("stderr", line))

            out_thread = threading.Thread(target=_read_stdout, daemon=True)
            err_thread = threading.Thread(target=_read_stderr, daemon=True)
            out_thread.start()
            err_thread.start()

            deadline = time.monotonic() + timeout_seconds

            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    # Process finished
                    break

                try:
                    stream, line = output_queue.get(timeout=0.1)
                    if stream == "stdout":
                        output_lines.append(line)
                    else:
                        error_lines.append(line)
                except queue.Empty:
                    continue

            # Check for timeout
            if proc.poll() is None:
                proc.kill()
                raise TimeoutError(f"Timed out waiting for Claude Code response after {timeout_seconds}s.")

            # Join any remaining output
            out_thread.join(timeout=1)
            err_thread.join(timeout=1)

            stderr_text = "".join(error_lines)
            if proc.returncode != 0:
                # Non-zero exit — always treat as failure, even if stderr is
                # empty (could be a crash or无声 error). Surface the stderr
                # if available, otherwise use a generic message.
                if stderr_text:
                    raise RuntimeError(f"Claude Code failed (exit {proc.returncode}): {stderr_text}")
                else:
                    raise RuntimeError(f"Claude Code CLI exited with code {proc.returncode} (no stderr)")

            return "".join(output_lines)

        finally:
            self.close()
            for _temp_path in (self._tools_manifest_path, self._mcp_config_path):
                if _temp_path:
                    try:
                        if os.path.exists(_temp_path):
                            os.unlink(_temp_path)
                    except Exception:
                        pass
            self._tools_manifest_path = None
            self._mcp_config_path = None
