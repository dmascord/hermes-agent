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
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_logger = logging.getLogger(__name__)

# Anthropic OAuth constants (decoded from base64: Claude Code clientId)
_CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_CLAUDE_TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"

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
        return ["-p", "--output-format", "stream-json", "--verbose"]
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

    Mirrors the OMP implementation at
    ``omp-src/packages/ai/src/utils/oauth/anthropic.ts:refreshAnthropicToken``.
    Returns a dict with at least ``access_token``, ``refresh_token``,
    ``expires_in`` seconds.  Raises on HTTP failure.
    """
    body = (
        f"grant_type=refresh_token"
        f"&client_id={_CLAUDE_CLIENT_ID}"
        f"&refresh_token={urllib.parse.quote(refresh_token, safe='')}"
    ).encode("utf-8")
    req = urllib.request.Request(
        _CLAUDE_TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "anthropic-sdk-typescript/0.94.0 userOAuthProvider",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
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
    }


def _maybe_refresh_claude_oauth() -> bool:
    """Refresh the Claude OAuth token if it's expired.

    The Claude Code CLI does NOT auto-refresh expired access tokens — when the
    stored ``accessToken`` in ``.credentials.json`` is past its ``expiresAt``,
    the CLI returns "Not logged in · Please run /login" and fails every request.
    To keep the subprocess working, we proactively refresh using the stored
    ``refreshToken`` and write the new ``accessToken`` / ``expiresAt`` back to
    ``.credentials.json`` in place before each Claude CLI invocation.

    The function is best-effort: any failure (no credentials, no refresh token,
    HTTP error) is swallowed and logged, so a broken refresh can never break
    the request path — it just means the next CLI invocation will hit the
    same "Not logged in" error and the operator can fix credentials manually.

    Returns True if a refresh was actually performed, False otherwise.
    """
    home = _resolve_home_dir()
    creds_path = Path(home) / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return False
    with _claude_oauth_refresh_lock:
        try:
            data = json.loads(creds_path.read_text(encoding="utf-8"))
        except Exception as exc:
            _logger.debug("claude_oauth: failed to read %s: %s", creds_path, exc)
            return False
        if not isinstance(data, dict):
            return False
        auth = data.get("claudeAiOauth")
        if not isinstance(auth, dict):
            return False
        if not _claude_oauth_needs_refresh(auth):
            return False
        refresh_token = auth.get("refreshToken")
        if not refresh_token:
            _logger.debug("claude_oauth: no refresh_token in %s", creds_path)
            return False
        try:
            result = _claude_oauth_refresh_token(refresh_token)
        except urllib.error.HTTPError as exc:
            # Anthropic returns 400 with body {"error":"invalid_grant",...}
            # when the refresh token has been consumed/revoked.  This is
            # unrecoverable until the user re-runs `claude /login` (or the
            # equivalent OAuth device flow).  Surface the body in the log
            # and persist it to .credentials.json so the gateway can show
            # the operator which Claude account needs re-auth.
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            auth["last_refresh_error"] = {
                "code": exc.code,
                "body": err_body,
                "at": int(time.time() * 1000),
            }
            _logger.warning(
                "claude_oauth: refresh failed (HTTP %s); body=%s. "
                "Run `claude /login` to re-authorise.",
                exc.code, err_body,
            )
            # Best-effort: persist the error so subsequent gateway
            # calls / health checks can surface it without re-trying.
            try:
                creds_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
            return False
        except Exception as exc:
            _logger.warning(
                "claude_oauth: refresh failed (%s); CLI may return 'Not logged in'",
                exc,
            )
            return False
        _logger.info(
            "claude_oauth: refreshed access token, expires in %s seconds",
            result["expires_in"],
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

        # Add allowed tools if tools are provided
        if tools:
            tool_names = []
            for t in tools:
                if isinstance(t, dict) and "function" in t:
                    fn = t.get("function", {})
                    if isinstance(fn, dict) and fn.get("name"):
                        tool_names.append(fn["name"])
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