"""OpenAI-compatible shim that forwards Hermes requests to `claude -p`.

This adapter lets Hermes treat Claude Code CLI as a chat-style backend.
Each request runs Claude in print mode with a formatted conversation,
collects the result, and converts it back into the minimal shape
Hermes expects from an OpenAI client.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CLAUDE_CODE_BASE_URL = "claude://codex"
_DEFAULT_TIMEOUT_SECONDS = 300.0  # 5 minutes for print mode

# Model mapping: hermes model hint → Claude Code --model flag
MODEL_MAP = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4.6",
    "haiku": "claude-haiku-4-5-20250520",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4.6": "claude-opus-4.6",
    "claude-haiku-4-5-20250520": "claude-haiku-4-5-20250520",
}


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CODE_ARGS", "").strip()
    if not raw:
        return ["-p", "--output-format", "json", "--no-stream"]
    return shlex.split(raw)


def _resolve_home_dir() -> str:
    """Return a stable HOME for child Claude Code processes."""
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


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
) -> str:
    """Format messages as a prompt for Claude Code print mode."""
    sections: list[str] = [
        "You are being used as the active coding agent backend for Hermes.",
        "Use your coding capabilities to complete tasks.",
        "Provide a clear, actionable response.",
    ]

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        if not content:
            continue

        # Handle tool results
        if role == "tool":
            content = f"Tool result: {content}"

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "context": "Context",
        }.get(role, role.title())

        transcript.append(f"{label}: {content}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Complete the user's request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _parse_json_output(output: str) -> tuple[str, str]:
    """Parse JSON output from Claude Code print mode."""
    try:
        data = json.loads(output)
        if isinstance(data, dict):
            # Success response
            result = data.get("result", "")
            reason = data.get("reasoning", "") or data.get("reason", "")
            return str(result) if result else "", str(reason) if reason else ""
    except json.JSONDecodeError:
        pass

    # Not JSON - return as-is
    return output.strip(), ""


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
        # Map model hint to Claude Code model
        model_flag = MODEL_MAP.get(model or "sonnet", "claude-sonnet-4-6")

        prompt_text = _format_messages_as_prompt(messages or [], model=model)

        # Build command args
        cmd_args = [self._claude_command] + self._claude_args + [prompt_text]

        # Add model flag if not already in args
        if not any("--model" in arg for arg in cmd_args):
            cmd_args.insert(1, "--model")
            cmd_args.insert(2, model_flag)

        # Add max turns if tools are provided
        if tools and not any("--max-turns" in arg for arg in cmd_args):
            cmd_args.insert(1, "--max-turns")
            cmd_args.insert(2, "10")

        # Add allowed tools if tools are provided
        if tools:
            tool_names = []
            for t in tools:
                if isinstance(t, dict) and "function" in t:
                    fn = t.get("function", {})
                    if isinstance(fn, dict) and fn.get("name"):
                        tool_names.append(fn["name"])
            if tool_names:
                if not any("--allowedTools" in arg for arg in cmd_args):
                    cmd_args.insert(1, "--allowedTools")
                    cmd_args.insert(2, ",".join(tool_names))

        # Timeout handling
        effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        if max_tokens:
            # Rough estimate: 4 tokens per token, plus overhead
            effective_timeout = max(effective_timeout, max_tokens / 2.0)

        response_text = self._run_prompt(
            cmd_args,
            timeout_seconds=effective_timeout,
        )

        result_text, reasoning_text = _parse_json_output(response_text)

        # Extract tool calls if any (Claude Code uses JSON in result)
        tool_calls = []
        try:
            # Try to parse tool calls from the result
            if "tool_call" in result_text.lower() or "<tool_call>" in result_text:
                # Extract JSON tool calls
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

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
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
            model=model or "claude-code",
        )

    def _run_prompt(
        self,
        cmd_args: list[str],
        *,
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
            # Claude Code in print mode reads from stdin if no prompt is given
            if proc.stdin:
                proc.stdin.close()

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
            if proc.returncode != 0 and stderr_text:
                # Non-zero exit with stderr - check for errors
                if "error" in stderr_text.lower():
                    raise RuntimeError(f"Claude Code failed: {stderr_text}")

            return "".join(output_lines)

        finally:
            self.close()