"""OpenAI-compatible shim that forwards Hermes requests to `mimo run`.

Each request runs MiMoCode in JSON event mode, collects text and tool
events, and converts them back into the minimal shape Hermes expects.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

_logger = logging.getLogger(__name__)

MIMOCODE_BASE_URL = "mimocode://codex"

DEFAULT_TIMEOUT_SECONDS = 300.0


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_MIMOCODE_COMMAND", "").strip()
        or os.getenv("MIMOCODE_CLI_PATH", "").strip()
        or "mimo"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_MIMOCODE_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    return [
        "run",
        "--format", "json",
        "--pure",
    ]


def _resolve_home_dir() -> str:
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
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    return env


class MiMoCodeClient:
    """Thin wrapper around the `mimo run` CLI."""

    def __init__(
        self,
        *,
        api_key: str = "mimocode-cli",
        base_url: str = MIMOCODE_BASE_URL,
        command: str | None = None,
        args: list[str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._command = command or _resolve_command()
        self._args = args or _resolve_args()

    def _create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Any:
        """Run `mimo run` in JSON mode and return a SimpleNamespace mimicking
        an OpenAI ChatCompletion response."""
        # Build the prompt from messages — extract system + user content
        instructions = ""
        user_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "system":
                instructions = str(content).strip()
            elif role in ("user", "developer"):
                user_parts.append(str(content).strip())

        prompt = "\n\n".join(user_parts) if user_parts else ""

        cmd = [self._command] + self._args
        if model:
            cmd += ["--model", model]
        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()

        _logger.info("[mimocode-cli] running: %s", " ".join(cmd[:8]))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=False,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"MiMoCode CLI timed out after {timeout}s")

        if proc.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"MiMoCode CLI exited {proc.returncode}: {stderr}")

        # Parse JSON events from stdout
        text_parts = []
        tool_calls = []
        usage = {}
        total_tokens = 0

        for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            part = event.get("part", {})

            if etype == "text":
                text_parts.append(part.get("text", ""))
            elif etype == "tool_use":
                tool_input = part.get("state", {}).get("input", {})
                tool_calls.append({
                    "id": part.get("callID", f"call_{len(tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": part.get("tool", "unknown"),
                        "arguments": json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input),
                    },
                })
            elif etype == "step_finish":
                tokens = part.get("tokens", {})
                usage = {
                    "prompt_tokens": tokens.get("input", 0),
                    "completion_tokens": tokens.get("output", 0),
                    "total_tokens": tokens.get("total", 0),
                }

        content_text = "\n".join(text_parts) if text_parts else ""

        message = SimpleNamespace(
            role="assistant",
            content=content_text if not tool_calls else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        choices = [SimpleNamespace(index=0, message=message, finish_reason="stop" if not tool_calls else "tool_calls")]
        return SimpleNamespace(choices=choices, usage=SimpleNamespace(**usage), model=model)

    def run_with_tool_bridge(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        """Generator that yields events for tool-call bridging.

        Yields dicts with type: text, tool_call, tool_result, or final.
        """
        # For MiMoCode, the CLI handles tools internally, so we just run
        # the full request and yield the results.
        result = self._create_chat_completion(
            model=model,
            messages=messages,
            tools=tools,
            timeout=timeout,
        )

        # Yield text content
        if result.choices[0].message.content:
            yield {
                "type": "assistant_text",
                "text": result.choices[0].message.content,
            }

        # Yield tool calls
        if result.choices[0].message.tool_calls:
            for tc in result.choices[0].message.tool_calls:
                yield {
                    "type": "tool_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                }

        # Yield final
        usage = {}
        if hasattr(result, "usage") and result.usage:
            usage = {
                "input_tokens": getattr(result.usage, "prompt_tokens", 0),
                "output_tokens": getattr(result.usage, "completion_tokens", 0),
                "total_tokens": getattr(result.usage, "total_tokens", 0),
            }
        yield {
            "type": "final",
            "model": model,
            "usage": usage,
        }
