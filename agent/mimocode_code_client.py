"""OpenAI-compatible shim that forwards Hermes requests to `mimo run`.

Supports two modes:
1. Simple mode: runs `mimo run --format json` and parses JSON events.
2. MCP bridge mode: runs `mimo run --mcp-config <config>` with a proxy
   that lets Hermes intercept and execute tool calls.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path
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
    return ["run", "--format", "json", "--pure"]


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
        """Simple mode: run `mimo run --format json` and parse events."""
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
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, text=False,
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

        text_parts = []
        tool_calls = []
        usage = {}

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
        """MCP bridge mode: run with MCP proxy for tool interception.

        Yields events: text, tool_call, tool_result, final.
        """
        # Build the prompt from messages
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

        # Create temp files for MCP bridge communication
        session_id = f"hermes_{int(time.time())}_{os.getpid()}"
        tools_file = tempfile.mktemp(suffix=".json", prefix="hermes_tools_")
        queue_in = f"/tmp/hermes_queue_{session_id}.in"
        queue_out_dir = f"/tmp/hermes_result_{session_id}"

        # Write tools manifest
        os.makedirs(queue_out_dir, exist_ok=True)
        tool_schemas = []
        if tools:
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    tool_schemas.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    })
        with open(tools_file, "w") as f:
            json.dump(tool_schemas, f)

        # Create MCP config
        bridge_script = str(Path(__file__).parent / "mimocode_mcp_bridge.py")
        mcp_config = {
            "mcpServers": {
                "hermes-tools": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [bridge_script],
                    "env": {
                        "HERMES_TOOLS_FILE": tools_file,
                        "HERMES_QUEUE_IN": queue_in,
                        "HERMES_QUEUE_OUT_DIR": queue_out_dir,
                    },
                }
            }
        }
        config_file = tempfile.mktemp(suffix=".json", prefix="mimocode_mcp_")
        with open(config_file, "w") as f:
            json.dump(mcp_config, f)

        cmd = [self._command] + self._args
        if model:
            cmd += ["--model", model]
        cmd += ["--mcp-config", config_file]
        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()
        _logger.info("[mimocode-cli] MCP bridge: %s", " ".join(cmd[:6]))

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, text=False,
        )

        try:
            # Read stdout line by line for streaming
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                part = event.get("part", {})

                if etype == "text":
                    yield {"type": "text", "text": part.get("text", "")}
                elif etype == "tool_use":
                    tool_name = part.get("tool", "unknown")
                    call_id = part.get("callID", "")
                    tool_input = part.get("state", {}).get("input", {})

                    yield {
                        "type": "tool_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": tool_input if isinstance(tool_input, dict) else {},
                    }
                elif etype == "step_finish":
                    tokens = part.get("tokens", {})
                    yield {
                        "type": "final",
                        "model": model,
                        "usage": {
                            "input_tokens": tokens.get("input", 0),
                            "output_tokens": tokens.get("output", 0),
                            "total_tokens": tokens.get("total", 0),
                        },
                    }

        except Exception as exc:
            _logger.error("[mimocode-cli] MCP bridge error: %s", exc)
            yield {"type": "error", "message": str(exc)}
        finally:
            proc.kill()
            proc.wait()
            # Cleanup temp files
            for f in [tools_file, config_file]:
                try:
                    os.unlink(f)
                except Exception:
                    pass
            try:
                import shutil
                shutil.rmtree(queue_out_dir, ignore_errors=True)
            except Exception:
                pass
