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
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_logger = logging.getLogger(__name__)

MIMOCODE_BASE_URL = "mimocode://codex"
DEFAULT_TIMEOUT_SECONDS = 300.0

MODEL_MAP = {
    "mimocode-cli": "mimo/mimo-auto",
    "mimo-auto": "mimo/mimo-auto",
    "mimo": "mimo/mimo-auto",
}


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
        self._queue_in_path: str | None = None
        self._queue_out_dir: str | None = None

    def _setup_mcp_workspace(self, tools: list[dict[str, Any]]) -> str:
        """Create a temp workspace with MCP bridge config and agent definition.

        Creates:
        - .mcp.json pointing to mimocode_mcp_bridge.py
        - .mimocode/agents/hermes.md with tool_allowlist to disable built-in tools
        - tools.json manifest for the MCP bridge

        Returns the workspace path to use as CWD for the mimo process.
        """
        workspace = tempfile.mkdtemp(prefix="hermes_mcp_")

        # Write MCP config
        bridge_script = str(Path(__file__).parent / "mimocode_mcp_bridge.py")
        mcp_config = {
            "mcpServers": {
                "hermes-tools": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [bridge_script],
                    "env": {
                        "HERMES_TOOLS_FILE": os.path.join(workspace, "tools.json"),
                        "HERMES_QUEUE_IN": os.path.join(workspace, "queue.in"),
                        "HERMES_QUEUE_OUT_DIR": os.path.join(workspace, "result"),
                    }
                }
            }
        }
        with open(os.path.join(workspace, ".mcp.json"), "w") as f:
            json.dump(mcp_config, f)

        # Write tools manifest
        tool_schemas = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_schemas.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        with open(os.path.join(workspace, "tools.json"), "w") as f:
            json.dump(tool_schemas, f)

        # Create agent definition with tool_allowlist
        agent_dir = os.path.join(workspace, ".mimocode", "agents")
        os.makedirs(agent_dir, exist_ok=True)

        tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("type") == "function"]
        allowlist_yaml = "\n".join(f"  - {name}" for name in tool_names if name)

        with open(os.path.join(agent_dir, "hermes.md"), "w") as f:
            f.write("---\n")
            f.write("name: hermes\n")
            f.write("description: Hermes tool proxy agent\n")
            f.write("tool_allowlist:\n")
            f.write(allowlist_yaml + "\n")
            f.write("---\n\n")
            f.write("You are a helpful assistant. Use tools when needed.\n")

        # Create queue and result dirs
        os.makedirs(os.path.join(workspace, "result"), exist_ok=True)

        self._queue_in_path = os.path.join(workspace, "queue.in")
        self._queue_out_dir = os.path.join(workspace, "result")

        _logger.info("[mimocode-cli] MCP workspace: %s tools=%s", workspace, tool_names)
        return workspace

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
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
        return "\n\n".join(user_parts) if user_parts else ""

    def _create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Any:
        """Simple mode: run `mimo run --format json` and parse events.

        When tools are provided, creates a temporary working directory with:
        - .mcp.json pointing to mimocode_mcp_bridge.py for MCP tool interception
        - .mimocode/agents/hermes.md with tool_allowlist to disable built-in tools
        This forces the mimo CLI to call MCP-proxied tools instead of built-ins.
        """
        prompt = self._build_prompt(messages)
        model_name = MODEL_MAP.get(model, model) if model else "mimo/mimo-auto"

        cmd = [self._command] + self._args
        if model_name:
            cmd += ["--model", model_name]

        cwd = None
        if tools:
            # Set up MCP bridge workspace for tool interception
            cwd = self._setup_mcp_workspace(tools)
            cmd += ["--agent", "hermes"]

        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()
        _logger.info("[mimocode-cli] simple mode: %s cwd=%s", " ".join(cmd[:8]), cwd)

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, text=False,
            cwd=cwd,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(f"MiMoCode CLI timed out after {timeout}s")
        finally:
            if cwd:
                try:
                    import shutil
                    shutil.rmtree(cwd, ignore_errors=True)
                except Exception:
                    pass

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
            content=content_text if content_text else None,
            tool_calls=tool_calls if tool_calls and not content_text else None,
        )
        choices = [SimpleNamespace(index=0, message=message, finish_reason="stop")]
        return SimpleNamespace(choices=choices, usage=SimpleNamespace(**usage), model=model)

    def _write_tools_manifest(self, tools: list[dict[str, Any]]) -> str:
        """Write tools manifest to a temp file. Returns the path."""
        path = tempfile.mktemp(suffix=".json", prefix="hermes_tools_")
        tool_schemas = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_schemas.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        with open(path, "w") as f:
            json.dump(tool_schemas, f)
        return path

    def _build_mcp_config(self, tools_manifest: str) -> str:
        """Write MCP config and return the path."""
        bridge_script = str(Path(__file__).parent / "mimocode_mcp_bridge.py")
        mcp_config = {
            "mcpServers": {
                "hermes-tools": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [bridge_script],
                    "env": {
                        "HERMES_TOOLS_FILE": tools_manifest,
                        "HERMES_QUEUE_IN": self._queue_in_path or "",
                        "HERMES_QUEUE_OUT_DIR": self._queue_out_dir or "",
                    },
                }
            }
        }
        path = tempfile.mktemp(suffix=".json", prefix="mimocode_mcp_")
        with open(path, "w") as f:
            json.dump(mcp_config, f)
        return path

    def run_with_tool_bridge(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        """MCP bridge mode: run with MCP proxy for tool interception.

        Yields a sequence of dicts:
          {"type": "tool_call", "call_id": ..., "name": ..., "arguments": ...}
          {"type": "text", "text": ...}
          {"type": "final", "model": ..., "usage": ...}
          {"type": "error", "message": ...}

        Mirrors the Claude Code CLI bridge pattern: the mimo CLI is spawned
        with --mcp-config pointing to mimocode_mcp_bridge.py.  A background
        thread monitors the inbound queue file for tool calls from the MCP
        proxy, yielding them as events.  The caller writes placeholder results
        to the output directory so the MCP proxy unblocks.
        """
        prompt = self._build_prompt(messages)
        model_name = MODEL_MAP.get(model, model) if model else "mimo/mimo-auto"

        # Set up MCP bridge communication channels
        session_id = f"hermes_{int(time.time())}_{os.getpid()}"
        self._queue_in_path = f"/tmp/hermes_queue_{session_id}.in"
        self._queue_out_dir = f"/tmp/hermes_result_{session_id}"
        os.makedirs(self._queue_out_dir, exist_ok=True)

        tools_manifest = self._write_tools_manifest(tools or [])
        mcp_config = self._build_mcp_config(tools_manifest)

        # --pure disables external plugins (MCP).  Strip it when using MCP bridge.
        _args = [a for a in self._args if a != "--pure"]
        cmd = [self._command] + _args
        if model_name:
            cmd += ["--model", model_name]
        cmd += ["--mcp-config", mcp_config]
        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()
        _logger.info("[mimocode-cli] MCP bridge: %s", " ".join(cmd[:6]))

        # Truncate queue file
        try:
            if os.path.exists(self._queue_in_path):
                os.unlink(self._queue_in_path)
        except Exception:
            pass

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, text=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start mimo command '{self._command}'. "
                "Install MiMoCode CLI: npm install -g @mimo-ai/cli"
            ) from exc

        # Queue monitor thread — watches $HERMES_QUEUE_IN for tool calls
        # from the MCP bridge proxy, yielding them as events.
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
                        # Write empty placeholder so MCP proxy unblocks
                        result_path = os.path.join(self._queue_out_dir, f"{call_id}.json")
                        if not os.path.exists(result_path):
                            with open(result_path, "w", encoding="utf-8") as f:
                                json.dump({"content": ""}, f)
                except Exception as exc:
                    _logger.warning("[mimocode-cli] MCP queue monitor error: %s", exc)
                time.sleep(0.05)

        mcp_thread: threading.Thread | None = None
        if tools and self._queue_in_path and self._queue_out_dir:
            mcp_thread = threading.Thread(target=_watch_mcp_queue, daemon=True)
            mcp_thread.start()

        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Drain any pending MCP queue events before processing stdout
                for ev in _drain_mcp_events():
                    yield ev

                etype = event.get("type", "")
                part = event.get("part", {})

                if etype == "text":
                    yield {"type": "text", "text": part.get("text", "")}
                elif etype == "tool_use":
                    call_id = part.get("callID", uuid.uuid4().hex)
                    tool_name = part.get("tool", "unknown")
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

            proc.wait(timeout=timeout)

            # Drain any remaining MCP queue events
            for ev in _drain_mcp_events():
                yield ev

            if proc.returncode != 0 and proc.stderr:
                err_text = proc.stderr.read()[:500]
                if err_text:
                    yield {"type": "error", "message": f"exit {proc.returncode}: {err_text}"}

        except Exception as exc:
            _logger.error("[mimocode-cli] MCP bridge error: %s", exc)
            yield {"type": "error", "message": str(exc)}
        finally:
            mcp_stop.set()
            if mcp_thread is not None:
                mcp_thread.join(timeout=1.0)
            # Cleanup temp files
            for _path in [tools_manifest, mcp_config, self._queue_in_path]:
                if _path and os.path.exists(_path):
                    try:
                        os.unlink(_path)
                    except Exception:
                        pass
            if self._queue_out_dir and os.path.isdir(self._queue_out_dir):
                try:
                    import shutil
                    shutil.rmtree(self._queue_out_dir, ignore_errors=True)
                except Exception:
                    pass
            self._queue_in_path = None
            self._queue_out_dir = None
