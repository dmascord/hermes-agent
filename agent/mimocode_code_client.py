"""OpenAI-compatible shim that forwards Hermes requests to `mimo run`.

Supports two modes:
1. Simple mode: runs `mimo run --format json` and parses JSON events.
2. MCP bridge mode: runs `mimo run --mcp-config <config>` with a proxy
   that lets Hermes intercept and execute tool calls.
"""

from __future__ import annotations

import json
import hashlib
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

# Built-in mimo tools that MCP tools can replace
BUILTIN_TOOLS = [
    "bash", "read", "write", "edit", "glob", "grep",
    "webfetch", "websearch", "codesearch", "lsp",
    "actor", "skill", "memory", "history", "task", "workflow",
]


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
        """Create a temp workspace that forces mimo to use MCP tools only.

        Architecture (from MiMo-Code source analysis):
        - MCP server named "mcp" → tools become mcp_bash, mcp_read, etc.
        - Permission { "*": "deny", "mcp_*": "allow" } suppresses built-in tools
        - Agent prompt instructs model to use MCP-backed tools

        Creates:
        - .mcp.json with server named "mcp" (produces mcp_* tool IDs)
        - .mimocode/agents/hermes.md with permission deny-all + allow mcp_*
        - tools.json manifest for the MCP bridge
        - queue.in and result/ for tool call proxying
        """
        workspace = tempfile.mkdtemp(prefix="hermes_mcp_")
        bridge_script = str(Path(__file__).parent / "mimocode_mcp_bridge.py")

        # Server named "mcp" → tool IDs become mcp_bash, mcp_read, etc.
        # MCP config goes in .mimocode/mimocode.json (not .mcp.json — mimo CLI reads this format)
        mimocode_dir = os.path.join(workspace, ".mimocode")
        os.makedirs(mimocode_dir, exist_ok=True)
        mimocode_config = {
            "mcp": {
                "mcp": {
                    "type": "local",
                    "command": [sys.executable, bridge_script],
                    "environment": {
                        "HERMES_TOOLS_FILE": os.path.join(workspace, "tools.json"),
                        "HERMES_QUEUE_IN": os.path.join(workspace, "queue.in"),
                        "HERMES_QUEUE_OUT_DIR": os.path.join(workspace, "result"),
                    }
                }
            }
        }
        with open(os.path.join(mimocode_dir, "mimocode.json"), "w") as f:
            json.dump(mimocode_config, f)

        # Build tool schemas (gateway already added mcp_ prefix, so just record names)
        tool_schemas = []
        mcp_tool_names = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_name = func.get("name", "")
                tool_schemas.append({
                    "name": tool_name,
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
                # The gateway has already prefixed the tool name with mcp_ for
                # mimocode-cli. The agent.md allowlist should use the SAME
                # prefixed name (NOT add another mcp_ on top — that would
                # produce mcp_mcp_bash and the model would call a tool that
                # doesn't exist).
                mcp_tool_names.append(tool_name)

        with open(os.path.join(workspace, "tools.json"), "w") as f:
            json.dump(tool_schemas, f)

        # Create agent definition
        # Permission: deny all tools (*), allow only mcp_* prefixed tools
        # This forces the model to use MCP-proxied tools exclusively
        agent_dir = os.path.join(workspace, ".mimocode", "agents")
        os.makedirs(agent_dir, exist_ok=True)

        tool_list = ", ".join(mcp_tool_names)

        with open(os.path.join(agent_dir, "hermes.md"), "w") as f:
            f.write("---\n")
            f.write("name: hermes\n")
            f.write("description: Hermes tool proxy agent\n")
            f.write("permission:\n")
            f.write("  '*': deny\n")
            for name in mcp_tool_names:
                f.write(f"  '{name}': allow\n")
            f.write("tool_allowlist:\n")
            for name in mcp_tool_names:
                f.write(f"  - {name}\n")
            f.write("---\n\n")
            f.write(
                "You are running through the hermes-agent gateway "
                "(https://hermes.tusker.net.au), which proxies your responses to a "
                "connected client. The connected client's filesystem and workspace is "
                f"accessible to you via the MCP tools exposed by this gateway: {tool_list}. "
                "Do not claim you are in a sandbox or that you cannot access local files — "
                "use the MCP tools to read, edit, and run commands. If a specific tool you "
                "need is not registered, ask the user to add it via the hermes tool "
                "registry.\n\n"
                f"You have access to tools: {tool_list}.\n"
                "Use these tools to help the user. "
                "When a task requires running a command, reading a file, "
                "or any code operation, use the appropriate mcp_ prefixed tool.\n\n"
                "Important: invoke the MCP-prefixed tool names (e.g. mcp_bash, "
                "mcp_read, mcp_write, mcp_edit) rather than the bare names. "
                "The bridge routes the MCP tools to the connected client for "
                "real filesystem access.\n"
            )

        # Create queue and result dirs
        os.makedirs(os.path.join(workspace, "result"), exist_ok=True)
        with open(os.path.join(workspace, "queue.in"), "w"):
            pass

        self._queue_in_path = os.path.join(workspace, "queue.in")
        self._queue_out_dir = os.path.join(workspace, "result")

        _logger.info("[mimocode-cli] MCP workspace: %s tools=%s", workspace, mcp_tool_names)
        return workspace

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result."""
        # Strip mcp_ prefix to get the real tool name
        real_name = tool_name[4:] if tool_name.startswith("mcp_") else tool_name

        if real_name == "bash":
            import subprocess as sp
            cmd = arguments.get("command", "")
            try:
                result = sp.run(
                    ["sh", "-c", cmd],
                    capture_output=True, text=True, timeout=120,
                )
                output = result.stdout
                if result.returncode != 0:
                    output += f"\n[exit code {result.returncode}]"
                    if result.stderr:
                        output += f"\n{result.stderr}"
                return output.strip()
            except sp.TimeoutExpired:
                return "[timeout: command took too long]"
            except Exception as exc:
                return f"[error executing command: {exc}]"

        return f"[tool {tool_name} not implemented in gateway]"

    def _watch_mcp_queue(self, stop_event: threading.Event) -> None:
        """Monitor the MCP queue file and execute tool calls."""
        if not self._queue_in_path or not self._queue_out_dir:
            return
        offset = 0
        while not stop_event.is_set():
            try:
                if os.path.exists(self._queue_in_path):
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
                        if not call_id:
                            continue
                        result_path = os.path.join(self._queue_out_dir, f"{call_id}.json")
                        if not os.path.exists(result_path):
                            tool_name = str(call.get("tool") or "")
                            arguments = call.get("arguments") or {}
                            output = self._execute_tool(tool_name, arguments)
                            with open(result_path, "w", encoding="utf-8") as f:
                                json.dump({"content": output}, f)
                            _logger.info("[mimocode-cli] executed %s for %s", tool_name, call_id)
            except Exception as exc:
                _logger.warning("[mimocode-cli] watcher error: %s", exc)
            time.sleep(0.05)

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
        continue_session: bool = False,
        session_id: str | None = None,
    ) -> Any:
        """Run mimo CLI and parse events.

        When tools are provided:
        - Creates temp workspace with MCP bridge (server named "mcp")
        - Agent permission denies all built-in tools, allows only mcp_* tools
        - Uses streaming reader with watcher thread to handle MCP tool results

        When continue_session=True, passes --continue (or --session <id>) to
        mimo so it loads the existing session from disk and continues the
        conversation. This preserves the full agent loop including tool
        execution history.
        """
        prompt = self._build_prompt(messages)
        model_name = MODEL_MAP.get(model, model) if model else "mimo/mimo-auto"

        cmd = [self._command] + self._args
        if model_name:
            cmd += ["--model", model_name]

        cwd = None
        if tools:
            cwd = self._setup_mcp_workspace(tools)
            cmd += ["--agent", "hermes"]

        if continue_session:
            if session_id:
                cmd += ["--session", session_id]
            else:
                cmd += ["--continue"]

        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()
        _logger.info("[mimocode-cli] running: %s cwd=%s", " ".join(cmd[:8]), cwd)

        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, text=False,
            cwd=cwd,
        )

        text_parts = []
        tool_calls = []
        usage = {}

        # Drain stderr in a background thread to prevent pipe buffer deadlock
        stderr_lines: list[str] = []
        def _drain_stderr():
            try:
                while True:
                    chunk = proc.stderr.read(4096)
                    if not chunk:
                        break
                    stderr_lines.append(chunk.decode("utf-8", errors="replace"))
            except Exception:
                pass
        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        try:
            if tools and self._queue_in_path and self._queue_out_dir:
                # MCP mode: read stdout line-by-line with watcher thread for tool results
                watcher_stop = threading.Event()
                watcher_thread = threading.Thread(
                    target=self._watch_mcp_queue, args=(watcher_stop,), daemon=True
                )
                watcher_thread.start()

                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    raw_line = proc.stdout.readline()
                    if not raw_line:
                        break
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
                        text_parts.append(part.get("text", ""))
                        # mimo CLI sometimes emits tool_call as raw XML text
                        # instead of a structured tool_use event. Parse it.
                        _xml_tc = _parse_tool_call_xml(part.get("text", ""))
                        if _xml_tc:
                            tool_calls.append(_xml_tc)
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

                watcher_stop.set()
                watcher_thread.join(timeout=2)
                proc.wait(timeout=10)
            else:
                # Simple mode: no tools, just read all output
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)

                if proc.returncode != 0:
                    stderr = stderr_bytes.decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"MiMoCode CLI exited {proc.returncode}: {stderr}")

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
                        # mimo CLI sometimes emits tool_call as raw XML text
                        # instead of a structured tool_use event. Parse it.
                        _xml_tc = _parse_tool_call_xml(part.get("text", ""))
                        if _xml_tc:
                            tool_calls.append(_xml_tc)
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
        finally:
            proc.kill()
            proc.wait()
            stderr_thread.join(timeout=5)
            if stderr_lines:
                _logger.warning("[mimocode-cli] stderr: %s", "".join(stderr_lines)[:500])
            if cwd:
                try:
                    import shutil
                    shutil.rmtree(cwd, ignore_errors=True)
                except Exception:
                    pass

        content_text = "\n".join(text_parts) if text_parts else ""
        # Strip the <tool_call>...</tool_call> XML out of content_text if we
        # parsed it as a tool_call — the model emitted both forms and we
        # don't want the raw XML leaking into the response content.
        if tool_calls and "<tool_call>" in content_text:
            import re as _re
            content_text = _re.sub(r"<tool_call>.*?</tool_call>", "", content_text, flags=_re.DOTALL).strip()
        message = SimpleNamespace(
            role="assistant",
            content=content_text if content_text else None,
            # If we parsed tool calls (either from structured events or from
            # XML in the text), include them. Otherwise fall back to text only.
            tool_calls=tool_calls if tool_calls else None,
        )
        choices = [SimpleNamespace(index=0, message=message, finish_reason="stop")]
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

        Yields events: text, tool_call, final, error.
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

        # --pure disables external plugins (MCP). Strip it when using MCP bridge.
        _args = [a for a in self._args if a != "--pure"]
        cmd = [self._command] + _args
        if model_name:
            cmd += ["--model", model_name]
        cmd += ["--mcp-config", mcp_config]
        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()
        _logger.info("[mimocode-cli] MCP bridge: %s", " ".join(cmd[:6]))

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

                for ev in _drain_mcp_events():
                    yield ev

                etype = event.get("type", "")
                part = event.get("part", {})

                if etype == "text":
                    text_payload = part.get("text", "")
                    # mimo CLI sometimes emits tool_call as raw XML text
                    # instead of a structured tool_use event. Parse it and
                    # yield as tool_call so the gateway can route it to
                    # the connected client via tool_call_hub.
                    _xml_tc = _parse_tool_call_xml(text_payload)
                    if _xml_tc:
                        yield _xml_tc
                    else:
                        yield {"type": "text", "text": text_payload}
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


# Module-level helper, moved here so it doesn't break the class body.
def _parse_tool_call_xml(text: str) -> dict | None:
    """Parse mimo CLI's text-formatted tool_call XML.

    The mimo CLI emits tool_call as raw XML text in many different shapes.
    Five formats observed:
        <tool_call>
        {"name": "mcp_bash", "arguments": {"command": "..."}}
        </tool_call>
    or
        <tool_call>
        <function=mcp_bash>
        <parameter=command>...</parameter>
        </function>
        </tool_call>
    or (newer):
        <tool_call>
        <tool_name>mcp_bash</tool_name>
        <parameters>
        <command>...</command>
        </parameters>
        </tool_call>
    or (compact):
        <tool_call>
        <name>mcp_bash</name>
        <args>{"command": "..."}</args>
        </tool_call>
or (claude-style):
        <function_calls>
        <invoke name="mcp__X">
        <parameter name="command">...</parameter>
        </invoke>
        </function_calls>
    or (newer mimo, format 6):
        <tool_invocation name="mcp__hermes-tools__bash" arguments={"command": "..."} />
    Returns an OpenAI-format tool_call dict or None.
    """
    import re
    if (
        "<tool_call>" not in text
        and "<function_calls>" not in text
        and "<invoke name=" not in text
        and "<tool_invocation" not in text
    ):
        return None

    # Format 1: JSON-style tool_call
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            return {
                "id": f"call_{hashlib.md5(m.group(1).encode()).hexdigest()[:16]}",
                "type": "function",
                "function": {
                    "name": obj.get("name", "unknown"),
                    "arguments": json.dumps(args),
                },
            }
        except (json.JSONDecodeError, KeyError):
            pass

    # Format 2: <function=name><parameter=key>value</parameter></function>
    m2 = re.search(r"<function=([^>]+)>.*?<parameter=([^>]+)>(.*?)</parameter>.*?</function>", text, re.DOTALL)
    if m2:
        import uuid as _uuid
        name, key, value = m2.group(1), m2.group(2), m2.group(3).strip()
        return {
            "id": f"call_{_uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({key: value}),
            },
        }

    # Format 3: <tool_name>name</tool_name><parameters><key>value</key>...</parameters>
    m3 = re.search(r"<tool_name>([^<]+)</tool_name>", text)
    if m3:
        import uuid as _uuid
        name = m3.group(1).strip()
        # Find the <parameters>...</parameters> block (if present)
        params_m = re.search(r"<parameters>(.*?)</parameters>", text, re.DOTALL)
        params_block = params_m.group(1) if params_m else text
        # Strip the <tool_name>...</tool_name> if it's inside params_block
        params_block = re.sub(r"<tool_name>.*?</tool_name>", "", params_block, flags=re.DOTALL)
        # Extract all <key>value</key> pairs (non-nested)
        args = {}
        for am in re.finditer(r"<([a-zA-Z_][a-zA-Z0-9_]*)>\s*([^<]*?)\s*</\1>", params_block, re.DOTALL):
            key = am.group(1)
            if key in ("parameters",):
                continue
            args[key] = am.group(2).strip()
        return {
            "id": f"call_{_uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }

    # Format 4: <name>name</name><args>{json}</args>
    m4 = re.search(r"<name>([^<]+)</name>", text)
    if m4 and "<args>" in text:
        import uuid as _uuid
        name = m4.group(1).strip()
        args_m = re.search(r"<args>(.*?)</args>", text, re.DOTALL)
        args_text = args_m.group(1).strip() if args_m else "{}"
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {"raw": args_text}
        return {
            "id": f"call_{_uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }

    # Format 5: <function_calls><invoke name="mcp__X"><parameter name="key">value</parameter></invoke></function_calls>
    m5 = re.search(r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>", text, re.DOTALL)
    if m5:
        import uuid as _uuid
        name = m5.group(1).strip()
        args_block = m5.group(2)
        args = {}
        for am in re.finditer(r"<parameter\s+name=\"([^\"]+)\">(.*?)</parameter>", args_block, re.DOTALL):
            args[am.group(1)] = am.group(2).strip()
        return {
            "id": f"call_{_uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }

    # Format 6: <tool_invocation name="mcp__X" arguments={json} /> (self-closing)
    # Arguments are raw JSON, no quotes around them. The closing /> after
    # the JSON stops at the next > character.
    m6 = re.search(r'<tool_invocation\s+name="([^"]+)"\s+arguments=(\{[^>]+?)\s*/?>', text, re.DOTALL)
    if m6:
        import uuid as _uuid
        name = m6.group(1).strip()
        args_text = m6.group(2).strip()
        # Strip any trailing / that got included
        if args_text.endswith("/"):
            args_text = args_text[:-1].rstrip()
        # Try both raw and HTML-entity-decoded
        for candidate in (args_text,
                          args_text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")):
            try:
                args = json.loads(candidate)
                return {
                    "id": f"call_{_uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            except json.JSONDecodeError:
                continue
        return {
            "id": f"call_{_uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"raw": args_text}),
            },
        }

    return None
