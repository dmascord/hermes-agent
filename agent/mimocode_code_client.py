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
        - .mimocode/agent/hermes.md with tools: deny-all-builtins
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
        # Permission: deny all built-in tools, allow only MCP bridge tools.
        # The MCP bridge registers tools at runtime (mcp_bash, mcp_read, etc.)
        # which are NOT controlled by the agent's built-in tool permissions.
        agent_dir = os.path.join(workspace, ".mimocode", "agent")
        os.makedirs(agent_dir, exist_ok=True)

        tool_list = ", ".join(mcp_tool_names)

        with open(os.path.join(agent_dir, "hermes.md"), "w") as f:
            f.write("---\n")
            f.write("name: hermes\n")
            f.write("mode: primary\n")
            f.write("description: Hermes tool proxy agent - MCP bridge tools only\n")
            f.write("tools:\n")
            f.write("  bash: false\n")
            f.write("  read: false\n")
            f.write("  write: false\n")
            f.write("  edit: false\n")
            f.write("  glob: false\n")
            f.write("  grep: false\n")
            f.write("  webfetch: false\n")
            f.write("  actor: false\n")
            f.write("  task: false\n")
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
                "real filesystem access.\n\n"
                "CRITICAL — Use this EXACT format when invoking a tool:\n"
                '  <tool_invocation name="mcp_bash" arguments={"command": "ls -la"} />\n'
                '  <tool_invocation name="mcp_read" arguments={"path": "/path/to/file"} />\n'
                '  <tool_invocation name="mcp_write" arguments={"path": "/path/to/file", "content": "file contents here"} />\n'
                '  <tool_invocation name="mcp_edit" arguments={"path": "/path/to/file", "input": "EDIT_SPEC_HERE"} />\n'
                '  <tool_invocation name="mcp_find" arguments={"paths": ["src/**/*.py"]} />\n'
                '  <tool_invocation name="mcp_search_tool_bm25" arguments={"query": "search terms here"} />\n\n'
                "Rules:\n"
                "1. Emit ONE tool invocation per response (do not chain).\n"
                "2. The `arguments` value is raw JSON (no HTML entities, no escaping).\n"
                "3. ALL required parameters MUST be included — never emit empty arguments {}.\n"
                "   - mcp_bash REQUIRES \"command\" (the shell command string)\n"
                "   - mcp_read REQUIRES \"path\" (absolute file path)\n"
                "   - mcp_write REQUIRES \"path\" and \"content\"\n"
                "   - mcp_edit REQUIRES \"path\" and \"input\"\n"
                "   - mcp_find REQUIRES \"paths\" (array of glob patterns)\n"
                "   - mcp_search_tool_bm25 REQUIRES \"query\" (search string)\n"
                "4. Close with ` />` (self-closing), NOT `></tool_invocation>`.\n"
                "5. Do NOT wrap the invocation in code blocks (no ``` fences).\n"
                "6. Do NOT describe what you'll do in prose — emit the invocation directly.\n"
                "7. After invoking, STOP. The bridge will execute and return a result.\n"
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
            elif role == "assistant" and not content and msg.get("tool_calls"):
                # Preserve assistant tool_calls as text so they're not
                # silently dropped in multi-turn restart scenarios.
                tc_texts = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tc_texts.append(
                        f"[tool_call: {fn.get('name', '?')} "
                        f"args={fn.get('arguments', '{}')}]"
                    )
                if tc_texts:
                    user_parts.append("Assistant tool calls:\n" + "\n".join(tc_texts))
            elif role == "tool":
                # Preserve tool results as text context.
                tc_id = msg.get("tool_call_id", "")
                c = content or ""
                user_parts.append(f"Tool result ({tc_id}):\n{c}")
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

        cmd = [self._command]
        # When tools are provided, strip --pure (disables MCP/external plugins)
        # so the MCP bridge can work. --pure must be removed because it
        # prevents mimo from loading the .mimocode/mimocode.json config that
        # _setup_mcp_workspace() creates.
        _effective_args = [a for a in self._args if a != "--pure"] if tools else self._args
        cmd += _effective_args
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
                if time.monotonic() >= deadline:
                    _logger.warning(
                        "[mimocode-cli] MCP mode timeout after %.1fs — process may be stuck in polling loop",
                        timeout,
                    )
                    raise RuntimeError(
                        f"MiMoCode CLI exceeded {timeout}s timeout in MCP mode — "
                        "the process may be stuck in a polling loop or "
                        "the API may be rate-limiting this IP"
                    )
            else:
                # Simple mode: no tools — use deadline-based stdout reader
                # with hard kill-timer. Replaces proc.communicate(timeout=timeout)
                # which could hang for the full timeout if the mimo binary enters
                # its polling-loop bug (GET /session/status every 750ms).
                deadline = time.monotonic() + timeout
                _stdout_lines: list[str] = []
                while time.monotonic() < deadline:
                    raw_line = proc.stdout.readline()
                    if not raw_line:
                        break
                    _stdout_lines.append(raw_line.decode("utf-8", errors="replace"))
                else:
                    # Deadline reached — force-kill the process
                    _logger.warning(
                        "[mimocode-cli] timeout after %.1fs — force-killing process",
                        timeout,
                    )
                    proc.kill()
                    proc.wait(timeout=10)
                    # Drain any remaining stderr before raising
                    stderr_thread.join(timeout=2)
                    if stderr_lines:
                        _logger.warning(
                            "[mimocode-cli] stderr at timeout: %s",
                            "".join(stderr_lines)[:500],
                        )
                    raise RuntimeError(
                        f"MiMoCode CLI exceeded {timeout}s timeout — "
                        "the process may be stuck in a polling loop or "
                        "the API may be rate-limiting this IP"
                    )

                if proc.returncode not in (0, None):
                    _drain_stderr_rem = ""
                    try:
                        _drain_stderr_rem += proc.stderr.read(4096).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    stderr_msg = ("".join(stderr_lines) + _drain_stderr_rem)[:500]
                    raise RuntimeError(f"MiMoCode CLI exited {proc.returncode}: {stderr_msg}")

                for line in _stdout_lines:
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

    def stream_events(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        """Generator that runs mimo CLI and yields events.

        Yields event dicts with types:
          - {"type": "text", "text": "..."}
          - {"type": "tool_call", "call_id": "...", "name": "...", "arguments": {...}}
          - {"type": "final", "model": "...", "usage": {...}}
          - {"type": "error", "message": "..."}

        Unlike _create_chat_completion(), this does NOT use a watcher thread
        that executes tools locally. Instead, tool_call events are yielded to
        the caller, which should handle them by:
          1. Registering with tool_call_hub
          2. Waiting for the connected client to POST the result
          3. Writing the result to <self._queue_out_dir>/<call_id>.json
          4. The MCP bridge reads the result and returns it to the subprocess
          5. The subprocess continues producing events

        This avoids the race where the watcher executes tools locally (with
        wrong results for non-bash tools) before the gateway can proxy them
        to the connected client.
        """
        prompt = self._build_prompt(messages)
        model_name = MODEL_MAP.get(model, model) if model else "mimo/mimo-auto"

        cmd = [self._command]
        _effective_args = [a for a in self._args if a != "--pure"] if tools else self._args
        cmd += _effective_args
        if model_name:
            cmd += ["--model", model_name]

        cwd = None
        if tools:
            cwd = self._setup_mcp_workspace(tools)
            cmd += ["--agent", "hermes"]

        if prompt:
            cmd += [prompt]

        env = _build_subprocess_env()
        _logger.info("[mimocode-cli] stream_events: %s cwd=%s", " ".join(cmd[:8]), cwd)

        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, text=False,
            cwd=cwd,
        )

        # Drain stderr in a background thread
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

        # Character-level StreamSieve to catch tool calls that span multiple
        # SSE JSON lines. Without this, a long <tool_call>...</tool_call> that the
        # mimo CLI emits across two `proc.stdout.readline()` boundaries is
        # missed because each half fails the regex match in
        # _parse_tool_call_xml. The sieve buffers text internally and yields
        # (text, tool_call) events as soon as a complete block is detected.
        # We initialise it lazily on the first text event (need the tool_names
        # at that point; we read them from any tool_use event's part if seen).
        _sieve: _StreamSieve | None = None
        _known_tool_names: list[str] = []
        # Pre-populate tool_names from the tools list (best-effort, may be empty)
        if tools:
            for t in tools:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                n = fn.get("name") if isinstance(fn, dict) else None
                if not n and isinstance(t, dict):
                    n = t.get("name")
                if n:
                    _known_tool_names.append(n)

        def _sieve_parse(buf: str, names: list[str]) -> tuple[list[dict] | None, str]:
            tc = _parse_tool_call_xml(buf, names)
            if tc:
                return [tc], _clean_tool_text(buf)
            return None, buf

        try:
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
                    text_payload = part.get("text", "")
                    if not text_payload:
                        continue
                    # Lazy-init the sieve on first text event so we have
                    # tool_names for name resolution.
                    if _sieve is None:
                        _sieve = _StreamSieve(_sieve_parse, _known_tool_names)
                    # Feed the text through the sieve; the sieve catches
                    # tool calls that span multiple text events.
                    for kind, data in _sieve.feed(text_payload):
                        if kind == "text":
                            # Clean up any tool-call residue (e.g. partial
                            # TOOL_CALL: lines that the sieve flushed)
                            yield {"type": "text", "text": _clean_tool_text(data)}
                        elif kind == "tool_call":
                            for tc in data:
                                _func = tc.get("function", {})
                                _raw_args = _func.get("arguments", "{}")
                                if isinstance(_raw_args, str):
                                    try:
                                        _parsed_args = json.loads(_raw_args)
                                    except json.JSONDecodeError:
                                        _parsed_args = {"raw": _raw_args}
                                else:
                                    _parsed_args = _raw_args
                                yield {
                                    "type": "tool_call",
                                    "call_id": tc.get("id", uuid.uuid4().hex),
                                    "name": _func.get("name", "unknown"),
                                    "arguments": _parsed_args
                                    if isinstance(_parsed_args, dict)
                                    else {},
                                }
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

            # Flush any buffered text/tool_call from the sieve
            if _sieve is not None:
                for kind, data in _sieve.flush():
                    if kind == "text":
                        if data:
                            yield {"type": "text", "text": _clean_tool_text(data)}
                    elif kind == "tool_call":
                        for tc in data:
                            _func = tc.get("function", {})
                            _raw_args = _func.get("arguments", "{}")
                            if isinstance(_raw_args, str):
                                try:
                                    _parsed_args = json.loads(_raw_args)
                                except json.JSONDecodeError:
                                    _parsed_args = {"raw": _raw_args}
                            else:
                                _parsed_args = _raw_args
                            yield {
                                "type": "tool_call",
                                "call_id": tc.get("id", uuid.uuid4().hex),
                                "name": _func.get("name", "unknown"),
                                "arguments": _parsed_args
                                if isinstance(_parsed_args, dict)
                                else {},
                            }

            # After events: check timeout
            if time.monotonic() >= deadline:
                _logger.warning(
                    "[mimocode-cli] stream_events timeout after %.1fs — force-killing process",
                    timeout,
                )
                proc.kill()
                proc.wait(timeout=10)
                stderr_thread.join(timeout=2)
                if stderr_lines:
                    _logger.warning(
                        "[mimocode-cli] stream_events stderr at timeout: %s",
                        "".join(stderr_lines)[:500],
                    )
                yield {"type": "error", "message": f"MiMoCode CLI exceeded {timeout}s timeout"}
                return

            proc.wait(timeout=10)
            stderr_thread.join(timeout=2)

            if proc.returncode not in (0, None):
                stderr_msg = ("".join(stderr_lines))[:500]
                yield {"type": "error", "message": f"MiMoCode CLI exited {proc.returncode}: {stderr_msg}"}

        except Exception as exc:
            _logger.error("[mimocode-cli] stream_events error: %s", exc)
            yield {"type": "error", "message": str(exc)}
        finally:
            proc.kill()
            proc.wait(timeout=5)
            stderr_thread.join(timeout=2)
            # Clean up MCP workspace
            if cwd:
                try:
                    import shutil
                    shutil.rmtree(cwd, ignore_errors=True)
                except Exception:
                    pass


# Module-level helpers, moved here so they don't break the class body.
# These were adopted from Fly143/MiMo2API (MIT licensed) to handle additional
# tool call formats the mimo CLI has been observed to emit, plus general
# noise tolerance and markdown-fence skipping. Algorithms adapted, not copied.


def _find_balanced_json(text: str, start: int) -> str:
    """Find balanced JSON object starting at `start`.

    Tracks brace depth with string-aware escape handling. Returns the
    substring (including the surrounding braces) or empty string if no
    balanced object exists.

    Used by Format 10 and 11 to extract JSON payloads that may contain
    nested objects/arrays (e.g. arguments with nested structure).
    """
    if start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _auto_type(val: str) -> Any:
    """Auto-detect type: bool, int, float, list, dict, or string.

    Used after XML parameter extraction to coerce values to the type
    the tool schema expects. Recognises:
      - booleans: true / false (case-insensitive)
      - null: null / none
      - integers and floats
      - JSON arrays and objects: [1, 2] / {"k": "v"}
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("null", "none"):
        return None
    # JSON array or object
    if s.startswith(("[", "{")):
        try:
            parsed = json.loads(s)
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return val


def _resolve_tool_name(name: str, tool_names: list[str] | None) -> str | None:
    """Resolve tool name with 4-level matching.

    1. Exact match
    2. Case-insensitive match
    3. camelCase -> snake_case match
    4. snake_case case-insensitive match

    Returns the canonical name from `tool_names` or None if not found.
    `tool_names` may be None (in which case `name` is returned as-is).
    """
    if not name:
        return None
    if not tool_names:
        return name
    if name in tool_names:
        return name
    name_lower = name.lower()
    for tn in tool_names:
        if tn.lower() == name_lower:
            return tn
    # camelCase -> snake_case
    import re as _re_r
    snake = _re_r.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()
    if snake in tool_names:
        return snake
    for tn in tool_names:
        if tn.lower() == snake:
            return tn
    return None


def _skip_fenced_block(text: str, i: int) -> tuple[int, str | None]:
    """Skip markdown code blocks (``` and ~~~) starting at position i.

    Returns (new_position, skipped_text or None).
    If the position is at a fence opener, returns the position past the
    matching fence close (or end of text) and the skipped text.
    Otherwise returns (i, None).
    """
    n = len(text)
    if i >= n:
        return i, None
    char = text[i]
    if char not in ("`", "~"):
        return i, None
    # Count fence length
    fence_len = 0
    while i + fence_len < n and text[i + fence_len] == char:
        fence_len += 1
    if fence_len < 3:
        return i, None
    # Find matching close
    j = i + fence_len
    nl = text.find("\n", j)
    if nl < 0:
        return i, None
    j = nl + 1
    while j < n:
        nl2 = text.find("\n", j)
        if nl2 < 0:
            return n, text[i:n]
        line_start = nl2 + 1
        close_len = 0
        while (
            line_start + close_len < n
            and text[line_start + close_len] == char
        ):
            close_len += 1
        if close_len >= fence_len:
            end = line_start + fence_len
            if end < n and text[end] == "\n":
                end += 1
            return end, text[i:end]
        j = line_start + 1
    return n, text[i:n]


def _strip_mimoml(text: str) -> str:
    """Strip MiMoML noise from tool call tags, return normalised XML.

    Handles 8+ variants observed in mimo's native MiMoML format:
      <|MiMoML|tool_calls>     <tool_calls>
      <MiMoML|tool_calls>      <tool_calls>     (missing leading |)
      <|MiMoML tool_calls>     <tool_calls>     (space instead of |)
      <｜MiMoML｜tool_calls>    <tool_calls>     (full-width pipes ｜)
      <MiMoMLtool_calls>       <tool_calls>     (no separator)
      <mimoml-tool_calls>      <tool_calls>     (hyphen)
      <mimoml_tool_calls>      <tool_calls>     (underscore)
      <|MiMoML|function_calls> <function_calls> (function_calls variant)

    Skips content inside ``` and ~~~ markdown fences (those are
    documentation examples, not actual tool calls).
    """
    if not text:
        return text
    result = []
    i = 0
    n = len(text)
    while i < n:
        # Markdown fence? skip the whole block
        new_i, skipped = _skip_fenced_block(text, i)
        if skipped is not None:
            result.append(skipped)
            i = new_i
            continue
        c = text[i]
        if c != "<":
            result.append(c)
            i += 1
            continue
        # Find end of tag
        end = text.find(">", i)
        if end == -1:
            result.append(text[i:])
            break
        inner = text[i + 1 : end]
        closing = inner.startswith("/")
        rest = inner[1:] if closing else inner
        # Consume MiMoML noise
        j = 0
        is_mimoml = False
        rest_len = len(rest)
        while j < rest_len:
            ch = rest[j]
            if ch in "| ｜\t\r\n ":
                j += 1
                is_mimoml = True
                continue
            if (
                rest[j : j + 6].lower() == "mimoml"
                or rest[j : j + 4].lower() == "dsml"
            ):
                kw_len = 4 if rest[j : j + 4].lower() == "dsml" else 6
                j += kw_len
                is_mimoml = True
                if j < rest_len and rest[j] in ("-", "_"):
                    j += 1
                continue
            break
        if is_mimoml:
            # Match canonical tag name
            name_end = j
            while name_end < rest_len and (
                rest[name_end].isalnum() or rest[name_end] == "_"
            ):
                name_end += 1
            tag = rest[j:name_end].lower()
            if tag in ("tool_calls", "function_calls", "invoke", "parameter"):
                # Preserve any attributes (e.g. name="X") and skip the
                # trailing | or ｜ after the closing >
                attrs = rest[name_end:]
                # Strip a leading |, |, or ｜ from attrs (e.g. |name="x")
                if attrs and attrs[0] in ("|", "｜"):
                    attrs = attrs[1:]
                prefix = "</" if closing else "<"
                result.append(prefix + tag + attrs + ">")
                k = end + 1
                if k < n and text[k] in ("|", "｜"):
                    k += 1
                i = k
                continue
        result.append(text[i : end + 1])
        i = end + 1
    return "".join(result)


def _clean_tool_text(text: str) -> str:
    """Strip tool call residue from text after extraction.

    Removes leftover:
      - TOOL_CALL: name(...) lines
      - <|MiMoML|*> tags
      - <tool_call>...</tool_call>
      - <function=...>...</function>
      - <parameter=...>...</parameter>
      - <tool_invocation .../> self-closing tags
      - Empty ```code fences```
      - Excess blank lines
    """
    if not text:
        return text
    import re as _re_c
    text = _re_c.sub(r"TOOL_CALL:\s*\w+\s*\([^)]*(?:\([^)]*\)[^)]*)*\)", "", text, flags=_re_c.IGNORECASE)
    text = _re_c.sub(r"TOOL_CALL:.*$", "", text, flags=_re_c.MULTILINE | _re_c.IGNORECASE)
    text = _re_c.sub(r"</?\|?MiMoML\|?[^>]*>", "", text)
    text = _re_c.sub(r"<tool_calls?>.*?</tool_calls?>", "", text, flags=_re_c.DOTALL | _re_c.IGNORECASE)
    text = _re_c.sub(r"<(?:\|?MiMoML\|?)?invoke[^>]*>.*?</(?:\|?MiMoML\|?)?invoke>", "", text, flags=_re_c.DOTALL | _re_c.IGNORECASE)
    text = _re_c.sub(r"<(?:\|?MiMoML\|?)?parameter[^>]*>.*?</(?:\|?MiMoML\|?)?parameter>", "", text, flags=_re_c.DOTALL | _re_c.IGNORECASE)
    text = _re_c.sub(r"<tool_call>.*?</tool_call>", "", text, flags=_re_c.DOTALL | _re_c.IGNORECASE)
    text = _re_c.sub(r"<function=\w+>.*?</function>", "", text, flags=_re_c.DOTALL | _re_c.IGNORECASE)
    text = _re_c.sub(r"<parameter=\w+>.*?</parameter>", "", text, flags=_re_c.DOTALL | _re_c.IGNORECASE)
    text = _re_c.sub(r"<tool_invocation[^>]*/>", "", text)
    text = _re_c.sub(r"```\w*\s*\n?\s*```", "", text)
    text = _re_c.sub(r"\n{3,}", "\n\n", text)
    return text


class _StreamSieve:
    """Character-level sieve that separates text from tool call blocks.

    Used to catch tool calls that span multiple SSE JSON lines (e.g. when
    mimo CLI emits a long tool_call XML that the readline() boundary
    splits mid-tag). Adapted from Fly143/MiMo2API's StreamSieve.

    Modes:
        feed(chunk)  -> list of (kind, data) events
                        kind = "text" | "tool_call"
        flush()      -> emit any remaining buffered text/tool_call
    """

    _TOOL_STARTS = (
        "TOOL_CALL:",
        "<tool_call>",
        "<function_call",
        "<function=",
        "[调用工具:",
        "<|MiMoML|tool_calls>",
        "<｜MiMoML｜tool_calls>",
        "<|MiMoML|function_calls>",
        "<｜MiMoML｜function_calls>",
        "<tool_calls>",
        "<function_calls>",
        "<tool_invocation",
    )

    def __init__(self, parse_fn, tool_names: list[str] | None = None):
        self._parse_fn = parse_fn
        self._tool_names = tool_names or []
        self._buf = ""
        self._capturing = False
        self._capture_buf = ""

    def feed(self, chunk: str) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        if self._capturing:
            self._capture_buf += chunk
            result = self._try_finish()
            if result is not None:
                prefix, tool_calls, suffix = result
                if prefix:
                    events.append(("text", prefix))
                if tool_calls:
                    events.append(("tool_call", tool_calls))
                if suffix:
                    self._buf = suffix
                self._capturing = False
                self._capture_buf = ""
            return events
        self._buf += chunk
        start = self._find_tool_start(self._buf)
        if start >= 0:
            prefix = self._buf[:start]
            rest = self._buf[start:]
            self._buf = ""
            if prefix:
                events.append(("text", prefix))
            self._capture_buf = rest
            self._capturing = True
            result = self._try_finish()
            if result is not None:
                pre, tcs, suf = result
                if pre:
                    events.append(("text", pre))
                if tcs:
                    events.append(("tool_call", tcs))
                if suf:
                    self._buf = suf
                self._capturing = False
                self._capture_buf = ""
        else:
            safe, hold = self._split_safe(self._buf)
            if safe:
                events.append(("text", safe))
            self._buf = hold
        return events

    def flush(self) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        if self._capturing:
            result = self._try_finish()
            if result is not None:
                pre, tcs, suf = result
                if pre:
                    events.append(("text", pre))
                if tcs:
                    events.append(("tool_call", tcs))
                if suf:
                    events.append(("text", suf))
            else:
                if self._capture_buf:
                    events.append(("text", self._capture_buf))
            self._capturing = False
            self._capture_buf = ""
        if self._buf:
            events.append(("text", self._buf))
            self._buf = ""
        return events

    def _find_tool_start(self, text: str) -> int:
        idx = -1
        # Skip content inside markdown fences
        i = 0
        n = len(text)
        while i < n:
            new_i, _ = _skip_fenced_block(text, i)
            if new_i != i:
                # Fence skipped; look only in non-fence portions
                # But for tool call detection, fences should block all
                # detection in their range, so we mask them out.
                # For simplicity, just check the first non-fence portion.
                break
            i += 1
        # Simple approach: find earliest tool start not inside a fence
        best = -1
        for tag in self._TOOL_STARTS:
            pos = text.find(tag)
            if pos >= 0 and (best < 0 or pos < best):
                # Check if inside a fence
                inside_fence = False
                j = 0
                while j < pos:
                    new_j, _ = _skip_fenced_block(text, j)
                    if new_j == j:
                        j += 1
                        continue
                    if new_j > pos:
                        inside_fence = True
                        break
                    j = new_j
                if not inside_fence:
                    best = pos
        return best

    def _split_safe(self, text: str) -> tuple[str, str]:
        """Release safe text, hold suspicious trailing chars."""
        # Check last 20 chars for partial tool start
        for i in range(len(text) - 1, max(len(text) - 25, -1), -1):
            tail = text[i:]
            for tag in self._TOOL_STARTS:
                if tag.startswith(tail) and len(tail) >= 1:
                    return text[:i], tail
        return text, ""

    def _try_finish(self) -> tuple[str, list[dict] | None, str] | None:
        if not self._capture_buf:
            return None
        if not self._is_capture_complete():
            return None
        result = self._parse_fn(self._capture_buf, self._tool_names)
        if result is None:
            return None
        if isinstance(result, tuple) and len(result) == 2:
            tool_calls, cleaned = result
        elif isinstance(result, list):
            tool_calls, cleaned = result, ""
        else:
            return None
        prefix, suffix = self._extract_non_tool_parts(self._capture_buf)
        if tool_calls:
            return (prefix, tool_calls, suffix)
        return (cleaned or self._capture_buf, None, "")

    def _is_capture_complete(self) -> bool:
        buf = self._capture_buf
        if buf.lstrip().upper().startswith("TOOL_CALL:"):
            return ")" in buf or "\n" in buf
        if "[调用工具:" in buf:
            return "\n" in buf or "]" in buf
        if buf.lstrip().startswith("<"):
            # Must have an opening fence closer, or markdown close
            if "<tool_invocation" in buf and "/>" in buf:
                return True
            if "<tool_call" in buf and "</tool_call>" in buf:
                return True
            if "<function_call" in buf and "</function_call>" in buf:
                return True
            if "<tool_calls>" in buf and "</tool_calls>" in buf:
                return True
            if "<|MiMoML|tool_calls>" in buf and "</|MiMoML|tool_calls>" in buf:
                return True
            if "<function=" in buf and "</function>" in buf:
                return True
            return False
        return False

    def _extract_non_tool_parts(self, text: str) -> tuple[str, str]:
        start = -1
        for tag in self._TOOL_STARTS:
            pos = text.find(tag)
            if pos >= 0 and (start < 0 or pos < start):
                start = pos
        if start < 0:
            return text, ""
        prefix = text[:start]
        rest = text[start:]
        end = -1
        if rest.lstrip().upper().startswith("TOOL_CALL:"):
            nl = rest.find("\n")
            if nl >= 0:
                end = start + nl + 1
        elif "[调用工具:" in rest:
            end = start + rest.find("]") + 1
        elif "<tool_call" in rest:
            end = start + rest.find("</tool_call>") + len("</tool_call>")
        elif "<function=" in rest:
            end = start + rest.find("</function>") + len("</function>")
        elif "<function_call" in rest:
            end = start + rest.find("</function_call>") + len("</function_call>")
        elif "<tool_calls>" in rest:
            end = start + rest.find("</tool_calls>") + len("</tool_calls>")
        elif "<|MiMoML|tool_calls>" in rest:
            end = start + rest.find("</|MiMoML|tool_calls>") + len("</|MiMoML|tool_calls>")
        elif "<tool_invocation" in rest:
            end = start + rest.find("/>") + 2
        if end < 0:
            return prefix, ""
        return prefix, text[end:]


def _parse_tool_call_xml(text: str, tool_names: list[str] | None = None) -> dict | None:
    """Parse mimo CLI's text-formatted tool_call XML.

    The mimo CLI emits tool_call as raw XML text in many different shapes.
    Twelve formats observed:
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
        and "```" not in text
        and "<mcp_" not in text
        and '"command"' not in text
        and '"name"' not in text
        and "TOOL_CALL:" not in text
        and "MiMoML" not in text
        and "mimoml" not in text
        and "<function=" not in text
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
            resolved = _resolve_tool_name(
                obj.get("name", "unknown"), tool_names
            ) or obj.get("name", "unknown")
            return {
                "id": f"call_{hashlib.md5(m.group(1).encode()).hexdigest()[:16]}",
                "type": "function",
                "function": {
                    "name": resolved,
                    "arguments": json.dumps(args),
                },
            }
        except (json.JSONDecodeError, KeyError):
            pass

    # Format 2: <function=name><parameter=key>value</parameter>...</function>
    # Handles multiple <parameter=key>value</parameter> pairs.
    m2 = re.search(r"<function=([^>]+)>(.*?)</function>", text, re.DOTALL)
    if m2:
        import uuid as _uuid
        name = m2.group(1).strip()
        inner = m2.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, re.DOTALL):
            args[pm.group(1).strip()] = pm.group(2).strip()
        if args:
            return {
                "id": f"call_{_uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
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
    # Flexible: handles attributes in any order, with double or single quotes,
    # and with or without space before />.
    import uuid as _uuid  # noqa: PLC0415 — used by multiple formats below
    _ti_name = None
    _ti_args_text = None
    _ti_m = re.search(
        r'<tool_invocation\s+'
        r'(?:name=["\']([^"\']+)["\']\s*|\s+)*'
        r'(?:arguments=(\{[^>]+?)\}\s*|\s+)*'
        r'/?\s*>',
        text, re.DOTALL,
    )
    if not _ti_m:
        # Try with attributes reversed (arguments first, then name)
        _ti_m = re.search(
            r'<tool_invocation\s+'
            r'(?:arguments=(\{[^>]+?\})\s*|\s+)*'
            r'(?:name=["\']([^"\']+)["\']\s*|\s+)*'
            r'/?\s*>',
            text, re.DOTALL,
        )
    if _ti_m:
        name = (_ti_m.group(1) or _ti_m.group(2) or "").strip()
        args_text = (_ti_m.group(2) or _ti_m.group(1) or "").strip()
        # Group 1 is name in first pattern, group 2 is name in second pattern
        # Check which group is the name vs args based on whether it looks like JSON
        if name.startswith("{"):
            # Swapped — first group is actually args, second is name
            name, args_text = (_ti_m.group(2) or "").strip(), name
        if name and args_text:
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

    # Format 7: JSON in code block (```json {"name":"X","arguments":{...}} ```)
    # Match the code block content and try to parse as JSON
    m7 = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m7:
        try:
            obj = json.loads(m7.group(1))
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                return {
                    "id": f"call_{_uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": obj.get("name", "unknown"),
                        "arguments": json.dumps(args if isinstance(args, dict) else {"raw": str(args)}),
                    },
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Format 8: tool_invocation wrapped in code blocks (any language tag)
    # ```xml <tool_invocation name="..." arguments={...} /> ```
    # or just ``` <tool_invocation ... /> ```
    if "<tool_invocation" in text and "```" in text:
        _ti_code = re.search(r'```(?:\w*)\s*(<tool_invocation[^`]+?/>)\s*```', text, re.DOTALL)
        if _ti_code:
            inner = _ti_code.group(1).strip()
            # Recurse into format 6 parser on the inner XML
            return _parse_tool_call_xml(inner)

    # Format 9: <mcp_toolname>content</mcp_toolname> (bare XML tag with tool name)
    # The model sometimes emits tool calls as bare XML tags where the tag name
    # IS the tool name and the content is the command/argument.
    # Examples:
    #   <mcp_bash>head -3 /path/to/file</mcp_bash>
    #   <mcp_read>/path/to/file</mcp_read>
    m9 = re.search(r'<(mcp_[a-zA-Z_][a-zA-Z0-9_]*)>([^<]+)</\1>', text)
    if m9:
        name = m9.group(1).strip()
        content = m9.group(2).strip()
        # Map the content to the appropriate parameter based on tool name
        if name.endswith("_bash"):
            args = {"command": content}
        elif name.endswith("_read"):
            args = {"path": content}
        elif name.endswith("_write"):
            args = {"path": content, "content": content}
        elif name.endswith("_edit"):
            args = {"path": content, "input": content}
        elif name.endswith("_find"):
            args = {"paths": [content]}
        elif name.endswith("_search_tool_bm25"):
            args = {"query": content}
        else:
            args = {"raw": content}
        return {
            "id": f"call_{_uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }
    # Format 10: Bare JSON tool call in text (no XML wrapper, no code fence)
    # The model sometimes emits JSON tool calls as plain text, e.g.:
    #   {"command": "head -3 /path/to/file"}
    #   {"name": "mcp_bash", "arguments": {"command": "ls"}}
    # Only attempt on short text (< 500 chars) to avoid false positives.
    if len(text) < 500:
        # Try {"command": "..."} pattern (bash shorthand)
        m10 = re.search(r'\{\s*"command"\s*:\s*"[^"]{2,}"\s*\}', text)
        if m10:
            try:
                obj = json.loads(m10.group(0))
                if isinstance(obj, dict) and "command" in obj:
                    return {
                        "id": f"call_{_uuid.uuid4().hex[:16]}",
                        "type": "function",
                        "function": {
                            "name": "mcp_bash",
                            "arguments": json.dumps(obj),
                        },
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        # Try {"name": "...", "arguments": {...}} pattern using balanced-brace
        # scanner so we handle nested objects/arrays (depth-aware).
        m10b = re.search(
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{', text
        )
        if m10b:
            start = m10b.start()
            js = _find_balanced_json(text, start)
            if js:
                try:
                    obj = json.loads(js)
                    if (
                        isinstance(obj, dict)
                        and "name" in obj
                        and "arguments" in obj
                    ):
                        args = obj.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"raw": args}
                        # Apply tool name resolution
                        resolved = _resolve_tool_name(
                            obj.get("name", "unknown"), tool_names
                        ) or obj.get("name", "unknown")
                        return {
                            "id": f"call_{_uuid.uuid4().hex[:16]}",
                            "type": "function",
                            "function": {
                                "name": resolved,
                                "arguments": json.dumps(
                                    args if isinstance(args, dict) else {"raw": str(args)}
                                ),
                            },
                        }
                except (json.JSONDecodeError, ValueError):
                    pass

    # Format 11: TOOL_CALL: name(args) plain-text format
    # Older mimo CLI versions (and some prompts) emit tool calls as:
    #   TOOL_CALL: mcp_bash(command="ls -la")
    # or with python-style args:
    #   TOOL_CALL: mcp_read(path="/etc/hostname")
    # Uses depth-aware parenthesis matching to handle nested calls/quotes.
    if "TOOL_CALL:" in text:
        m11 = re.search(
            r"(?:^|\n)\s*TOOL_CALL:\s*(\w+)\s*\(", text
        )
        if m11:
            fname = m11.group(1)
            paren_start = m11.end() - 1
            depth = 1
            in_str = False
            esc = False
            end_paren = -1
            for k in range(paren_start + 1, len(text)):
                c = text[k]
                if esc:
                    esc = False
                    continue
                if c == "\\" and in_str:
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        end_paren = k
                        break
            if end_paren > 0:
                args_raw = text[paren_start + 1 : end_paren]
                # Parse python-style kwargs: key="value", key2=123
                args = _parse_python_kwargs(args_raw)
                resolved = _resolve_tool_name(fname, tool_names) or fname
                return {
                    "id": f"call_{_uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": resolved,
                        "arguments": json.dumps(args),
                    },
                }

    # Format 12: MiMoML native format (with noise tolerance)
    # <|MiMoML|tool_calls>
    #   <|MiMoML|invoke name="X">
    #     <|MiMoML|parameter name="Y"><![CDATA[V]]></|MiMoML|parameter>
    #   </|MiMoML|invoke>
    # </|MiMoML|tool_calls>
    if "MiMoML" in text or "mimoml" in text:
        normalised = _strip_mimoml(text)
        # Now look for normalised <tool_calls><invoke name="X">...</invoke></tool_calls>
        m12 = re.search(
            r"<tool_calls>(.*?)</tool_calls>", normalised, re.DOTALL | re.IGNORECASE
        )
        if m12:
            inner = m12.group(1)
            m12i = re.search(
                r'<invoke\s+name=["\']([^"\']+)["\']>(.*?)</invoke>',
                inner,
                re.DOTALL | re.IGNORECASE,
            )
            if m12i:
                fname = m12i.group(1).strip()
                args = _parse_mimoml_params(m12i.group(2))
                resolved = _resolve_tool_name(fname, tool_names) or fname
                return {
                    "id": f"call_{_uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {
                        "name": resolved,
                        "arguments": json.dumps(args),
                    },
                }
    return None


def _parse_python_kwargs(raw: str) -> dict[str, Any]:
    """Parse python-style kwargs: key1="v1", key2=123, key3=True.

    Handles balanced parentheses/brackets/braces within values.
    """
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    args: dict[str, Any] = {}
    parts = _smart_split_args(raw, ",")
    for part in parts:
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Strip surrounding quotes
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]
        args[k] = _auto_type(v)
    return args


def _smart_split_args(text: str, sep: str) -> list[str]:
    """Split `text` by `sep` respecting parens/brackets/braces and quotes."""
    parts: list[str] = []
    current: list[str] = []
    dp = db = dbr = 0
    in_str = False
    quote_char = ""
    esc = False
    for ch in text:
        if esc:
            current.append(ch)
            esc = False
            continue
        if ch == "\\" and in_str:
            current.append(ch)
            esc = True
            continue
        if in_str:
            current.append(ch)
            if ch == quote_char:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote_char = ch
            current.append(ch)
            continue
        if ch == "(":
            dp += 1
        elif ch == ")":
            dp -= 1
        elif ch == "[":
            db += 1
        elif ch == "]":
            db -= 1
        elif ch == "{":
            dbr += 1
        elif ch == "}":
            dbr -= 1
        elif ch == sep and dp == 0 and db == 0 and dbr == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_mimoml_params(inner: str) -> dict[str, Any]:
    """Parse MiMoML <parameter name="X">VALUE</parameter> children into dict.

    Extracts CDATA, auto-types booleans/numbers, and merges duplicate keys
    into lists.
    """
    import re as _re_p
    args: dict[str, Any] = {}
    for m in _re_p.finditer(
        r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>',
        inner,
        _re_p.DOTALL | _re_p.IGNORECASE,
    ):
        key = m.group(1).strip()
        raw = m.group(2).strip()
        # Strip CDATA wrapper if present
        if raw.startswith("<![CDATA[") and raw.endswith("]]>"):
            raw = raw[len("<![CDATA[") : -len("]]>")]
        val: Any = _auto_type(raw)
        if key in args:
            existing = args[key]
            if isinstance(existing, list):
                existing.append(val)
            else:
                args[key] = [existing, val]
        else:
            args[key] = val
    return args
