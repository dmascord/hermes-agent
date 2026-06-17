"""MCP bridge proxy for Claude Code CLI.

This script is spawned by `claude -p --mcp-config` as a stdio MCP server.
It registers the OMP client's tools (read from a manifest file) and proxies
tool calls through a shared file queue to the Hermes gateway, which forwards
them to the Mac OMP client for local execution.

Protocol:
  - Hermes writes tool definitions to $HERMES_TOOLS_FILE (JSON array).
  - When Claude CLI calls a tool, we write the call to $HERMES_QUEUE_IN.
  - The gateway reads the call, forwards to the Mac, writes the result to
    $HERMES_QUEUE_OUT_DIR/<call_id>.json.
  - We block on that file, read the result, and return it to Claude CLI.
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging
import uuid
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("claude-mcp-bridge")

TOOLS_FILE = os.environ.get("HERMES_TOOLS_FILE", "/tmp/hermes_tools.json")
QUEUE_IN = os.environ.get("HERMES_QUEUE_IN", "/tmp/hermes_queue.in")
QUEUE_OUT_DIR = os.environ.get("HERMES_QUEUE_OUT_DIR", "/tmp/hermes_queue_out")
RESULT_TIMEOUT = float(os.environ.get("HERMES_RESULT_TIMEOUT", "300"))

# Ensure output directory exists
Path(QUEUE_OUT_DIR).mkdir(parents=True, exist_ok=True)


def _load_tools() -> list[dict]:
    try:
        with open(TOOLS_FILE) as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load tools from %s: %s", TOOLS_FILE, exc)
        return []


def _proxy_tool_call(tool_name: str, arguments: dict) -> str:
    """Write tool call to queue, block for result, return result text."""
    call_id = uuid.uuid4().hex
    call_payload = {
        "call_id": call_id,
        "tool": tool_name,
        "arguments": arguments,
        "timestamp": time.time(),
    }
    result_path = os.path.join(QUEUE_OUT_DIR, f"{call_id}.json")

    # Write call to inbound queue (append mode for concurrent safety)
    try:
        with open(QUEUE_IN, "a") as f:
            f.write(json.dumps(call_payload) + "\n")
            f.flush()
    except Exception as exc:
        logger.error("Failed to write call to queue: %s", exc)
        return json.dumps({"error": f"queue write failed: {exc}"})

    logger.info("Tool call queued: %s(%s) call_id=%s", tool_name, arguments, call_id)

    # Block on result file
    deadline = time.monotonic() + RESULT_TIMEOUT
    poll_interval = 0.05
    while time.monotonic() < deadline:
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    result = json.load(f)
                try:
                    os.unlink(result_path)
                except Exception:
                    pass
                logger.info("Got result for call_id=%s", call_id)
                # Return result as a string for MCP
                if isinstance(result, dict):
                    if "error" in result:
                        return json.dumps({"error": result["error"]})
                    if "content" in result:
                        content = result["content"]
                        if isinstance(content, str):
                            return content
                        return json.dumps(content)
                    return json.dumps(result)
                return str(result)
            except Exception as exc:
                logger.error("Failed to read result for %s: %s", call_id, exc)
                return json.dumps({"error": f"result read failed: {exc}"})
        time.sleep(poll_interval)

    logger.error("Timeout waiting for result: call_id=%s", call_id)
    return json.dumps({"error": f"timeout waiting for tool result ({RESULT_TIMEOUT}s)"})


class _ProxyToolMetadata:
    async def call_fn_with_arg_validation(
        self,
        fn,
        is_async: bool,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        return fn(**(arguments or {}))

    def convert_result(self, result: Any) -> Any:
        return result


def main() -> None:
    tools = _load_tools()
    if not tools:
        logger.warning("No tools loaded; starting empty MCP server")

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("mcp package not installed; cannot start bridge")
        sys.exit(1)

    mcp = FastMCP("hermes-tools")

    # Dynamically register each tool as a proxy
    for tool_def in tools:
        tool_name = tool_def.get("name")
        if not tool_name:
            continue
        tool_desc = tool_def.get("description", "")
        # Use a factory to capture the name in a closure.
        def _make_handler(name: str):
            def handler(**kwargs) -> str:
                return _proxy_tool_call(name, kwargs)
            return handler

        from mcp.server.fastmcp.tools import Tool
        proxy_tool = Tool.from_function(
            _make_handler(tool_name),
            name=tool_name,
            description=tool_desc,
            structured_output=False,
        )
        # FastMCP derives a generic **kwargs schema. Override it with the
        # actual OpenAI-compatible tool schema and matching validation metadata.
        proxy_tool.parameters = tool_def.get(
            "input_schema", {"type": "object", "properties": {}}
        )
        proxy_tool.fn_metadata = _ProxyToolMetadata()
        mcp._tool_manager._tools[tool_name] = proxy_tool
        logger.info("Registered proxy tool: %s", tool_name)

    # Run stdio MCP server
    logger.info("Starting stdio MCP server (tools=%d)", len(tools))
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
