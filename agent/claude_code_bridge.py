#!/usr/bin/env python3
"""
Claude Code CLI MCP Bridge -- uses Claude Code as a tool-calling LLM provider.

Architecture:
  1. Start `claude mcp serve` as a subprocess (Claude Code = MCP server).
     Claude Code will:
       - Advertise its built-in tools to us
       - Make sampling/createMessage requests to us when it needs LLM reasoning
  2. We connect as the MCP client and respond to sampling/createMessage
     requests using our own LLM (via call_llm).
  3. If our LLM responds with tool calls, we return them as
     CreateMessageResultWithTools -- Claude Code executes those tools
     and sends results back via another sampling/createMessage.
  4. Loop continues until we get a final text response.

In this model:
  - Claude Code is the TOOL EXECUTOR (executes whatever tools we decide to use)
  - We are the LLM PROVIDER (we decide what to do via our LLM)
  - Claude Code has access to its own built-in tools PLUS any MCP tools
    we connect to it (via extra_mcp_servers config)

This lets OpenCode drive a Claude Code session where OpenCode decides what
to do (via its own LLM) while Claude Code handles the tool execution and
provides a rich interactive environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time as time_module
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP types
# ---------------------------------------------------------------------------
_MCP_AVAILABLE = False
try:
    from mcp.types import (
        CreateMessageResult,
        CreateMessageResultWithTools,
        ErrorData,
        SamplingMessage,
        TextContent,
        ToolUseContent,
    )
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    _MCP_AVAILABLE = True
except ImportError:
    logger.debug("mcp package not installed -- Claude Code MCP bridge disabled")
    _MCP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Claude Code session
# ---------------------------------------------------------------------------

class ClaudeCodeMCPBridge:
    """Connect to Claude Code running `mcp serve` as an MCP client.

    Claude Code acts as the MCP server, advertising its tools and making
    sampling/createMessage requests. We handle those requests with our own
    LLM, and Claude Code executes any resulting tool calls.

    This lets Claude Code function as a tool-executing backend while we
    (OpenCode) provide the LLM reasoning via the sampling protocol.
    """

    def __init__(
        self,
        *,
        claude_cli_path: str = "",
        model: str = "claude-sonnet-4-6",
        extra_mcp_servers: dict[str, Any] | None = None,
        request_timeout: float = 120.0,
        verbose: bool = False,
    ):
        self.claude_cli_path = claude_cli_path or os.environ.get("CLAUDE_CLI_PATH", "claude")
        self.model = model
        self.extra_mcp_servers = extra_mcp_servers or {}
        self.request_timeout = request_timeout
        self.verbose = verbose

        self._closed = False
        self._final_result = ""
        self._tool_calls_from_last: list[dict] = []
        self._pending_sampling_result: Any = None
        self._sampling_response_available = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Sampling callback (sync, called by MCP SDK on its event loop)
    # ------------------------------------------------------------------

    def _sampling_callback(
        self,
        messages: list[SamplingMessage],
        max_tokens: int,
        model_preferences: Any,
    ) -> CreateMessageResult | CreateMessageResultWithTools | ErrorData:
        """Handle sampling/createMessage from Claude Code (sync callback).

        We use our LLM to decide what to do. If the LLM produces tool calls,
        we return them so Claude Code executes the tools and continues.
        """
        # Convert MCP messages to OpenAI format for our LLM
        openai_messages = self._convert_messages(messages)

        if self.verbose:
            logger.info(
                "[claude-code-bridge] sampling request: %d messages, max_tokens=%d",
                len(openai_messages), max_tokens,
            )

        # Call our LLM asynchronously from the sync callback
        # We use a future/result queue pattern
        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def _llm_thread():
            try:
                from agent.auxiliary_client import call_llm
                resp = call_llm(
                    task="claude_code_sampling",
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    timeout=self.request_timeout,
                )
                result_queue.put(("ok", resp))
            except Exception as exc:
                logger.error("[claude-code-bridge] LLM call failed: %s", exc)
                result_queue.put(("error", str(exc)))

        t = threading.Thread(target=_llm_thread, daemon=True)
        t.start()
        t.join(timeout=self.request_timeout + 10)
        if t.is_alive():
            logger.error("[claude-code-bridge] LLM call timed out after %ss", self.request_timeout)
            return ErrorData(code=-1, message="LLM call timed out")

        try:
            status, resp = result_queue.get_nowait()
        except queue.Empty:
            return ErrorData(code=-1, message="LLM call produced no result")

        if status == "error":
            return ErrorData(code=-1, message=f"LLM error: {resp}")

        # Extract response data
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
        raw_content = choice.message.content if hasattr(choice.message, "content") else ""
        tool_calls_attr = getattr(choice.message, "tool_calls", None)

        if (
            isinstance(tool_calls_attr, list)
            and tool_calls_attr
            and finish_reason == "tool_calls"
        ):
            # Return tool calls so Claude Code executes them
            tool_use_contents: list[ToolUseContent] = []
            for i, tc in enumerate(tool_calls_attr):
                fn = getattr(tc, "function", None) or {}
                name = str(getattr(fn, "name", "unknown"))
                raw_args = getattr(fn, "arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {"_raw": raw_args}
                else:
                    args = raw_args if isinstance(raw_args, dict) else {"_raw": str(raw_args)}
                call_id = str(getattr(tc, "id", f"call_{i}"))
                tool_use_contents.append(ToolUseContent(
                    type="tool_use",
                    id=call_id,
                    name=name,
                    input=args,
                ))
            self._tool_calls_from_last = [
                {"id": str(getattr(tc, "id", f"call_{i}")),
                 "name": str(getattr(getattr(tc, "function", None) or {}, "name", "unknown")),
                 "arguments": json.dumps(args) if isinstance(args, dict) else str(args)}
                for i, tc in enumerate(tool_calls_attr)
            ]
            resp_model = getattr(resp, "model", self.model)
            return CreateMessageResultWithTools(
                role="assistant",
                content=[TextContent(type="text", text=raw_content or "Using tools...")],
                model=resp_model,
                stopReason="toolUse",
            )
        else:
            # Text response -- final answer
            self._final_result = raw_content or ""
            if self.verbose:
                logger.info("[claude-code-bridge] final text result: %s", self._final_result[:200])
            resp_model = getattr(resp, "model", self.model)
            stop = "endTurn" if finish_reason in ("stop", "eos") else finish_reason
            return CreateMessageResult(
                role="assistant",
                content=[TextContent(type="text", text=raw_content or "")],
                model=resp_model,
                stopReason=stop,
            )

    # ------------------------------------------------------------------
    # Message format conversion (MCP SamplingMessage -> OpenAI format)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[SamplingMessage]) -> list[dict]:
        """Convert MCP SamplingMessage list to OpenAI messages format."""
        result: list[dict] = []
        for msg in messages:
            role = str(getattr(msg, "role", "user"))
            blocks: list = getattr(msg, "content", []) or []
            if not isinstance(blocks, list):
                blocks = [blocks]

            assistant_parts: list[str] = []
            tool_results: list[dict] = []

            for block in blocks:
                if not hasattr(block, "_type") and hasattr(block, "toolUseId"):
                    # Tool result content block
                    tool_id = str(block.toolUseId)
                    text = ""
                    if hasattr(block, "content"):
                        block_content = block.content
                        if isinstance(block_content, list):
                            for item in block_content:
                                text += getattr(item, "text", "")
                        else:
                            text = str(getattr(block_content, "text", ""))
                    tool_results.append({"tool_call_id": tool_id, "content": text})
                elif hasattr(block, "text"):
                    assistant_parts.append(block.text)

            if tool_results:
                for tr in tool_results:
                    result.append({"role": "tool", **tr})
            if assistant_parts:
                result.append({"role": role, "content": "\n".join(assistant_parts)})

        return result

    # ------------------------------------------------------------------
    # Connect and run
    # ------------------------------------------------------------------

    async def _connect_and_run(self, prompt: str, system_prompt: str = ""):
        """Connect to Claude Code MCP server and send the prompt via sampling."""
        # Build environment for the subprocess
        env = {
            k: v for k, v in os.environ.items()
            if k in {"HOME", "USER", "PATH", "TERM", "SHELL", "LANG", "LC_ALL", "TMPDIR"}
        }
        # Extra env vars for this bridge
        for k, v in {
            "CLAUDE_CODE_SIMPLE": "1",
            "HOME": os.environ.get("HOME", "/home/tusker"),
        }.items():
            env[k] = v

        # Build extra MCP servers config file if needed
        mcp_config_file: str | None = None
        if self.extra_mcp_servers:
            fd, mcp_config_file = tempfile.mkstemp(suffix=".json", prefix="claude_mcp_")
            with os.fdopen(fd, "w") as f:
                json.dump({"mcpServers": self.extra_mcp_servers}, f)
            env["CLAUDE_MCP_CONFIG_FILE"] = mcp_config_file

        server_params = StdioServerParameters(
            command=self.claude_cli_path,
            args=["mcp", "serve"],
            env=env,
        )

        if self.verbose:
            logger.info("[claude-code-bridge] starting Claude Code MCP server...")

        try:
            read_stream, write_stream = await stdio_client(server_params)
        except Exception as exc:
            logger.error("[claude-code-bridge] failed to start Claude Code: %s", exc)
            return

        if self.verbose:
            logger.info("[claude-code-bridge] connected, initializing session...")

        sampling_cb = self._sampling_callback
        async with ClientSession(
            read_stream,
            write_stream,
            sampling_callback=sampling_cb,
        ) as session:
            await session.initialize()
            if self.verbose:
                logger.info("[claude-code-bridge] MCP session initialized")

            # Build initial prompt as a sampling message
            from mcp.types import SamplingMessage
            system_content = [{"type": "text", "text": system_prompt}] if system_prompt else []
            user_content = [{"type": "text", "text": prompt}]
            # Send the prompt as the first (and only) user sampling message
            # This triggers Claude Code to start reasoning + potentially call tools
            sampling_messages = [
                SamplingMessage(role="user", content=user_content),
            ]
            if system_content:
                sampling_messages.insert(
                    0,
                    SamplingMessage(role="system", content=system_content),
                )

            if self.verbose:
                logger.info("[claude-code-bridge] sending prompt via sampling...")

            # Wait for final result with timeout
            timeout_at = time_module.time() + self.request_timeout
            while not self._final_result and not self._closed:
                remaining = timeout_at - time_module.time()
                if remaining <= 0:
                    logger.warning("[claude-code-bridge] request timed out")
                    break
                await asyncio.sleep(0.5)

        # Cleanup
        if mcp_config_file:
            try:
                os.unlink(mcp_config_file)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, prompt: str, *, system: str = "") -> str:
        """Send a prompt and get a final text response (blocking).

        This starts Claude Code as an MCP server, sends the prompt via the
        sampling protocol, and returns the final text response.
        """
        if self._closed:
            raise RuntimeError("ClaudeCodeMCPBridge is closed")

        self._final_result = ""
        self._tool_calls_from_last = []

        async def _main():
            await self._connect_and_run(prompt, system_prompt=system)

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            logger.info("[claude-code-bridge] interrupted")
        except Exception as exc:
            logger.error("[claude-code-bridge] error: %s", exc)

        return self._final_result

    def get_tool_calls(self) -> list[dict]:
        """Return tool calls from the last LLM response (if any)."""
        return list(self._tool_calls_from_last)

    def close(self):
        """Close the bridge."""
        self._closed = True


# ---------------------------------------------------------------------------
# Synchronous auxiliary-client-compatible wrapper
# ---------------------------------------------------------------------------

class ClaudeCodeAuxiliaryClient:
    """Synchronous wrapper compatible with auxiliary_client's client interface.

    Exposes:
      - .chat.completions.create(**kwargs) -> response
    """

    def __init__(
        self,
        *,
        claude_cli_path: str = "",
        model: str = "claude-sonnet-4-6",
        request_timeout: float = 120.0,
    ):
        self.model = model
        self._bridge = ClaudeCodeMCPBridge(
            claude_cli_path=claude_cli_path,
            model=model,
            request_timeout=request_timeout,
            verbose=False,
        )

    def chat(self, prompt: str, *, system: str = "") -> str:
        return self._bridge.chat(prompt, system=system)

    def get_tool_calls(self) -> list[dict]:
        return self._bridge.get_tool_calls()

    def close(self):
        self._bridge.close()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Test Claude Code MCP bridge")
    parser.add_argument("prompt", nargs="?", default="Say hello in exactly 3 words.")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    bridge = ClaudeCodeMCPBridge(model=args.model, request_timeout=args.timeout, verbose=True)
    try:
        result = bridge.chat(args.prompt)
        print("\n=== FINAL RESULT ===")
        print(result)
    finally:
        bridge.close()
