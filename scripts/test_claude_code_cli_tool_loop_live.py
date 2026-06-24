#!/usr/bin/env python3
"""Live end-to-end test for claude-code-cli tool calling through Hermes gateway.

This script acts as the OpenAI-compatible client:
- calls Hermes /v1/chat/completions with model=claude-code-cli/sonnet
- receives tool_calls
- executes local read/edit tools in a temp workspace
- sends tool results back in follow-up requests
- proves the multi-turn tool loop works for read + edit

Usage:
  export API_SERVER_KEY=<your gateway API key>
  python3 scripts/test_claude_code_cli_tool_loop_live.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests
import urllib3

# Live test defaults match the Hermes cluster, but credentials must come from
# the environment so this script is safe to commit and share.
API_BASE = os.getenv("API_BASE", "https://hermes.tusker.net.au").rstrip("/")
API_URL = f"{API_BASE}/v1/chat/completions"
API_KEY = os.getenv("API_SERVER_KEY", "")
MODEL = os.getenv("CLAUDE_CODE_CLI_MODEL", "claude-code-cli/sonnet")
VERIFY_TLS = os.getenv("API_VERIFY_TLS", "0").strip().lower() in ("1", "true", "yes")

if not VERIFY_TLS:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

if not API_KEY:
    print("FATAL: Set API_SERVER_KEY env var", file=sys.stderr)
    sys.exit(1)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path to read"}
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace exact text in a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path to edit"},
                    "old": {"type": "string", "description": "Exact old text to replace"},
                    "new": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old", "new"],
                "additionalProperties": False,
            },
        },
    },
]


def call_gateway(messages: list[dict[str, Any]]) -> dict[str, Any]:
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "stream": False,
            "messages": messages,
            "tools": TOOLS,
        },
        timeout=120,
        verify=VERIFY_TLS,
    )
    resp.raise_for_status()
    return resp.json()


def run_tool(workspace: Path, tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "read":
        rel = arguments["path"]
        text = (workspace / rel).read_text(encoding="utf-8")
        return text
    if tool_name == "edit":
        rel = arguments["path"]
        old = arguments["old"]
        new = arguments["new"]
        path = workspace / rel
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise AssertionError(f"edit old text not found in {rel!r}: {old!r}")
        updated = text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        return f"edited {rel}"
    raise AssertionError(f"unexpected tool {tool_name!r}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hermes_claude_cli_live_") as td:
        workspace = Path(td)
        sample = workspace / "sample.txt"
        sample.write_text("hello world\n", encoding="utf-8")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "You are operating in a tiny test workspace. "
                    "First read sample.txt. Then replace the word world with Claude using the edit tool. "
                    "When finished, reply with exactly DONE."
                ),
            }
        ]

        observed_tool_names: list[str] = []
        max_turns = 8
        final_text = None

        for turn in range(max_turns):
            data = call_gateway(messages)
            choice = data["choices"][0]
            message = choice["message"]
            tool_calls = message.get("tool_calls") or []
            content = message.get("content")
            finish_reason = choice.get("finish_reason")

            print(f"TURN {turn+1} finish_reason={finish_reason} tool_calls={len(tool_calls)} content={content!r}")

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"])
                    observed_tool_names.append(name)
                    result = run_tool(workspace, name, args)
                    print(f"  TOOL {name} args={args} result={result!r}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                continue

            final_text = content
            break

        final_file = sample.read_text(encoding="utf-8")
        print("FINAL FILE:", repr(final_file))
        print("OBSERVED TOOLS:", observed_tool_names)
        print("FINAL TEXT:", repr(final_text))

        assert "read" in observed_tool_names, f"expected read tool call, got {observed_tool_names}"
        assert "edit" in observed_tool_names, f"expected edit tool call, got {observed_tool_names}"
        assert final_file == "hello Claude\n", f"expected edited file, got {final_file!r}"
        assert final_text == "DONE", f"expected final response DONE, got {final_text!r}"

        print("PASS: claude-code-cli completed read + edit tool loop end-to-end")


if __name__ == "__main__":
    main()
