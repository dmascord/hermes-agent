#!/usr/bin/env python3
"""Live end-to-end tests for claude-code-cli through Hermes gateway.

This script acts as an OpenAI-compatible client and validates both Claude Code
CLI gateway paths:

1. Non-streaming multi-turn tool loop:
   - calls /v1/chat/completions with model=claude-code-cli/sonnet
   - receives tool_calls
   - executes local read/edit tools in a temp workspace
   - sends tool results back in follow-up requests
   - proves read + edit round-trip works

2. Streaming tool_call_hub round-trip:
   - calls /v1/chat/completions with stream=true and X-Hermes-Session-Id
   - receives tool_call delta + side-channel tool_call_request in real time
   - posts the tool result to /v1/sessions/{session}/tool_responses
   - verifies the stream reaches [DONE]

Usage:
  export API_SERVER_KEY=<your gateway API key>
  python3 scripts/test_claude_code_cli_tool_loop_live.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

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


def log(label: str, msg: str = "") -> None:
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] {label:>24s} | {msg}")


def auth_headers(*, session_id: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    if session_id:
        # Server reads X-Hermes-Session-Id, not X-Session-Id.
        headers["X-Hermes-Session-Id"] = session_id
    return headers


def call_gateway(messages: list[dict[str, Any]]) -> dict[str, Any]:
    resp = requests.post(
        API_URL,
        headers=auth_headers(),
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


def post_tool_result(session_id: str, call_id: str, result: str) -> None:
    resp = requests.post(
        f"{API_BASE}/v1/sessions/{session_id}/tool_responses",
        headers=auth_headers(),
        json={"call_id": call_id, "status": "ok", "result": result},
        timeout=15,
        verify=VERIFY_TLS,
    )
    resp.raise_for_status()
    log("hub", f"accepted result for {call_id}: {resp.text[:120]}")


def iter_sse_json(resp: requests.Response) -> Iterable[dict[str, Any] | str]:
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            yield "[DONE]"
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            log("stream", f"non-json SSE payload: {payload[:200]!r}")


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


def test_non_streaming_tool_loop() -> None:
    print("\n--- non-streaming claude-code-cli tool loop ---")
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

            log("non-stream", f"turn={turn + 1} finish={finish_reason} tool_calls={len(tool_calls)} content={content!r}")

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
                    log("non-stream", f"TOOL {name} args={args} result={result!r}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                continue

            final_text = content
            break

        final_file = sample.read_text(encoding="utf-8")
        log("non-stream", f"FINAL FILE: {final_file!r}")
        log("non-stream", f"OBSERVED TOOLS: {observed_tool_names}")
        log("non-stream", f"FINAL TEXT: {final_text!r}")

        assert "read" in observed_tool_names, f"expected read tool call, got {observed_tool_names}"
        assert "edit" in observed_tool_names, f"expected edit tool call, got {observed_tool_names}"
        assert final_file == "hello Claude\n", f"expected edited file, got {final_file!r}"
        assert final_text == "DONE", f"expected final response DONE, got {final_text!r}"


def test_streaming_tool_hub_round_trip() -> None:
    print("\n--- streaming claude-code-cli tool_call_hub round-trip ---")
    with tempfile.TemporaryDirectory(prefix="hermes_claude_cli_stream_") as td:
        workspace = Path(td)
        sample = workspace / "sample.txt"
        sample.write_text("hello streaming claude\n", encoding="utf-8")
        session_id = f"test-claude-{uuid.uuid4().hex[:12]}"

        body = {
            "model": MODEL,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the read tool to read sample.txt. "
                        "After the tool result is available, reply with exactly STREAM-DONE."
                    ),
                }
            ],
            "tools": TOOLS,
        }

        log("stream", f"POST {API_URL} session={session_id}")
        resp = requests.post(
            API_URL,
            headers=auth_headers(session_id=session_id),
            json=body,
            stream=True,
            timeout=180,
            verify=VERIFY_TLS,
        )
        resp.raise_for_status()
        log("stream", f"HTTP {resp.status_code}, SSE started")

        saw_tool_delta = False
        saw_tool_request = False
        saw_done = False
        hub_posts = 0
        text_parts: list[str] = []

        for event in iter_sse_json(resp):
            if event == "[DONE]":
                saw_done = True
                log("stream", "[DONE]")
                break
            if not isinstance(event, dict):
                continue
            data_event = event.get("data")
            if isinstance(data_event, dict) and data_event.get("type") == "tool_call_request":
                saw_tool_request = True
                call_id = str(data_event.get("call_id") or "")
                name = str(data_event.get("name") or "")
                args = data_event.get("arguments") or {}
                log("stream", f"tool_call_request {name}({args}) id={call_id}")
                result = run_tool(workspace, name, args if isinstance(args, dict) else {})
                post_tool_result(session_id, call_id, result)
                hub_posts += 1
                continue
            for choice in event.get("choices", []) or []:
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if delta.get("tool_calls"):
                    saw_tool_delta = True
                    log("stream", f"tool_call delta: {delta['tool_calls']}")
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if choice.get("finish_reason"):
                    log("stream", f"finish_reason={choice.get('finish_reason')}")

        final_text = "".join(text_parts)
        log("stream", f"text_len={len(final_text)} tool_delta={saw_tool_delta} tool_request={saw_tool_request} hub_posts={hub_posts} done={saw_done}")

        assert saw_tool_delta, "expected streaming tool_call delta"
        assert saw_tool_request, "expected real-time tool_call_request side-channel event"
        assert hub_posts >= 1, "expected at least one hub tool result POST"
        assert saw_done, "expected [DONE]"


def main() -> None:
    test_non_streaming_tool_loop()
    test_streaming_tool_hub_round_trip()
    print("\nPASS: claude-code-cli non-streaming and streaming tool loops completed")


if __name__ == "__main__":
    main()
