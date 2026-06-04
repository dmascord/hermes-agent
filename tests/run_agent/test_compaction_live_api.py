#!/usr/bin/env python3
"""
Live API test: verify hermes-swarm handles the post-compaction assistant message
correctly on all three API paths (/v1/runs, /v1/responses, /v1/chat/completions).

Sends a payload where the last message is an assistant-role compaction summary,
as pi/opencode would send after context compaction. Verifies:
  - Server does NOT return 400 / 500
  - Server does NOT respond with the canned "message came through empty" error
  - Server starts a real agent response (any non-error text)

Run via:
    python -m pytest tests/run_agent/test_compaction_live_api.py -v
Or directly:
    python tests/run_agent/test_compaction_live_api.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

BASE_URL = os.environ.get("HERMES_API_URL", "http://localhost:8642")

COMPACTION_SUMMARY = """\
[CONTEXT SUMMARY]:

## Active Task
Write a Python function that returns the nth Fibonacci number using memoization.

## Completed Steps
- Discussed approach with user
- Chose memoization as the preferred strategy

## Current Status
In progress — implementation not yet written.
"""

EMPTY_RESPONSE_PHRASES = [
    "message came through empty",
    "your message came through empty",
    "empty message",
]


def _post(path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read())
        except Exception:
            pass
        return e.code, body


def _extract_text(body: dict) -> str:
    """Pull reply text from any response shape."""
    # /v1/runs SSE result (non-streaming we get the full run object)
    for key in ("output", "result", "response", "message"):
        if key in body:
            val = body[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        c = item.get("content") or item.get("text", "")
                        if c:
                            return c if isinstance(c, str) else str(c)
    # /v1/chat/completions shape
    choices = body.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "") or ""
    return str(body)


def check_health() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def test_runs_path() -> bool:
    print("\n[TEST] /v1/runs — post-compaction assistant message as last input")
    payload = {
        "model": "github-copilot-enterprise/claude-sonnet-4.6",
        "stream": False,
        "input": [
            {"role": "user", "content": "Write me a Fibonacci function with memoization."},
            {"role": "assistant", "content": COMPACTION_SUMMARY},
        ],
    }
    status, body = _post("/v1/runs", payload)
    text = _extract_text(body)
    print(f"  status={status}")
    print(f"  reply_preview={repr(text[:200])}")

    if status not in (200, 201, 202):
        print(f"  FAIL: unexpected status {status} — body={body}")
        return False

    for phrase in EMPTY_RESPONSE_PHRASES:
        if phrase.lower() in text.lower():
            print(f"  FAIL: got canned empty-message response: {phrase!r}")
            return False

    print("  PASS")
    return True


def test_responses_path() -> bool:
    print("\n[TEST] /v1/responses — post-compaction assistant message as last input")
    payload = {
        "model": "github-copilot-enterprise/claude-sonnet-4.6",
        "stream": False,
        "input": [
            {"role": "user", "content": "Write me a Fibonacci function with memoization."},
            {"role": "assistant", "content": COMPACTION_SUMMARY},
        ],
    }
    status, body = _post("/v1/responses", payload)
    text = _extract_text(body)
    print(f"  status={status}")
    print(f"  reply_preview={repr(text[:200])}")

    if status not in (200, 201, 202):
        print(f"  FAIL: unexpected status {status} — body={body}")
        return False

    for phrase in EMPTY_RESPONSE_PHRASES:
        if phrase.lower() in text.lower():
            print(f"  FAIL: got canned empty-message response: {phrase!r}")
            return False

    print("  PASS")
    return True


def test_chat_completions_path() -> bool:
    print("\n[TEST] /v1/chat/completions — post-compaction assistant message as last message")
    payload = {
        "model": "github-copilot-enterprise/claude-sonnet-4.6",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Write me a Fibonacci function with memoization."},
            {"role": "assistant", "content": COMPACTION_SUMMARY},
        ],
    }
    status, body = _post("/v1/chat/completions", payload)
    text = _extract_text(body)
    print(f"  status={status}")
    print(f"  reply_preview={repr(text[:200])}")

    if status not in (200, 201, 202):
        print(f"  FAIL: unexpected status {status} — body={body}")
        return False

    for phrase in EMPTY_RESPONSE_PHRASES:
        if phrase.lower() in text.lower():
            print(f"  FAIL: got canned empty-message response: {phrase!r}")
            return False

    print("  PASS")
    return True


def test_empty_user_message_continuation() -> bool:
    print("\n[TEST] /v1/runs — empty user message continuation signal")
    payload = {
        "model": "github-copilot-enterprise/claude-sonnet-4.6",
        "stream": False,
        "input": [
            {"role": "user", "content": "Write me a Fibonacci function with memoization."},
            {"role": "assistant", "content": "I'll write that for you now..."},
            {"role": "user", "content": ""},
        ],
    }
    status, body = _post("/v1/runs", payload)
    text = _extract_text(body)
    print(f"  status={status}")
    print(f"  reply_preview={repr(text[:200])}")

    if status not in (200, 201, 202):
        print(f"  FAIL: unexpected status {status} — body={body}")
        return False

    for phrase in EMPTY_RESPONSE_PHRASES:
        if phrase.lower() in text.lower():
            print(f"  FAIL: got canned empty-message response: {phrase!r}")
            return False

    print("  PASS")
    return True


if __name__ == "__main__":
    print(f"Testing hermes-swarm at {BASE_URL}")
    if not check_health():
        print(f"ERROR: server not reachable at {BASE_URL}/health")
        sys.exit(1)
    print("Health check OK")

    results = [
        test_runs_path(),
        test_responses_path(),
        test_chat_completions_path(),
        test_empty_user_message_continuation(),
    ]

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if all(results) else 1)
