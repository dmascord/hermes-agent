#!/usr/bin/env python3
"""Provider parity test: Codex and Claude Code CLI in both streaming modes.

Usage:
    python3 scripts/test_provider_parity.py [--key KEY] [--url URL]

    KEY defaults to the API_SERVER_KEY env var.
    URL defaults to https://hermes.tusker.net.au/v1/chat/completions.

Exits 0 if all tests pass, 1 if any fail.  Each test is a single chat
completion with a known prompt that should produce a short answer.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

API_KEY = os.getenv("API_SERVER_KEY", "")
BASE_URL = os.getenv("HERMES_API_URL", "https://hermes.tusker.net.au/v1/chat/completions")

PROMPT = "Reply with just the word PASS"

# Models under test: (label, model_id, supports_tools)
TEST_SUITE: List[Tuple[str, str, bool]] = [
    ("codex-nonstreaming", "openai-codex/gpt-5.4", False),
    ("codex-streaming",    "openai-codex/gpt-5.4", False),
    ("claude-nonstreaming","claude-code-cli",       False),
    ("claude-streaming",   "claude-code-cli",       False),
]


def _call(
    model: str,
    prompt: str,
    stream: bool,
    timeout: int = 120,
) -> Tuple[int, Dict[str, Any]]:
    """POST a chat completion and return (http_status, parsed_json)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "stream": stream,
    }).encode()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(BASE_URL, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        status = exc.code

    # Streaming responses come as SSE lines; concatenate data: lines.
    if stream:
        collected = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = line[6:]
                try:
                    chunk = json.loads(payload)
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    collected += delta
                except json.JSONDecodeError:
                    pass
        return status, {"content": collected, "choices": [
            {"message": {"content": collected}, "finish_reason": "stop"},
        ]}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"error": raw}
    return status, parsed


def _test_provider(label: str, model: str, stream: bool) -> Tuple[bool, str]:
    """Run one test and return (passed, detail)."""
    start = time.time()
    status, data = _call(model, PROMPT, stream=stream)
    elapsed = time.time() - start

    failures: List[str] = []

    if status != 200:
        err_msg = data.get("error", data.get("message", str(data)[:200]))
        failures.append(f"HTTP {status}: {err_msg}")

    choices = data.get("choices", [])
    if not choices:
        failures.append("no choices in response")
    else:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        # Claude's stream-json output has a different envelope
        if not content and stream and model == "claude-code-cli":
            # Claude streaming returns result directly, not OpenAI-format
            content = data.get("content", "")
        if not content:
            failures.append("empty content")
        finish = (msg.get("finish_reason") if isinstance(msg, dict)
                  else choices[0].get("finish_reason", ""))
        if finish and finish not in ("stop", "tool_calls"):
            failures.append(f"unexpected finish_reason={finish}")

    status_emoji = "PASS" if not failures else "FAIL"
    detail = "; ".join(failures) if failures else f"OK ({elapsed:.1f}s)"
    print(f"  [{status_emoji}] {label}  {detail}")
    return len(failures) == 0, detail


def main() -> int:
    if not API_KEY:
        print("Error: API_SERVER_KEY env var required")
        return 1

    print(f"Base URL:       {BASE_URL}")
    print(f"Prompt:         {PROMPT!r}")
    print()

    results: List[bool] = []
    for label, model, supports_tools in TEST_SUITE:
        stream = "streaming" in label
        ok, detail = _test_provider(label, model, stream)
        results.append(ok)

    print()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    # Also check credential pool health
    print()
    print("--- Pool health (Codex) ---")
    try:
        from agent.credential_pool import load_pool
        pool = load_pool("openai-codex")
        active = sum(1 for e in pool.entries() if e.last_error_code != 401)
        total_entries = len(pool.entries())
        print(f"  openai-codex: {active}/{total_entries} entries healthy")
    except Exception as exc:
        print(f"  (pool check skipped: {exc})")

    print()
    print("--- Claude token expiry ---")
    try:
        import os as _os
        from pathlib import Path
        home = _os.environ.get("HERMES_HOME", "")
        creds_paths = [
            Path(home) / ".claude" / ".credentials.json",
            Path(home) / ".claude_backup" / ".credentials.json",
            Path(_os.path.expanduser("~")) / ".claude" / ".credentials.json",
            Path("/home/tusker") / ".claude" / ".credentials.json",
        ]
        for cp in creds_paths:
            if cp.exists():
                with open(cp) as _f:
                    _cj = json.load(_f)
                _oauth = _cj.get("claudeAiOauth", {})
                _exp = _oauth.get("expiresAt", 0)
                _rem = (_exp / 1000) - time.time() if _exp else 0
                print(f"  {cp}: {'VALID' if _rem > 0 else 'EXPIRED'} ({_rem/3600:.1f}h)")
                break
        else:
            print("  (no claude credentials found)")
    except Exception as exc:
        print(f"  (skipped: {exc})")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())