#!/usr/bin/env python3
"""Man-in-the-middle proxy between pi/opencode and hermes-swarm.

Sits on a local port, forwards every request upstream to the real hermes
server, and dumps both the raw request body and the full streamed response
to a JSONL log file so we can replay and inspect exactly what pi sends.

Usage:
    python3 tests/gateway/mitm_proxy.py \
        --upstream http://localhost:8642 \
        --port 8643 \
        --log /tmp/mitm.jsonl

Then point pi at http://localhost:8643 instead of :8642.

Each log entry is a JSON object with:
  ts          - ISO timestamp
  direction   - "request" or "response"
  method      - HTTP method
  path        - request path
  headers     - dict of headers (request) or status (response)
  body        - parsed JSON body (request) or full streamed text (response)
  session_id  - extracted from x-session-id or messages if present
  trigger     - detected scenario tag (empty_user, tool_loop, compaction, normal)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web, ClientSession, ClientTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MITM] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mitm")


# ── helpers ────────────────────────────────────────────────────────────────────

def _detect_trigger(body: Any) -> str:
    """Classify the request scenario so interesting cases are easy to grep."""
    if not isinstance(body, dict):
        return "unknown"

    messages = body.get("messages") or body.get("input") or []
    if not isinstance(messages, list) or not messages:
        return "no_messages"

    last = messages[-1]
    if not isinstance(last, dict):
        return "unknown"

    role = last.get("role", "")
    content = last.get("content", "") or ""

    # Flatten content if it's a list of blocks
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )

    if role == "user" and not content.strip():
        # Check if prior assistant had tool_calls
        for msg in reversed(messages[:-1]):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                if isinstance(msg.get("tool_calls"), list) and msg["tool_calls"]:
                    return "empty_user_after_tool_calls"
                return "empty_user_plain"
            if msg.get("role") == "user":
                break
        return "empty_user_no_prior"

    if role == "assistant":
        markers = ["## Active Task", "context was compacted", "<summary>"]
        if any(m.lower() in content.lower() for m in markers):
            return "compaction_summary"

    if role == "tool":
        return "tool_result_cycle"

    return "normal"


def _extract_session(body: Any, headers: Dict) -> str:
    sid = headers.get("x-session-id", "")
    if sid:
        return sid
    if isinstance(body, dict):
        # openai-style: look in messages for a system message with session hint
        for msg in (body.get("messages") or []):
            if isinstance(msg, dict) and msg.get("role") == "system":
                c = msg.get("content", "")
                if "session" in c.lower():
                    return c[:80]
    return "unknown"


def _append_log(log_path: Path, entry: Dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ── proxy handler ──────────────────────────────────────────────────────────────

async def _proxy(request: web.Request, upstream: str, log_path: Path) -> web.StreamResponse:
    ts = datetime.now(timezone.utc).isoformat()
    path = request.path
    if request.query_string:
        path = f"{path}?{request.query_string}"

    # Read and parse request body
    raw_body = await request.read()
    try:
        body = json.loads(raw_body) if raw_body else {}
    except Exception:
        body = raw_body.decode(errors="replace")

    req_headers = dict(request.headers)
    session_id = _extract_session(body, req_headers)
    trigger = _detect_trigger(body)

    log.info(
        "%s %s  session=%s  trigger=%s  body_bytes=%d",
        request.method, request.path, session_id, trigger, len(raw_body),
    )

    if trigger.startswith("empty_user"):
        log.warning("⚠️  EMPTY USER MESSAGE detected — trigger=%s", trigger)
        # Log the last few messages for context
        messages = (body.get("messages") or body.get("input") or []) if isinstance(body, dict) else []
        for i, m in enumerate(messages[-5:]):
            if isinstance(m, dict):
                role = m.get("role", "?")
                c = m.get("content", "") or ""
                tc = m.get("tool_calls")
                if isinstance(c, list):
                    c = str(c)[:120]
                log.warning(
                    "  msg[-%d] role=%s content_len=%d tool_calls=%s preview=%r",
                    5 - i, role, len(c), bool(tc), c[:80],
                )

    # Log request
    _append_log(log_path, {
        "ts": ts,
        "direction": "request",
        "method": request.method,
        "path": request.path,
        "session_id": session_id,
        "trigger": trigger,
        "headers": {k: v for k, v in req_headers.items() if k.lower() not in ("authorization", "x-api-key")},
        "body": body,
    })

    # Forward to upstream
    upstream_url = f"{upstream.rstrip('/')}{path}"
    forward_headers = {
        k: v for k, v in req_headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    t0 = time.monotonic()
    try:
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async with session.request(
                method=request.method,
                url=upstream_url,
                headers=forward_headers,
                data=raw_body,
                allow_redirects=False,
            ) as upstream_resp:
                # Stream the response back
                resp = web.StreamResponse(
                    status=upstream_resp.status,
                    headers={
                        k: v for k, v in upstream_resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "content-length")
                    },
                )
                await resp.prepare(request)

                chunks = []
                async for chunk in upstream_resp.content.iter_any():
                    await resp.write(chunk)
                    chunks.append(chunk)

                await resp.write_eof()

                elapsed = time.monotonic() - t0
                full_body = b"".join(chunks).decode(errors="replace")

                log.info(
                    "%s %s → %d  %.2fs  trigger=%s",
                    request.method, request.path,
                    upstream_resp.status, elapsed, trigger,
                )

                # Log response
                _append_log(log_path, {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "direction": "response",
                    "method": request.method,
                    "path": request.path,
                    "session_id": session_id,
                    "trigger": trigger,
                    "status": upstream_resp.status,
                    "elapsed_s": round(elapsed, 3),
                    "body": full_body,
                })

                return resp

    except Exception as exc:
        elapsed = time.monotonic() - t0
        log.error("Upstream error after %.2fs: %s", elapsed, exc)
        _append_log(log_path, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "direction": "error",
            "method": request.method,
            "path": request.path,
            "session_id": session_id,
            "trigger": trigger,
            "error": str(exc),
            "elapsed_s": round(elapsed, 3),
        })
        return web.Response(status=502, text=f"Upstream error: {exc}")


# ── replay tool ────────────────────────────────────────────────────────────────

def replay(log_path: Path, upstream: str, filter_trigger: Optional[str] = None) -> None:
    """Replay logged requests from a JSONL file back at the upstream server.

    Useful for reproducing a bug: capture a session with the proxy, then
    replay just the interesting requests to test a fix.

    Usage:
        python3 mitm_proxy.py replay /tmp/mitm.jsonl http://localhost:8642 \
            --trigger empty_user_after_tool_calls
    """
    import urllib.request as _urllib

    entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    requests = [e for e in entries if e["direction"] == "request"]

    if filter_trigger:
        requests = [r for r in requests if r.get("trigger") == filter_trigger]

    log.info("Replaying %d requests (filter=%s) against %s", len(requests), filter_trigger, upstream)

    for i, req in enumerate(requests):
        body_bytes = json.dumps(req["body"]).encode()
        url = f"{upstream.rstrip('/')}{req['path']}"
        headers = {k: v for k, v in req.get("headers", {}).items()
                   if k.lower() not in ("host",)}
        headers["Content-Type"] = "application/json"

        log.info("[%d/%d] %s %s  trigger=%s  body_bytes=%d",
                 i + 1, len(requests), req["method"], req["path"],
                 req.get("trigger"), len(body_bytes))

        try:
            r = _urllib.Request(url, data=body_bytes, headers=headers, method=req["method"])
            with _urllib.urlopen(r, timeout=60) as resp:
                resp_body = resp.read().decode(errors="replace")
                log.info("  → %d  %d bytes", resp.status, len(resp_body))
                # Print first line of response (SSE or JSON)
                first_line = resp_body.splitlines()[0] if resp_body else ""
                log.info("  preview: %r", first_line[:120])
        except Exception as exc:
            log.error("  → ERROR: %s", exc)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MITM proxy for hermes-swarm debugging")
    sub = parser.add_subparsers(dest="cmd")

    # proxy subcommand
    proxy_p = sub.add_parser("proxy", help="Run the proxy")
    proxy_p.add_argument("--upstream", default="http://localhost:8642")
    proxy_p.add_argument("--port", type=int, default=8643)
    proxy_p.add_argument("--log", type=Path, default=Path("/tmp/mitm.jsonl"))

    # replay subcommand
    replay_p = sub.add_parser("replay", help="Replay captured requests")
    replay_p.add_argument("log_file", type=Path)
    replay_p.add_argument("upstream", nargs="?", default="http://localhost:8642")
    replay_p.add_argument("--trigger", help="Only replay requests with this trigger tag")

    # analyse subcommand
    analyse_p = sub.add_parser("analyse", help="Summarise a capture log")
    analyse_p.add_argument("log_file", type=Path)

    args = parser.parse_args()

    if args.cmd == "proxy" or args.cmd is None:
        upstream = getattr(args, "upstream", "http://localhost:8642")
        port = getattr(args, "port", 8643)
        log_path = getattr(args, "log", Path("/tmp/mitm.jsonl"))

        log.info("MITM proxy: localhost:%d → %s", port, upstream)
        log.info("Logging to: %s", log_path)

        app = web.Application()

        async def handler(request: web.Request) -> web.StreamResponse:
            return await _proxy(request, upstream, log_path)

        app.router.add_route("*", "/{path_info:.*}", handler)
        web.run_app(app, host="127.0.0.1", port=port, print=None)

    elif args.cmd == "replay":
        replay(args.log_file, args.upstream, filter_trigger=args.trigger)

    elif args.cmd == "analyse":
        entries = [json.loads(l) for l in args.log_file.read_text().splitlines() if l.strip()]
        from collections import Counter
        triggers = Counter(e.get("trigger") for e in entries if e.get("direction") == "request")
        print("Trigger distribution:")
        for trigger, count in triggers.most_common():
            print(f"  {trigger:40s} {count}")
        empties = [e for e in entries if e.get("trigger", "").startswith("empty_user")]
        print(f"\nEmpty user message events: {len(empties)}")
        for e in empties:
            print(f"  {e['ts']}  trigger={e['trigger']}  session={e.get('session_id','?')}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
