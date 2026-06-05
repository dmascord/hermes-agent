#!/usr/bin/env python3
"""Provider/model test harness — hits the live Hermes gateway.

Tests every model through the actual passthrough chain, exercising all
of Hermes' provider routing: credential resolution, URL normalization,
reasoning echo, tool_call_id sanitization, context filtering, etc.

Run inside the pod:
    python3 scripts/test_provider_models.py
    python3 scripts/test_provider_models.py --provider arliai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Tool calling test fixtures (same as real Hermes passthrough uses) ─

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_MESSAGES = [
    {
        "role": "system",
        "content": "You are a file operations assistant. Always use the provided tools to answer. Never respond with plain text when a tool can be used.",
    },
    {
        "role": "user",
        "content": "List the files in /tmp",
    },
]

SIMPLE_MESSAGES = [
    {"role": "user", "content": "Reply with just the number 42."},
]


# ── Discover models from HERMES_CODE env vars and OpenRouter free ─────

def _get_hermes_code_models() -> List[Dict[str, Any]]:
    """Return all models configured in HERMES_CODE_FALLBACK_* env vars."""
    models = []
    for k, v in sorted(os.environ.items()):
        m = re.match(r"HERMES_CODE_FALLBACK_(\d+)", k)
        if m:
            model = v.strip()
            if model and "/" in model:
                models.append({
                    "source": "HERMES_CODE_FALLBACK",
                    "model": model,
                    "order": int(m.group(1)),
                })
    return models


def _get_swarm_models() -> List[Dict[str, Any]]:
    """Return models configured for swarm fallback."""
    models = []
    for k, v in sorted(os.environ.items()):
        m = re.match(r"HERMES_SWARM_FALLBACK_(\d+)", k)
        if m:
            model = v.strip()
            if model and "/" in model:
                models.append({
                    "source": "HERMES_SWARM_FALLBACK",
                    "model": model,
                    "order": int(m.group(1)),
                })
    return models


def _get_openrouter_free_models() -> List[Dict[str, Any]]:
    """Return free OpenRouter models (queried live)."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        models = []

        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            prompt = pricing.get("prompt", "0")
            completion = pricing.get("completion", "0")
            is_free = prompt == "0" and completion == "0"
            ctx = m.get("context_length", 0)

            if not is_free or ctx < 8000:
                continue

            model_id = m.get("id", "")
            models.append({
                "source": "OPENROUTER_FREE",
                "model": f"openrouter/{model_id}",
                "original_model_id": model_id,
                "context": ctx,
            })

        return sorted(models, key=lambda x: x["context"], reverse=True)
    except Exception as e:
        print(f"[WARN] Failed to fetch OpenRouter models: {e}", file=sys.stderr)
        return []


def _get_all_models() -> List[Dict[str, Any]]:
    """Return all models to test."""
    seen: set[str] = set()
    models = []

    for m in _get_hermes_code_models():
        if m["model"] not in seen:
            seen.add(m["model"])
            models.append(m)

    for m in _get_swarm_models():
        if m["model"] not in seen:
            seen.add(m["model"])
            models.append(m)

    for m in _get_openrouter_free_models():
        if m["model"] not in seen:
            seen.add(m["model"])
            models.append(m)

    return models


# ── Test helpers ───────────────────────────────────────────────────────

def _gateway_url() -> str:
    """Return the Hermes gateway URL."""
    host = os.getenv("HERMES_HOST", "http://localhost:8080")
    key = os.getenv("API_SERVER_KEY", "")
    return host


def _call_gateway(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Make a chat completions call through the Hermes gateway."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0,
    }
    if tools:
        body["tools"] = tools

    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('API_SERVER_KEY', '')}",
    }

    url = f"{_gateway_url()}/v1/chat/completions"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed_ms = (time.time() - start) * 1000
        resp_body = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "latency_ms": elapsed_ms, "body": resp_body, "status": resp.status}
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.time() - start) * 1000
        raw = b""
        try:
            raw = e.read()
        except Exception:
            pass
        return {"ok": False, "latency_ms": elapsed_ms, "status": e.code, "body_raw": raw.decode("utf-8", errors="replace")[:1000]}
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {"ok": False, "latency_ms": elapsed_ms, "error": str(e)[:500]}


def _test_tool_calling(model: str) -> Dict[str, Any]:
    """Test if a model returns tool calls through the gateway."""
    result = _call_gateway(model, TOOL_MESSAGES, tools=TOOLS, timeout=60)

    if not result.get("ok"):
        body = result.get("body_raw", "")
        status = result.get("status", 0)
        return {
            "tool_calling": False,
            "error": body or result.get("error", "") or f"HTTP {status}",
            "latency_ms": result.get("latency_ms", 0),
            "status_code": status,
            "details": body[:300] if body else "",
        }

    body = result.get("body", {})
    choices = body.get("choices", [])
    if not choices:
        return {"tool_calling": False, "error": "No choices in response", "latency_ms": result["latency_ms"]}

    choice = choices[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "") or ""
    finish = choice.get("finish_reason", "stop")

    has_tools = bool(tool_calls) or finish == "tool_calls"
    tool_names = [tc.get("function", {}).get("name", "?") for tc in (tool_calls or [])]

    return {
        "tool_calling": has_tools,
        "text_only": not has_tools and bool(content),
        "finish_reason": finish,
        "tool_names": tool_names,
        "content_preview": content[:200] if content else "",
        "latency_ms": result["latency_ms"],
        "usage": body.get("usage", {}),
    }


def _test_simple_text(model: str) -> Dict[str, Any]:
    """Test basic text generation (no tools)."""
    result = _call_gateway(model, SIMPLE_MESSAGES, timeout=30)

    if not result.get("ok"):
        body = result.get("body_raw", "")
        return {
            "ok": False,
            "error": body or result.get("error", ""),
            "latency_ms": result.get("latency_ms", 0),
        }

    body = result.get("body", {})
    choices = body.get("choices", [])
    if not choices:
        return {"ok": False, "error": "No choices", "latency_ms": result["latency_ms"]}

    content = choices[0].get("message", {}).get("content", "") or ""
    return {
        "ok": bool(content.strip()),
        "content": content.strip()[:200],
        "latency_ms": result["latency_ms"],
        "usage": body.get("usage", {}),
    }


# ── Test runner ────────────────────────────────────────────────────────

def _test_model(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Run all tests on a single model through the gateway."""
    model = entry["model"]
    source = entry.get("source", "")

    print(f"  {model:70s}", end="", flush=True)

    tc = _test_tool_calling(model)
    st = _test_simple_text(model)

    # Determine status
    if tc.get("tool_calling"):
        status = "TOOLS_OK"
        detail = f"tools: {','.join(tc.get('tool_names', []))[:40]}"
    elif tc.get("text_only"):
        status = "TXT_ONLY"
        detail = "text-only (tools provided)"
    elif st.get("ok"):
        status = "TEXT_OK"
        detail = st.get("content", "")[:40]
    elif tc.get("status_code") == 413:
        status = "CTX_OVFL"
        detail = "context too large"
    else:
        status = "FAIL"
        err = tc.get("error", "") or st.get("error", "") or "unknown"
        sc = tc.get("status_code", 0)
        if sc == 400:
            status = "BAD_REQ"
        elif sc == 401 or sc == 403:
            status = "AUTH_ER"
        elif sc == 429:
            status = "RATE_LI"
        elif sc == 503:
            status = "UNAVAIL"
        detail = (err or str(tc.get("status_code", "")) or "?")[:80]

    latency = tc.get("latency_ms", 0) or st.get("latency_ms", 0)
    print(f" {status:8s} {latency:7.0f}ms  {detail[:60]}")

    # Record to quality DB
    try:
        from agent.model_quality_db import record_success, record_failure, record_text_only
        provider = model.split("/")[0] if "/" in model else ""

        if tc.get("tool_calling"):
            record_success(provider, model, latency_ms=latency)
        elif tc.get("text_only"):
            record_text_only(provider, model, latency_ms=latency)
        else:
            err_msg = tc.get("error", "") or st.get("error", "") or "all tests failed"
            record_failure(provider, model, latency_ms=latency,
                          error_code=tc.get("status_code", 0), error_message=err_msg)
    except Exception:
        pass

    # Record context window if we got usage data
    try:
        usage = tc.get("usage", {}) or {}
        pt = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
        ct = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
        if pt and ct:
            from agent.model_quality_db import record_success
            pass  # already done above
    except Exception:
        pass

    return {
        "model": model,
        "source": source,
        "status": status,
        "tool_calling": tc,
        "simple_text": st,
        "latency_ms": latency,
        "tested_at": time.time(),
    }


# ── Main ───────────────────────────────────────────────────────────────

def run_all_tests(
    output_path: Optional[str] = None,
    provider_filter: Optional[str] = None,
) -> List[Dict]:
    """Run tests on all models through the gateway."""
    models = _get_all_models()

    if provider_filter:
        pf = provider_filter.lower()
        models = [m for m in models if pf in m["model"].lower()]

    # Group by source for nice output
    from collections import defaultdict
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for m in models:
        by_source[m.get("source", "other")].append(m)

    n_models = len(models)
    n_sources = len(by_source)

    print(f"{'='*120}")
    print(f"Gateway Model Test Harness — {n_models} models across {n_sources} sources")
    print(f"{'='*120}")

    results = []
    for source_name, source_models in sorted(by_source.items()):
        print(f"\n── {source_name} ({len(source_models)} models) {'─'*60}")
        for entry in source_models:
            try:
                result = _test_model(entry)
                results.append(result)
            except Exception as e:
                print(f"  {entry['model']:70s} ERROR  {e}")
                results.append({
                    "model": entry["model"],
                    "source": entry.get("source", ""),
                    "status": "ERROR",
                    "error": str(e),
                    "tested_at": time.time(),
                })

    # Summary
    tools_ok = sum(1 for r in results if r.get("status") == "TOOLS_OK")
    text_only = sum(1 for r in results if r.get("status") == "TXT_ONLY")
    text_ok = sum(1 for r in results if r.get("status") == "TEXT_OK")
    bad_req = sum(1 for r in results if r.get("status") == "BAD_REQ")
    auth_err = sum(1 for r in results if r.get("status") == "AUTH_ER")
    rate_ltd = sum(1 for r in results if r.get("status") == "RATE_LI")
    ctx_ovfl = sum(1 for r in results if r.get("status") == "CTX_OVFL")
    unavailable = sum(1 for r in results if r.get("status") == "UNAVAIL")
    failed = sum(1 for r in results if r.get("status") in ("FAIL", "ERROR"))

    print(f"\n{'='*120}")
    print(f"SUMMARY: {len(results)} models tested")
    print(f"  ✓ Tool calling works:     {tools_ok}")
    print(f"  ⚠ Text only (no tools):    {text_only}")
    print(f"  ○ Text works (no tool):    {text_ok}")
    print(f"  ✗ Bad request (400):      {bad_req}")
    print(f"  ✗ Auth error (401/403):   {auth_err}")
    print(f"  ✗ Rate limited (429):     {rate_ltd}")
    print(f"  ✗ Context overflow (413): {ctx_ovfl}")
    print(f"  ✗ Unavailable (503):      {unavailable}")
    print(f"  ✗ Failed/other:           {failed}")
    print(f"{'='*120}")

    # Text-only models (need investigation)
    text_only_models = [r for r in results if r.get("status") == "TXT_ONLY"]
    if text_only_models:
        print(f"\n── Models returning text-only (tools provided but ignored) ──")
        for r in sorted(text_only_models, key=lambda x: x.get("latency_ms", 0)):
            tc = r.get("tool_calling", {})
            preview = tc.get("content_preview", "")[:80]
            print(f"  {r['model']:65s} {r.get('latency_ms',0):7.0f}ms  \"{preview}\"")

    # Auth/bad request models (check env)
    bad_models = [r for r in results if r.get("status") in ("AUTH_ER", "BAD_REQ", "NOT_FOUND")]
    if bad_models:
        print(f"\n── Models with auth/config errors ──")
        for r in bad_models:
            tc = r.get("tool_calling", {})
            err = tc.get("error", "") or r.get("error", "") or ""
            print(f"  {r['model']:65s} {err[:80]}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test models through Hermes gateway")
    parser.add_argument("-o", "--output", default="/tmp/provider_test_results.json")
    parser.add_argument("-p", "--provider", default=None,
                       help="Test only this provider (substring match)")
    args = parser.parse_args()

    run_all_tests(output_path=args.output, provider_filter=args.provider)


if __name__ == "__main__":
    main()