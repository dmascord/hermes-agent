#!/usr/bin/env python3
"""Comprehensive provider/model test harness.

Hits every known provider/model with real API calls, testing:
1. Tool calling — does the model return tool_calls when tools are provided?
2. Simple text — does the model respond at all?
3. Latency — how fast is each model?
4. Error patterns — which models fail and why?

Results are stored in model_quality_db and optionally as JSON.

Run inside the pod:
    python3 scripts/test_provider_models.py -o /tmp/provider_test_results.json
    python3 scripts/test_provider_models.py --provider arliai   # test one provider
    python3 scripts/test_provider_models.py --provider openrouter --free  # free OR models
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the hermes-agent root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Tool calling test fixtures ─────────────────────────────────────────

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


# ── Provider definitions ───────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_all_provider_configs() -> Dict[str, List[Dict[str, Any]]]:
    """Return all providers and their models with credentials."""
    providers: Dict[str, List[Dict[str, Any]]] = {}

    def _add(provider: str, model: str, base_url: str, api_key: str, **kw):
        if not api_key:
            return
        providers.setdefault(provider, []).append({
            "provider": provider,
            "model": model,
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            **kw,
        })

    # ── ARLIAI ──────────────────────────────────────────────────────
    arliai_key = _env("ARLIAI_API_KEY") or _env("ARLI_API_KEY")
    arliai_url = _env("ARLIAI_BASE_URL", "https://api.arliai.com/v1")
    if arliai_key:
        for m in [
            "Mistral-Medium-3.5-128B",
            "GLM-4.6-Derestricted-v5",
            "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted",
            "Qwen3.5-27B-BlueStar-v3-Derestricted-Lite",
            "DeepSeek-V3.2-Arli",
        ]:
            _add("arliai", m, arliai_url, arliai_key)

    # ── MINIMAX ─────────────────────────────────────────────────────
    minimax_key = _env("MINIMAX_API_KEY")
    minimax_url = _env("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    if minimax_key:
        for m in ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5"]:
            _add("minimax", m, minimax_url, minimax_key)

    # ── ZAI (Zhipu) ────────────────────────────────────────────────
    zai_key = _env("ZAI_API_KEY") or _env("GLM_API_KEY")
    zai_url = _env("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    if zai_key:
        for m in ["glm-4.7", "glm-4.7-flash", "GLM-5"]:
            _add("zai", m, zai_url, zai_key)

    # ── OPENCODE-GO ─────────────────────────────────────────────────
    ocgo_key = _env("OPENCODE_GO_API_KEY")
    ocgo_url = _env("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
    if ocgo_key:
        for m in [
            "mimo-v2.5", "mimo-v2.5-pro", "deepseek-v4-pro",
            "deepseek-v4-flash", "glm-5", "kimi-k2.6", "qwen3.6-plus",
            "gemini-2.5-flash",
        ]:
            _add("opencode-go", m, ocgo_url, ocgo_key)

    # ── OPENCODE-ZEN ────────────────────────────────────────────────
    oczn_key = _env("OPENCODE_ZEN_API_KEY")
    oczn_url = _env("OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1")
    if oczn_key:
        for m in [
            "mimo-v2.5-free", "deepseek-v4-flash-free", "big-pickle",
            "ling-2.6-flash-free",
        ]:
            _add("opencode-zen", m, oczn_url, oczn_key)

    # ── GOOGLE ──────────────────────────────────────────────────────
    google_key = _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")
    google_url = _env("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    if google_key:
        for m in ["gemini-2.5-flash", "gemini-2.0-flash", "gemma-4-26b-a4b-it"]:
            _add("google", m, google_url, google_key)

    # ── GROQ ────────────────────────────────────────────────────────
    groq_key = _env("GROQ_API_KEY")
    groq_url = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    if groq_key:
        for m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]:
            _add("groq", m, groq_url, groq_key)

    # ── CEREBRAS ────────────────────────────────────────────────────
    cerebras_key = _env("CEREBRAS_API_KEY")
    cerebras_url = _env("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    if cerebras_key:
        for m in ["llama-3.3-70b", "llama-3.1-8b"]:
            _add("cerebras", m, cerebras_url, cerebras_key)

    # ── COHERE ──────────────────────────────────────────────────────
    cohere_key = _env("COHERE_API_KEY")
    cohere_url = _env("COHERE_BASE_URL", "https://api.cohere.com/compatibility/v1")
    if cohere_key:
        for m in ["command-a-03-2025", "command-r-plus-08-2024", "command-r-08-2024"]:
            _add("cohere", m, cohere_url, cohere_key)

    # ── SYNTHETIC ───────────────────────────────────────────────────
    synth_key = _env("SYNTHETIC_API_KEY")
    synth_url = _env("SYNTHETIC_BASE_URL", "https://api.synthetic.new/openai/v1")
    if synth_key:
        for m in ["synthetic-large", "synthetic-medium"]:
            _add("synthetic", m, synth_url, synth_key)

    # ── OLLAMA (cloud) ─────────────────────────────────────────────
    ollama_key = _env("OLLAMA_API_KEY")
    ollama_url = _env("OLLAMA_BASE_URL", "https://ollama.com/v1")
    if ollama_key:
        for m in [
            "qwen3-coder", "glm-5.1", "deepseek-v4-flash",
            "kimi-k2-thinking", "qwen3-coder-next",
        ]:
            _add("ollama", m, ollama_url, ollama_key)

    # ── OPENROUTER (free tier) ──────────────────────────────────────
    or_key = _env("OPENROUTER_API_KEY")
    or_url = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if or_key:
        # Free models with context >= 32K
        free_models = [
            "openrouter/owl-alpha",
            "openrouter/qwen/qwen3-coder:free",
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
            "openrouter/poolside/laguna-xs.2:free",
            "openrouter/poolside/laguna-m.1:free",
            "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
            "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
            "openrouter/openai/gpt-oss-120b:free",
            "openrouter/openai/gpt-oss-20b:free",
            "openrouter/z-ai/glm-4.5-air:free",
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",
            "openrouter/nvidia/nemotron-nano-9b-v2:free",
            "openrouter/google/gemma-3-27b-it:free",
            "openrouter/deepseek/deepseek-r1-0528:free",
            "openrouter/deepseek/deepseek-chat-v3-0324:free",
            "openrouter/qwen/qwen3-235b-a22b:free",
            "openrouter/microsoft/mai-ds-r1:free",
        ]
        for m in free_models:
            # Extract the model name after "openrouter/"
            model_id = m.split("/", 1)[1] if "/" in m else m
            _add("openrouter", m, or_url, or_key, or_model_id=model_id)

    return providers


# ── Test runner ────────────────────────────────────────────────────────

def _call_openai_compat(
    model_cfg: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Make a raw OpenAI-compatible API call using urllib (no SDK dependency)."""
    import urllib.request
    import urllib.error

    base_url = model_cfg["base_url"]
    api_key = model_cfg["api_key"]
    model = model_cfg.get("or_model_id") or model_cfg["model"]

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
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://hermes.tusker.net.au",
        "X-Title": "Hermes Model Test",
    }

    url = f"{base_url}/chat/completions"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        elapsed_ms = (time.time() - start) * 1000
        body = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "latency_ms": elapsed_ms, "body": body}
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.time() - start) * 1000
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:500]
        except Exception:
            pass
        return {"ok": False, "latency_ms": elapsed_ms, "status": e.code, "error": error_body}
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {"ok": False, "latency_ms": elapsed_ms, "error": str(e)[:500]}


def _test_tool_calling(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Test if a model returns tool calls when tools are provided."""
    result = _call_openai_compat(model_cfg, TOOL_MESSAGES, tools=TOOLS, timeout=45)

    if not result.get("ok"):
        return {
            "tool_calling": False,
            "error": result.get("error", "") or f"HTTP {result.get('status')}: {(result.get('error', ''))[:200]}",
            "latency_ms": result.get("latency_ms", 0),
            "error_code": result.get("status", 0),
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

    # Extract tool names if available
    tool_names = []
    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_names.append(fn.get("name", "unknown"))

    return {
        "tool_calling": has_tools,
        "text_only": not has_tools and bool(content),
        "finish_reason": finish,
        "tool_names": tool_names,
        "content_preview": content[:200] if content else "",
        "latency_ms": result["latency_ms"],
        "usage": body.get("usage", {}),
    }


def _test_simple_text(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Test basic text generation (no tools)."""
    result = _call_openai_compat(model_cfg, SIMPLE_MESSAGES, timeout=30)

    if not result.get("ok"):
        return {
            "ok": False,
            "error": result.get("error", "") or f"HTTP {result.get('status')}",
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


def _test_model(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run all tests on a single model."""
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    full_name = f"{provider}/{model}"

    print(f"  {full_name:70s}", end="", flush=True)

    tc = _test_tool_calling(model_cfg)
    st = _test_simple_text(model_cfg)

    # Determine status
    if tc.get("tool_calling"):
        status = "TOOLS_OK"
        score_delta = "tool calling works"
    elif tc.get("text_only"):
        status = "TEXT_ONLY"
        score_delta = "returned text instead of tool calls"
    elif st.get("ok"):
        status = "TEXT_OK"
        score_delta = "text works, no tool calls"
    elif tc.get("error") or st.get("error"):
        err = tc.get("error", "") or st.get("error", "")
        if "401" in err or "403" in err or "unauthorized" in err.lower():
            status = "AUTH_ERR"
            score_delta = f"authentication failed: {err[:80]}"
        elif "429" in err or "rate" in err.lower():
            status = "RATE_LTD"
            score_delta = f"rate limited: {err[:80]}"
        elif "404" in err or "not found" in err.lower():
            status = "NOT_FOUND"
            score_delta = f"model not found: {err[:80]}"
        elif "503" in err or "unavailable" in err.lower():
            status = "UNAVAIL"
            score_delta = f"unavailable: {err[:80]}"
        else:
            status = "FAIL"
            score_delta = err[:120]
    else:
        status = "FAIL"
        score_delta = "all tests failed"

    latency = tc.get("latency_ms", 0) or st.get("latency_ms", 0)
    print(f" {status:10s} {latency:7.0f}ms  {score_delta[:60]}")

    # Record to quality DB
    try:
        from agent.model_quality_db import record_success, record_failure, record_text_only
        base_url = model_cfg.get("base_url", "")
        if tc.get("tool_calling"):
            record_success(provider, model, base_url=base_url, latency_ms=latency)
        elif tc.get("text_only"):
            record_text_only(provider, model, base_url=base_url, latency_ms=latency)
        else:
            err_msg = tc.get("error", "") or st.get("error", "") or "unknown"
            record_failure(provider, model, base_url=base_url, latency_ms=latency,
                          error_code=tc.get("error_code", 0), error_message=err_msg)
    except Exception:
        pass

    return {
        "provider": provider,
        "model": model,
        "full_name": full_name,
        "status": status,
        "tool_calling": tc,
        "simple_text": st,
        "latency_ms": latency,
        "tested_at": time.time(),
        "summary": score_delta,
    }


# ── Main ───────────────────────────────────────────────────────────────

def run_all_tests(
    output_path: Optional[str] = None,
    provider_filter: Optional[str] = None,
    free_only: bool = False,
) -> List[Dict]:
    """Run tests on all models and return results."""
    providers = _get_all_provider_configs()

    if provider_filter:
        pf = provider_filter.lower()
        providers = {k: v for k, v in providers.items() if pf in k.lower()}

    total_models = sum(len(v) for v in providers.values())
    print(f"{'='*100}")
    print(f"Provider/Model Test Harness — {total_models} models across {len(providers)} providers")
    print(f"{'='*100}")

    results = []
    for provider_name, models in sorted(providers.items()):
        print(f"\n── {provider_name} ({len(models)} models) {'─'*60}")
        for model_cfg in models:
            try:
                result = _test_model(model_cfg)
                results.append(result)
            except Exception as e:
                print(f"  {model_cfg['provider']}/{model_cfg['model']:60s} ERROR  {e}")
                results.append({
                    "provider": model_cfg["provider"],
                    "model": model_cfg["model"],
                    "full_name": f"{model_cfg['provider']}/{model_cfg['model']}",
                    "status": "ERROR",
                    "error": str(e),
                    "tested_at": time.time(),
                })

    # Summary
    tools_ok = sum(1 for r in results if r.get("status") == "TOOLS_OK")
    text_only = sum(1 for r in results if r.get("status") == "TEXT_ONLY")
    text_ok = sum(1 for r in results if r.get("status") == "TEXT_OK")
    auth_err = sum(1 for r in results if r.get("status") == "AUTH_ERR")
    rate_ltd = sum(1 for r in results if r.get("status") == "RATE_LTD")
    not_found = sum(1 for r in results if r.get("status") == "NOT_FOUND")
    fail = sum(1 for r in results if r.get("status") in ("FAIL", "ERROR", "UNAVAIL"))

    print(f"\n{'='*100}")
    print(f"SUMMARY: {len(results)} models tested")
    print(f"  ✓ Tool calling works:  {tools_ok}")
    print(f"  ⚠ Text only (no tools): {text_only}")
    print(f"  ○ Text works (no tool test): {text_ok}")
    print(f"  ✗ Auth error:  {auth_err}")
    print(f"  ✗ Rate limited: {rate_ltd}")
    print(f"  ✗ Not found:  {not_found}")
    print(f"  ✗ Failed/unavailable: {fail}")
    print(f"{'='*100}")

    # List models that need work
    print("\n── Models that returned text-only (need tool-calling fix or removal) ──")
    for r in results:
        if r.get("status") == "TEXT_ONLY":
            print(f"  {r['full_name']:60s} {r.get('summary', '')[:60]}")

    print("\n── Models with auth/key errors (check env) ──")
    for r in results:
        if r.get("status") in ("AUTH_ERR", "NOT_FOUND"):
            print(f"  {r['full_name']:60s} {r.get('summary', '')[:60]}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test all Hermes provider models")
    parser.add_argument("-o", "--output", default=None, help="JSON output path")
    parser.add_argument("-p", "--provider", default=None, help="Test only this provider (substring match)")
    parser.add_argument("--free", action="store_true", help="Only test free-tier models")
    args = parser.parse_args()

    run_all_tests(output_path=args.output, provider_filter=args.provider, free_only=args.free)


if __name__ == "__main__":
    main()
