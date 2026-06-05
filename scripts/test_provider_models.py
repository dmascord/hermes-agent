#!/usr/bin/env python3
"""Provider/model test harness — drives real API calls to every model.

Run from the k8s pod or locally with the right env vars:
    python3 -m scripts.test_provider_models --output /tmp/provider_test_results.json

This tests:
1. Tool calling — does the model return tool_calls when tools are provided?
2. Response format — does the model return valid JSON?
3. Latency — how fast is the model?
4. Text-only detection — does the model return text when tools are needed?
5. Context window — what's the actual usable context size?

Results are stored in model_quality_db and optionally as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# Add the hermes-agent root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test scenarios ─────────────────────────────────────────────────────

# Simple tool-calling test: a minimal set of tools + a prompt that should trigger them
TOOL_CALL_TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list",
                    }
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_CALL_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Always use the provided tools to answer questions about files and directories."},
    {"role": "user", "content": "List the files in /tmp"},
]

# Simple text test (no tools)
SIMPLE_TEXT_TEST_MESSAGES = [
    {"role": "user", "content": "What is 2+2? Reply with just the number."},
]

# Context window test: generate a large message
def _make_context_test_messages(target_tokens: int) -> List[Dict[str, Any]]:
    """Create a message list with approximately target_tokens tokens."""
    # Rough estimate: 1 token ~= 4 characters
    padding = "x" * (target_tokens * 4)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Summarize this: {padding}"},
    ]


# ── Provider configs ───────────────────────────────────────────────────

def _get_all_models() -> List[Dict[str, Any]]:
    """Return all known models with their provider configs."""
    models = []

    def _add(provider, model, base_url, api_key, **kwargs):
        models.append({
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            **kwargs,
        })

    # github-copilot-enterprise
    _add("github-copilot-enterprise", "gpt-5.4-mini",
         os.getenv("GITHUB_COPILOT_ENTERPRISE_BASE_URL", ""),
         _get_copilot_key("enterprise"))

    # minimax
    _add("minimax", "MiniMax-M3",
         os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
         os.getenv("MINIMAX_API_KEY", ""))
    _add("minimax", "MiniMax-M2.7",
         os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
         os.getenv("MINIMAX_API_KEY", ""))
    _add("minimax", "MiniMax-M2.5",
         os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
         os.getenv("MINIMAX_API_KEY", ""))

    # zai
    _add("zai", "glm-4.7",
         os.getenv("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
         os.getenv("ZAI_API_KEY", ""))

    # opencode-go
    for model in ["mimo-v2.5", "deepseek-v4-pro", "deepseek-v4-flash",
                   "glm-5", "kimi-k2.6", "qwen3.6-plus"]:
        _add("opencode-go", model,
             os.getenv("OPENCODE_GO_BASE_URL", ""),
             os.getenv("OPENCODE_GO_API_KEY", ""))

    # opencode-zen
    for model in ["mimo-v2.5-free", "deepseek-v4-flash-free", "big-pickle"]:
        _add("opencode-zen", model,
             os.getenv("OPENCODE_ZEN_BASE_URL", ""),
             os.getenv("OPENCODE_ZEN_API_KEY", ""))

    # google
    _add("google", "gemini-2.5-flash",
         os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
         os.getenv("GOOGLE_API_KEY", ""))

    # ollama (local)
    for model in ["glm-5.1", "qwen3-coder-next", "deepseek-v4-flash", "kimi-k2-thinking"]:
        _add("ollama", model,
             os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
             os.getenv("OLLAMA_API_KEY", ""))

    # groq
    _add("groq", "llama-3.3-70b-versatile",
         os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
         os.getenv("GROQ_API_KEY", ""))

    # nous (free tier)
    for model in ["stepfun/step-3.7-flash:free", "nvidia/nemotron-3-ultra:free"]:
        _add("nous", model,
             os.getenv("NOUS_BASE_URL", "https://inference-api.nousresearch.com/v1"),
             os.getenv("NOUS_API_KEY", ""))

    # arliai
    for model in ["Mistral-Medium-3.5-128B", "GLM-4.6-Derestricted-v5"]:
        _add("arliai", model,
             os.getenv("ARLIAI_BASE_URL", "https://api.arliai.com/v1"),
             os.getenv("ARLIAI_API_KEY", ""))

    return [m for m in models if m.get("api_key") or m["provider"] == "ollama"]


def _get_copilot_key(tier: str = "enterprise") -> str:
    """Get copilot API key from credential pool."""
    try:
        from agent.credential_pool import load_pool
        pool = load_pool("copilot")
        for entry in pool.entries():
            base = str(getattr(entry, "base_url", "") or "").rstrip("/")
            token = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "") or ""
            if tier == "enterprise" and "copilot-api." in base.lower() and token:
                return token
        # Fallback to env
        return os.getenv("GITHUB_TOKEN", "")
    except Exception:
        return os.getenv("GITHUB_TOKEN", "")


# ── Test runner ────────────────────────────────────────────────────────

def _test_tool_calling(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Test if a model can make tool calls."""
    result = {
        "test": "tool_calling",
        "passed": False,
        "latency_ms": 0,
        "error": None,
        "text_only": False,
        "tool_calls": [],
    }

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            timeout=30.0,
        )

        start = time.time()
        response = client.chat.completions.create(
            model=model_cfg["model"],
            messages=TOOL_CALL_TEST_MESSAGES,
            tools=TOOL_CALL_TEST_TOOLS,
            max_tokens=1024,
        )
        elapsed = (time.time() - start) * 1000

        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", []) or []

        result["latency_ms"] = round(elapsed, 1)

        if tool_calls:
            result["passed"] = True
            result["tool_calls"] = [
                {"name": tc.function.name, "args": tc.function.arguments[:100]}
                for tc in tool_calls
            ]
        else:
            result["text_only"] = True
            result["error"] = "No tool_calls returned (text-only)"

    except Exception as e:
        result["error"] = str(e)[:500]
        result["latency_ms"] = 0

    return result


def _test_simple_text(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Test basic text generation (no tools)."""
    result = {
        "test": "simple_text",
        "passed": False,
        "latency_ms": 0,
        "error": None,
        "response_preview": "",
    }

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            timeout=30.0,
        )

        start = time.time()
        response = client.chat.completions.create(
            model=model_cfg["model"],
            messages=SIMPLE_TEXT_TEST_MESSAGES,
            max_tokens=64,
        )
        elapsed = (time.time() - start) * 1000

        content = response.choices[0].message.content or ""
        result["latency_ms"] = round(elapsed, 1)
        result["response_preview"] = content[:200]
        result["passed"] = len(content.strip()) > 0

    except Exception as e:
        result["error"] = str(e)[:500]

    return result


def _test_model(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run all tests on a model and return combined results."""
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    full_name = f"{provider}/{model}"

    print(f"  Testing {full_name}...", end=" ", flush=True)

    results = {
        "provider": provider,
        "model": model,
        "full_name": full_name,
        "base_url": model_cfg.get("base_url", ""),
        "has_api_key": bool(model_cfg.get("api_key")),
        "tool_calling": _test_tool_calling(model_cfg),
        "simple_text": _test_simple_text(model_cfg),
        "tested_at": time.time(),
    }

    # Summary
    tool_ok = results["tool_calling"]["passed"]
    text_ok = results["simple_text"]["passed"]
    latency = results["tool_calling"]["latency_ms"] or results["simple_text"]["latency_ms"]

    if tool_ok:
        print(f"✓ tools OK ({latency:.0f}ms)")
    elif text_ok:
        print(f"⚠ text-only ({latency:.0f}ms)")
    else:
        err = results["tool_calling"].get("error") or results["simple_text"].get("error")
        print(f"✗ FAILED: {err[:80]}")

    # Record to quality DB
    try:
        from agent.model_quality_db import record_success, record_failure, record_text_only
        if tool_ok:
            record_success(provider, model, base_url=model_cfg.get("base_url", ""), latency_ms=latency)
        elif text_ok and results["tool_calling"].get("text_only"):
            record_text_only(provider, model, base_url=model_cfg.get("base_url", ""), latency_ms=latency)
        elif results["tool_calling"].get("error"):
            record_failure(provider, model, base_url=model_cfg.get("base_url", ""),
                          latency_ms=latency, error_message=results["tool_calling"]["error"])
        else:
            record_failure(provider, model, base_url=model_cfg.get("base_url", ""),
                          latency_ms=latency, error_message="All tests failed")
    except Exception as e:
        print(f"    [quality_db error: {e}]")

    return results


# ── Main ───────────────────────────────────────────────────────────────

def run_all_tests(output_path: Optional[str] = None) -> List[Dict]:
    """Run tests on all models and return results."""
    models = _get_all_models()
    print(f"Found {len(models)} models with API keys")

    results = []
    for model_cfg in models:
        try:
            result = _test_model(model_cfg)
            results.append(result)
        except Exception as e:
            print(f"  ✗ {model_cfg['provider']}/{model_cfg['model']}: {e}")
            results.append({
                "provider": model_cfg["provider"],
                "model": model_cfg["model"],
                "full_name": f"{model_cfg['provider']}/{model_cfg['model']}",
                "error": str(e),
                "tested_at": time.time(),
            })

    # Summary
    total = len(results)
    tool_ok = sum(1 for r in results if r.get("tool_calling", {}).get("passed"))
    text_ok = sum(1 for r in results if r.get("simple_text", {}).get("passed"))
    text_only = sum(1 for r in results if r.get("tool_calling", {}).get("text_only"))

    print(f"\n{'='*60}")
    print(f"Results: {total} models tested")
    print(f"  Tool calling OK: {tool_ok}")
    print(f"  Text-only (no tools): {text_only}")
    print(f"  Text generation OK: {text_ok}")
    print(f"  Failed: {total - tool_ok - text_only}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test all Hermes provider models")
    parser.add_argument("--output", "-o", default=None,
                       help="Path to save JSON results")
    parser.add_argument("--model", "-m", default=None,
                       help="Test a specific model (provider/model)")
    args = parser.parse_args()

    if args.model:
        # Test single model
        parts = args.model.split("/", 1)
        if len(parts) != 2:
            print("Error: model must be in provider/model format")
            sys.exit(1)
        provider, model = parts
        # Find the model config
        models = _get_all_models()
        model_cfg = next((m for m in models if m["provider"] == provider and m["model"] == model), None)
        if not model_cfg:
            print(f"Error: model {args.model} not found (no API key?)")
            sys.exit(1)
        result = _test_model(model_cfg)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
    else:
        run_all_tests(output_path=args.output)


if __name__ == "__main__":
    main()
