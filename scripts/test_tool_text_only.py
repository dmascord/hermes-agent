#!/usr/bin/env python3
"""Test harness: reproduce text-only failures across all fallback models.

Tests each model with 24 tools (OMP default), 7 tools (Hermes fallback),
and 3 tools (OMP discoveryMode=all). Reports text-only rate for each.

Usage:
    # From hermes pod (has API_SERVER_KEY and env):
    python3 /opt/hermes/scripts/test_tool_text_only.py --output results.json

    # Test specific model:
    python3 test_tool_text_only.py --model minimax/MiniMax-M2.7 --runs 5
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.error, urllib.request
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Tool sets ──────────────────────────────────────────────────────────

# All 24 active built-in tools OMP sends (matches BUILTIN_TOOLS minus
# conditionally-gated ones like goal, resolve, report_tool_issue).
TOOLS_24 = [
    {"type":"function","function":{"name":"read","description":"Read files, directories, archives, SQLite databases, images, documents, and web URLs through a single path string","parameters":{"type":"object","properties":{"path":{"type":"string","description":"the file or resource path"}},"required":["path"]}}},
    {"type":"function","function":{"name":"bash","description":"Execute shell command, with per-command working directory and environment support","parameters":{"type":"object","properties":{"command":{"type":"string","description":"shell command to run"},"cwd":{"type":"string","description":"working directory path"}},"required":["command"]}}},
    {"type":"function","function":{"name":"edit","description":"Your patch language names lines to replace, delete, or insert at, then lists the new content","parameters":{"type":"object","properties":{"input":{"type":"string","description":"patch input"},"path":{"type":"string","description":"target file"}},"required":["input","path"]}}},
    {"type":"function","function":{"name":"ast_grep","description":"Performs structural code search using AST matching via native ast-grep","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"AST pattern to find"},"paths":{"type":"array","items":{"type":"string"},"description":"file or directory paths"}},"required":["pattern","paths"]}}},
    {"type":"function","function":{"name":"ast_edit","description":"Performs structural AST-aware rewrites via native ast-grep","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"AST pattern to match"},"replacement":{"type":"string","description":"replacement template"},"paths":{"type":"array","items":{"type":"string"},"description":"file or directory paths"}},"required":["pattern","replacement","paths"]}}},
    {"type":"function","function":{"name":"render_mermaid","description":"Render a Mermaid diagram to an image for visual inspection","parameters":{"type":"object","properties":{"diagram":{"type":"string","description":"the Mermaid diagram definition"},"type":{"type":"string","enum":["image","flowchart","sequence","class","state","gantt","pie"]}},"required":["diagram"]}}},
    {"type":"function","function":{"name":"ask","description":"Ask the user a question when you need clarification or input during task execution","parameters":{"type":"object","properties":{"question":{"type":"string","description":"question to ask"},"options":{"type":"array","items":{"type":"string"},"description":"answer choices"}},"required":["question"]}}},
    {"type":"function","function":{"name":"debug","description":"Provides debugger access through the Debug Adapter Protocol (DAP)","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["launch","attach","continue","step_in","step_out"]},"file":{"type":"string","description":"source file"},"line":{"type":"integer","description":"line number"}},"required":["action"]}}},
    {"type":"function","function":{"name":"eval","description":"Run code in a persistent kernel using cells","parameters":{"type":"object","properties":{"cells":{"type":"array","items":{"type":"object","properties":{"language":{"type":"string"},"code":{"type":"string"}}},"description":"cells to execute"}},"required":["cells"]}}},
    {"type":"function","function":{"name":"ssh","description":"Execute commands on remote hosts via SSH","parameters":{"type":"object","properties":{"host":{"type":"string","description":"remote hostname"},"command":{"type":"string","description":"command to execute"}},"required":["host","command"]}}},
    {"type":"function","function":{"name":"github","description":"GitHub API operations - issues, PRs, repos","parameters":{"type":"object","properties":{"action":{"type":"string","description":"github action to perform"},"repo":{"type":"string","description":"repository name"}},"required":["action"]}}},
    {"type":"function","function":{"name":"find","description":"Finds files and directories using fast pattern matching","parameters":{"type":"object","properties":{"paths":{"type":"array","items":{"type":"string"},"description":"glob patterns to match"},"limit":{"type":"integer","description":"max results"}},"required":["paths"]}}},
    {"type":"function","function":{"name":"search","description":"Searches files using powerful regex matching","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"regex pattern"},"paths":{"type":"array","items":{"type":"string"},"description":"paths to search"}},"required":["pattern","paths"]}}},
    {"type":"function","function":{"name":"lsp","description":"Interacts with Language Server Protocol servers for code intelligence","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["definition","references","hover","rename"]},"file":{"type":"string","description":"file path"},"symbol":{"type":"string","description":"symbol name"}},"required":["action","file"]}}},
    {"type":"function","function":{"name":"inspect_image","description":"Read and inspect image files for visual analysis","parameters":{"type":"object","properties":{"path":{"type":"string","description":"image file path"}},"required":["path"]}}},
    {"type":"function","function":{"name":"browser","description":"Drives a real Chromium tab with full puppeteer access","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["open","run","close"]},"url":{"type":"string","description":"URL to navigate to"}},"required":["action"]}}},
    {"type":"function","function":{"name":"checkpoint","description":"Save current state as a checkpoint for later rewind","parameters":{"type":"object","properties":{"label":{"type":"string","description":"checkpoint label"}},"required":["label"]}}},
    {"type":"function","function":{"name":"rewind","description":"Rewind session state to a previous checkpoint","parameters":{"type":"object","properties":{"target":{"type":"string","description":"checkpoint to rewind to"}},"required":["target"]}}},
    {"type":"function","function":{"name":"task","description":"Launch subagents to parallelize workflows","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"task instructions"},"context":{"type":"string","description":"shared context"}},"required":["prompt"]}}},
    {"type":"function","function":{"name":"job","description":"Background job operations - manage long-running processes","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["list","cancel","status"]},"job_id":{"type":"string","description":"job id"}},"required":["action"]}}},
    {"type":"function","function":{"name":"irc","description":"Inter-process communication with sibling subagents","parameters":{"type":"object","properties":{"target":{"type":"string","description":"recipient agent id"},"message":{"type":"string","description":"message text"}},"required":["target","message"]}}},
    {"type":"function","function":{"name":"todo_write","description":"Manages a phased task list","parameters":{"type":"object","properties":{"ops":{"type":"array","items":{"type":"object"},"description":"todo operations"}},"required":["ops"]}}},
    {"type":"function","function":{"name":"web_search","description":"Searches the web for up-to-date information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}}},
    {"type":"function","function":{"name":"search_tool_bm25","description":"Search hidden tool metadata to discover and activate tools","parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"},"limit":{"type":"integer","description":"max results"}},"required":["query"]}}},
    {"type":"function","function":{"name":"write","description":"Creates or overwrites file at specified path","parameters":{"type":"object","properties":{"path":{"type":"string","description":"file path"},"content":{"type":"string","description":"file content"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"memory_edit","description":"Edit a memory entry in long-term storage","parameters":{"type":"object","properties":{"id":{"type":"string","description":"memory entry id"},"content":{"type":"string","description":"new content"}},"required":["id","content"]}}},
    {"type":"function","function":{"name":"retain","description":"Store information in long-term memory for future sessions","parameters":{"type":"object","properties":{"content":{"type":"string","description":"information to remember"}},"required":["content"]}}},
    {"type":"function","function":{"name":"recall","description":"Search long-term memory for relevant information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"search query"}},"required":["query"]}}},
    {"type":"function","function":{"name":"reflect","description":"Generate a synthesised answer by reasoning over long-term memory","parameters":{"type":"object","properties":{"query":{"type":"string","description":"question to answer"}},"required":["query"]}}},
]

# 7 essential tools (Hermes fallback filter)
_TOOL_NAMES_ESSENTIAL = {"bash", "read", "edit", "find", "search", "write", "search_tool_bm25"}
TOOLS_7 = [t for t in TOOLS_24 if t["function"]["name"] in _TOOL_NAMES_ESSENTIAL]

# 3 essential tools (OMP discoveryMode=all default)
_TOOL_NAMES_MINIMAL = {"bash", "read", "edit"}
TOOLS_3 = [t for t in TOOLS_24 if t["function"]["name"] in _TOOL_NAMES_MINIMAL]

# ── Messages ───────────────────────────────────────────────────────────

# Realistic coding task that should trigger tool calls
TOOL_TEST_MSGS = [
    {"role": "system", "content": "You are a helpful coding assistant. You have file system tools available — use them when you need to read or manipulate files."},
    {"role": "user", "content": "Find all Python files in /tmp that contain the word 'test' in them. Read one of them and summarize what it does."},
]

# Simple text-only baseline (no tools)
SIMPLE_MSGS = [
    {"role": "user", "content": "Reply: 42"},
]

# ── Client ─────────────────────────────────────────────────────────────

API_SERVER_KEY = os.getenv("API_SERVER_KEY", "")
HERMES_URL = os.getenv("HERMES_URL", "https://hermes.tusker.net.au")

def call_model(model: str, messages: list, tools: list = None, timeout: int = 60) -> dict:
    """Make a chat completion request to the gateway."""
    body = {"model": model, "messages": messages, "max_tokens": 512, "temperature": 0}
    if tools: body["tools"] = tools
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_SERVER_KEY}"}
    req = urllib.request.Request(
        f"{HERMES_URL}/v1/chat/completions",
        data=json.dumps(body).encode(), headers=headers, method="POST",
    )
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        r = json.loads(resp.read())
        return {"ok": True, "ms": (time.time()-start)*1000, "body": r, "status": resp.status}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "ms": (time.time()-start)*1000, "status": e.code, "raw": raw}
    except Exception as e:
        return {"ok": False, "ms": (time.time()-start)*1000, "err": str(e)[:200]}

def analyze_response(resp: dict) -> dict:
    """Analyze response: text-only vs tool_calls vs error."""
    if not resp.get("ok"):
        return {"status": "error", "text_only": False, "tool_calls": False,
                "content": resp.get("raw","") or resp.get("err","")}
    choices = (resp.get("body") or {}).get("choices", [])
    if not choices:
        return {"status": "no_choices", "text_only": False, "tool_calls": False}
    msg = choices[0].get("message", {})
    tc = msg.get("tool_calls", [])
    content = msg.get("content", "") or ""
    finish = choices[0].get("finish_reason", "")
    return {"status": "ok", "text_only": bool(content and not tc), "tool_calls": bool(tc),
            "finish_reason": finish,
            "tool_names": [t.get("function",{}).get("name","?") for t in (tc or [])],
            "content_preview": content[:200] if content else "", "content_len": len(content)}

def get_fallback_chain() -> list:
    """Get fallback chain from environment."""
    models = []
    for key, val in sorted(os.environ.items()):
        if key.startswith("HERMES_CODE_FALLBACK_") or key == "HERMES_CODE_MODEL":
            if val and val.strip():
                if val not in models:
                    models.append(val.strip())
    # Reorder: primary first, fallbacks after
    primary = os.getenv("HERMES_CODE_MODEL", "")
    if primary in models:
        models.remove(primary)
        models.insert(0, primary)
    return models

# ── Test runner ────────────────────────────────────────────────────────

def test_model(model: str, tools: list, runs: int = 3, timeout: int = 60) -> dict:
    """Test a model with a specific tool set."""
    results = []
    for i in range(runs):
        resp = call_model(model, TOOL_TEST_MSGS, tools, timeout=timeout)
        analysis = analyze_response(resp)
        analysis["ms"] = resp.get("ms", 0)
        analysis["status_code"] = resp.get("status", 200)
        results.append(analysis)

    total = len(results)
    text_only = sum(1 for r in results if r.get("text_only"))
    tool_calls = sum(1 for r in results if r.get("tool_calls"))
    errors = sum(1 for r in results if r.get("status") == "error")
    no_choices = sum(1 for r in results if r.get("status") == "no_choices")
    avg_ms = sum(r.get("ms", 0) for r in results) / total if total else 0

    return {"model": model, "runs": results, "tool_count": len(tools),
            "summary": {"total": total, "text_only": text_only, "tool_calls": tool_calls,
                        "errors": errors, "no_choices": no_choices,
                        "text_only_rate": text_only / total if total else 0,
                        "tool_calls_rate": tool_calls / total if total else 0,
                        "avg_latency_ms": avg_ms}}

def test_models(models: list, runs: int = 3, output: str = None):
    """Test all models with 24, 7, and 3 tools."""
    configs = [
        ("24 tools (OMP default)", TOOLS_24),
        (" 7 tools (Hermes fallback)", TOOLS_7),
        (" 3 tools (OMP discoveryMode=all)", TOOLS_3),
    ]

    all_by_config = {}
    for label, tools in configs:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        config_results = []
        for model in models:
            print(f"  {model:45s} ...", end=" ", flush=True)
            result = test_model(model, tools, runs=runs)
            s = result["summary"]
            pct = s["text_only_rate"] * 100
            if s["text_only_rate"] > 0.5:
                status = "TEXT"
            elif s["tool_calls_rate"] > 0.5:
                status = "TOOL"
            else:
                status = "MIX"
            print(f"{status:4s}  {s['tool_calls']}/{s['total']} tool  {pct:3.0f}% text  {s['avg_latency_ms']:.0f}ms")
            config_results.append(result)

        config_results.sort(key=lambda r: -(r["summary"]["text_only_rate"]))
        all_by_config[len(tools)] = config_results

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Text-only rate comparison")
    print(f"{'='*70}")
    print(f"  {'Model':45s} {'24T':>6s} {'7T':>6s} {'3T':>6s}")
    print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*6}")
    for model in models:
        short = model.split("/", 1)[-1] if "/" in model else model
        r24 = next((r for r in all_by_config.get(24, []) if r["model"] == model), None)
        r7 = next((r for r in all_by_config.get(7, []) if r["model"] == model), None)
        r3 = next((r for r in all_by_config.get(3, []) if r["model"] == model), None)
        p24 = f"{r24['summary']['text_only_rate']*100:4.0f}%" if r24 else "  N/A"
        p7 = f"{r7['summary']['text_only_rate']*100:4.0f}%" if r7 else "  N/A"
        p3 = f"{r3['summary']['text_only_rate']*100:4.0f}%" if r3 else "  N/A"
        print(f"  {short:45s} {p24:>6s} {p7:>6s} {p3:>6s}")

    if output:
        with open(output, "w") as f:
            json.dump({"configs": all_by_config, "runs": runs, "timestamp": time.time()}, f, indent=2)
        print(f"\n  Saved to {output}")

def main():
    parser = argparse.ArgumentParser(description="Test tool-calling across fallback chain")
    parser.add_argument("--model", help="Test a single model")
    parser.add_argument("--output", default="tool_text_only_results.json", help="Output JSON file")
    parser.add_argument("--runs", type=int, default=3, help="Runs per model (default 3)")
    args = parser.parse_args()

    if args.model:
        for label, tools in [("24 tools", TOOLS_24), ("7 tools", TOOLS_7), ("3 tools", TOOLS_3)]:
            print(f"\n{label}:")
            result = test_model(args.model, tools, runs=args.runs)
            s = result["summary"]
            print(f"  tool_calls={s['tool_calls']}/{s['total']}  text_only={s['text_only']}/{s['total']}  "
                  f"errors={s['errors']}  avg={s['avg_latency_ms']:.0f}ms")
        return

    models = get_fallback_chain()
    if not models:
        print("No models in fallback chain.")
        sys.exit(1)
    print(f"Testing {len(models)} models, {args.runs} runs each, 3 tool configurations")
    test_models(models, runs=args.runs, output=args.output)

if __name__ == "__main__":
    main()
