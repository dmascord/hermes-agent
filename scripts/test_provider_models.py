#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.error, urllib.request
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOOLS = [{"type":"function","function":{"name":"read_file","description":"Read file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},{"type":"function","function":{"name":"list_dir","description":"List dir","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
TOOL_MSGS = [{"role":"system","content":"Use tools."},{"role":"user","content":"List /tmp"}]
SIMPLE_MSGS = [{"role":"user","content":"Reply: 42"}]

def gw_url(): return os.getenv("HERMES_HOST","http://0.0.0.0:8642")

def call_gw(model, msgs, tools=None, timeout=60):
    body = {"model": model, "messages": msgs, "max_tokens": 256, "temperature": 0}
    if tools: body["tools"] = tools
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('API_SERVER_KEY','')}"}
    req = urllib.request.Request(f"{gw_url()}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST")
    start = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return {"ok": True, "ms": (time.time()-start)*1000, "body": json.loads(r.read()), "sc": r.status}
    except urllib.error.HTTPError as e:
        raw = ""
        try: raw = e.read().decode(errors="replace")[:500]
        except: pass
        return {"ok": False, "ms": (time.time()-start)*1000, "sc": e.code, "raw": raw}
    except Exception as e:
        return {"ok": False, "ms": (time.time()-start)*1000, "err": str(e)[:200]}

def test_tc(model):
    r = call_gw(model, TOOL_MSGS, TOOLS, 60)
    if not r.get("ok"): return {"tc": False, "err": r.get("raw","") or r.get("err",""), "ms": r["ms"], "sc": r.get("sc",0)}
    c = r["body"].get("choices",[])
    if not c: return {"tc": False, "err": "no choices", "ms": r["ms"]}
    m = c[0].get("message",{}); tcs = m.get("tool_calls",[]); ct = (m.get("content","") or ""); fn = c[0].get("finish_reason","stop")
    return {"tc": bool(tcs) or fn=="tool_calls", "txt_only": not bool(tcs or fn=="tool_calls") and bool(ct),
            "tools": [x.get("function",{}).get("name","?") for x in (tcs or [])], "ct": ct[:200], "ms": r["ms"]}

def test_txt(model):
    r = call_gw(model, SIMPLE_MSGS)
    if not r.get("ok"): return {"ok": False, "err": r.get("raw","") or r.get("err",""), "ms": r["ms"]}
    c = (r["body"].get("choices",[{}])[0].get("message",{}).get("content","") or "")
    return {"ok": bool(c.strip()), "ct": c[:200], "ms": r["ms"]}

def get_models():
    seen, models = set(), []
    for prefix, src in [("HERMES_CODE_FALLBACK_","CODE"), ("HERMES_SWARM_FALLBACK_","SWARM")]:
        for k,v in sorted(os.environ.items()):
            m = re.match(prefix+r"(\d+)", k)
            if m and v.strip() and "/" in v.strip() and v.strip() not in seen:
                seen.add(v.strip()); models.append({"src": src, "model": v.strip()})
    key = os.getenv("OPENROUTER_API_KEY","").strip()
    if key:
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"})
            for m in json.loads(urllib.request.urlopen(req, timeout=15).read()).get("data",[]):
                p = m.get("pricing",{}); mid = m.get("id","")
                if p.get("prompt","0")!="0" or p.get("completion","0")!="0": continue
                if m.get("context_length",0)<8000: continue
                full = f"openrouter/{mid}"
                if full not in seen: seen.add(full); models.append({"src": "OR_FREE", "model": full})
        except Exception as e:
            print(f"[WARN] OR: {e}", file=sys.stderr)
    return models

def test_one(entry):
    model = entry["model"]
    print(f"  {model:65s}", end="", flush=True)
    tc = test_tc(model); st = test_txt(model)
    status, detail = "FAIL", "?"
    if tc["tc"]: status, detail = "TOOLS_OK", "tools:"+",".join(tc.get("tools",[]))[:40]
    elif tc.get("txt_only"): status, detail = "TXT_ONLY", "text-only"
    elif st["ok"]: status, detail = "TEXT_OK", st["ct"][:40]
    else:
        sc = tc.get("sc",0); err = tc.get("err","") or st.get("err","") or ""
        if sc==400: status="BAD_REQ"
        elif sc in(401,403): status="AUTH_ER"
        elif sc==429: status="RATE_LI"
        elif sc==413: status="CTX_OVFL"
        elif sc==503: status="UNAVAIL"
        detail = err[:80] or f"HTTP {sc}"
    ms = tc.get("ms",0) or st.get("ms",0)
    print(f" {status:8s} {ms:7.0f}ms  {detail[:60]}")
    try:
        from agent.model_quality_db import record_success, record_failure, record_text_only
        prov = model.split("/")[0] if "/" in model else ""
        bare = model.split("/",1)[1] if "/" in model else model
        if tc["tc"]: record_success(prov, bare, latency_ms=ms)
        elif tc.get("txt_only"): record_text_only(prov, bare, latency_ms=ms)
        else: record_failure(prov, bare, latency_ms=ms, error_code=tc.get("sc",0), error_message=tc.get("err",""))
    except: pass
    return {"model": model, "status": status, "latency_ms": ms}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="/tmp/test_results.json")
    parser.add_argument("-p", "--provider", default=None)
    args = parser.parse_args()
    models = get_models()
    if args.provider: models = [m for m in models if args.provider.lower() in m["model"].lower()]
    srcs = defaultdict(list)
    for m in models: srcs[m["src"]].append(m)
    print(f"{'='*100}\nGateway Test — {len(models)} models\n{'='*100}")
    results = []
    for src, ms in sorted(srcs.items()):
        print(f"\n── {src} ({len(ms)})")
        for e in ms:
            try: results.append(test_one(e))
            except Exception as x:
                print(f"  {e['model']:65s} ERROR {x}")
                results.append({"model": e["model"], "status": "ERROR"})
    counts = defaultdict(int)
    for r in results: counts[r.get("status","FAIL")] += 1
    print(f"\n{'='*100}\nSUMMARY: {len(results)} models tested")
    for s in ["TOOLS_OK","TXT_ONLY","TEXT_OK","BAD_REQ","AUTH_ER","RATE_LI","CTX_OVFL","UNAVAIL","FAIL","ERROR"]:
        if counts[s]: print(f"  {s:10s}: {counts[s]}")
    print(f"{'='*100}")
    if args.output:
        with open(args.output,"w") as f: json.dump(results,f,indent=2,default=str)
        print(f"Saved to {args.output}")

if __name__=="__main__": main()
