# Hermes Gateway Optimization — 2026-06-06

## Deployment

- **Image**: `swarm-alpine-20260606131928` (commit `7be8f59c1`)
- **Pod**: `hermes-8479c55fbc-7hkts` on `wyrm`
- **Health**: OK

## Changes Applied

### Arliai Provider Plugin (NEW)

Created `plugins/model-providers/arliai/__init__.py` — registers 66 open-weight models via `api.arliai.com/v1`. **Constraint**: 2 concurrent requests max (ADVANCED tier).

### Code Passthrough Chain (1 primary + 27 fallbacks = 28 unique models)

Reordered by measured quality data (text-only rates), added 3 new providers (synthetic, ollama-cloud, arliai), moved gemini-2.5-flash to lower priority (20 req/day free tier limit).

**Provider breakdown**: opencode-go (5), ollama (4), arliai (4), minimax (3), synthetic (3), nous (2), ollama-cloud (2), github-copilot-enterprise (1), opencode-zen (1), zai (1), groq (1), google (1)

| # | Model | Provider | Notes |
|---|-------|----------|-------|
| 1 | `github-copilot-enterprise/gpt-5.4-mini` | copilot | PRIMARY |
| 2 | `minimax/MiniMax-M3` | minimax | Best reliability (100%, 348 calls) |
| 3 | `minimax/MiniMax-M2.7` | minimax | Low text-only (5.7%), fast |
| 4 | `minimax/MiniMax-M2.5` | minimax | 37% text-only, 43 calls |
| 5 | `opencode-go/mimo-v2.5` | opencode-go | Low text-only (7.1%) |
| 6 | `opencode-zen/mimo-v2.5-free` | opencode-zen | Free, low text-only (7.0%) |
| 7 | `opencode-go/deepseek-v4-pro` | opencode-go | 100% score |
| 8 | `opencode-go/deepseek-v4-flash` | opencode-go | Low text-only (9.6%) |
| 9 | `ollama/deepseek-v4-flash` | ollama | Low text-only (8.3%) |
| 10 | `zai/glm-4.7` | zai | 20% text-only |
| 11 | `opencode-go/qwen3.6-plus` | opencode-go | 100% score |
| 12 | `groq/llama-3.3-70b-versatile` | groq | Fast, reliable |
| 13 | `opencode-go/kimi-k2.6` | opencode-go | 21% text-only |
| 14 | `nous/stepfun/step-3.7-flash:free` | nous | Free tier |
| 15 | `ollama/qwen3-coder-next` | ollama | 31% text-only |
| 16 | `ollama/glm-5.1` | ollama | 23% text-only |
| 17 | `google/gemini-2.5-flash` | google | ⚠️ 20 req/day free tier |
| 18 | `nous/nvidia/nemotron-3-ultra:free` | nous | Free tier |
| 19 | `ollama/kimi-k2-thinking` | ollama | Last resort (75% text-only) |
| 20 | `arliai/GLM-4.7` | arliai | **NEW** — verified tools OK |
| 21 | `synthetic/syn:large:text` | synthetic | **NEW** — HF distillations |
| 22 | `synthetic/hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | synthetic | **NEW** — 120B param |
| 23 | `synthetic/hf:moonshotai/Kimi-K2.6` | synthetic | **NEW** — Kimi 2.6 |
| 24 | `ollama-cloud/deepseek-v4-flash` | ollama-cloud | **NEW** — 1M ctx |
| 25 | `ollama-cloud/glm-5.1` | ollama-cloud | **NEW** |
| 26 | `arliai/Mistral-Medium-3.5-128B` | arliai | **NEW** — 128B model |
| 27 | `arliai/Qwen3.5-27B-Anko` | arliai | **NEW** — Qwen-based |
| 28 | `arliai/Gemma-4-31B-Claude-4.6-Opus-Reasoning-Distilled` | arliai | **NEW** — Reasoning distilled |

### Swarm Chain (22 unique models)

Removed 3 duplicates, replaced 3 stale free models, added 3 arliai models:
- `openrouter/google/gemma-3-5b-eu:free` → `openrouter/google/gemma-4-26b-a4b-it:free`
- `openrouter/meta-llama/llama-3.1-8b-instruct:free` → `openrouter/google/gemma-4-31b-it:free`
- `openrouter/mistralai/mistral-7b-instruct:free` → `openrouter/poolside/laguna-xs.2:free`

### Large Context Chain (5 models)

Reordered by context window:
1. `ollama-cloud/deepseek-v4-flash` (1M ctx) — **NEW**
2. `google/gemini-2.5-flash` (1M ctx)
3. `openai/gpt-5.3-codex` (400K ctx)
4. `openai/gpt-5.2-codex` (400K ctx)
5. `openrouter/qwen/qwen3-coder:free` (262K ctx)

## Known Constraints

### Google Gemini Free Tier (20 req/day)

Gemini's free tier allows only 20 requests/day for `gemini-2.5-flash`. With 24/7 gateway usage this is easily exceeded. Moved from fallback #3 to #17 in code chain. 23h cooldown correctly applied on 429 quota exhaustion.

### Arliai ADVANCED Tier (2 concurrent requests)

Arliai's ADVANCED tier limits to 2 concurrent requests. Failures are 403 "exceeded parallel requests" — models are tool-capable. Placed at end of chain as bursty reserve.

### Text-Only Models

Models returning text instead of tool_calls when tools are provided:

| Model | Text-Only % | Calls | Effect |
|-------|------------|-------|--------|
| `ollama/kimi-k2-thinking` | 74.6% | 67 | Last resort, gets reduced tools |
| `minimax/MiniMax-M2.5` | 37.2% | 43 | Gets reduced tools (>30%) |
| `opencode-go/glm-5` | 36.7% | 30 | Gets reduced tools (>30%) |
| `ollama/qwen3-coder-next` | 30.8% | 26 | Borderline (>30%) |
| `ollama/glm-5.1` | 22.5% | 40 | Safe |
| `zai/glm-4.7` | 20.0% | 75 | Safe |

Models above 30% get quality-aware essential tool reduction (7 tools) via `HERMES_FALLBACK_ESSENTIAL_TOOLS`.

## Verification

- 19/19 key models: 100% tool calling pass rate
- All new providers (arliai, synthetic, ollama-cloud) verified working with tools
- Provider plugin: arliai ✓, ollama-cloud ✓, synthetic (no plugin needed, uses env convention)
- Gateway health: OK
