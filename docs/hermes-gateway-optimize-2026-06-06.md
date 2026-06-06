# Hermes Gateway Optimization — 2026-06-06

## Changes Applied

### Arliai Provider Plugin (NEW)

Created `plugins/model-providers/arliai/__init__.py` — registers 66 open-weight models via `api.arliai.com/v1`.
Deployed as image `swarm-alpine-20260606131928` (commit `7be8f59c1`).

### Code Passthrough Chain (1 primary + 28 fallbacks = 29 unique models)

Reordered from quality data, added 3 new providers (synthetic, ollama-cloud, arliai).
Removed `kimi-k2-thinking` from primary positions (82.5% text-only rate) — kept only as last resort.

**Provider breakdown:** opencode-go (6), ollama (4), arliai (4), minimax (3), synthetic (3), nous (2), ollama-cloud (2), google (1), opencode-zen (1), zai (1), groq (1), copilot (1)

| # | Model | Provider | Notes |
|---|-------|----------|-------|
| 1 | `github-copilot-enterprise/gpt-5.4-mini` | copilot | PRIMARY |
| 2 | `minimax/MiniMax-M3` | minimax | Best reliability (100%, 348 calls) |
| 3 | `minimax/MiniMax-M2.7` | minimax | Low text-only (5.7%), fast |
| 4 | `google/gemini-2.5-flash` | google | High reliability, 1M ctx |
| 5 | `opencode-go/mimo-v2.5` | opencode-go | Low text-only (7.1%) |
| 6 | `opencode-zen/mimo-v2.5-free` | opencode-zen | Free, low text-only (7.0%) |
| 7 | `opencode-go/deepseek-v4-pro` | opencode-go | 100% score |
| 8 | `opencode-go/deepseek-v4-flash` | opencode-go | Low text-only (9.6%) |
| 9 | `ollama/deepseek-v4-flash` | ollama | Low text-only (8.3%) |
| 10 | `zai/glm-4.7` | zai | 20% text-only |
| 11 | `opencode-go/qwen3.6-plus` | opencode-go | 100% score |
| 12 | `groq/llama-3.3-70b-versatile` | groq | Fast, reliable |
| 13 | `opencode-go/kimi-k2.6` | opencode-go | 21.4% text-only |
| 14 | `nous/stepfun/step-3.7-flash:free` | nous | Free tier |
| 15 | `ollama/qwen3-coder-next` | ollama | 30.8% text-only |
| 16 | `ollama/glm-5.1` | ollama | 22.5% text-only |
| 17 | `minimax/MiniMax-M2.5` | minimax | 34.4% text-only |
| 18 | `opencode-go/glm-5` | opencode-go | 35.0% text-only |
| 19 | `nous/nvidia/nemotron-3-ultra:free` | nous | Free tier |
| 20 | `ollama/kimi-k2-thinking` | ollama | Last resort (82.5% text-only) |
| 21 | `arliai/GLM-4.7` | arliai | **NEW** — verified tool calling |
| 22 | `synthetic/syn:large:text` | synthetic | **NEW** — HF distillations |
| 23 | `synthetic/hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | synthetic | **NEW** — 120B param |
| 24 | `synthetic/hf:moonshotai/Kimi-K2.6` | synthetic | **NEW** — Kimi 2.6 |
| 25 | `ollama-cloud/deepseek-v4-flash` | ollama-cloud | **NEW** — 1M ctx |
| 26 | `ollama-cloud/glm-5.1` | ollama-cloud | **NEW** |
| 27 | `arliai/Mistral-Medium-3.5-128B` | arliai | **NEW** — 128B model |
| 28 | `arliai/Qwen3.5-27B-Anko` | arliai | **NEW** — Qwen-based |
| 29 | `arliai/Gemma-4-31B-Claude-4.6-Opus-Reasoning-Distilled` | arliai | **NEW** — Reasoning distilled |

### Swarm Chain (1 primary + 21 fallbacks = 22 unique models)

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

## Quality Data

### Models with High Text-Only Rates

| Model | Text-Only % | Calls |
|-------|------------|-------|
| `ollama/kimi-k2-thinking` | 82.5% | 40 |
| `opencode-go/glm-5` | 35.0% | 20 |
| `minimax/MiniMax-M2.5` | 34.4% | 32 |
| `ollama/qwen3-coder-next` | 30.8% | 26 |
| `ollama/glm-5.1` | 22.5% | 40 |
| `zai/glm-4.7` | 20.0% | 75 |
| `opencode-go/kimi-k2.6` | 21.4% | 14 |

Models above 30% get quality-aware essential tool reduction (7 tools) via `HERMES_FALLBACK_ESSENTIAL_TOOLS`.

### Verification Results

- 19/19 key models tested: 100% pass rate on tool calling
- All new providers (arliai, synthetic, ollama-cloud) verified working with tools
- Gateway health: OK
- Active cooldowns: 7 (24h on claude-sonnet, 43m on nemotron, 7m on gemma-4)
