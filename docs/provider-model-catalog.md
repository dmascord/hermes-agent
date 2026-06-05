# Hermes Gateway — Provider & Model Catalog

Generated 2026-06-05. Source of truth: live deployment + code audit.

## Provider Catalog

### github-copilot-enterprise
- **Base URL**: `copilot-api.sita.ghe.com` (SITA GHE Copilot)
- **Auth**: OAuth token exchange via credential pool (gho_* → copilot token)
- **API Modes**: `anthropic_messages` (Claude models), `chat_completions` (GPT models), `codex_responses` (GPT-5.x Responses API)
- **Tool Calling**: ✅ All models
- **Reasoning Content**: ❌ Not used
- **Headers**: Custom via `copilot_request_headers()` (includes `copilot-*` headers, `Editor-Version`, `OpenAI-Organization`)
- **Vision**: GPT-5.x ✅, Claude ✅, codex models ❌
- **Special**: AIU budget tracking per base_url. `max_tokens` always 16384. Credential pool with multi-account rotation.
- **Known 400**: GPT-5.x models reject `tool` in tools array → 3600s cooldown
- **Models**:
  - `gpt-5.4-mini` — 400K context, tool calling, vision. PRIMARY model.
  - `claude-sonnet-4.6` — 1M context, tool calling, vision. Currently on extended cooldown.
  - `claude-opus-4.6` — 1M context, tool calling, vision.

### minimax
- **Base URL**: `api.minimax.io/v1`
- **Auth**: `MINIMAX_API_KEY` env var
- **API Mode**: `openai_chat` (OpenAI-compatible)
- **Tool Calling**: M3 ✅, M2.7 ✅, M2.5 ✅ (but M2.7/M2.5 sometimes return text-only)
- **Reasoning Content**: ❌ Strip before sending
- **Context**: M3=1M, M2.7=204K, M2.5=204K
- **Special**: M3 is the most reliable tool-caller. M2.7/M2.5 intermittently return text-only when tools provided.
- **Models**:
  - `MiniMax-M3` — 1M context, tool calling. Reliable.
  - `MiniMax-M2.7` — 204K context, tool calling. Sometimes text-only.
  - `MiniMax-M2.5` — 204K context, tool calling. Sometimes text-only.

### zai (Zhipu AI / GLM)
- **Base URL**: `https://open.bigmodel.cn/api/paas/v4` (or configured)
- **Auth**: `ZAI_API_KEY` env var
- **API Mode**: `openai_chat`
- **Tool Calling**: GLM-4.7 ✅ (reliable)
- **Reasoning Content**: ❌ Strip before sending
- **Context**: GLM-4.7 = 202,752
- **Special**: Reliable tool caller, but context limit means ~178K tokens is borderline.
- **Models**:
  - `glm-4.7` — 202K context, tool calling. Reliable.

### opencode-go (OpenCode aggregator)
- **Base URL**: Configured in env (aggregator proxy)
- **Auth**: Provider-specific key
- **API Mode**: `openai_chat`
- **Tool Calling**: Varies by backend model
- **Reasoning Content**: ✅ Echo preserved (aggregator may forward to DeepSeek)
- **Context**: Depends on backend (mimo-v2.5=1M, deepseek-v4-pro=1M, glm-5=202K, kimi-k2.6=262K, qwen3.6-plus=1M)
- **Special**: Aggregator that routes to multiple backends. Always preserve reasoning_content since some backends (DeepSeek) need it. Some models return text-only.
- **Models**:
  - `mimo-v2.5` — 1M context, tool calling, reasoning echo. Reliable when not on cooldown.
  - `deepseek-v4-pro` — 1M context, tool calling, reasoning echo.
  - `deepseek-v4-flash` — 1M context, tool calling, reasoning echo. Sometimes text-only.
  - `glm-5` — 202K context, tool calling.
  - `kimi-k2.6` — 262K context, tool calling, reasoning echo.
  - `qwen3.6-plus` — 1M context, tool calling.

### opencode-zen (OpenCode Zen aggregator)
- **Base URL**: Configured in env (aggregator proxy)
- **Auth**: Provider-specific key
- **API Mode**: `openai_chat`
- **Tool Calling**: Varies by backend
- **Reasoning Content**: ✅ Echo preserved (same as opencode-go)
- **Context**: mimo-v2.5-free=262K, deepseek-v4-flash-free=varies, big-pickle=varies
- **Special**: Free tier models. Often text-only. Less reliable than opencode-go.
- **Models**:
  - `mimo-v2.5-free` — 262K context, tool calling. Often text-only despite tools.
  - `deepseek-v4-flash-free` — tool calling. Often text-only.
  - `big-pickle` — tool calling. Often text-only.

### ollama (local + cloud)
- **Base URL**: Local Orin (10.0.0.212:11434) or cloud
- **Auth**: None (local) or OLLAMA_API_KEY
- **API Mode**: `openai_chat`
- **Tool Calling**: Varies. GLM-5.1 ✅, qwen3-coder-next ⚠️ (text-only), deepseek-v4-flash ✅, kimi-k2-thinking ⚠️
- **Reasoning Content**: Strip for local models, echo for DeepSeek/Kimi variants
- **Context**: Varies by model
- **Special**: Local models (Orin) are fast but have limited context. Cloud models may have longer context but higher latency.
- **Models**:
  - `glm-5.1` — 131K context, tool calling. Reliable but context-limited.
  - `qwen3-coder-next` — tool calling but often text-only.
  - `deepseek-v4-flash` — 1M context, tool calling, reasoning echo.
  - `kimi-k2-thinking` — 262K context, tool calling, reasoning echo. Often text-only.

### google
- **Base URL**: `generativelanguage.googleapis.com/v1beta/openai`
- **Auth**: `GOOGLE_API_KEY` env var
- **API Mode**: `openai_chat` (via OpenAI-compatible endpoint) or native Gemini for audio
- **Tool Calling**: ✅ gemini-2.5-flash
- **Reasoning Content**: ❌ Strip before sending
- **Context**: gemini-2.5-flash = 1M
- **Special**: Native audio support via GeminiNativeClient. Fast and reliable.
- **Models**:
  - `gemini-2.5-flash` — 1M context, tool calling. Reliable.

### groq
- **Base URL**: `api.groq.com/openai/v1`
- **Auth**: `GROQ_API_KEY` env var
- **API Mode**: `openai_chat`
- **Tool Calling**: llama-3.3-70b-versatile ✅
- **Reasoning Content**: ❌ Strip
- **Context**: 131K
- **Special**: Very fast inference. Rate-limited.
- **Models**:
  - `llama-3.3-70b-versatile` — 131K context, tool calling.

### nous
- **Base URL**: Portal API
- **Auth**: `NOUS_API_KEY` env var
- **API Mode**: `openai_chat`
- **Tool Calling**: Free tier models via OpenRouter backend
- **Reasoning Content**: Strip
- **Context**: Varies
- **Special**: Rate-limited free tier. Cross-session rate limit guard.
- **Models**:
  - `stepfun/step-3.7-flash:free` — free tier, tool calling.
  - `nvidia/nemotron-3-ultra:free` — free tier, tool calling.

### arliai
- **Base URL**: `api.arliai.com/v1`
- **Auth**: `ARLIAI_API_KEY` env var
- **API Mode**: `openai_chat`
- **Tool Calling**: ✅ (but tool_call_ids must be ≤9 characters!)
- **Reasoning Content**: Strip before sending
- **Context**: Varies (Mistral-Medium-3.5-128B = 128K, GLM-4.6-Derestricted-v5 = varies)
- **Special**: 
  - **MAX 2 CONCURRENT STREAMS** — parallel requests beyond 2 cause 429
  - tool_call_id sanitization via ToolCallIdMapper (bidirectional: sanitize→unsanitize)
  - Often returns 503 when overloaded
- **Models**:
  - `Mistral-Medium-3.5-128B` — 128K context, tool calling. Circuit breaker prone.
  - `GLM-4.6-Derestricted-v5` — context varies, tool calling.
  - `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted` — context varies, tool calling.
  - `Qwen3.5-27B-BlueStar-v3-Derestricted-Lite` — context varies, tool calling.

---

## Code Path Audit

### Bug: Operator precedence on line 6066
```python
if prov and "opencode" in prov.lower() or "deepseek" in resolved_model.lower():
```
This evaluates as `(prov and "opencode" in prov.lower()) or "deepseek" in resolved_model.lower()` — it always logs the deep trace for deepseek models regardless of provider. Should be:
```python
if prov and ("opencode" in prov.lower() or "deepseek" in resolved_model.lower()):
```

### Text-only model waste
When models return text-only (no tool calls) despite tools being provided:
- 120s cooldown per attempt
- Full request + response round-trip wasted (5-20s)
- Models that consistently return text-only should be in a "no tools" pool

### Cooldown cascade
Models with 3600s cooldown after 400 errors:
- `github-copilot-enterprise/claude-sonnet-4.6` — 54000s remaining (multiple 400s accumulated)
- This blocks the most capable model in the chain

### Unknown context models bypass guard
Models with `_model_context_length() == 0` (unknown) are skipped by the context overflow guard but may still fail at runtime if context is too small.

### Reasoning echo for opencode-zen/go
The code echoes reasoning_content for ALL opencode-zen/go models. This is correct for DeepSeek backend but unnecessary overhead for other backends. Could be model-specific.

---

## Model Classification (Capability Pools)

### Tier 1: Large-context + reliable tool calling (≥200K, tools ✅, low text-only rate)
| Model | Context | Provider | Notes |
|-------|---------|----------|-------|
| github-copilot-enterprise/gpt-5.4-mini | 400K | copilot | PRIMARY, AIU budget |
| minimax/MiniMax-M3 | 1M | minimax | Most reliable non-copilot |
| opencode-go/mimo-v2.5 | 1M | opencode-go | Reliable when available |
| google/gemini-2.5-flash | 1M | google | Fast, reliable |
| opencode-go/qwen3.6-plus | 1M | opencode-go | Reliable |
| opencode-go/deepseek-v4-pro | 1M | opencode-go | Reliable |

### Tier 2: Medium-context + tool calling (128K-200K)
| Model | Context | Provider | Notes |
|-------|---------|----------|-------|
| zai/glm-4.7 | 202K | zai | Reliable, borderline context |
| minimax/MiniMax-M2.7 | 204K | minimax | Sometimes text-only |
| minimax/MiniMax-M2.5 | 204K | minimax | Sometimes text-only |
| opencode-go/glm-5 | 202K | opencode-go | Reliable |
| ollama/glm-5.1 | 131K | ollama | Reliable, local |

### Tier 3: Large context but unreliable tool calling
| Model | Context | Provider | Notes |
|-------|---------|----------|-------|
| opencode-zen/mimo-v2.5-free | 262K | opencode-zen | Often text-only |
| opencode-zen/deepseek-v4-flash-free | varies | opencode-zen | Often text-only |
| opencode-zen/big-pickle | varies | opencode-zen | Often text-only |
| ollama/qwen3-coder-next | varies | ollama | Often text-only |
| ollama/kimi-k2-thinking | 262K | ollama | Often text-only |
| opencode-go/deepseek-v4-flash | 1M | opencode-go | Sometimes text-only |

### Tier 4: Small context or free tier (fallback only)
| Model | Context | Provider | Notes |
|-------|---------|----------|-------|
| groq/llama-3.3-70b-versatile | 131K | groq | Fast, rate-limited |
| nous/stepfun/step-3.7-flash:free | varies | nous | Free, rate-limited |
| nous/nvidia/nemotron-3-ultra:free | 131K | nous | Free, rate-limited |
| opencode-go/kimi-k2.6 | 262K | opencode-go | Reliable |

### Tier 5: arliai (limited parallel, fragile)
| Model | Context | Provider | Notes |
|-------|---------|----------|-------|
| arliai/Mistral-Medium-3.5-128B | 128K | arliai | Max 2 concurrent, 503-prone |
| arliai/GLM-4.6-Derestricted-v5 | varies | arliai | Max 2 concurrent |
| arliai/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Derestricted | varies | arliai | Max 2 concurrent |
| arliai/Qwen3.5-27B-BlueStar-v3-Derestricted-Lite | varies | arliai | Max 2 concurrent |

### Known "no tools" (skip when tools are provided)
None identified as absolute — all models in the chain are tried with tools. The 120s cooldown handles text-only models.
