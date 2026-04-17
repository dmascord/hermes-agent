# Hermes Code Final Audit

## Source variants audited

### Local / synced repo
- `/Users/tusker/dev/opencode/hermes-agent/gateway/platforms/api_server.py`
- `/Users/tusker/dev/opencode/hermes-agent/run_agent.py`
- `/Users/tusker/dev/opencode/hermes-agent/gateway/run.py`
- `/Users/tusker/dev/opencode/hermes-agent/Dockerfile.custom`

### Server recovery sources
- `/srv/opencode/hermes-agent-new/`
- `/tmp/api_server.py`
- `/tmp/api_server_custom.py`
- `/tmp/fix_broker_mode.py`
- `/tmp/fix_client_mode2.py`
- `/tmp/implement_hermes_code.py`
- `/tmp/add_swarm_routing.py`
- `/tmp/fix_swarm_*`
- Docker images: `hermes-swarm:latest`, `hermes-custom:latest`, `nousresearch/hermes-agent:latest`

## What was missing before

The synced `hermes-agent-new` tree had the swarm/model-mode work, but it was missing important OpenCode tool-routing pieces that existed in the `/tmp` and OpenCode variants:

- request `tools` extraction in chat/responses handlers
- assistant/tool message normalization for multi-turn OpenAI tool loops
- `_extract_openai_tool_calls()` normalization
- `_compact_message_history()` with better preservation of tool-call payloads
- `tool_choice` passthrough
- `tool_gen_callback` passthrough
- broker/in-band external tool mode handling
- SSE tool-call chunk emission
- `/v1/sessions/{session_id}/tool_responses` ingestion route
- non-stream `finish_reason: tool_calls` synthesis when tool gen happened but provider parsing was lossy
- deployment/runtime path consistency for `.hermes` vs `hermes-data`
- gateway runtime provider resolution respecting configured `model.provider`

## Fixes merged

### `gateway/platforms/api_server.py`
Added / fixed:
- `hermes-code`, `hermes-agentic-full`, `hermes-agentic-remote`, `hermes-swarm` model routing
- tool extraction from request body
- client tool marking (`_from_client`)
- assistant/tool-role message parsing for follow-up turns
- `_extract_openai_tool_calls()`
- `_is_opencode_user_agent()`
- bounded history compaction
- `tool_choice`, `external_tool_mode`, `tool_gen_callback` plumbing
- SSE tool-call chunk emission (`__tool_call_start__`)
- non-streaming tool-call response synthesis with `finish_reason: "tool_calls"`
- `/v1/sessions/{session_id}/tool_responses`

### `gateway/run.py`
Fixed `_resolve_runtime_agent_kwargs()` so it respects configured `model.provider` from gateway config instead of silently drifting to OpenRouter auto-detection.

### `run_agent.py`
Confirmed the important client-mode logic is present:
- `enabled_toolsets == [] and tools is not None` => all API tools treated as client tools

### `Dockerfile.custom`
Fixed container runtime support:
- default `HERMES_HOME` is `/home/tusker/.hermes`
- `HOME=/home/tusker`
- legacy `/home/tusker/hermes-data` is symlinked to `/home/tusker/.hermes`

## Runtime / deployment fixes applied on server

### Container deployment
Rebuilt and deployed `hermes-swarm:latest`.

### Server config
Updated `/home/tusker/.hermes/config.yaml` so the gateway default runtime model is:
- provider: `openrouter`
- model: `anthropic/claude-sonnet-4-6`

This was necessary because the previous configured model/provider combination was not returning valid tool-capable completions in this deployment.

## Validations performed

### Basic API health
- `GET /health` => `200 OK`
- `GET /v1/models` with auth => returns all 5 models:
  - `hermes-agent`
  - `hermes-code`
  - `hermes-agentic-full`
  - `hermes-agentic-remote`
  - `hermes-swarm`

### Basic model execution
Validated successful basic chat completion on `hermes-agent`.

### Forced tool-call validation across all models
Using an OpenCode-style client-provided function `get_magic_number`, validated non-stream responses return:
- `finish_reason: "tool_calls"`
- `tool_call_count: 1`

Validated models:
- `hermes-agent`
- `hermes-code`
- `hermes-agentic-full`
- `hermes-agentic-remote`
- `hermes-swarm`

### Hermes-code OpenAI-style two-turn tool loop
Validated:
1. first turn returns assistant `tool_calls`
2. second turn, when assistant tool call + tool result are sent back in `messages`, returns final answer using the tool output

Important note:
- this worked correctly when the client sent the full message history back
- using `X-Hermes-Session-Id` for that second turn caused the server-side stored history to override the explicit tool result history, so the safer supportable path is the standard OpenAI-style message replay used by OpenCode

### Hermes-code streaming validation
Validated streaming SSE now includes:
- `tool_calls` chunks
- final `finish_reason: "tool_calls"`
- no final `finish_reason: "stop"` on the forced tool-call turn

## Final supportable state

The current synced source tree now has a solid hermes-code implementation with working OpenCode-style client tool routing.

### Supported behavior
- `hermes-code` routes tools back to the client using OpenAI-compatible tool-call turns
- `hermes-agent` / `full` / `remote` / `swarm` all expose working tool-call responses for client-supplied tools
- streaming and non-streaming tool-call responses both work
- container runtime paths are consistent for future deployments

### Recommended client pattern
For highest reliability, OpenCode should:
1. send tools in the request
2. consume returned assistant `tool_calls`
3. execute locally on the Mac
4. send the full updated `messages` array back on the next turn

That path is validated.
