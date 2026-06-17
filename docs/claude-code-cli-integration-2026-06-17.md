# Claude Code CLI Integration

Date: 2026-06-17

This note documents the working Hermes integration for routing OpenAI-compatible
chat requests through Claude Code CLI, including OAuth refresh and MCP-backed
tool calls.

## Goals

- Treat Claude Code CLI as an OpenAI-compatible provider:
  `claude-code-cli/sonnet`, `claude-code-cli/opus`, and
  `claude-code-cli/haiku`.
- Keep Claude OAuth credentials refreshable inside the Kubernetes pod.
- Accept normal OpenAI `tools` payloads and expose them to Claude Code through
  MCP.
- Return OpenAI-style `tool_calls` using the original client tool names.

## Main Code Paths

| File | Responsibility |
|---|---|
| `agent/claude_code_client.py` | Claude CLI subprocess adapter, OAuth refresh, MCP config generation, MCP queue pump |
| `agent/claude_mcp_bridge.py` | Stdio MCP server spawned by Claude CLI; proxies tool calls through per-request files |
| `agent/anthropic_adapter.py` | Shared Anthropic OAuth refresh helper |
| `gateway/platforms/api_server.py` | Dispatches `claude-code-cli/*` requests into `ClaudeCodeClient.run_with_tool_bridge()` |
| `Dockerfile.swarm` | Installs Claude CLI plus `mcp==1.26.0` runtime dependencies |

## Credential Model

Claude Code stores OAuth credentials in a JSON object shaped like:

```json
{
  "claudeAiOauth": {
    "accessToken": "...",
    "refreshToken": "...",
    "expiresAt": 1781700000000,
    "scopes": [
      "user:file_upload",
      "user:inference",
      "user:mcp_servers",
      "user:profile",
      "user:sessions:claude_code"
    ]
  }
}
```

Runtime locations in the pod:

- Current CLI credentials: `/root/.claude/.credentials.json`
- PVC-backed backup: `/home/tusker/.hermes/.claude_backup/.credentials.json`
- Hermes auth store provider state: `claude-code-cli`

The adapter calls `_maybe_refresh_claude_oauth()` before each Claude CLI
subprocess. Refresh uses the current Anthropic OAuth endpoint first:

```text
https://api.anthropic.com/v1/oauth/token
```

Requests include the Claude OAuth beta header:

```text
anthropic-beta: oauth-2025-04-20
```

If the current credentials are missing a refresh token, Hermes tries to recover a
refreshable copy from the PVC backup. This matters because some Claude CLI
operations can leave an access-token-only credentials file behind; that file can
work temporarily but cannot refresh.

Credential files are written atomically with `0600` permissions.

## MCP Tool Bridge

Claude Code CLI does not consume OpenAI `tools` directly. Hermes translates
OpenAI tool definitions into a temporary MCP server.

For each request with tools:

1. Hermes writes a tools manifest to a temp file.
2. Hermes writes a Claude MCP config:

   ```json
   {
     "mcpServers": {
       "hermes-tools": {
         "type": "stdio",
         "command": "python3",
         "args": ["agent/claude_mcp_bridge.py"],
         "env": {
           "HERMES_TOOLS_FILE": "...",
           "HERMES_QUEUE_IN": "/tmp/hermes_queue_<session>.in",
           "HERMES_QUEUE_OUT_DIR": "/tmp/hermes_result_<session>"
         }
       }
     }
   }
   ```

3. Hermes starts Claude CLI with:

   ```text
   claude -p \
     --input-format stream-json \
     --output-format stream-json \
     --verbose \
     --model <sonnet|opus|haiku> \
     --mcp-config <config> \
     --strict-mcp-config \
     --allowedTools mcp__hermes-tools__<tool-name>
   ```

4. Claude starts `claude_mcp_bridge.py` as a stdio MCP server.
5. The MCP server registers each OpenAI function as a FastMCP tool.
6. When Claude calls a tool, the MCP server appends a JSON line to
   `HERMES_QUEUE_IN` and waits for a result file in `HERMES_QUEUE_OUT_DIR`.
7. `ClaudeCodeClient.run_with_tool_bridge()` watches that queue file, converts
   queued MCP calls into OpenAI-style `tool_call` events, and writes a placeholder
   result to the exact MCP call id so Claude can finish the turn.

The queue pump is important. Claude also emits its own stream-json `tool_use`
events with Claude-internal ids, but the MCP server waits on its separate queue
call id. Writing results to the Claude stream-json id does not unblock MCP. The
working integration watches the MCP queue itself and uses those call ids.

## FastMCP Compatibility

`mcp==1.26.0` expects `Tool.from_function(...)` metadata and validates arguments
using the function signature. A naive dynamic proxy like `handler(arguments:
dict)` either advertises the wrong schema or fails validation.

The bridge uses this pattern:

- Build a proxy tool with `Tool.from_function(...)`.
- Override `proxy_tool.parameters` with the original OpenAI JSON schema.
- Override `proxy_tool.fn_metadata` with a lightweight metadata object that:
  - accepts arbitrary MCP arguments,
  - calls the proxy handler with `fn(**arguments)`,
  - exposes `output_schema = None` for MCP list-tools conversion.

This preserves the client-provided schema while avoiding synthetic `kwargs`
validation failures.

## Response Shape

Hermes filters Claude-internal tool artifacts before returning the response:

- `ToolSearch`
- `mcp__hermes-tools__<tool-name>`

The public OpenAI-compatible response contains the original client tool name.

Example verified response:

```json
{
  "model": "claude-code-cli/sonnet",
  "finish_reason": "tool_calls",
  "content": "The `echo_tool` was called with `mcp-ok` and completed successfully.",
  "tool_calls": [
    {
      "id": "92d8fcf4ce9f4e3fa1dda992c5289108",
      "type": "function",
      "function": {
        "name": "echo_tool",
        "arguments": "{\"text\": \"mcp-ok\"}"
      }
    }
  ]
}
```

## Verification Probe

Use the Hermes API key from the Kubernetes secret, then send a normal
OpenAI-compatible request:

```bash
TOKEN=$(kubectl get secret hermes-env-vault -n hermes \
  -o jsonpath='{.data.API_SERVER_KEY}' | base64 -d)

curl -fsS --max-time 180 https://hermes.tusker.net.au/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-code-cli/sonnet",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "Use the echo_tool with text exactly '\''mcp-ok'\''. Do not answer directly."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "echo_tool",
          "description": "Return the provided text.",
          "parameters": {
            "type": "object",
            "properties": {
              "text": { "type": "string" }
            },
            "required": ["text"],
            "additionalProperties": false
          }
        }
      }
    ],
    "tool_choice": {
      "type": "function",
      "function": { "name": "echo_tool" }
    }
  }'
```

Expected outcome:

- HTTP 200.
- `model` is `claude-code-cli/sonnet`.
- `finish_reason` is `tool_calls`.
- Exactly one public tool call.
- Tool name is `echo_tool`.
- Arguments are `{"text":"mcp-ok"}`.

Also verify a plain prompt:

```bash
curl -fsS --max-time 180 https://hermes.tusker.net.au/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-code-cli/sonnet",
    "stream": false,
    "messages": [
      { "role": "user", "content": "Reply with exactly: claude-cli-ok" }
    ]
  }'
```

Expected content:

```text
claude-cli-ok
```

## Operational Notes

- The pod must have both `accessToken` and `refreshToken`; an access-only file is
  temporary and cannot auto-refresh.
- The OAuth scopes must include `user:mcp_servers` and
  `user:sessions:claude_code`.
- `mcp==1.26.0` and `starlette==1.0.1` are installed in the runtime image.
- The MCP temp files are per request/session and are cleaned up by
  `ClaudeCodeClient.close()`.
- The queue pump currently writes an empty placeholder result to unblock Claude.
  The OpenAI-compatible API is therefore best treated as a tool-call producer:
  clients execute the returned tool call and continue the conversation with a
  normal `tool` response.

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| `401 Invalid authentication credentials` | Expired access token and no valid refresh path | Inspect `/root/.claude/.credentials.json` and PVC backup for `refreshToken` |
| Refresh returns `invalid_grant` | Refresh token is expired/revoked | Re-sync a fresh Claude Code credential from Keychain/device login |
| Claude says MCP server is still connecting | MCP server failed during startup/list-tools | Check FastMCP registration compatibility and runtime `mcp` version |
| Tool call hangs until timeout | Result was written to Claude stream-json id, not MCP queue id | Inspect `/tmp/hermes_queue_*.in` and `/tmp/hermes_result_*` |
| Public response includes `ToolSearch` | Internal Claude artifacts are not filtered | Ensure `run_with_tool_bridge()` filters `ToolSearch` and `mcp__hermes-tools__*` |

## Known Boundary

This integration intentionally produces OpenAI-style tool calls. It does not
execute arbitrary client tools inside Hermes. The client remains responsible for
executing the returned tool call and sending the result back in the next request,
which matches the normal OpenAI chat-completions tool loop.
