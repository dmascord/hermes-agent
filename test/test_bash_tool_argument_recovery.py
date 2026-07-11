import json

from agent.mimocode_code_client import _coerce_args_to_schema
from gateway.platforms.api_server import _external_tool_call_arguments


MALFORMED_BASH_ARGS = (
    '{"command":"TOKEN=$(grep \'^HA_TOKEN=\' /Volumes/dev/dev/homeassistant/.env | cut -d= -f2); '
    'curl -s -X POST -H "Authorization: Bearer $TOKEN" -H \'Content-Type: application/json\' '
    'https://ha.tusker.net.au/api/config/config_entries/entry/01KT3YENC2JHJNPR5C56BSX4XV/reload; echo",'
    '"i":"Reload bt-proxy config entry"}'
)


def test_external_bash_arguments_recover_command_with_unescaped_inner_quotes():
    args = _external_tool_call_arguments("mcp__hermes-tools__bash", MALFORMED_BASH_ARGS)

    assert args["command"].startswith("TOKEN=$(grep '^HA_TOKEN='")
    assert '-H "Authorization: Bearer $TOKEN"' in args["command"]
    assert args["command"].endswith("; echo")


def test_mimocode_schema_coercion_recovers_raw_malformed_bash_command():
    tools = [
        {
            "function": {
                "name": "mcp__hermes-tools__bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        }
    ]

    args = _coerce_args_to_schema(
        {"raw": MALFORMED_BASH_ARGS},
        "mcp__hermes-tools__bash",
        tools,
    )

    assert args["command"].startswith("TOKEN=$(grep '^HA_TOKEN='")
    assert '-H "Authorization: Bearer $TOKEN"' in args["command"]
    json.dumps(args)
