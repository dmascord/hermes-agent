import json

from gateway.platforms.api_server import _enrich_client_tool_calls


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _call(name: str, args: dict) -> dict:
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def test_maps_grep_tool_call_to_advertised_search_tool():
    calls = _enrich_client_tool_calls(
        [_call("grep", {"regex": "needle", "path": "."})],
        advertised_tools=[_tool("search")],
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["regex"] == "needle"
    assert args["pattern"] == "needle"


def test_keeps_directly_advertised_tool_name():
    calls = _enrich_client_tool_calls(
        [_call("grep", {"pattern": "needle"})],
        advertised_tools=[_tool("grep"), _tool("search")],
    )

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "grep"


def test_drops_unmapped_unadvertised_tool_call():
    calls = _enrich_client_tool_calls(
        [_call("grep", {"pattern": "needle"})],
        advertised_tools=[_tool("read")],
    )

    assert calls == []
