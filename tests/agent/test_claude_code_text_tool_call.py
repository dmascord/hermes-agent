from __future__ import annotations

from agent.claude_code_client import _parse_text_tool_call


def test_parse_text_tool_call_accepts_xai_function_call() -> None:
    parsed = _parse_text_tool_call(
        '<xai:function_call name="mcp__hermes-tools__bash">'
        '<xai:parameter name="command">echo ok</xai:parameter>'
        "</xai:function_call>"
    )

    assert parsed is not None
    assert parsed["name"] == "mcp__hermes-tools__bash"
    assert parsed["arguments"] == {"command": "echo ok"}
