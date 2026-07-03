from __future__ import annotations

import json
from types import SimpleNamespace

from agent.transports.chat_completions import ChatCompletionsTransport


def _response(content: str, *, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=None),
            )
        ],
        usage=None,
    )


def test_normalize_response_converts_dsml_content_to_tool_call():
    content = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="i" string="true">Check container health</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="command" string="true">'
        'ssh wildduck.tusker.net.au "sudo docker ps --filter '
        "'name=immich_server' --format '{{.Status}}'; echo '--- logs tail ---'; "
        'sudo docker logs immich_server --tail 5"'
        "</｜DSML｜parameter>\n"
        '<｜DSML｜parameter name="timeout" string="false">15</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )

    normalized = ChatCompletionsTransport().normalize_response(_response(content))

    assert normalized.finish_reason == "tool_calls"
    assert normalized.content is None
    assert normalized.tool_calls is not None
    assert len(normalized.tool_calls) == 1
    tool_call = normalized.tool_calls[0]
    assert tool_call.name == "bash"
    args = json.loads(tool_call.arguments)
    assert args["i"] == "Check container health"
    assert "immich_server" in args["command"]
    assert args["timeout"] == 15


def test_normalize_response_leaves_plain_text_alone():
    normalized = ChatCompletionsTransport().normalize_response(
        _response("Use JSON examples carefully.")
    )

    assert normalized.finish_reason == "stop"
    assert normalized.content == "Use JSON examples carefully."
    assert normalized.tool_calls is None
