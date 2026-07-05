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


def test_normalize_response_converts_dsml_multi_invoke_content_to_tool_calls():
    content = (
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="command" string="true">'
        'curl -sS "http://10.94.14.132:8080/api/v1/etl/backfill/'
        '551d2ecc-d066-4c52-82cf-94692120147b" | python3 -m json.tool 2>&1 | head\n'
        "-40"
        "</｜DSML｜parameter>\n"
        '<｜DSML｜parameter name="cwd" string="true">/tmp</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="description" string="true">Check status of the failed job</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="command" string="true">'
        'curl -sS "http://10.94.14.132:8080/api/v1/etl/backfill/'
        'b55d1a12-efcd-40e0-aa87-05d8d91faf11" | python3 -m json.tool 2>&1 | head\n'
        "-40"
        "</｜DSML｜parameter>\n"
        '<｜DSML｜parameter name="cwd" string="true">/tmp</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="description" string="true">Check status of successful July 4 job</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )

    normalized = ChatCompletionsTransport().normalize_response(_response(content))

    assert normalized.finish_reason == "tool_calls"
    assert normalized.content is None
    assert normalized.tool_calls is not None
    assert len(normalized.tool_calls) == 2
    first_args = json.loads(normalized.tool_calls[0].arguments)
    second_args = json.loads(normalized.tool_calls[1].arguments)
    assert normalized.tool_calls[0].name == "bash"
    assert normalized.tool_calls[1].name == "bash"
    assert "551d2ecc-d066-4c52-82cf-94692120147b" in first_args["command"]
    assert first_args["cwd"] == "/tmp"
    assert first_args["description"] == "Check status of the failed job"
    assert "b55d1a12-efcd-40e0-aa87-05d8d91faf11" in second_args["command"]
    assert second_args["cwd"] == "/tmp"
    assert second_args["description"] == "Check status of successful July 4 job"


def test_normalize_response_converts_tool_use_content_to_tool_call():
    content = (
        "※ recap: Fixed the Immich 404.\n\n"
        '<tool_use id="bash">\n'
        '<parameter name="i">Check container health</parameter>\n'
        '<parameter name="command">'
        'ssh wildduck.tusker.net.au "sudo docker ps --filter '
        "'name=immich_server' --format '{{.Status}}'; echo '--- logs tail ---'; "
        'sudo docker logs immich_server --tail 5"'
        "</parameter>\n"
        '<parameter name="timeout">15</parameter>\n'
        "</tool_use>"
    )

    normalized = ChatCompletionsTransport().normalize_response(_response(content))

    assert normalized.finish_reason == "tool_calls"
    assert normalized.content == "※ recap: Fixed the Immich 404."
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
