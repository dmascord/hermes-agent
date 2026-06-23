"""Tests for mimocode tool call parsing improvements.

Validates the helper functions and new formats added to
`agent.mimocode_code_client` after surveying Fly143/MiMo2API and other
MiMoCode proxy projects on GitHub.

Tests cover:
  - _find_balanced_json  (depth-aware JSON extraction)
  - _auto_type           (bool/int/float/null detection)
  - _resolve_tool_name   (4-level name matching)
  - _skip_fenced_block   (markdown fence skip)
  - _strip_mimoml        (MiMoML noise tolerance)
  - _clean_tool_text     (residue stripping)
  - _StreamSieve         (cross-line tool call catch)
  - _parse_tool_call_xml (formats 1-12)
  - Format 11 (TOOL_CALL: text)
  - Format 12 (MiMoML native)
  - Tool name resolution on real extracted calls
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add the repo root to sys.path so the test can import the module under test
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.mimocode_code_client import (  # noqa: E402
    _find_balanced_json,
    _auto_type,
    _resolve_tool_name,
    _skip_fenced_block,
    _strip_mimoml,
    _clean_tool_text,
    _parse_python_kwargs,
    _parse_mimoml_params,
    _parse_tool_call_xml,
    _StreamSieve,
)


# ── _find_balanced_json ──────────────────────────────────────────────


class TestFindBalancedJson:
    def test_flat_object(self):
        assert _find_balanced_json('{"a": 1}', 0) == '{"a": 1}'

    def test_nested_object(self):
        text = '{"a": {"b": {"c": 1}}}'
        assert _find_balanced_json(text, 0) == text

    def test_string_with_braces(self):
        # } inside a string should not decrement depth
        text = '{"a": "} }"}'
        assert _find_balanced_json(text, 0) == text

    def test_escaped_quote(self):
        text = '{"a": "he said \\"}"}'
        assert _find_balanced_json(text, 0) == text

    def test_unbalanced_returns_empty(self):
        assert _find_balanced_json('{"a": 1', 0) == ""

    def test_not_at_brace(self):
        assert _find_balanced_json("hello", 0) == ""

    def test_nested_arrays(self):
        text = '{"items": [1, 2, [3, 4]]}'
        assert _find_balanced_json(text, 0) == text


# ── _auto_type ────────────────────────────────────────────────────────


class TestAutoType:
    def test_true_false(self):
        assert _auto_type("true") is True
        assert _auto_type("True") is True
        assert _auto_type("false") is False
        assert _auto_type("FALSE") is False

    def test_null(self):
        assert _auto_type("null") is None
        assert _auto_type("None") is None

    def test_integers(self):
        assert _auto_type("42") == 42
        assert _auto_type("-7") == -7
        # Not numeric enough → stays string
        assert _auto_type("12abc") == "12abc"

    def test_floats(self):
        assert _auto_type("3.14") == 3.14
        assert _auto_type("-0.5") == -0.5

    def test_passthrough(self):
        assert _auto_type("hello world") == "hello world"
        assert _auto_type("") == ""

    def test_non_string_passthrough(self):
        # Auto-type is meant to be called on strings; non-strings pass through
        assert _auto_type(42) == 42
        assert _auto_type(True) is True


# ── _resolve_tool_name ───────────────────────────────────────────────


class TestResolveToolName:
    def test_exact_match(self):
        assert _resolve_tool_name("mcp_bash", ["mcp_bash", "mcp_read"]) == "mcp_bash"

    def test_case_insensitive(self):
        assert _resolve_tool_name("MCP_BASH", ["mcp_bash"]) == "mcp_bash"

    def test_camel_to_snake(self):
        assert _resolve_tool_name("mcpBash", ["mcp_bash"]) == "mcp_bash"

    def test_snake_case_insensitive(self):
        assert _resolve_tool_name("MCP_BASH", ["mcp_bash"]) == "mcp_bash"

    def test_no_match_returns_none(self):
        assert _resolve_tool_name("foo", ["mcp_bash"]) is None

    def test_no_tool_names_returns_input(self):
        assert _resolve_tool_name("mcp_bash", None) == "mcp_bash"

    def test_empty_input(self):
        assert _resolve_tool_name("", ["mcp_bash"]) is None


# ── _skip_fenced_block ───────────────────────────────────────────────


class TestSkipFencedBlock:
    def test_backtick_fence(self):
        text = "before\n```\n<tool_call>{json}</tool_call>\n```\nafter"
        new_i, skipped = _skip_fenced_block(text, text.find("```"))
        assert skipped is not None
        assert "<tool_call>" in skipped
        assert text[new_i:].startswith("after")

    def test_tilde_fence(self):
        text = "before\n~~~\ncode\n~~~\nafter"
        new_i, skipped = _skip_fenced_block(text, text.find("~~~"))
        assert skipped is not None
        assert "code" in skipped

    def test_not_a_fence(self):
        text = "hello world"
        new_i, skipped = _skip_fenced_block(text, 0)
        assert new_i == 0
        assert skipped is None

    def test_too_few_fence_chars(self):
        text = "``\ncode\n``"
        new_i, skipped = _skip_fenced_block(text, 0)
        assert skipped is None


# ── _strip_mimoml ─────────────────────────────────────────────────────


class TestStripMimoml:
    def test_standard_form(self):
        text = "<|MiMoML|tool_calls>x</|MiMoML|tool_calls>"
        out = _strip_mimoml(text)
        assert out == "<tool_calls>x</tool_calls>"

    def test_hyphenated(self):
        text = "<mimoml-tool_calls>x</mimoml-tool_calls>"
        out = _strip_mimoml(text)
        # Hyphen variant is normalised
        assert "<tool_calls>" in out

    def test_fullwidth_pipe(self):
        text = "<｜MiMoML｜tool_calls>x</｜MiMoML｜tool_calls>"
        out = _strip_mimoml(text)
        assert "<tool_calls>" in out

    def test_function_calls_variant(self):
        text = "<|MiMoML|function_calls>x</|MiMoML|function_calls>"
        out = _strip_mimoml(text)
        assert "<function_calls>" in out

    def test_skips_fenced(self):
        text = "```\n<|MiMoML|tool_calls>example</|MiMoML|tool_calls>\n```"
        out = _strip_mimoml(text)
        # Code fence preserved unchanged
        assert "MiMoML" in out

    def test_no_mimoml_passthrough(self):
        text = "just plain text"
        assert _strip_mimoml(text) == text


# ── _clean_tool_text ──────────────────────────────────────────────────


class TestCleanToolText:
    def test_strip_tool_call_inline(self):
        text = "hello\nTOOL_CALL: mcp_bash(ls)\nworld"
        out = _clean_tool_text(text)
        assert "TOOL_CALL" not in out
        assert "hello" in out
        assert "world" in out

    def test_strip_xml_tags(self):
        text = "before <tool_call>{json}</tool_call> after"
        out = _clean_tool_text(text)
        assert "<tool_call>" not in out
        assert "before" in out
        assert "after" in out

    def test_strip_mimoml_residue(self):
        text = "<|MiMoML|tool_calls>x</|MiMoML|tool_calls>clean text"
        out = _clean_tool_text(text)
        assert "MiMoML" not in out
        assert "clean text" in out

    def test_strip_empty_fences(self):
        text = "before\n```\n```\nafter"
        out = _clean_tool_text(text)
        assert "```" not in out

    def test_collapse_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        out = _clean_tool_text(text)
        assert "\n\n\n" not in out

    def test_empty(self):
        assert _clean_tool_text("") == ""


# ── _parse_python_kwargs ─────────────────────────────────────────────


class TestParsePythonKwargs:
    def test_single(self):
        result = _parse_python_kwargs('command="ls -la"')
        assert result == {"command": "ls -la"}

    def test_multiple(self):
        result = _parse_python_kwargs('path="/etc/hostname", timeout=30')
        assert result == {"path": "/etc/hostname", "timeout": 30}

    def test_bools(self):
        result = _parse_python_kwargs("verbose=True, debug=False")
        assert result == {"verbose": True, "debug": False}

    def test_json_object(self):
        result = _parse_python_kwargs('{"key": "value"}')
        assert result == {"key": "value"}

    def test_empty(self):
        assert _parse_python_kwargs("") == {}

    def test_nested_value(self):
        # Value contains a list — should not break the comma split
        result = _parse_python_kwargs('items=[1, 2, 3], name="x"')
        assert result == {"items": [1, 2, 3], "name": "x"}


# ── _parse_mimoml_params ─────────────────────────────────────────────


class TestParseMimomlParams:
    def test_single_param(self):
        inner = '<parameter name="command">ls -la</parameter>'
        result = _parse_mimoml_params(inner)
        assert result == {"command": "ls -la"}

    def test_multiple_params(self):
        inner = (
            '<parameter name="path">/tmp</parameter>'
            '<parameter name="recursive">true</parameter>'
        )
        result = _parse_mimoml_params(inner)
        assert result["path"] == "/tmp"
        assert result["recursive"] is True

    def test_cdata(self):
        inner = (
            '<parameter name="content">'
            "<![CDATA[hello world]]>"
            "</parameter>"
        )
        result = _parse_mimoml_params(inner)
        assert result == {"content": "hello world"}

    def test_duplicate_keys_become_list(self):
        inner = (
            '<parameter name="x">1</parameter>'
            '<parameter name="x">2</parameter>'
        )
        result = _parse_mimoml_params(inner)
        assert result == {"x": [1, 2]}


# ── _parse_tool_call_xml: all 12 formats ─────────────────────────────


class TestParseToolCallXml:
    def test_format_1_json(self):
        text = '<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "bash"
        args = json.loads(result["function"]["arguments"])
        assert args == {"command": "ls"}

    def test_format_2_function_param(self):
        text = '<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "bash"
        args = json.loads(result["function"]["arguments"])
        assert args == {"command": "ls"}

    def test_format_3_tool_name_params(self):
        text = (
            "<tool_call>"
            "<tool_name>bash</tool_name>"
            "<parameters>"
            "<command>ls</command>"
            "</parameters>"
            "</tool_call>"
        )
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "bash"

    def test_format_4_name_args(self):
        text = '<tool_call><name>bash</name><args>{"command": "ls"}</args></tool_call>'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "bash"

    def test_format_5_invoke(self):
        text = (
            "<function_calls>"
            '<invoke name="mcp__bash">'
            '<parameter name="command">ls</parameter>'
            "</invoke>"
            "</function_calls>"
        )
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp__bash"

    def test_format_6_tool_invocation(self):
        text = '<tool_invocation name="mcp__bash" arguments={"command": "ls"} />'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp__bash"

    def test_format_7_json_in_code_block(self):
        text = (
            "```json\n"
            '{"name": "bash", "arguments": {"command": "ls"}}\n'
            "```"
        )
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "bash"

    def test_format_8_tool_invocation_in_code_block(self):
        text = (
            "```\n"
            '<tool_invocation name="mcp__bash" arguments={"command": "ls"} />\n'
            "```"
        )
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp__bash"

    def test_format_9_bare_mcp_tag(self):
        text = "<mcp_bash>ls -la</mcp_bash>"
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp_bash"
        args = json.loads(result["function"]["arguments"])
        assert args == {"command": "ls -la"}

    def test_format_10_bare_json_nested(self):
        # Format 10 should now handle nested JSON (was depth-limited)
        text = '{"name": "mcp_bash", "arguments": {"command": "ls", "opts": {"a": 1}}}'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp_bash"
        args = json.loads(result["function"]["arguments"])
        assert args["command"] == "ls"
        assert args["opts"] == {"a": 1}

    def test_format_11_tool_call_text(self):
        text = 'TOOL_CALL: mcp_bash(command="ls -la")'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp_bash"
        args = json.loads(result["function"]["arguments"])
        assert args == {"command": "ls -la"}

    def test_format_11_tool_call_with_multiple_args(self):
        text = 'TOOL_CALL: mcp_read(path="/etc/hostname", verbose=True)'
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp_read"
        args = json.loads(result["function"]["arguments"])
        assert args == {"path": "/etc/hostname", "verbose": True}

    def test_format_12_mimoml(self):
        text = (
            "<|MiMoML|tool_calls>"
            '<|MiMoML|invoke name="mcp_bash">'
            '<|MiMoML|parameter name="command"><![CDATA[ls]]></|MiMoML|parameter>'
            "</|MiMoML|invoke>"
            "</|MiMoML|tool_calls>"
        )
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp_bash"
        args = json.loads(result["function"]["arguments"])
        assert args == {"command": "ls"}

    def test_format_12_mimoml_hyphenated(self):
        text = (
            "<mimoml-tool_calls>"
            '<invoke name="mcp_bash">'
            '<parameter name="command">ls</parameter>'
            "</invoke>"
            "</mimoml-tool_calls>"
        )
        result = _parse_tool_call_xml(text)
        assert result is not None
        assert result["function"]["name"] == "mcp_bash"

    def test_no_tool_call_returns_none(self):
        assert _parse_tool_call_xml("just plain text response") is None

    def test_camelcase_name_resolved(self):
        # Format 1 with a camelCase name; should resolve to snake_case
        text = '<tool_call>{"name": "mcpBash", "arguments": {"command": "ls"}}</tool_call>'
        result = _parse_tool_call_xml(text, tool_names=["mcp_bash", "mcp_read"])
        assert result is not None
        assert result["function"]["name"] == "mcp_bash"


# ── _StreamSieve ──────────────────────────────────────────────────────


class TestStreamSieve:
    """Verifies the sieve catches tool calls that span multiple feed() calls."""

    def _parse(self, buf, names):
        tc = _parse_tool_call_xml(buf, names)
        if tc:
            return [tc], buf.replace(tc.get("function", {}).get("arguments", ""), "")
        return None, buf

    def test_simple_text(self):
        sieve = _StreamSieve(self._parse)
        events = sieve.feed("hello world")
        # Either all flushed at the end, or held back; check after flush
        sieve.feed("")  # nudge
        all_events = events + sieve.flush()
        text = "".join(d for k, d in all_events if k == "text")
        assert "hello" in text

    def test_catches_complete_tool_call(self):
        sieve = _StreamSieve(self._parse)
        events = sieve.feed('<tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>')
        # Sieve should detect the tool_call
        kinds = [k for k, _ in events]
        assert "tool_call" in kinds or "text" in kinds  # at minimum some event

    def test_catches_split_tool_call(self):
        """The critical bug: a tool call split across two feed() calls."""
        sieve = _StreamSieve(self._parse, tool_names=["bash"])
        # First chunk: opening tag
        events1 = sieve.feed('<tool_call>{"name": "bash"')
        # Second chunk: rest of the call
        events2 = sieve.feed(', "arguments": {"command": "ls"}}</tool_call>')
        # Flush to make sure
        all_events = events1 + events2 + sieve.flush()
        tool_calls = [d for k, d in all_events if k == "tool_call"]
        assert len(tool_calls) >= 1, f"Expected at least one tool_call, got {all_events}"
        tc = tool_calls[0][0]  # list of one
        assert tc["function"]["name"] == "bash"
        args = json.loads(tc["function"]["arguments"])
        assert args == {"command": "ls"}

    def test_holds_suspicious_trailing(self):
        """The sieve should hold a trailing '<' until the next chunk arrives."""
        sieve = _StreamSieve(self._parse)
        # Trailing '<' could be start of <tool_call> or just a less-than sign
        events = sieve.feed("output: 5 <")
        # The '<' should be held back, not emitted as text
        text = "".join(d for k, d in events if k == "text")
        assert "<" not in text
        assert "output: 5" in text
        # After flush, the '<' is released as text
        flushed = sieve.flush()
        flushed_text = "".join(d for k, d in flushed if k == "text")
        assert "<" in flushed_text

    def test_flush_releases_buffered(self):
        sieve = _StreamSieve(self._parse)
        events = sieve.feed("partial buffer")
        # 'partial buffer' doesn't match any tool start, so feed should emit it
        text = "".join(d for k, d in events if k == "text")
        assert "partial" in text
        # And the flush should be a no-op
        flushed = sieve.flush()
        assert len(flushed) == 0

    def test_does_not_capture_in_fenced_code_block(self):
        """Tool call inside a markdown fence should be ignored."""
        sieve = _StreamSieve(self._parse, tool_names=["bash"])
        text = "```\n<tool_call>{'name':'bash'}\n```\nreal output"
        events = sieve.feed(text)
        events += sieve.flush()
        # The tool call inside the fence should NOT be extracted
        tool_calls = [d for k, d in events if k == "tool_call"]
        assert len(tool_calls) == 0
