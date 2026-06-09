"""Tests for _strip_hermes_ts_packed_ids and _strip_call_id_from_tool_calls."""

import json
from gateway.platforms.api_server import (
    _strip_hermes_ts_packed_ids,
    _strip_call_id_from_tool_calls,
    _unpack_hermes_ts_and_inject_signatures,
)


class TestStripHermesTsPackedIds:
    """_strip_hermes_ts_packed_ids strips :hermes_ts: suffix from tool_call ids."""

    def test_no_packed_ids_returns_zero(self):
        msgs = [{"role": "assistant", "tool_calls": [{"id": "call_abc", "type": "function"}]}]
        assert _strip_hermes_ts_packed_ids(msgs) == 0

    def test_strips_from_assistant_tool_call_id(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_abc:hermes_ts:YWJjZGVm", "type": "function", "call_id": "call_abc:hermes_ts:YWJjZGVm"},
                ],
            },
        ]
        fixed = _strip_hermes_ts_packed_ids(msgs)
        assert fixed == 1
        assert msgs[0]["tool_calls"][0]["id"] == "call_abc"
        assert msgs[0]["tool_calls"][0]["call_id"] == "call_abc"

    def test_strips_from_tool_result_tool_call_id(self):
        msgs = [
            {"role": "tool", "tool_call_id": "call_abc:hermes_ts:YWJjZGVm", "content": "ok"},
        ]
        fixed = _strip_hermes_ts_packed_ids(msgs)
        assert fixed == 1
        assert msgs[0]["tool_call_id"] == "call_abc"

    def test_strips_from_both_assistant_and_tool_result(self):
        """Both the assistant tool_call and the tool result must be fixed."""
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "tc1:hermes_ts:abc", "type": "function", "call_id": "tc1:hermes_ts:abc"}]},
            {"role": "tool", "tool_call_id": "tc1:hermes_ts:abc", "content": "result"},
        ]
        fixed = _strip_hermes_ts_packed_ids(msgs)
        assert fixed == 2
        assert msgs[0]["tool_calls"][0]["id"] == "tc1"
        assert msgs[0]["tool_calls"][0]["call_id"] == "tc1"
        assert msgs[1]["tool_call_id"] == "tc1"

    def test_no_hermes_ts_in_id_leaves_unchanged(self):
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call_123", "content": "ok"},
        ]
        assert _strip_hermes_ts_packed_ids(msgs) == 0
        assert msgs[0]["tool_calls"][0]["id"] == "call_123"
        assert msgs[1]["tool_call_id"] == "call_123"

    def test_multiple_packed_ids(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "a:hermes_ts:x", "type": "function", "call_id": "a:hermes_ts:x"},
                {"id": "b:hermes_ts:y", "type": "function", "call_id": "b:hermes_ts:y"},
            ]},
            {"role": "tool", "tool_call_id": "a:hermes_ts:x", "content": "r1"},
            {"role": "tool", "tool_call_id": "b:hermes_ts:y", "content": "r2"},
        ]
        fixed = _strip_hermes_ts_packed_ids(msgs)
        assert fixed == 4
        assert msgs[0]["tool_calls"][0]["id"] == "a"
        assert msgs[0]["tool_calls"][1]["id"] == "b"
        assert msgs[1]["tool_call_id"] == "a"
        assert msgs[2]["tool_call_id"] == "b"


class TestUnpackHermesTsAndInjectSignatures:
    """_unpack_hermes_ts_and_inject_signatures extracts thought_signature for Google."""

    def test_unpacks_base64_signature(self):
        import base64
        sig = "test_signature_123"
        packed = base64.urlsafe_b64encode(sig.encode()).decode().rstrip("=")
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": f"call_abc:hermes_ts:{packed}", "type": "function", "call_id": f"call_abc:hermes_ts:{packed}"},
                ],
            },
            {"role": "tool", "tool_call_id": f"call_abc:hermes_ts:{packed}", "content": "ok"},
        ]
        injected, unpacked = _unpack_hermes_ts_and_inject_signatures(msgs)
        assert injected == 1
        assert unpacked == 1
        assert msgs[0]["tool_calls"][0]["id"] == "call_abc"
        assert msgs[0]["tool_calls"][0]["call_id"] == "call_abc"
        assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == sig
        assert msgs[1]["tool_call_id"] == "call_abc"

    def test_preserves_existing_extra_content(self):
        """If extra_content.google.thought_signature already exists, don't overwrite."""
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_abc:hermes_ts:xyz", "type": "function",
                     "extra_content": {"google": {"thought_signature": "existing_sig"}}},
                ],
            },
        ]
        injected, unpacked = _unpack_hermes_ts_and_inject_signatures(msgs)
        # The existing signature is kept, but the id is NOT stripped because
        # we found _ts early and didn't enter the unpack branch
        assert injected == 1
        assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "existing_sig"

    def test_tool_result_id_always_stripped(self):
        """Tool result tool_call_id is always stripped of :hermes_ts:."""
        msgs = [
            {"role": "tool", "tool_call_id": "call_abc:hermes_ts:xyz", "content": "ok"},
        ]
        injected, unpacked = _unpack_hermes_ts_and_inject_signatures(msgs)
        assert injected == 0
        assert unpacked == 0
        assert msgs[0]["tool_call_id"] == "call_abc"


class TestStripCallIdFromToolCalls:
    """_strip_call_id_from_tool_calls removes call_id field from tool_calls."""

    def test_removes_call_id(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "call_id": "call_1", "type": "function"},
                {"id": "call_2", "call_id": "call_2", "type": "function"},
            ]},
        ]
        removed = _strip_call_id_from_tool_calls(msgs)
        assert removed == 2
        assert "call_id" not in msgs[0]["tool_calls"][0]
        assert "call_id" not in msgs[0]["tool_calls"][1]
        assert msgs[0]["tool_calls"][0]["id"] == "call_1"

    def test_no_call_id_returns_zero(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function"},
            ]},
        ]
        assert _strip_call_id_from_tool_calls(msgs) == 0

    def test_preserves_other_fields(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "call_id": "call_1", "type": "function",
                 "function": {"name": "read", "arguments": "{}"}},
            ]},
        ]
        _strip_call_id_from_tool_calls(msgs)
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "read"
        assert msgs[0]["tool_calls"][0]["id"] == "call_1"
