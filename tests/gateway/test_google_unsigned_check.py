"""Tests for Google thought_signature helpers."""

import base64

from gateway.platforms.api_server import (
    _count_unsigned_google_tool_calls,
    _has_unsigned_google_tool_calls,
    _inject_google_sentinel_signatures,
    _strip_hermes_ts_packed_ids,
    _strip_call_id_from_tool_calls,
    _unpack_hermes_ts_and_inject_signatures,
    _GOOGLE_THOUGHT_SIGNATURE_SENTINEL,
)


class TestCountUnsignedGoogleToolCalls:
    """Count assistant tool_calls that lack Google thought_signature."""

    def test_no_messages_returns_zero(self):
        assert _count_unsigned_google_tool_calls([]) == 0

    def test_no_tool_calls_returns_zero(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 0

    def test_one_unsigned_tool_call(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read"}},
                ],
            },
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 1

    def test_signed_tool_call_returns_zero(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read"},
                        "extra_content": {"google": {"thought_signature": "sig123"}},
                    },
                ],
            },
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 0

    def test_mixed_signed_and_unsigned(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read"}},
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "write"},
                        "extra_content": {"google": {"thought_signature": "sig"}},
                    },
                ],
            },
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 1

    def test_multiple_assistant_messages(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read"}},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_2", "type": "function", "function": {"name": "write"}},
                    {"id": "call_3", "type": "function", "function": {"name": "bash"}},
                ],
            },
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 3

    def test_empty_thought_signature_treated_as_unsigned(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read"},
                        "extra_content": {"google": {"thought_signature": ""}},
                    },
                ],
            },
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 1

    def test_extra_content_without_google(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "x"},
                        "extra_content": {"other_provider": {"thought_signature": "x"}},
                    },
                ],
            },
        ]
        assert _count_unsigned_google_tool_calls(msgs) == 1

    def test_alias_has_unsigned_works(self):
        """_has_unsigned_google_tool_calls is an alias for _count_unsigned."""
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read"}},
            ]},
        ]
        assert _has_unsigned_google_tool_calls(msgs) == _count_unsigned_google_tool_calls(msgs)


class TestInjectGoogleSentinelSignatures:
    """Inject sentinel thought_signature on unsigned tool_calls."""

    def test_injects_on_unsigned(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read"}},
            ]},
        ]
        modified = _inject_google_sentinel_signatures(msgs)
        assert modified == 1
        ec = msgs[0]["tool_calls"][0]["extra_content"]
        assert ec["google"]["thought_signature"] == _GOOGLE_THOUGHT_SIGNATURE_SENTINEL

    def test_preserves_existing_signature(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read"},
                 "extra_content": {"google": {"thought_signature": "real_sig"}}},
            ]},
        ]
        modified = _inject_google_sentinel_signatures(msgs)
        assert modified == 0
        assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "real_sig"

    def test_injects_only_unsigned_in_mixed(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read"}},
                {"id": "c2", "type": "function", "function": {"name": "write"},
                 "extra_content": {"google": {"thought_signature": "existing"}}},
            ]},
        ]
        modified = _inject_google_sentinel_signatures(msgs)
        assert modified == 1
        assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == _GOOGLE_THOUGHT_SIGNATURE_SENTINEL
        assert msgs[0]["tool_calls"][1]["extra_content"]["google"]["thought_signature"] == "existing"

    def test_no_tool_calls_returns_zero(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        assert _inject_google_sentinel_signatures(msgs) == 0

    def test_empty_messages_returns_zero(self):
        assert _inject_google_sentinel_signatures([]) == 0

    def test_multiple_unsigned(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read"}},
                {"id": "c2", "type": "function", "function": {"name": "write"}},
            ]},
        ]
        modified = _inject_google_sentinel_signatures(msgs)
        assert modified == 2
        for tc in msgs[0]["tool_calls"]:
            assert tc["extra_content"]["google"]["thought_signature"] == _GOOGLE_THOUGHT_SIGNATURE_SENTINEL

    def test_overwrites_non_google_extra_content(self):
        """If extra_content exists but has no google key, sentinel overwrites it."""
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read"},
                 "extra_content": {"other": "data"}},
            ]},
        ]
        modified = _inject_google_sentinel_signatures(msgs)
        assert modified == 1
        assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == _GOOGLE_THOUGHT_SIGNATURE_SENTINEL


class TestStripHermesTsPackedIds:
    """_strip_hermes_ts_packed_ids strips :hermes_ts: suffix from tool_call ids."""

    def test_strips_from_assistant_and_tool_result(self):
        msgs = [
            {"role": "assistant", "tool_calls": [{"id": "tc1:hermes_ts:abc", "type": "function", "call_id": "tc1:hermes_ts:abc"}]},
            {"role": "tool", "tool_call_id": "tc1:hermes_ts:abc", "content": "result"},
        ]
        fixed = _strip_hermes_ts_packed_ids(msgs)
        assert fixed == 2
        assert msgs[0]["tool_calls"][0]["id"] == "tc1"
        assert msgs[0]["tool_calls"][0]["call_id"] == "tc1"
        assert msgs[1]["tool_call_id"] == "tc1"


class TestUnpackHermesTsAndInjectSignatures:
    """Unpacks :hermes_ts: and restores thought_signature for Google."""

    def test_unpacks_base64_and_strips_ids(self):
        sig = "test_signature_123"
        packed = base64.urlsafe_b64encode(sig.encode()).decode().rstrip("=")
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": f"call_abc:hermes_ts:{packed}", "type": "function", "call_id": f"call_abc:hermes_ts:{packed}"},
            ]},
            {"role": "tool", "tool_call_id": f"call_abc:hermes_ts:{packed}", "content": "ok"},
        ]
        injected, unpacked = _unpack_hermes_ts_and_inject_signatures(msgs)
        assert injected == 1
        assert unpacked == 1
        assert msgs[0]["tool_calls"][0]["id"] == "call_abc"
        assert msgs[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == sig
        assert msgs[1]["tool_call_id"] == "call_abc"


class TestStripCallIdFromToolCalls:
    """_strip_call_id_from_tool_calls removes call_id field."""

    def test_removes_call_id(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "call_id": "call_1", "type": "function"},
            ]},
        ]
        removed = _strip_call_id_from_tool_calls(msgs)
        assert removed == 1
        assert "call_id" not in msgs[0]["tool_calls"][0]
        assert msgs[0]["tool_calls"][0]["id"] == "call_1"
