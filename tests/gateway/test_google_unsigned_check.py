"""Tests for _has_unsigned_google_tool_calls."""

import base64

from gateway.platforms.api_server import _has_unsigned_google_tool_calls


class TestHasUnsignedGoogleToolCalls:
    """Detects assistant tool_calls that lack Google thought_signature."""

    def test_no_messages_returns_zero(self):
        assert _has_unsigned_google_tool_calls([]) == 0

    def test_no_tool_calls_returns_zero(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _has_unsigned_google_tool_calls(msgs) == 0

    def test_one_unsigned_tool_call(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read"}},
                ],
            },
        ]
        assert _has_unsigned_google_tool_calls(msgs) == 1

    def test_signed_tool_call_returns_zero(self):
        """A tool_call with extra_content.google.thought_signature is signed."""
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
        assert _has_unsigned_google_tool_calls(msgs) == 0

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
        assert _has_unsigned_google_tool_calls(msgs) == 1

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
        assert _has_unsigned_google_tool_calls(msgs) == 3

    def test_empty_thought_signature_treated_as_unsigned(self):
        """Empty string is falsy — counts as unsigned."""
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
        assert _has_unsigned_google_tool_calls(msgs) == 1

    def test_non_dict_tool_call_skipped(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": ["not_a_dict", {"id": "call_1", "function": {"name": "x"}}],
            },
        ]
        assert _has_unsigned_google_tool_calls(msgs) == 1

    def test_extra_content_without_google(self):
        """extra_content present but no google key — still unsigned."""
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
        assert _has_unsigned_google_tool_calls(msgs) == 1
