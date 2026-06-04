"""Regression test for empty user message at end of a tool cycle.

pi/opencode sometimes sends an empty user message (content="") at the end of
a tool-call cycle as a "please continue" signal.  Before this fix, Hermes
would pass the empty string directly to run_agent, causing the model to
produce the canned "It looks like your message came through empty!" response
instead of continuing the active task.

Fix: _find_last_nonempty_user_message() walks back through conversation_messages
to find the most recent non-blank user message, which is then used as the
effective user_message for that cycle.
"""

from __future__ import annotations

import pytest
from gateway.platforms.api_server import _find_last_nonempty_user_message, _prior_assistant_has_pending_tool_calls, _sanitise_compaction_summary


class TestFindLastNonemptyUserMessage:
    def test_walks_back_past_empty_user_to_prior_content(self):
        msgs = [
            {"role": "user", "content": "Build me a Bruno collection"},
            {"role": "assistant", "content": "Sure, here is what I will do..."},
        ]
        result = _find_last_nonempty_user_message(msgs)
        assert result == "Build me a Bruno collection"

    def test_none_content_is_treated_as_empty(self):
        msgs = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": None},
        ]
        # Pass the prior messages (excluding the empty-content last one)
        result = _find_last_nonempty_user_message(msgs[:-1])
        assert result == "What is 2+2?"

    def test_whitespace_only_content_skipped(self):
        msgs = [
            {"role": "user", "content": "Do the thing"},
            {"role": "assistant", "content": "Done"},
            {"role": "user", "content": "   \t\n"},
        ]
        # Walk through all — whitespace-only entries are skipped
        result = _find_last_nonempty_user_message(msgs)
        assert result == "Do the thing"

    def test_returns_empty_string_when_no_prior_user_messages(self):
        msgs = [
            {"role": "assistant", "content": "Hello"},
        ]
        result = _find_last_nonempty_user_message(msgs)
        assert result == ""

    def test_returns_empty_string_on_empty_list(self):
        result = _find_last_nonempty_user_message([])
        assert result == ""

    def test_returns_most_recent_nonempty_user_message(self):
        """Multiple user messages — should return the LAST non-empty one."""
        msgs = [
            {"role": "user", "content": "First task"},
            {"role": "assistant", "content": "Done first"},
            {"role": "user", "content": "Second task"},
            {"role": "assistant", "content": "Done second"},
            {"role": "user", "content": "Third task"},
            {"role": "assistant", "content": "Done third"},
        ]
        result = _find_last_nonempty_user_message(msgs)
        assert result == "Third task"

    def test_ignores_non_user_roles(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Hello"},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "My actual question"},
        ]
        result = _find_last_nonempty_user_message(msgs)
        assert result == "My actual question"

    def test_empty_string_content_is_skipped(self):
        msgs = [
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": ""},  # empty continuation signal
        ]
        result = _find_last_nonempty_user_message(msgs)
        assert result == "Original question"


class TestPriorAssistantHasPendingToolCalls:
    def test_returns_true_when_last_assistant_has_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Do the thing"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]
        assert _prior_assistant_has_pending_tool_calls(msgs) is True

    def test_returns_false_when_last_assistant_has_no_tool_calls(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        assert _prior_assistant_has_pending_tool_calls(msgs) is False

    def test_returns_false_when_last_assistant_has_empty_tool_calls_list(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!", "tool_calls": []},
        ]
        assert _prior_assistant_has_pending_tool_calls(msgs) is False

    def test_returns_false_when_no_assistant_message(self):
        msgs = [
            {"role": "user", "content": "First message"},
        ]
        assert _prior_assistant_has_pending_tool_calls(msgs) is False

    def test_returns_false_on_empty_list(self):
        assert _prior_assistant_has_pending_tool_calls([]) is False

    def test_skips_tool_messages_to_find_assistant(self):
        """tool messages between assistant and the end should be skipped."""
        msgs = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc2", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "file contents", "tool_call_id": "tc2"},
            {"role": "tool", "content": "more contents", "tool_call_id": "tc2"},
        ]
        assert _prior_assistant_has_pending_tool_calls(msgs) is True

    def test_stops_at_user_message_before_assistant(self):
        """If a user message appears before finding any assistant, return False."""
        msgs = [
            {"role": "user", "content": "Earlier message"},
            {"role": "user", "content": ""},  # the empty trigger
        ]
        assert _prior_assistant_has_pending_tool_calls(msgs) is False

    def test_multiple_turns_finds_most_recent_assistant(self):
        """With multiple assistant turns, only the most recent matters."""
        msgs = [
            {"role": "user", "content": "Task 1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc3", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "result1", "tool_call_id": "tc3"},
            {"role": "assistant", "content": "Done with task 1."},  # plain reply, no tool_calls
        ]
        # Most recent assistant has no tool_calls
        assert _prior_assistant_has_pending_tool_calls(msgs) is False


class TestSanitiseCompactionSummary:
    def test_read_files_rendered_as_bullet_list(self):
        content = "<read-files>\n /path/to/a.sql\n /path/to/b.sql\n</read-files>"
        result = _sanitise_compaction_summary(content)
        assert "<read-files>" not in result
        assert "Files read:" in result
        assert "  - /path/to/a.sql" in result
        assert "  - /path/to/b.sql" in result

    def test_modified_files_rendered_as_bullet_list(self):
        content = "<modified-files>\n /path/to/c.sql\n</modified-files>"
        result = _sanitise_compaction_summary(content)
        assert "<modified-files>" not in result
        assert "Files modified:" in result
        assert "  - /path/to/c.sql" in result

    def test_both_blocks_replaced(self):
        content = (
            "Summary text.\n"
            "<read-files>\n /a.py\n</read-files>\n"
            "<modified-files>\n /b.py\n</modified-files>\n"
            "## Active Task\nDo the thing."
        )
        result = _sanitise_compaction_summary(content)
        assert "<read-files>" not in result
        assert "<modified-files>" not in result
        assert "Files read:" in result
        assert "Files modified:" in result
        assert "Do the thing." in result

    def test_empty_blocks_produce_no_output(self):
        content = "<read-files>\n</read-files>"
        result = _sanitise_compaction_summary(content)
        assert result.strip() == ""

    def test_no_xml_passthrough_unchanged(self):
        content = "Plain summary with no XML annotations."
        result = _sanitise_compaction_summary(content)
        assert result == content

    def test_case_insensitive(self):
        content = "<Read-Files>\n /x.sql\n</Read-Files>"
        result = _sanitise_compaction_summary(content)
        assert "<Read-Files>" not in result
        assert "Files read:" in result
