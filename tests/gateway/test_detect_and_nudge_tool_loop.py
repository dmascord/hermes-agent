"""Tests for _detect_and_nudge_tool_loop — the passthrough loop-breaker.

The function mutates passthrough_messages in place:
- Returns True and appends a user nudge at the TAIL when a loop is detected.
- Returns False and leaves the list unchanged when no loop is found.

Critical invariant: the nudge is always the *last* element (after all tool
results), never inserted between an assistant turn and its tool results,
which is what broke the reverted implementation.
"""

import pytest
from gateway.platforms.api_server import _detect_and_nudge_tool_loop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tc(name: str, tc_id: str = "tc0") -> dict:
    return {"id": tc_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _loop(msgs: list, fn: str, n: int) -> list:
    """Append n identical (assistant→tool_calls, tool-result) round-trips."""
    for i in range(n):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc(fn, f"tc_{fn}_{i}")]})
        msgs.append({"role": "tool", "content": f"result {i}", "tool_call_id": f"tc_{fn}_{i}"})
    return msgs


# ---------------------------------------------------------------------------
# No loop → returns False, list unchanged
# ---------------------------------------------------------------------------

class TestNoLoop:
    def test_empty_list(self):
        msgs = []
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == []

    def test_only_user_and_system(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_one_round_trip_below_threshold(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 1)
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_two_round_trips_below_default_threshold(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 2)
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_three_different_functions_no_loop(self):
        msgs = [{"role": "user", "content": "go"}]
        for i, fn in enumerate(["bash", "read_file", "bash"]):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc(fn, f"tc{i}")]})
            msgs.append({"role": "tool", "content": "ok", "tool_call_id": f"tc{i}"})
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_user_message_resets_window(self):
        """2 rounds, user message, 2 more rounds — total below threshold in each window."""
        msgs = []
        _loop(msgs, "bash", 2)
        msgs.append({"role": "user", "content": "try again"})
        _loop(msgs, "bash", 2)
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_plain_assistant_text_ends_window(self):
        """An assistant message without tool_calls stops the backward scan."""
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "thinking..."},  # no tool_calls
        ]
        _loop(msgs, "bash", 2)
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_custom_threshold_not_yet_reached(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 4)
        assert _detect_and_nudge_tool_loop(msgs, threshold=5) is False

    def test_empty_tool_calls_list_not_counted_as_round_trip(self):
        """assistant with tool_calls=[] is effectively a text reply; must not count."""
        msgs = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "thinking", "tool_calls": []},
            {"role": "assistant", "content": "thinking", "tool_calls": []},
            {"role": "assistant", "content": "thinking", "tool_calls": []},
        ]
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_none_tool_calls_not_counted(self):
        """assistant with tool_calls=None is a text turn; must not count."""
        msgs = [{"role": "user", "content": "go"}]
        for _ in range(4):
            msgs.append({"role": "assistant", "content": "hmm", "tool_calls": None})
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_orphaned_tool_messages_no_preceding_assistant(self):
        """Malformed: tool results with no preceding assistant must not raise."""
        msgs = [
            {"role": "tool", "content": "r0", "tool_call_id": "tc0"},
            {"role": "tool", "content": "r1", "tool_call_id": "tc1"},
            {"role": "tool", "content": "r2", "tool_call_id": "tc2"},
        ]
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_large_history_before_loop_not_triggered_early(self):
        """Noise before the looping segment must not inflate the round-trip count."""
        msgs = [{"role": "user", "content": "start"}]
        _loop(msgs, "search", 5)          # 5 rounds of 'search'
        msgs.append({"role": "assistant", "content": "OK, done searching"})
        msgs.append({"role": "user", "content": "now read"})
        _loop(msgs, "read_file", 2)       # only 2 rounds of 'read_file' — below threshold
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_loop_of_different_superset_not_counted(self):
        """Two rounds with {bash, read} and one with just {bash} — sets differ, no loop."""
        msgs = [{"role": "user", "content": "go"}]
        for i in range(2):
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [_tc("bash", f"b{i}"), _tc("read_file", f"r{i}")],
            })
            msgs.append({"role": "tool", "content": "ok", "tool_call_id": f"b{i}"})
            msgs.append({"role": "tool", "content": "ok", "tool_call_id": f"r{i}"})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc("bash", "b2")]})
        msgs.append({"role": "tool", "content": "ok", "tool_call_id": "b2"})
        before = list(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is False
        assert msgs == before

    def test_threshold_one_single_call_triggers(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 1)
        assert _detect_and_nudge_tool_loop(msgs, threshold=1) is True



# ---------------------------------------------------------------------------
# Loop detected → returns True, messages NOT mutated (audit mode)
# ---------------------------------------------------------------------------

class TestLoopDetected:
    def test_three_identical_calls_triggers(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        assert _detect_and_nudge_tool_loop(msgs) is True

    def test_messages_not_mutated_on_detection(self):
        """Audit mode: list must be identical before and after detection."""
        import copy
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        snapshot = copy.deepcopy(msgs)
        _detect_and_nudge_tool_loop(msgs)
        assert msgs == snapshot

    def test_five_calls_with_threshold_five(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 5)
        assert _detect_and_nudge_tool_loop(msgs, threshold=5) is True

    def test_more_than_threshold_still_triggers(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 7)
        assert _detect_and_nudge_tool_loop(msgs, threshold=3) is True

    def test_multiple_tools_per_round(self):
        """Same set of two tools called 3× in a row should trigger."""
        msgs = [{"role": "user", "content": "go"}]
        for i in range(3):
            msgs.append({
                "role": "assistant", "content": "",
                "tool_calls": [_tc("bash", f"b{i}"), _tc("read_file", f"r{i}")],
            })
            msgs.append({"role": "tool", "content": "out", "tool_call_id": f"b{i}"})
            msgs.append({"role": "tool", "content": "out", "tool_call_id": f"r{i}"})
        import copy
        snapshot = copy.deepcopy(msgs)
        assert _detect_and_nudge_tool_loop(msgs) is True
        assert msgs == snapshot  # no mutation

    def test_loop_after_user_reset(self):
        """Loop window resets at user messages; loop after reset is detected."""
        msgs = [{"role": "user", "content": "start"}]
        _loop(msgs, "search", 5)
        msgs.append({"role": "user", "content": "now try bash"})
        _loop(msgs, "bash", 3)
        assert _detect_and_nudge_tool_loop(msgs) is True

    def test_system_messages_transparent_to_scan(self):
        msgs = [{"role": "system", "content": "sys"}]
        _loop(msgs, "bash", 3)
        assert _detect_and_nudge_tool_loop(msgs) is True

    def test_repeated_calls_both_detect(self):
        """Audit mode has no memory; calling twice on the same loop returns
        True both times — the loop is still there, just not suppressed."""
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        assert _detect_and_nudge_tool_loop(msgs) is True
        assert _detect_and_nudge_tool_loop(msgs) is True
        assert len(msgs) == 7  # nothing appended


# ---------------------------------------------------------------------------
# Structural invariants — audit mode: read-only, log only
# ---------------------------------------------------------------------------

class TestAuditModeInvariants:
    def test_no_message_appended(self):
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        before_len = len(msgs)
        _detect_and_nudge_tool_loop(msgs)
        assert len(msgs) == before_len

    def test_list_not_mutated_at_all(self):
        import copy
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        snapshot = copy.deepcopy(msgs)
        _detect_and_nudge_tool_loop(msgs)
        assert msgs == snapshot

    def test_last_message_still_tool_after_detection(self):
        """Last message must remain a tool result — no user nudge appended."""
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        _detect_and_nudge_tool_loop(msgs)
        assert msgs[-1]["role"] == "tool"

    def test_returns_true_on_stuck_loop(self, caplog):
        import logging
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "bash", 3)
        with caplog.at_level(logging.WARNING):
            result = _detect_and_nudge_tool_loop(msgs)
        assert result is True
        assert "AUDIT" in caplog.text
        assert "bash" in caplog.text

    def test_polling_loop_logged_as_polling(self, caplog):
        import logging
        msgs = [{"role": "user", "content": "go"}]
        for i in range(3):
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [_tc("bash", f"tc{i}")]})
            msgs.append({"role": "tool", "content": "InProgress", "tool_call_id": f"tc{i}"})
        with caplog.at_level(logging.WARNING):
            result = _detect_and_nudge_tool_loop(msgs)
        assert result is True
        assert "polling-loop" in caplog.text

    def test_stuck_loop_logged_as_stuck(self, caplog):
        import logging
        msgs = [{"role": "user", "content": "go"}]
        _loop(msgs, "read_file", 3)
        with caplog.at_level(logging.WARNING):
            _detect_and_nudge_tool_loop(msgs)
        assert "stuck-loop" in caplog.text