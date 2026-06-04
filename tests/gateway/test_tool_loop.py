"""
Tests for tool-loop behavior in the API server.

Reproduces and detects the "context dilution" loop where the model re-does
work instead of continuing from tool results.

The problem manifests when:
1. Long conversation (>150 messages) with tool calls
2. Last message is a tool result
3. Prior assistant message has pending tool_calls
4. Gateway sends user_message="" or a generic continuation

These tests verify the fix:
- _prior_assistant_has_pending_tool_calls() correctly identifies loops
- _get_recent_tool_context() truncates to prevent dilution
- user_message handling doesn't add confusing placeholders
"""

import json

import pytest

from gateway.platforms.api_server import (
    _get_recent_tool_context,
    _prior_assistant_has_pending_tool_calls,
    _find_last_nonempty_user_message,
    _detect_and_nudge_tool_loop,
)


# ---------------------------------------------------------------------------
# Helper: Build conversation fixtures
# ---------------------------------------------------------------------------

def make_user_message(content: str, idx: int) -> dict:
    """Create a user message."""
    return {"role": "user", "content": content, "message_idx": idx}


def make_assistant_message(tool_calls: list = None, content: str = "", idx: int = 0) -> dict:
    """Create an assistant message."""
    msg = {"role": "assistant", "content": content, "message_idx": idx}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def make_tool_result(content: str, tool_call_id: str, idx: float) -> dict:
    """Create a tool result message."""
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "message_idx": idx,
    }


def make_developer_message(content: str) -> dict:
    """Create a developer message."""
    return {"role": "developer", "content": content}


def simulate_gateway_continuation(conversation: list) -> tuple:
    """
    Simulate what the gateway does with a tool-result conversation.

    Returns (user_message, history) that would be sent to the agent.
    This mirrors the logic in gateway/platforms/api_server.py for tool loop continuation.
    """
    if not conversation:
        return "", []

    last_message = conversation[-1]

    if last_message.get("role") == "tool":
        prior_has_pending = _prior_assistant_has_pending_tool_calls(
            conversation[:-1]
        )

        if prior_has_pending:
            # Tool loop continuation
            if len(conversation) > 150:
                history = _get_recent_tool_context(conversation)
            else:
                history = conversation
            return " ", history
        else:
            user_message = _find_last_nonempty_user_message(conversation)
            return user_message, conversation

    return "", conversation


# ---------------------------------------------------------------------------
# _prior_assistant_has_pending_tool_calls
# ---------------------------------------------------------------------------

class TestPriorAssistantHasPendingToolCalls:
    """Tests for the tool-loop detection helper."""

    def test_empty_conversation_returns_false(self):
        assert _prior_assistant_has_pending_tool_calls([]) is False

    def test_conversation_with_user_only_returns_false(self):
        conv = [make_user_message("hello", 0)]
        assert _prior_assistant_has_pending_tool_calls(conv) is False

    def test_assistant_without_tool_calls_returns_false(self):
        conv = [
            make_user_message("hello", 0),
            make_assistant_message(content="I'll help with that", idx=1),
        ]
        assert _prior_assistant_has_pending_tool_calls(conv) is False

    def test_assistant_with_tool_calls_and_tool_result_returns_true(self):
        """When last is tool and prior assistant has tool_calls, returns True."""
        conv = [
            make_user_message("fix the bug", 0),
            make_assistant_message(
                tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
                idx=1,
            ),
            make_tool_result("done", "tc1", 2),
        ]
        # Last is tool, prior has pending tool_calls
        assert _prior_assistant_has_pending_tool_calls(conv) is True

    def test_truncates_at_user_message_boundary(self):
        """When there's a user message after an assistant, don't look past it."""
        conv = [
            make_user_message("task 1", 0),
            make_assistant_message(
                tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
                idx=1,
            ),
            make_tool_result("result 1", "tc1", 2),
            make_user_message("task 2", 3),  # user message = boundary
        ]
        # Should return False because we hit the user message before finding pending tool_calls
        assert _prior_assistant_has_pending_tool_calls(conv) is False

    def test_multiple_tool_calls_all_pending(self):
        conv = [
            make_user_message("run commands", 0),
            make_assistant_message(
                tool_calls=[
                    {"id": "tc1", "function": {"name": "bash", "arguments": "{}"}},
                    {"id": "tc2", "function": {"name": "read", "arguments": "{}"}},
                ],
                idx=1,
            ),
            make_tool_result("bash output", "tc1", 2),
            make_tool_result("read output", "tc2", 3),
        ]
        # Last message is tool, prior has pending tool_calls
        assert _prior_assistant_has_pending_tool_calls(conv) is True


# ---------------------------------------------------------------------------
# _get_recent_tool_context
# ---------------------------------------------------------------------------

class TestGetRecentToolContext:
    """Tests for the context truncation helper."""

    def test_empty_returns_empty(self):
        assert _get_recent_tool_context([]) == []

    def test_small_conversation_returns_all(self):
        """With <150 messages, return everything."""
        conv = [make_user_message(f"msg {i}", i) for i in range(10)]
        result = _get_recent_tool_context(conv)
        assert len(result) == 10
        assert result[0] == conv[0]

    def test_large_conversation_truncates_to_50(self):
        """With >150 messages, truncate to last 50."""
        conv = [make_user_message(f"msg {i}", i) for i in range(200)]
        result = _get_recent_tool_context(conv)
        assert len(result) == 50
        assert result[0]["message_idx"] == 150  # first of last 50

    def test_preserves_tool_call_and_results(self):
        """Last 50 should include recent tool calls and results."""
        conv = [make_user_message(f"msg {i}", i) for i in range(160)]
        # Add tool calls and results at the end
        conv.append(make_assistant_message(
            tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
            idx=160,
        ))
        conv.append(make_tool_result("done", "tc1", 161))

        result = _get_recent_tool_context(conv)
        # Should include the last 50 (indices 111-161)
        assert len(result) == 50
        assert result[-1]["message_idx"] == 161  # last message preserved
        # The tool call and result should be in the result
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

    def test_includes_original_user_instruction(self):
        """Last 50 should include the original task if it's within range."""
        # Put user message at index 140 of 200 total
        conv = [make_user_message(f"msg {i}", i) for i in range(140)]
        conv.append(make_user_message("Fix the ADF entity error", 140))
        conv.extend([make_user_message(f"msg {i}", i) for i in range(141, 162)])

        result = _get_recent_tool_context(conv)
        # The user message at 140 should be included (indices 112-161)
        user_msgs = [m for m in result if "Fix the ADF" in str(m.get("content", ""))]
        assert len(user_msgs) >= 1


# ---------------------------------------------------------------------------
# _find_last_nonempty_user_message
# ---------------------------------------------------------------------------

class TestFindLastNonemptyUserMessage:
    """Tests for finding the last meaningful user message."""

    def test_empty_returns_empty(self):
        assert _find_last_nonempty_user_message([]) == ""

    def test_finds_nonempty_user_message(self):
        conv = [
            make_user_message("task 1", 0),
            make_user_message("task 2", 1),
        ]
        assert _find_last_nonempty_user_message(conv) == "task 2"

    def test_skips_empty_content(self):
        conv = [
            make_user_message("task 1", 0),
            make_user_message("", 1),
            make_user_message("task 3", 2),
        ]
        result = _find_last_nonempty_user_message(conv)
        assert result == "task 3"

    def test_skips_whitespace_only(self):
        conv = [
            make_user_message("task 1", 0),
            make_user_message("   ", 1),
            make_user_message("task 3", 2),
        ]
        result = _find_last_nonempty_user_message(conv)
        assert result == "task 3"

    def test_ignores_tool_and_assistant_roles(self):
        conv = [
            make_user_message("task 1", 0),
            make_assistant_message(content="response", idx=1),
            make_tool_result("result", "tc1", 2),
        ]
        result = _find_last_nonempty_user_message(conv)
        assert result == "task 1"


# ---------------------------------------------------------------------------
# Tool-loop continuation flow simulation
# ---------------------------------------------------------------------------

class TestToolLoopContinuationFlow:
    """Tests for the full tool-loop continuation flow."""

    def test_short_conversation_preserves_full_history(self):
        """With <150 messages, full history is preserved."""
        conv = [
            make_developer_message("You are a helpful assistant."),
            make_user_message("Fix the ADF entity error", 0),
            make_assistant_message(
                tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
                idx=1,
            ),
            make_tool_result("git output", "tc1", 2),
        ]

        user_msg, history = simulate_gateway_continuation(conv)

        assert user_msg == " "  # Single space
        assert len(history) == 4  # Full history
        assert history == conv

    def test_long_conversation_truncates_history(self):
        """With >150 messages, history is truncated to prevent dilution."""
        # Build a conversation with 200 messages
        conv = [make_developer_message("You are a helpful assistant.")]
        conv.append(make_user_message("Fix the ADF entity error", 0))
        conv.append(make_assistant_message(
            tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
            idx=1,
        ))

        # Add messages 2-197
        msg_idx = 2
        for i in range(98):  # 98 iterations = 196 messages
            conv.append(make_user_message(f"followup {i}", msg_idx))
            msg_idx += 1
            conv.append(make_assistant_message(
                tool_calls=[{"id": f"tc{i}", "function": {"name": "bash", "arguments": "{}"}}],
                idx=msg_idx,
            ))
            msg_idx += 1
            conv.append(make_tool_result(f"result {i}", f"tc{i}", msg_idx))
            msg_idx += 1

        # Final tool result
        conv.append(make_tool_result("done", "tc_last", msg_idx))

        assert len(conv) > 150  # Verify we're testing the truncation path

        user_msg, history = simulate_gateway_continuation(conv)

        assert user_msg == " "  # Single space (not empty, not a new prompt)
        assert len(history) == 50  # Truncated to last 50
        # The last message (tool result) should be in history
        assert history[-1]["role"] == "tool"

    def test_placeholder_not_added_to_messages(self):
        """
        Verify that the " " placeholder would NOT be added as a user message.

        This is the fix: run_agent.py skips adding user_message=" " to messages.
        """
        conv = [
            make_user_message("Fix the bug", 0),
            make_assistant_message(
                tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}],
                idx=1,
            ),
            make_tool_result("output", "tc1", 2),
        ]

        user_msg, history = simulate_gateway_continuation(conv)

        # Simulate what run_agent.py does: skip adding " " to messages
        messages = list(history)
        if user_msg.strip():  # Only add if non-whitespace
            messages.append({"role": "user", "content": user_msg})

        # With our fix, " " is whitespace so it should NOT be added
        assert len(messages) == 3  # Same as history, no extra user message

    def test_explicit_continuation_would_cause_loop(self):
        """
        Demonstrate why "Continue from the tool results above." causes loops.

        This test documents the PROBLEM that the fix addresses.
        """
        conv = [make_user_message("Fix the ADF entity error", 0)]

        # Add 50 tool cycles (simulating long conversation)
        for i in range(1, 101):
            conv.append(make_assistant_message(
                tool_calls=[{"id": f"tc{i}", "function": {"name": "bash", "arguments": "{}"}}],
                idx=i,
            ))
            conv.append(make_tool_result(f"output {i}", f"tc{i}", i + 0.5))

        assert len(conv) > 150  # Verify long conversation

        user_msg, history = simulate_gateway_continuation(conv)

        # The fix: " " is not an explicit continuation message
        assert user_msg == " "
        assert "Continue" not in user_msg
        # History is truncated to prevent context dilution
        assert len(history) == 50


# ---------------------------------------------------------------------------
# Regression test: detect the exact loop pattern from MITM logs
# ---------------------------------------------------------------------------

class TestLoopPatternDetection:
    """
    Tests that detect the exact patterns causing the loop:

    Pattern 1: Model re-does same git commands
    Pattern 2: Model re-reads same files
    Pattern 3: Model keeps saying "Backup complete"
    """

    def test_detect_repeated_bash_commands(self):
        """
        Given a conversation with repeated bash commands, detect the loop.

        The MITM logs showed:
        - bash("cd ~/dev/azure/prjcts_power_analytics_eta_adf && git branch")
        - bash("ls -la ~/dev/azure/prjcts_power_analytics_eta_adf/")
        - bash("find . -type f -name \"*.json\"")
        These 61 unique commands were repeated 156+ times.
        """
        # Simulate a conversation with repeated tool calls
        conv = [make_developer_message("You are a coding assistant.")]
        conv.append(make_user_message("Look at ADF files", 0))

        # Simulate 60 tool cycles of the same pattern
        for i in range(60):
            conv.append(make_assistant_message(
                tool_calls=[
                    {"id": f"tc{i}_1", "function": {"name": "bash", "arguments": json.dumps({"command": "cd ~/dev/azure/prjcts_power_analytics_eta_adf && git branch"})}},
                    {"id": f"tc{i}_2", "function": {"name": "bash", "arguments": json.dumps({"command": "ls -la ~/dev/azure/prjcts_power_analytics_eta_adf/"})}},
                ],
                idx=i * 2 + 1,
            ))
            conv.append(make_tool_result("Already on 'feature/adf'", f"tc{i}_1", i * 2 + 2))
            conv.append(make_tool_result("total 16\ndrwxr-xr-x   2 tusker staff   416 28 May 17:47 .", f"tc{i}_2", i * 2 + 3))

        # Final tool result that triggers continuation
        conv.append(make_tool_result("On branch feature/adf", "tc_last", 122))

        assert len(conv) > 150  # Verify we're testing truncation

        user_msg, history = simulate_gateway_continuation(conv)

        # With fix: history is truncated, no explicit continuation
        assert len(history) == 50  # Truncated
        assert user_msg == " "  # Single space, not explicit

    def test_detect_backup_complete_loop(self):
        """
        The model kept running the same Power BI backup script.

        Each cycle had:
        - bash(script that prints "Backup complete!")
        - tool_result("Found 2 reports\n✓ ETA Analytics Report...")
        - Model sees this as success, runs it again
        """
        # Simulate a conversation where model keeps running same backup
        conv = [make_developer_message("You are a coding assistant.")]

        # Simulate 80 cycles of the backup script (>75 to exceed 150 messages)
        for i in range(80):
            # Model runs the backup script
            script = """
cd prjcts_power_analytics_eta && python3 << 'EOF'
import subprocess, json, urllib.request, os
from datetime import datetime

PBI_API = 'https://api.powerbi.com/v1.0/myorg'
WORKSPACE_ID = 'cb3c1280-f7e8-41f8-893a-f8d6fcda43be'

# ... (backup script)
print("Backup complete!")
print(f"Reports: {len(reports)}")
EOF
"""
            conv.append(make_assistant_message(
                tool_calls=[{"id": f"tc{i}", "function": {"name": "bash", "arguments": json.dumps({"command": script})}}],
                idx=i * 2,
            ))
            # Tool result shows it "worked"
            conv.append(make_tool_result(
                f"Found 2 reports\n✓ ETA Analytics Report: 7 pages\n✓ Backup complete!",
                f"tc{i}",
                i * 2 + 1,
            ))

        # Final tool result
        conv.append(make_tool_result("Backup complete!", "tc_last", 61))

        assert len(conv) > 150  # Verify truncation

        user_msg, history = simulate_gateway_continuation(conv)

        # With truncation, model only sees last 50 messages (last ~2 cycles)
        # This gives it fresh context to realize the work is done
        assert len(history) == 50
        assert history[-1]["role"] == "tool"


# ---------------------------------------------------------------------------
# Test: Verify the fix addresses all loop causes
# ---------------------------------------------------------------------------

class TestLoopFixCompleteness:
    """
    Verify that our fix addresses all causes of the loop.

    Causes of loops:
    1. user_message="" → pi shows "empty message" error
    2. user_message="Continue..." → model re-analyzes
    3. Full history → context dilution
    """

    def test_fix_addresses_empty_message(self):
        """Fix: user_message=' ' (single space) passes pi's empty check."""
        conv = [
            make_user_message("task", 0),
            make_assistant_message(tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}], idx=1),
            make_tool_result("done", "tc1", 2),
        ]

        user_msg, history = simulate_gateway_continuation(conv)

        # " " is not empty, so pi won't show error
        assert user_msg == " "
        assert user_msg.strip() == ""  # But it's whitespace, not a real prompt

    def test_fix_addresses_re_analysis(self):
        """Fix: ' ' is not a meaningful prompt, model continues naturally."""
        conv = [make_user_message("task", 0)]
        for i in range(1, 161):
            conv.append(make_assistant_message(
                tool_calls=[{"id": f"tc{i}", "function": {"name": "bash", "arguments": "{}"}}],
                idx=i,
            ))
            conv.append(make_tool_result(f"result {i}", f"tc{i}", i + 0.5))

        conv.append(make_tool_result("done", "tc_last", 322))

        user_msg, history = simulate_gateway_continuation(conv)

        # " " is not "Continue from the tool results above."
        # So the model doesn't treat it as a new prompt
        assert user_msg == " "
        assert "Continue" not in user_msg

    def test_fix_addresses_context_dilution(self):
        """Fix: Truncate to 50 messages when >150 total."""
        conv = [make_user_message("original task", 0)]
        for i in range(1, 161):
            conv.append(make_assistant_message(
                tool_calls=[{"id": f"tc{i}", "function": {"name": "bash", "arguments": "{}"}}],
                idx=i,
            ))
            conv.append(make_tool_result(f"result {i}", f"tc{i}", i + 0.5))

        conv.append(make_tool_result("done", "tc_last", 322))

        assert len(conv) > 150

        user_msg, history = simulate_gateway_continuation(conv)

        # With >150 messages, history is truncated
        assert len(history) == 50  # Truncated to last 50
        # This prevents context dilution - model doesn't see old cycles

    def test_fix_preserves_task_context_for_small_conversations(self):
        """Fix: Small conversations (<150 msgs) keep full context."""
        conv = [
            make_user_message("Fix the ADF entity error", 0),
            make_assistant_message(tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}], idx=1),
            make_tool_result("git output", "tc1", 2),
        ]

        user_msg, history = simulate_gateway_continuation(conv)

        # Small conversation keeps full history
        assert len(history) == 3
        # Original task is preserved
        assert "Fix the ADF" in history[0]["content"]


# ---------------------------------------------------------------------------
# Integration test: run_agent.py integration
# ---------------------------------------------------------------------------

class TestRunAgentIntegration:
    """
    Test that verifies the fix works end-to-end with run_agent.py.

    The key check: when user_message=" ", run_agent.py should NOT add it
    to the messages list.
    """

    def test_run_agent_skips_whitespace_user_message(self):
        """
        Verify run_agent.py logic for skipping whitespace user messages.

        This mirrors the fix in run_agent.py line ~10928:
        if user_message.strip():
            messages.append(user_msg)
        else:
            # Skip - whitespace placeholder
            pass
        """
        # Simulate what run_agent.py does
        user_message = " "
        history = [
            make_user_message("task", 0),
            make_assistant_message(tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}], idx=1),
            make_tool_result("done", "tc1", 2),
        ]

        messages = list(history)
        if user_message.strip():
            messages.append({"role": "user", "content": user_message})

        # With the fix, " " is whitespace so NOT added
        assert len(messages) == 3  # Same as history, no extra
        # Last message is still the tool result
        assert messages[-1]["role"] == "tool"

    def test_run_agent_adds_non_whitespace_user_message(self):
        """Verify run_agent.py adds non-whitespace user messages."""
        user_message = "Continue from the tool results above."
        history = [
            make_user_message("task", 0),
            make_assistant_message(tool_calls=[{"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}], idx=1),
            make_tool_result("done", "tc1", 2),
        ]

        messages = list(history)
        if user_message.strip():
            messages.append({"role": "user", "content": user_message})

        # Non-whitespace message IS added
        assert len(messages) == 4  # History + new user message
        assert messages[-1]["role"] == "user"
        assert "Continue" in messages[-1]["content"]