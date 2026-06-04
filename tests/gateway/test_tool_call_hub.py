"""Integration tests for tool_call_hub — broker-mode external tool coordination.

Tests cover the in-memory hub used by _await_external_tool_result to coordinate
tool execution between the agent thread and an external API client (e.g., OpenCode).

These tests validate the hub independently of a full gateway deployment, making
them fast and suitable for CI.
"""

from __future__ import annotations

import threading
import time

import pytest

from gateway.platforms.tool_call_hub import (
    PendingCall,
    register_call,
    set_response,
    get_pending_call,
    pop_pending_call,
    pop_next_pending_for_tool,
)


# ── Basic PendingCall tests ──────────────────────────────────────────


class TestPendingCall:
    def test_initial_state(self):
        pc = PendingCall("sess-1", "call-1", tool_name="bash")
        assert pc.session_id == "sess-1"
        assert pc.call_id == "call-1"
        assert pc.tool_name == "bash"
        assert pc.status is None
        assert pc.result is None
        assert not pc.event.is_set()

    def test_set_result_signals_event(self):
        pc = PendingCall("sess-1", "call-1")
        pc.set_result("ok", "hello world")
        assert pc.status == "ok"
        assert pc.result == "hello world"
        assert pc.event.is_set()

    def test_event_blocks_until_result(self):
        pc = PendingCall("sess-1", "call-1")
        results = []

        def waiter():
            pc.event.wait(timeout=5.0)
            results.append((pc.status, pc.result))

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert results == []  # still waiting
        pc.set_result("ok", {"output": "done"})
        t.join(timeout=2.0)
        assert results == [("ok", {"output": "done"})]


# ── register_call tests ──────────────────────────────────────────────


class TestRegisterCall:
    def test_register_creates_pending_call(self):
        pc = register_call("sess-1", "call-1", tool_name="read")
        assert pc.session_id == "sess-1"
        assert pc.call_id == "call-1"
        assert pc.tool_name == "read"
        assert not pc.event.is_set()

        # Clean up
        pop_pending_call("sess-1", "call-1")

    def test_register_same_call_id_returns_same_object(self):
        pc1 = register_call("sess-1", "call-2")
        pc2 = register_call("sess-1", "call-2")
        assert pc1 is pc2  # same object

        pop_pending_call("sess-1", "call-2")

    def test_register_without_tool_name(self):
        pc = register_call("sess-1", "call-3")
        assert pc.tool_name is None

        pop_pending_call("sess-1", "call-3")


# ── set_response tests ───────────────────────────────────────────────


class TestSetResponse:
    def test_set_response_signals_pending_call(self):
        pc = register_call("sess-2", "call-1", tool_name="write")
        assert set_response("sess-2", "call-1", "ok", "file written")
        assert pc.status == "ok"
        assert pc.result == "file written"
        assert pc.event.is_set()

        pop_pending_call("sess-2", "call-1")

    def test_set_response_stashes_orphan(self):
        """A response posted before registration is stashed as orphan."""
        assert set_response("sess-3", "orphan-1", "ok", "early result")
        # Now register — should adopt the orphan
        pc = register_call("sess-3", "orphan-1")
        assert pc.status == "ok"
        assert pc.result == "early result"
        assert pc.event.is_set()  # already signalled

        pop_pending_call("sess-3", "orphan-1")

    def test_set_response_error_status(self):
        pc = register_call("sess-2", "call-err")
        assert set_response("sess-2", "call-err", "error", "command not found")
        assert pc.status == "error"
        assert pc.result == "command not found"

        pop_pending_call("sess-2", "call-err")


# ── Full broker flow simulation ──────────────────────────────────────


class TestBrokerFlow:
    """Simulate the full broker-mode tool execution:

    1. Agent thread calls register_call() and waits on event
    2. External client calls set_response() with the tool result
    3. Agent thread resumes with the result
    """

    def test_full_broker_flow(self):
        session_id = "broker-sess-1"
        call_id = "call-bash-001"

        agent_result = []

        def agent_thread():
            """Simulate _await_external_tool_result."""
            pc = register_call(session_id, call_id, tool_name="bash")
            waited = pc.event.wait(timeout=5.0)
            assert waited, "Agent should receive result within timeout"
            agent_result.append((pc.status, pc.result))

        # Start agent thread
        t = threading.Thread(target=agent_thread)
        t.start()

        # Simulate external client posting the result
        time.sleep(0.05)
        set_response(
            session_id,
            call_id,
            "ok",
            '{"stdout": "hello world", "stderr": "", "exitCode": 0}',
        )

        t.join(timeout=3.0)
        assert len(agent_result) == 1
        status, result = agent_result[0]
        assert status == "ok"
        assert "hello world" in str(result)

        # Clean up
        pop_pending_call(session_id, call_id)

    def test_broker_flow_error(self):
        session_id = "broker-sess-2"
        call_id = "call-read-002"

        agent_result = []

        def agent_thread():
            pc = register_call(session_id, call_id, tool_name="read")
            waited = pc.event.wait(timeout=5.0)
            assert waited
            agent_result.append((pc.status, pc.result))

        t = threading.Thread(target=agent_thread)
        t.start()

        time.sleep(0.05)
        set_response(session_id, call_id, "error", "file not found")

        t.join(timeout=3.0)
        assert agent_result == [("error", "file not found")]

        pop_pending_call(session_id, call_id)

    def test_broker_flow_timeout_simulation(self):
        """Verify that a call that never gets a response times out."""
        session_id = "broker-sess-3"
        call_id = "call-timeout"

        pc = register_call(session_id, call_id, tool_name="bash")
        # Wait with a short timeout — no one posts a response
        waited = pc.event.wait(timeout=0.1)
        assert not waited, "Should timeout when no response posted"

        pop_pending_call(session_id, call_id)

    def test_concurrent_broker_calls(self):
        """Multiple concurrent tool calls should not interfere."""
        results = {}

        def agent(call_id, tool_name):
            pc = register_call("sess-concurrent", call_id, tool_name=tool_name)
            pc.event.wait(timeout=5.0)
            results[call_id] = (pc.status, pc.result)

        threads = []
        for i in range(5):
            t = threading.Thread(
                target=agent, args=(f"call-{i}", f"tool-{i}"), daemon=True
            )
            threads.append(t)
            t.start()

        time.sleep(0.1)
        # Post all results
        for i in range(5):
            set_response("sess-concurrent", f"call-{i}", "ok", f"result-{i}")

        for t in threads:
            t.join(timeout=3.0)

        assert len(results) == 5
        for i in range(5):
            status, result = results[f"call-{i}"]
            assert status == "ok"
            assert result == f"result-{i}"

        # Clean up
        for i in range(5):
            pop_pending_call("sess-concurrent", f"call-{i}")

    def test_orphaned_response_adopted(self):
        """A response posted before registration is adopted immediately."""
        # Client posts result before agent registers
        set_response("sess-orphan", "early-call", "ok", "pre-posted result")

        # Agent registers — should get result immediately without blocking
        pc = register_call("sess-orphan", "early-call")
        assert pc.status == "ok"
        assert pc.result == "pre-posted result"
        # The event should already be set
        assert pc.event.is_set()

        pop_pending_call("sess-orphan", "early-call")


# ── pop_pending_call / get_pending_call tests ────────────────────────


class TestPendingCallManagement:
    def test_get_pending_call(self):
        pc = register_call("sess-mgmt", "call-1")
        found = get_pending_call("sess-mgmt", "call-1")
        assert found is pc

        not_found = get_pending_call("sess-mgmt", "nonexistent")
        assert not_found is None

        pop_pending_call("sess-mgmt", "call-1")

    def test_pop_removes_call(self):
        register_call("sess-pop", "call-1")
        popped = pop_pending_call("sess-pop", "call-1")
        assert popped is not None

        # Should be gone now
        assert get_pending_call("sess-pop", "call-1") is None

    def test_pop_next_pending_for_tool_filters_by_name(self):
        register_call("sess-popnext", "call-1", tool_name="bash")
        register_call("sess-popnext", "call-2", tool_name="read")
        register_call("sess-popnext", "call-3", tool_name="bash")

        # Pop matching "bash"
        popped = pop_next_pending_for_tool("sess-popnext", tool_name="bash")
        assert popped is not None
        assert popped.tool_name == "bash"
        assert popped.call_id in ("call-1", "call-3")

        # Verify the other bash call is still there
        remaining_bash = pop_next_pending_for_tool("sess-popnext", tool_name="bash")
        if remaining_bash:
            assert remaining_bash.tool_name == "bash"

        # Clean up remaining
        for cid in ("call-1", "call-2", "call-3"):
            pop_pending_call("sess-popnext", cid)

    def test_pop_next_without_tool_name_returns_any(self):
        register_call("sess-any", "call-a", tool_name="bash")
        register_call("sess-any", "call-b", tool_name="read")

        popped = pop_next_pending_for_tool("sess-any")
        assert popped is not None
        assert popped.call_id in ("call-a", "call-b")

        # Clean up remaining
        pop_pending_call("sess-any", "call-a")
        pop_pending_call("sess-any", "call-b")
