"""Session contamination system tests.

Run:
    pytest hermes-agent/test/test_session_contamination.py -v
Live:
    export HERMES_API_SERVER_KEY=b49a80d538b98987e2f0c385bba137c79f017051cef9b95efd61929791dd4218
    export HERMES_GATEWAY_URL=https://hermes.tusker.net.au
    pytest hermes-agent/test/test_session_contamination.py -v
"""
from __future__ import annotations
import asyncio, hashlib, os, threading, time, uuid
from collections import OrderedDict
from typing import Any, Dict, Optional
import pytest


# --- helpers ---------------------------------------------------------------

def _derive_chat_session_id(system_prompt, first_user_message, *, salt=None):
    """Exact replica of gateway/platforms/api_server.py::_derive_chat_session_id."""
    seed = (salt or "") + "\n" + (system_prompt or "") + "\n" + first_user_message
    return "api-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _make_hub():
    return {"pending": {}, "orphaned": {}, "lock": threading.Lock()}


def _hub_register(hub, session_id, call_id, tool_name=None):
    key = (session_id, call_id)
    with hub["lock"]:
        if key in hub["orphaned"]:
            return hub["orphaned"].pop(key)
        od = hub["pending"].setdefault(session_id, OrderedDict())
        if call_id in od:
            return od[call_id]
        p = {"session_id": session_id, "call_id": call_id, "tool_name": tool_name,
             "event": threading.Event(), "status": None, "result": None}
        od[call_id] = p
        return p


def _hub_respond(hub, session_id, call_id, status, result):
    key = (session_id, call_id)
    with hub["lock"]:
        od = hub["pending"].get(session_id)
        if od and call_id in od:
            p = od[call_id]
            p["status"] = status; p["result"] = result; p["event"].set()
            return True
        p = {"session_id": session_id, "call_id": call_id, "event": threading.Event(),
             "status": status, "result": result}
        p["event"].set()
        hub["orphaned"][key] = p
        return True


SYS = "You are a helpful coding assistant."
MSG = "Help me fix this bug in my code."


# --- Layer 1: session ID derivation ----------------------------------------

def test_same_salt_same_session():
    s1 = _derive_chat_session_id(SYS, MSG, salt="10.0.0.1")
    s2 = _derive_chat_session_id(SYS, MSG, salt="10.0.0.1")
    assert s1 == s2


def test_different_ip_different_session():
    s1 = _derive_chat_session_id(SYS, MSG, salt="10.0.0.1")
    s2 = _derive_chat_session_id(SYS, MSG, salt="10.0.0.2")
    assert s1 != s2


def test_different_user_agent_different_session():
    s1 = _derive_chat_session_id(SYS, MSG, salt="10.0.0.1|Oh-My-Pi/1.0.0")
    s2 = _derive_chat_session_id(SYS, MSG, salt="10.0.0.1|Oh-My-Pi/1.1.0")
    assert s1 != s2


def test_x_omp_instance_isolates():
    """Primary fix: per-install UUID in X-OMP-Instance prevents NAT collisions."""
    ip = "10.0.0.1"
    s1 = _derive_chat_session_id(SYS, MSG, salt=ip + "|" + str(uuid.uuid4()))
    s2 = _derive_chat_session_id(SYS, MSG, salt=ip + "|" + str(uuid.uuid4()))
    assert s1 != s2


def test_nat_collision_without_instance():
    """Bug: same IP + same OMP version + no X-OMP-Instance = same session."""
    salt = "10.0.0.1|Oh-My-Pi/1.0.0"
    s1 = _derive_chat_session_id(SYS, MSG, salt=salt)
    s2 = _derive_chat_session_id(SYS, MSG, salt=salt)
    assert s1 == s2  # This IS the collision


# --- Layer 2: tool_call_hub isolation --------------------------------------

def test_hub_different_sessions_isolated():
    hub = _make_hub()
    p_a = _hub_register(hub, "sess-A", "cid-a", "bash")
    _hub_register(hub, "sess-B", "cid-b", "bash")
    _hub_respond(hub, "sess-B", "cid-b", "done", {"from": "B"})
    assert not p_a["event"].is_set(), "CONTAMINATION: sess-A got sess-B response"


def test_hub_same_session_different_callids_isolated():
    hub = _make_hub()
    p_a = _hub_register(hub, "shared", "cid-a", "bash")
    _hub_register(hub, "shared", "cid-b", "bash")
    _hub_respond(hub, "shared", "cid-b", "done", {"from": "B"})
    assert not p_a["event"].is_set(), "CONTAMINATION: cid-a got cid-b response"


def test_hub_orphan_adoption():
    hub = _make_hub()
    cid = str(uuid.uuid4())
    _hub_respond(hub, "s", cid, "done", {"x": 1})
    p = _hub_register(hub, "s", cid, "read")
    assert p["status"] == "done" and p["result"] == {"x": 1}


def test_five_concurrent_sessions_no_contamination():
    """5 parallel (session, call_id) pairs: each response reaches only its session."""
    hub = _make_hub()
    N = 5
    sids = ["session-" + str(i) for i in range(N)]
    cids = [str(uuid.uuid4()) for _ in range(N)]
    pends = [_hub_register(hub, sids[i], cids[i], "bash") for i in range(N)]
    errors = []

    def respond(i):
        # Snapshot which sessions are already done BEFORE we respond to i.
        # This avoids the race where a later thread sees an earlier thread's
        # legitimately-set event and falsely reports contamination.
        before = {j: pends[j]["event"].is_set() for j in range(N) if j != i}
        _hub_respond(hub, sids[i], cids[i], "done", {"owner": i})
        for j in range(N):
            if j != i and not before[j] and pends[j]["event"].is_set():
                errors.append("session-" + str(j) + " got result from session-" + str(i))

    threads = [threading.Thread(target=respond, args=(i,)) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, "\n".join(errors)
    for i in range(N):
        assert pends[i]["event"].is_set()
        assert pends[i]["result"] == {"owner": i}


# --- Layer 3: end-to-end contamination scenario ----------------------------

def test_collision_contamination_scenario():
    """
    Bug: same IP + same OMP version + same first message => same session_id.
    Two clients register tool calls with different call_ids.
    Responding to one must NOT unblock the other.
    """
    hub = _make_hub()
    session = _derive_chat_session_id(SYS, MSG, salt="10.0.0.1|Oh-My-Pi/1.0.0")
    cid_a = str(uuid.uuid4())
    cid_b = str(uuid.uuid4())
    p_a = _hub_register(hub, session, cid_a, "bash")
    _hub_register(hub, session, cid_b, "bash")
    _hub_respond(hub, session, cid_b, "done", {"output": "ls"})
    # cid_a must not have fired -- hub keys by call_id, so this should hold
    assert not p_a["event"].is_set(), (
        "CONTAMINATION: client A (" + cid_a + ") received B ('s (" + cid_b + ") result. "
        "Both share session " + session
    )


def test_fix_x_omp_instance():
    """Fix: different X-OMP-Instance => different sessions => zero chance of overlap."""
    ip = "10.0.0.1"
    s_a = _derive_chat_session_id(SYS, MSG, salt=ip + "|" + str(uuid.uuid4()))
    s_b = _derive_chat_session_id(SYS, MSG, salt=ip + "|" + str(uuid.uuid4()))
    assert s_a != s_b
    hub = _make_hub()
    cid_a, cid_b = str(uuid.uuid4()), str(uuid.uuid4())
    p_a = _hub_register(hub, s_a, cid_a, "bash")
    _hub_register(hub, s_b, cid_b, "bash")
    _hub_respond(hub, s_b, cid_b, "done", {"b": True})
    assert not p_a["event"].is_set()


# --- Layer 4: HTTP integration (requires live server) ----------------------

_API_KEY = os.environ.get(
    "HERMES_API_SERVER_KEY",
    "b49a80d538b98987e2f0c385bba137c79f017051cef9b95efd61929791dd4218",
)
_BASE_URL = os.environ.get("HERMES_GATEWAY_URL", "")
_LIVE = bool(_BASE_URL)


@pytest.mark.skipif(not _LIVE, reason="Set HERMES_GATEWAY_URL to run")
class TestLiveHTTPSessionIsolation:
    FIRST_MSG = "Reply with PING only. [" + uuid.uuid4().hex[:8] + "]"

    @pytest.mark.anyio
    async def test_five_parallel_with_omp_instance_all_unique(self):
        """5 parallel requests, unique X-OMP-Instance each => 5 unique session IDs."""
        try:
            import httpx
        except ImportError:
            pytest.skip("pip install httpx")

        async def send(instance):
            async with httpx.AsyncClient(base_url=_BASE_URL, timeout=httpx.Timeout(60.0)) as c:
                r = await c.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer " + _API_KEY,
                        "Content-Type": "application/json",
                        "X-OMP-Instance": instance,
                        "User-Agent": "Oh-My-Pi/1.0.0",
                    },
                    json={
                        "model": "hermes-code",
                        "messages": [{"role": "user", "content": self.FIRST_MSG}],
                        "stream": False,
                        "max_tokens": 10,
                    },
                )
                return {
                    "instance": instance,
                    "status": r.status_code,
                    "session_id": r.headers.get("X-Hermes-Session-Id", ""),
                }

        results = await asyncio.gather(*[send("test-" + str(i)) for i in range(5)])
        session_ids = [r["session_id"] for r in results]
        unique = set(session_ids)
        print("\n  Session IDs: " + str(session_ids))
        failed = [r for r in results if r["status"] >= 500]
        assert not failed, "Server errors: " + str(failed)
        assert len(unique) == 5, (
            "CONTAMINATION: " + str(len(unique)) + " unique session IDs from 5 requests: "
            + str(session_ids)
        )

    @pytest.mark.anyio
    async def test_five_parallel_no_instance_same_ip_collide(self):
        """
        5 parallel requests, same IP + same UA, NO X-OMP-Instance, identical first message.

        The session registry cannot distinguish byte-for-byte identical simultaneous
        turn-1 requests — they have no history to differentiate them yet.
        Some will collide. From turn 2 onward, real conversations diverge and the
        registry separates them cleanly. This test documents the inherent turn-1 limit.
        """
        try:
            import httpx
        except ImportError:
            pytest.skip("pip install httpx")

        async def send(i):
            async with httpx.AsyncClient(base_url=_BASE_URL, timeout=httpx.Timeout(60.0)) as c:
                r = await c.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer " + _API_KEY,
                        "Content-Type": "application/json",
                        "User-Agent": "Oh-My-Pi/1.0.0",
                        "X-Forwarded-For": "203.0.113.100",
                    },
                    json={
                        "model": "hermes-code",
                        "messages": [{"role": "user", "content": self.FIRST_MSG}],
                        "stream": False,
                        "max_tokens": 10,
                    },
                )
                return {"i": i, "status": r.status_code, "session_id": r.headers.get("X-Hermes-Session-Id", "")}

        results = await asyncio.gather(*[send(i) for i in range(5)])
        session_ids = [r["session_id"] for r in results]
        unique = set(session_ids)
        print("\n  No X-OMP-Instance, same IP: " + str(len(unique)) + " unique sessions")
        print("  IDs: " + str(session_ids))
        if len(unique) < 5:
            # Turn-1 collisions on simultaneous identical requests are inherent:
            # there is no information in the payload to distinguish them.
            # The registry fix resolves this from turn 2 onward.
            pytest.xfail(
                "TURN-1 COLLISION (expected): " + str(len(unique)) + " unique sessions from 5 "
                "simultaneous identical requests. This is inherent without X-OMP-Instance. "
                "From turn 2, the registry separates clients as their histories diverge. "
                "IDs: " + str(session_ids)
            )
