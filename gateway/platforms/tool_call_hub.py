"""Tool call coordination hub for API-server external tool responses.

Provides a tiny thread-safe in-memory registry for pending tool calls so
the API server (platform) can register a tool_call when it emits the
OpenAI-style function_call chunk to the client and later accept a POST
from the client with the tool result.  The agent thread waits on the
PendingCall.event and resumes when the response arrives.

This is intentionally simple and in-memory only: no persistence across
restarts.  It supports orphaned responses (client posted before the
agent registered) by stashing them until the agent registers the call.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional, Dict, Tuple


class PendingCall:
    def __init__(self, session_id: str, call_id: str, tool_name: Optional[str] = None, arguments: Optional[dict] = None):
        self.session_id = session_id
        self.call_id = call_id
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.event = threading.Event()
        self.status: Optional[str] = None
        self.result = None
        self.created_at = time.time()

    def set_result(self, status: str, result) -> None:
        self.status = status
        self.result = result
        self.event.set()


class _ToolCallHub:
    def __init__(self):
        self._lock = threading.Lock()
        # session_id -> OrderedDict[call_id, PendingCall]
        self._pending: Dict[str, OrderedDict[str, PendingCall]] = {}
        # Orphaned results keyed by (session_id, call_id)
        self._orphaned: Dict[Tuple[str, str], PendingCall] = {}

    def register_call(self, session_id: str, call_id: str, tool_name: Optional[str] = None, arguments: Optional[dict] = None) -> PendingCall:
        """Register and return a PendingCall for session_id/call_id.

        If a response was posted earlier (orphaned), adopt it so callers
        don't block.
        """
        key = (session_id, call_id)
        with self._lock:
            # Adopt orphaned response if present
            if key in self._orphaned:
                p = self._orphaned.pop(key)
                p.tool_name = p.tool_name or tool_name
                p.arguments = p.arguments or arguments or {}
                od = self._pending.setdefault(session_id, OrderedDict())
                od[call_id] = p
                return p

            od = self._pending.setdefault(session_id, OrderedDict())
            if call_id in od:
                # Update arguments if previously missing
                existing = od[call_id]
                if not existing.arguments and arguments:
                    existing.arguments = arguments
                if not existing.tool_name and tool_name:
                    existing.tool_name = tool_name
                return existing
            p = PendingCall(session_id, call_id, tool_name=tool_name, arguments=arguments)
            od[call_id] = p
            return p

    def set_response(self, session_id: str, call_id: str, status: str, result) -> bool:
        """Set a response for an existing pending call, or stash as orphan."""
        key = (session_id, call_id)
        with self._lock:
            od = self._pending.get(session_id)
            if od is not None and call_id in od:
                p = od[call_id]
                p.set_result(status, result)
                return True
            # Stash as orphaned result for later adoption
            p = PendingCall(session_id, call_id)
            p.set_result(status, result)
            self._orphaned[key] = p
            return True

    def get_pending_call(self, session_id: str, call_id: str) -> Optional[PendingCall]:
        with self._lock:
            return self._pending.get(session_id, {}).get(call_id)

    def pop_pending_call(self, session_id: str, call_id: str) -> Optional[PendingCall]:
        with self._lock:
            ses = self._pending.get(session_id)
            if not ses:
                return None
            return ses.pop(call_id, None)

    def pop_next_pending_for_tool(self, session_id: str, tool_name: Optional[str] = None) -> Optional[PendingCall]:
        """Pop the oldest pending call for session_id optionally matching tool_name.

        Returns None if no matching pending call exists.
        """
        with self._lock:
            ses = self._pending.get(session_id)
            if not ses:
                return None
            for cid, p in list(ses.items()):
                if tool_name is None or p.tool_name == tool_name:
                    return ses.pop(cid)
            return None


# Singleton hub
_hub = _ToolCallHub()


def register_call(session_id: str, call_id: str, tool_name: Optional[str] = None, arguments: Optional[dict] = None) -> PendingCall:
    return _hub.register_call(session_id, call_id, tool_name, arguments)


def set_response(session_id: str, call_id: str, status: str, result) -> bool:
    return _hub.set_response(session_id, call_id, status, result)


def get_pending_call(session_id: str, call_id: str) -> Optional[PendingCall]:
    return _hub.get_pending_call(session_id, call_id)


def pop_pending_call(session_id: str, call_id: str) -> Optional[PendingCall]:
    return _hub.pop_pending_call(session_id, call_id)


def pop_next_pending_for_tool(session_id: str, tool_name: Optional[str] = None) -> Optional[PendingCall]:
    return _hub.pop_next_pending_for_tool(session_id, tool_name)
