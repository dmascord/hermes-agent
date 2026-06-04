"""Shared tool call ID sanitization with bidirectional mapping.

Providers like arliai enforce a max length of 9 characters on tool_call_id
fields. This module provides:
  - ``sanitize_tool_call_id`` — one-shot sanitization (deterministic hash).
  - ``ToolCallIdMapper`` — bidirectional mapping so the connected client
    sees the same IDs it generated, while the upstream provider receives
    sanitized (≤9-char) IDs.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARLIAI_MAX_TOOL_ID_LENGTH = 9
_ARLIAI_BASE_URL_FRAGMENTS = ("arliai.com",)


# ---------------------------------------------------------------------------
# One-shot sanitization
# ---------------------------------------------------------------------------

def sanitize_tool_call_id(tool_id: str, *, max_length: int = 9) -> str:
    """Sanitize a single tool call ID.

    1. Replace non-``[a-zA-Z0-9_-]`` characters with ``_``.
    2. If the result exceeds *max_length*, deterministically hash to fit
       (using MD5 truncated to *max_length* hex chars).
    3. Return ``"tool_0"`` if the input is empty.
    """
    if not tool_id:
        return "tool_0"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)
    if len(sanitized) > max_length:
        sanitized = hashlib.md5(tool_id.encode()).hexdigest()[:max_length]
    return sanitized or "tool_0"


def needs_tool_id_sanitization(base_url: str = "") -> bool:
    """Return True if the provider enforces a short tool_call_id limit."""
    url = base_url.lower()
    return any(frag in url for frag in _ARLIAI_BASE_URL_FRAGMENTS)


# ---------------------------------------------------------------------------
# Bidirectional mapper
# ---------------------------------------------------------------------------

class ToolCallIdMapper:
    """Bidirectional original ↔ sanitized tool_call_id mapping.

    Usage::

        mapper = ToolCallIdMapper(max_length=9)

        # When building a request to arliai:
        messages = mapper.sanitize_messages(outgoing_messages)

        # When the provider responds with new tool_calls:
        response_tool_calls = mapper.unsanitize_tool_calls(raw_tool_calls)

    The mapper stores *all* IDs it sees (both directions) so that:
      - The upstream provider always receives sanitized IDs.
      - The client always receives original (unsanitized) IDs.
    """

    def __init__(self, *, max_length: int = ARLIAI_MAX_TOOL_ID_LENGTH) -> None:
        self.max_length = max_length
        self._to_sanitized: Dict[str, str] = {}
        self._to_original: Dict[str, str] = {}

    # ── single ID ────────────────────────────────────────────────────────

    def to_sanitized(self, tool_id: str) -> str:
        """Map original → sanitized.  If already short enough, return as-is."""
        if not tool_id:
            return "tool_0"
        if tool_id in self._to_sanitized:
            return self._to_sanitized[tool_id]
        if len(tool_id) <= self.max_length and re.fullmatch(r"[a-zA-Z0-9_-]+", tool_id):
            # Already compliant — keep it and remember the mapping.
            sanitized = tool_id
        else:
            sanitized = sanitize_tool_call_id(tool_id, max_length=self.max_length)
        self._to_sanitized[tool_id] = sanitized
        self._to_original[sanitized] = tool_id
        return sanitized

    def to_original(self, tool_id: str) -> str:
        """Map sanitized → original.  If no mapping exists, return as-is."""
        return self._to_original.get(tool_id, tool_id)

    # ── bulk helpers ─────────────────────────────────────────────────────

    def sanitize_tool_call(self, tc: dict) -> dict:
        """Return a copy of a single tool_call dict with sanitized ``id``."""
        if not isinstance(tc, dict):
            return tc
        original_id = tc.get("id", "")
        if not original_id:
            return tc
        sanitized = self.to_sanitized(original_id)
        if sanitized == original_id:
            return tc
        out = dict(tc)
        out["id"] = sanitized
        return out

    def sanitize_tool_result(self, msg: dict) -> dict:
        """Return a copy of a tool result message with sanitized ``tool_call_id``."""
        if not isinstance(msg, dict):
            return msg
        if msg.get("role") != "tool":
            return msg
        original_id = msg.get("tool_call_id", "")
        if not original_id:
            return msg
        sanitized = self.to_sanitized(original_id)
        if sanitized == original_id:
            return msg
        out = dict(msg)
        out["tool_call_id"] = sanitized
        return out

    def sanitize_messages(self, messages: list) -> list:
        """Sanitize tool_call_ids in a list of chat messages for upstream.

        Handles:
          - ``tool_calls`` arrays in assistant messages.
          - ``tool_call_id`` in tool result messages.
        """
        out = []
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            role = msg.get("role", "")
            if role == "assistant" and isinstance(msg.get("tool_calls"), list):
                new_tcs = [self.sanitize_tool_call(tc) for tc in msg["tool_calls"]]
                if new_tcs is not msg["tool_calls"]:
                    out.append({**msg, "tool_calls": new_tcs})
                else:
                    out.append(msg)
            elif role == "tool":
                sanitized = self.sanitize_tool_result(msg)
                out.append(sanitized)
            else:
                out.append(msg)
        return out

    def unsanitize_tool_calls(self, tool_calls: list) -> list:
        """Map sanitized IDs back to originals in the provider's response.

        When the upstream provider generates new tool_calls using sanitized
        IDs, the client needs to see the original IDs it sent earlier.
        """
        out = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                out.append(tc)
                continue
            raw_id = tc.get("id", "")
            original = self.to_original(raw_id)
            if original != raw_id:
                out.append({**tc, "id": original})
            else:
                out.append(tc)
        return out
