"""Tool name sanitization for GitHub Copilot API compatibility.

The Copilot API enforces a strict pattern on tool/function names:
``^[a-zA-Z0-9_-]+$``.  Tools defined with dots (e.g. ``browser.navigate``),
slashes, or other special characters are rejected with HTTP 400.

This module provides a single entry point ``sanitize_copilot_tool_names`` that
deep-copies an OpenAI-format tool list, rewrites function names to match the
Copilot pattern (replacing invalid characters with ``_``), and returns a
reverse-map so tool-call names can be mapped back to originals in responses.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

# Pattern for valid Copilot tool names — only alphanumeric, underscore, hyphen.
_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Characters NOT allowed in Copilot tool names — replaced with underscore.
_INVALID_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_tool_name(name: str) -> str:
    """Replace any character outside ``[a-zA-Z0-9_-]`` with ``_``."""
    return _INVALID_CHAR_RE.sub("_", name)


def _unique_name(candidate: str, used: set[str]) -> str:
    """Return ``candidate``, or ``candidate_N`` if already in ``used``."""
    if candidate not in used:
        return candidate
    idx = 1
    while f"{candidate}_{idx}" in used:
        idx += 1
    return f"{candidate}_{idx}"


def sanitize_copilot_tool_names(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Sanitize function names in an OpenAI-format tool list for Copilot API.

    Returns
    -------
    (sanitized_tools, reverse_map)
        sanitized_tools — a deep copy with names rewritten to match
        ``^[a-zA-Z0-9_-]+$``.  Returns the original list unchanged (identity)
        if no sanitization was needed.

        reverse_map — ``{sanitized_name: original_name}``.  Empty dict when no
        names needed changing.
    """
    if not tools:
        return tools, {}

    # Build the set of already-valid names that will stay in place.
    used_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name", "")).strip()
        if name and _VALID_NAME_RE.match(name):
            used_names.add(name)

    sanitized: List[Dict[str, Any]] = []
    reverse_map: Dict[str, str] = {}
    any_change = False

    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue

        fn = tool.get("function")
        if not isinstance(fn, dict):
            sanitized.append(tool)
            continue

        original_name = str(fn.get("name", "")).strip()
        if not original_name:
            sanitized.append(tool)
            continue

        # If already valid, pass through unchanged.
        if _VALID_NAME_RE.match(original_name):
            sanitized.append(tool)
            continue

        # Sanitize and ensure uniqueness.
        sanitized_name = _sanitize_tool_name(original_name)
        sanitized_name = _unique_name(sanitized_name, used_names)
        used_names.add(sanitized_name)

        reverse_map[sanitized_name] = original_name
        new_fn = {**fn, "name": sanitized_name}
        sanitized.append({**tool, "function": new_fn})
        any_change = True

    if not any_change:
        return tools, {}

    return sanitized, reverse_map
