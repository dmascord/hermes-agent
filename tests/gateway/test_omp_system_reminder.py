"""Tests for _unwrap_omp_system_reminder — stripping the OMP continuation wrapper.

oh-my-pi wraps every user text part on step > 1 with:

    <system-reminder>
    The user sent the following message:
    {original text}

    Please address this message and continue with your tasks.
    </system-reminder>

When this is forwarded verbatim to non-Claude providers via Hermes, the model
treats the wrapper as the driving prompt each turn and restarts work — the
looping bug.  _unwrap_omp_system_reminder strips the exact wrapper so the
provider sees only the original user text.

The function is called at parse time (line ~4226 of api_server.py), so tests
cover the core unwrapping logic independently.

Content format notes:
- OMP sends user messages as plain strings or multimodal arrays.
- Multimodal arrays contain a mix of plain strings (text) and dicts (images,
  audio).  The implementation checks isinstance(part, dict) and only touches
  dict parts with type="text"; plain string elements are also matched and
  unwrapped via the string branch.
- Source reference: opencode-src/packages/opencode/src/session/prompt.ts:1492
"""

import pytest
from gateway.platforms.api_server import (
    _unwrap_omp_system_reminder,
    _OMP_REMINDER_PREFIX,
    _OMP_REMINDER_SUFFIX,
)


ORIGINAL_TASK = "fix the ADF entity error in hermes-agent"


def _wrapped(text: str) -> str:
    """Return the exact OMP wire format for a given user text."""
    return _OMP_REMINDER_PREFIX + text + _OMP_REMINDER_SUFFIX


# ---------------------------------------------------------------------------
# String content
# ---------------------------------------------------------------------------

class TestStringUnwrap:
    def test_exact_wrapper_stripped(self):
        assert _unwrap_omp_system_reminder(_wrapped(ORIGINAL_TASK)) == ORIGINAL_TASK

    def test_idempotent_already_unwrapped(self):
        assert _unwrap_omp_system_reminder(ORIGINAL_TASK) == ORIGINAL_TASK

    def test_empty_string(self):
        assert _unwrap_omp_system_reminder("") == ""

    def test_only_prefix_no_suffix(self):
        """Partial match (suffix missing) — must not strip."""
        partial = _OMP_REMINDER_PREFIX + ORIGINAL_TASK
        assert _unwrap_omp_system_reminder(partial) is partial

    def test_only_suffix_no_prefix(self):
        """Partial match (prefix missing) — must not strip."""
        partial = ORIGINAL_TASK + _OMP_REMINDER_SUFFIX
        assert _unwrap_omp_system_reminder(partial) is partial

    def test_inner_text_contains_reminder_like_strings(self):
        """Inner text that contains the prefix string is preserved verbatim."""
        inner = f"use {_OMP_REMINDER_PREFIX} carefully"
        result = _unwrap_omp_system_reminder(_wrapped(inner))
        assert result == inner

    def test_whitespace_only_original(self):
        assert _unwrap_omp_system_reminder(_wrapped("   ")) == "   "

    def test_multiline_original(self):
        multi = "line one\nline two\n\nline four"
        assert _unwrap_omp_system_reminder(_wrapped(multi)) == multi

    def test_unicode_content(self):
        text = "Привет мир 你好世界 🔥"
        assert _unwrap_omp_system_reminder(_wrapped(text)) == text

    def test_plain_string_returns_same_object(self):
        """No-op case must return the same object, not a copy."""
        plain = "plain hello"
        assert _unwrap_omp_system_reminder(plain) is plain

    def test_wrapper_with_empty_inner_text(self):
        """Wrapper around an empty string yields empty string."""
        assert _unwrap_omp_system_reminder(_wrapped("")) == ""


# ---------------------------------------------------------------------------
# Multimodal array content (list of string and/or dict parts)
# ---------------------------------------------------------------------------

class TestMultimodalUnwrap:
    def test_empty_list(self):
        assert _unwrap_omp_system_reminder([]) == []

    def test_single_text_dict_part_unwrapped(self):
        """Dict part with type=text and wrapped text is unwrapped in-place."""
        part = {"type": "text", "text": _wrapped("analyse this image")}
        result = _unwrap_omp_system_reminder([part])
        assert result[0]["text"] == "analyse this image"

    def test_single_text_dict_part_plain_preserved(self):
        """Dict part with plain (non-wrapped) text is untouched."""
        part = {"type": "text", "text": ORIGINAL_TASK}
        result = _unwrap_omp_system_reminder([part])
        assert result[0]["text"] == ORIGINAL_TASK

    def test_dict_part_other_keys_preserved(self):
        """Unwrapping must not strip extra keys from the dict."""
        part = {"type": "text", "text": _wrapped("hi"), "extra_meta": True}
        result = _unwrap_omp_system_reminder([part])
        assert result[0]["extra_meta"] is True

    def test_image_dict_part_untouched(self):
        """Dict parts with non-text type are left completely unchanged."""
        img = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        result = _unwrap_omp_system_reminder([img])
        assert result[0] == img

    def test_audio_dict_part_untouched(self):
        audio = {"type": "input_audio", "input_audio": {"data": "YWJj", "format": "mp3"}}
        result = _unwrap_omp_system_reminder([audio])
        assert result[0] == audio

    def test_mixed_text_dict_and_image_dict(self):
        """Text dict gets unwrapped; image dict is preserved."""
        text_part = {"type": "text", "text": _wrapped("look at this")}
        img_part = {"type": "image_url", "image_url": {"url": "data:..."}}
        result = _unwrap_omp_system_reminder([text_part, img_part])
        assert result[0]["text"] == "look at this"
        assert result[1] == img_part

    def test_multiple_text_dicts_both_wrapped(self):
        parts = [
            {"type": "text", "text": _wrapped("task A")},
            {"type": "text", "text": _wrapped("task B")},
        ]
        result = _unwrap_omp_system_reminder(parts)
        assert result[0]["text"] == "task A"
        assert result[1]["text"] == "task B"

    def test_multiple_text_dicts_only_wrapped_one_changed(self):
        parts = [
            {"type": "text", "text": _wrapped("wrapped")},
            {"type": "text", "text": "plain"},
        ]
        result = _unwrap_omp_system_reminder(parts)
        assert result[0]["text"] == "wrapped"
        assert result[1]["text"] == "plain"

    def test_non_dict_elements_passthrough(self):
        """None and int list elements pass through unchanged."""
        result = _unwrap_omp_system_reminder([None, 42])
        assert result[0] is None
        assert result[1] == 42

    def test_original_dicts_not_mutated(self):
        """Unwrapping must return a new dict, not mutate the original."""
        part = {"type": "text", "text": _wrapped("original")}
        original_text = part["text"]
        _unwrap_omp_system_reminder([part])
        assert part["text"] == original_text  # untouched


# ---------------------------------------------------------------------------
# Non-string / non-list — pass through unchanged
# ---------------------------------------------------------------------------

class TestNonStringContent:
    def test_none_returns_none(self):
        assert _unwrap_omp_system_reminder(None) is None

    def test_integer_passthrough(self):
        assert _unwrap_omp_system_reminder(42) == 42

    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert _unwrap_omp_system_reminder(d) == d


# ---------------------------------------------------------------------------
# Constant correctness — guard against accidental drift from OMP source
# ---------------------------------------------------------------------------

class TestOmpConstants:
    def test_prefix_and_suffix_non_empty(self):
        assert _OMP_REMINDER_PREFIX
        assert _OMP_REMINDER_SUFFIX

    def test_prefix_contains_opening_tag(self):
        assert "<system-reminder>" in _OMP_REMINDER_PREFIX

    def test_suffix_contains_closing_tag(self):
        assert "</system-reminder>" in _OMP_REMINDER_SUFFIX

    def test_constants_match_opencode_source(self):
        """Exact strings from opencode-src/packages/opencode/src/session/prompt.ts:1492-1499.
        If OMP changes its wrapper format, this test will catch the drift."""
        assert _OMP_REMINDER_PREFIX == "<system-reminder>\nThe user sent the following message:\n"
        assert _OMP_REMINDER_SUFFIX == "\n\nPlease address this message and continue with your tasks.\n</system-reminder>"
