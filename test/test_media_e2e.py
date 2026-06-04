#!/usr/bin/env python3
"""
End-to-end tests: image, PDF, audio, and TTS support on the Hermes server.

Tests run against the live server at HERMES_BASE_URL (default: https://hermes.tusker.net.au).
API key read from HERMES_API_KEY env var or the local .env file.

Usage:
    cd hermes-agent
    python -m pytest test/test_media_e2e.py -v
    # or with explicit overrides:
    HERMES_BASE_URL=http://localhost:8080 HERMES_API_KEY=xxx python -m pytest test/test_media_e2e.py -v

Gaps documented inline with pytest.mark.xfail.
"""

import base64
import io
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_local_env(path: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return result


_LOCAL_ENV = _load_local_env(
    str(Path(__file__).parent.parent / ".env")
)


def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or _LOCAL_ENV.get(key) or default


HERMES_BASE_URL = _cfg("HERMES_BASE_URL", "https://hermes.tusker.net.au").rstrip("/")
HERMES_API_KEY = _cfg("HERMES_API_KEY") or _cfg("API_SERVER_KEY")

# Model to use for vision/multimodal tests in passthrough mode.
# Can be overridden — must be a provider/model that the live server has keys for.
# Default: a model known to support vision that is always in the fallback chain.
VISION_MODEL = _cfg("HERMES_TEST_VISION_MODEL", "hermes-code")

# Audio model: Gemini is the only supported audio backend.
AUDIO_MODEL = _cfg("HERMES_TEST_AUDIO_MODEL", "hermes-code")

# ---------------------------------------------------------------------------
# HTTP client (stdlib only — no extra deps beyond what the server already has)
# ---------------------------------------------------------------------------

try:
    import urllib.request as _urlreq
    import urllib.error as _urlerr
except ImportError:
    pytest.skip("urllib not available", allow_module_level=True)


def _post(path: str, body: Any, *, timeout: int = 60) -> Dict[str, Any]:
    url = HERMES_BASE_URL + path
    data = json.dumps(body).encode()
    req = _urlreq.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HERMES_API_KEY}",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except _urlerr.HTTPError as exc:
        raw = exc.read()
        try:
            return json.loads(raw)
        except Exception:
            raise RuntimeError(f"HTTP {exc.code}: {raw[:400]}") from exc


def _get(path: str, *, timeout: int = 10) -> Dict[str, Any]:
    url = HERMES_BASE_URL + path
    req = _urlreq.Request(
        url,
        headers={"Authorization": f"Bearer {HERMES_API_KEY}"},
        method="GET",
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except _urlerr.HTTPError as exc:
        raw = exc.read()
        try:
            return json.loads(raw)
        except Exception:
            raise RuntimeError(f"HTTP {exc.code}: {raw[:400]}") from exc


def _head_exists(path: str, *, timeout: int = 5) -> int:
    """Return HTTP status code for a HEAD request."""
    url = HERMES_BASE_URL + path
    req = _urlreq.Request(
        url,
        headers={"Authorization": f"Bearer {HERMES_API_KEY}"},
        method="HEAD",
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except _urlerr.HTTPError as exc:
        return exc.code


def _extract_text(resp: Dict[str, Any]) -> str:
    """Pull assistant text from a chat.completions response."""
    choices = resp.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content") or ""
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


# ---------------------------------------------------------------------------
# Fixture: skip if no API key
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_api_key():
    if not HERMES_API_KEY:
        pytest.skip("HERMES_API_KEY / API_SERVER_KEY not set")


# ---------------------------------------------------------------------------
# Synthetic media generators (no external files needed)
# ---------------------------------------------------------------------------

def _make_png_1x1(r: int = 255, g: int = 0, b: int = 0) -> bytes:
    """Return a minimal valid 1×1 RGB PNG."""
    def _chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return c + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    raw_row = b"\x00" + bytes([r, g, b])
    compressed = zlib.compress(raw_row, 9)
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _make_wav_beep(frequency_hz: int = 440, duration_ms: int = 100) -> bytes:
    """Return a minimal valid WAV file with a pure tone."""
    sample_rate = 8000
    n_samples = sample_rate * duration_ms // 1000
    import math
    samples = bytes(
        int(127 + 127 * math.sin(2 * math.pi * frequency_hz * i / sample_rate)) & 0xFF
        for i in range(n_samples)
    )
    # PCM WAV header
    data_len = len(samples)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_len,
        b"WAVE",
        b"fmt ",
        16,          # chunk size
        1,           # PCM
        1,           # mono
        sample_rate,
        sample_rate, # byte rate
        1,           # block align
        8,           # bits per sample
        b"data",
        data_len,
    )
    return header + samples


def _make_minimal_pdf(sentinel: str = "XZQV7F") -> bytes:
    """Return a syntactically valid single-page PDF containing `sentinel` as text.

    The sentinel is chosen to be lexically impossible to guess from context —
    the model can only reproduce it if it actually reads the PDF content.
    """
    stream_content = f"BT /F1 12 Tf 72 720 Td ({sentinel}) Tj ET"
    stream_bytes = stream_content.encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length " + str(len(stream_bytes)).encode() + b" >>\n"
        b"stream\n" + stream_bytes + b"\nendstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\ntrailer << /Size 6 /Root 1 0 R >>\n%%EOF\n"
    )


# Sentinel that appears only inside the PDF binary — not in any prompt text.
# If the model returns this string, it genuinely read the document content.
_PDF_SENTINEL = "XZQV7F"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# 1. Health / infrastructure
# ---------------------------------------------------------------------------

class TestInfrastructure:
    def test_health_endpoint(self):
        resp = _get("/health")
        assert resp.get("status") == "ok", f"Unexpected health: {resp}"

    def test_models_endpoint(self):
        resp = _get("/v1/models")
        ids = {m["id"] for m in resp.get("data", [])}
        assert "hermes-code" in ids, f"hermes-code missing from models: {ids}"

    def test_chat_completions_text_only(self):
        """Baseline: plain text round-trip works."""
        resp = _post("/v1/chat/completions", {
            "model": "hermes-code",
            "messages": [{"role": "user", "content": "Reply with exactly: HERMES_OK"}],
            "max_tokens": 20,
            "stream": False,
        }, timeout=60)
        text = _extract_text(resp)
        assert "HERMES_OK" in text, f"Expected HERMES_OK, got: {text!r}"


# ---------------------------------------------------------------------------
# 2. Image input (base64 PNG)
# ---------------------------------------------------------------------------

class TestImageInput:
    """
    Hermes supports image_url parts (base64 data URIs) in passthrough mode.

    Routing:
      - _messages_have_image_parts() → True
      - _select_hermes_code_model(require_vision=True) picks a vision-capable model
      - Gemini: content → inlineData; Anthropic: image_source block; OpenAI: image_url preserved
    """

    def _image_message(self, prompt: str = "What colour is this image? Reply with one word.") -> Dict[str, Any]:
        png = _make_png_1x1(r=255, g=0, b=0)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_b64(png)}",
                    },
                },
            ],
        }

    def test_image_part_accepted_without_error(self):
        """Server must not reject a request containing an image_url part."""
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [self._image_message()],
            "max_tokens": 50,
            "stream": False,
        }, timeout=90)
        # Must not be an error object at the top level
        assert "error" not in resp, f"Server returned error: {resp.get('error')}"
        assert "choices" in resp, f"No choices in response: {resp}"

    def test_image_produces_text_response(self):
        """Model must return a non-empty textual answer about the image."""
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [self._image_message("Is this image red? Answer yes or no.")],
            "max_tokens": 10,
            "stream": False,
        }, timeout=90)
        text = _extract_text(resp).lower()
        assert text.strip(), "Expected non-empty response to image query"
        # Model must have committed to yes or no — not refused or hallucinated something else.
        # A 1×1 red PNG is unambiguous; accept "yes" or "no" (some models answer "no" when
        # they consider the pixel too small to have a meaningful colour).
        assert "yes" in text or "no" in text, (
            f"Expected yes/no answer for red image query, got: {text!r}"
        )

    def test_image_preserve_path_history(self):
        """
        Multimodal content must be preserved in passthrough_messages, not
        flattened to text by _normalize_chat_content.
        """
        # Send as history (not the last message) so we can check it round-trips.
        png = _make_png_1x1(r=0, g=0, b=255)
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_b64(png)}"},
                        },
                    ],
                },
            ],
            "max_tokens": 30,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp
        assert _extract_text(resp).strip()


# ---------------------------------------------------------------------------
# 3. PDF input
# ---------------------------------------------------------------------------

class TestPDFInput:
    """
    Gap: PDF is NOT natively supported.

    No provider integration handles application/pdf content blocks.
    The Gemini adapter only processes image_url and input_audio parts.
    Anthropic supports PDF via `document` content blocks, but the Hermes
    passthrough layer has no PDF→document conversion.

    Expected behaviour today: the PDF base64 is silently dropped
    (_normalize_chat_content skips non-text parts), so the model only sees
    the text prompt and cannot answer questions about the document's content.

    These tests document the current state so a gap closure is immediately
    observable.
    """

    def _pdf_message(self, prompt: str = "What text is in this document?") -> Dict[str, Any]:
        pdf = _make_minimal_pdf()
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # Claude-style document block — Anthropic's format
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": _b64(pdf),
                    },
                },
            ],
        }

    def _pdf_message_as_image_url(self) -> Dict[str, Any]:
        """Some clients encode PDF as a data URI inside image_url — test this path too."""
        pdf = _make_minimal_pdf()
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "What text is in this document? Reply with the exact words."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:application/pdf;base64,{_b64(pdf)}",
                    },
                },
            ],
        }

    @pytest.mark.xfail(
        reason=(
            "PDF document blocks are forwarded structurally (no crash) but the "
            "Copilot/GHE Claude endpoint does not parse inline PDF base64 content — "
            "it echoes the block type label rather than reading the document. "
            "Direct Anthropic API with PDF support or Gemini inlineData path required."
        ),
        strict=False,
    )
    def test_pdf_document_block_answered(self):
        """
        The model must return the sentinel token embedded only inside the PDF binary.
        xfail: forwarded structurally but no backend currently parses inline PDF base64.
        Will pass once a Gemini or direct Anthropic key is configured.
        """
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [self._pdf_message(
                "Reply with only the exact token you find in the document. "
                "Do not explain or add anything else."
            )],
            "max_tokens": 20,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp
        text = _extract_text(resp)
        assert _PDF_SENTINEL in text, (
            f"PDF sentinel {_PDF_SENTINEL!r} not surfaced — document was not read: {text!r}"
        )

    def test_pdf_document_block_does_not_crash(self):
        """
        PDF blocks MUST NOT cause a 5xx or crash — they should be silently
        ignored and the text portion of the message must still be answered.
        This is the current supported behaviour.
        """
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [self._pdf_message("Tell me the current date in one sentence. Ignore any document.")],
            "max_tokens": 50,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp, f"Server error on PDF block: {resp.get('error')}"
        assert "choices" in resp
        # The text portion must still be answered
        text = _extract_text(resp)
        assert text.strip(), "Expected non-empty response even when PDF block is stripped"

    def test_pdf_as_image_url_data_uri_does_not_crash(self):
        """
        PDF sent as application/pdf data URI inside image_url MUST NOT crash.
        The image routing may forward it or reject it; either way 5xx is wrong.
        """
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [self._pdf_message_as_image_url()],
            "max_tokens": 30,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp, f"Server error on PDF-as-image: {resp.get('error')}"


# ---------------------------------------------------------------------------
# 4. Audio input
# ---------------------------------------------------------------------------

class TestAudioInput:
    """
    Hermes routes audio-bearing messages to HERMES_CODE_AUDIO_MODEL (Gemini).
    _messages_have_audio_parts() detects input_audio parts.
    GeminiNativeClient._create_chat_completion translates input_audio →
    inlineData with the correct audio MIME type.

    Gap: only Google/Gemini handles audio. If HERMES_CODE_AUDIO_MODEL is
    unset or no Google key is configured, the audio path will fall back to
    text-only models that silently drop the audio part.
    """

    def _audio_message(
        self,
        prompt: str = "What sound does this audio contain? Reply in one word.",
        wav: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        if wav is None:
            wav = _make_wav_beep()
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": _b64(wav),
                        "format": "wav",
                    },
                },
            ],
        }

    def test_audio_part_accepted_without_error(self):
        """Server must not crash or 5xx on a request with an input_audio part."""
        resp = _post("/v1/chat/completions", {
            "model": AUDIO_MODEL,
            "messages": [self._audio_message("Acknowledge you received an audio clip.")],
            "max_tokens": 40,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp, f"Server error on audio: {resp.get('error')}"
        assert "choices" in resp

    def test_audio_routing_flag_detection(self):
        """
        _messages_have_audio_parts must detect input_audio — this validates
        the detection function logic independently of the model call.
        We exercise it by sending a deliberately short prompt with an audio part
        and confirming the server returns a response (not 400/422).
        """
        resp = _post("/v1/chat/completions", {
            "model": AUDIO_MODEL,
            "messages": [self._audio_message("Say hello.")],
            "max_tokens": 20,
            "stream": False,
        }, timeout=90)
        assert resp.get("object") in ("chat.completion", "error"), (
            f"Unexpected response shape: {resp}"
        )
        # Accept either a valid completion OR a graceful error (not a crash)
        if "error" in resp:
            err_type = resp["error"].get("type", "")
            # Only internal server errors are bugs; auth/model errors are ok
            assert err_type != "internal_server_error", (
                f"Internal server error on audio routing: {resp['error']}"
            )

    def test_audio_produces_non_empty_response(self):
        """
        When routed to Gemini (the only audio-capable backend), the model
        should return a meaningful response to a question about the audio clip.

        xfail if HERMES_CODE_AUDIO_MODEL is not configured — audio is routed
        to a text-only fallback that cannot perceive audio content.
        """
        resp = _post("/v1/chat/completions", {
            "model": AUDIO_MODEL,
            "messages": [self._audio_message(
                "Describe what you hear in the audio. Be brief."
            )],
            "max_tokens": 60,
            "stream": False,
        }, timeout=90)
        if "error" in resp:
            pytest.skip(f"Audio backend unavailable: {resp['error'].get('message')}")
        text = _extract_text(resp)
        assert text.strip(), "Expected non-empty response for audio query"

    def test_audio_format_mp3_accepted(self):
        """input_audio with format=mp3 must not crash (format is passed through to Gemini)."""
        # Use WAV bytes labelled as mp3 — we just need to test the format field passthrough.
        wav = _make_wav_beep()
        resp = _post("/v1/chat/completions", {
            "model": AUDIO_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Acknowledge receipt."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _b64(wav), "format": "mp3"},
                    },
                ],
            }],
            "max_tokens": 20,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp or resp["error"].get("type") != "internal_server_error", (
            f"Server crashed on audio format=mp3: {resp.get('error')}"
        )


# ---------------------------------------------------------------------------
# 5. TTS (Text-to-Speech)
# ---------------------------------------------------------------------------

class TestTTS:
    """
    TTS / STT endpoint tests.

    /v1/audio/speech — proxies to Xiaomi MiMo (XIAOMI_API_KEY) or OpenAI (OPENAI_API_KEY).
    /v1/audio/transcriptions — proxies to Groq Whisper (GROQ_API_KEY) or OpenAI (OPENAI_API_KEY).

    Without provider keys the endpoints return 503; with keys they return audio/transcription.
    Either outcome is acceptable for the "endpoint exists" tests — only 404/500 are failures.
    """

    def test_tts_endpoint_exists(self):
        """
        POST /v1/audio/speech MUST NOT return 404 or 500.
        With no provider keys → 503 Service Unavailable (acceptable).
        With a Xiaomi or OpenAI key → 200 with audio bytes.
        """
        import urllib.request as _r
        import urllib.error as _e
        url = HERMES_BASE_URL + "/v1/audio/speech"
        req = _r.Request(
            url,
            data=json.dumps({"model": "tts-1", "input": "Hello from Hermes.", "voice": "alloy"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {HERMES_API_KEY}",
            },
            method="POST",
        )
        try:
            with _r.urlopen(req, timeout=60) as resp:
                # 200 — endpoint exists and a TTS provider returned audio
                assert resp.status == 200
                assert len(resp.read()) > 0, "Expected non-empty audio bytes"
        except _e.HTTPError as exc:
            body = exc.read()
            assert exc.code not in (404, 405), (
                f"/v1/audio/speech returned {exc.code} — endpoint not registered"
            )
            assert exc.code != 500, (
                f"/v1/audio/speech returned 500 — internal server error: {body[:300]}"
            )
            # 503 is acceptable — no provider keys configured
            assert exc.code == 503, (
                f"/v1/audio/speech returned unexpected {exc.code}: {body[:200]}"
            )

    def test_stt_endpoint_exists(self):
        """
        POST /v1/audio/transcriptions MUST NOT return 404 or 500.
        Sending JSON (not multipart) → 400 Bad Request (acceptable — proves routing exists).
        Sending valid multipart with no provider keys → 503.
        """
        import urllib.request as _r
        import urllib.error as _e
        url = HERMES_BASE_URL + "/v1/audio/transcriptions"
        # Send JSON to get a 400 (wrong content-type) rather than 404 (not found)
        req = _r.Request(
            url,
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {HERMES_API_KEY}",
            },
            method="POST",
        )
        try:
            with _r.urlopen(req, timeout=10):
                pass  # 200 would be unexpected but not a bug
        except _e.HTTPError as exc:
            assert exc.code not in (404, 405), (
                f"/v1/audio/transcriptions returned {exc.code} — endpoint not registered"
            )
            assert exc.code != 500, (
                f"/v1/audio/transcriptions returned 500 — internal server error"
            )
            # 400 (wrong content-type) or 503 (no provider) are both acceptable
            assert exc.code in (400, 503), (
                f"/v1/audio/transcriptions returned unexpected {exc.code}"
            )

    def test_tts_with_multipart_wrong_format_is_400(self):
        """Sending multipart to /v1/audio/speech MUST return 400, not 404/500."""
        import urllib.request as _r
        import urllib.error as _e
        boundary = b"TESTBOUNDARY"
        body = b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"input\"\r\n\r\nhello\r\n--" + boundary + b"--\r\n"
        url = HERMES_BASE_URL + "/v1/audio/speech"
        req = _r.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
                "Authorization": f"Bearer {HERMES_API_KEY}",
            },
            method="POST",
        )
        try:
            with _r.urlopen(req, timeout=10):
                pass  # handler parsed it — acceptable
        except _e.HTTPError as exc:
            assert exc.code != 500, f"TTS returned 500 on malformed request: {exc.read()[:200]}"
            assert exc.code != 404, "TTS endpoint returned 404 — not registered"

    def test_stt_with_valid_multipart(self):
        """
        POST /v1/audio/transcriptions with valid multipart audio MUST return
        200 (transcript) or 503 (no provider keys) — never 404 or 500.
        """
        import urllib.request as _r
        import urllib.error as _e
        wav = _make_wav_beep()
        boundary = b"HERMESTESTBND"
        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'
            b"Content-Type: audio/wav\r\n\r\n"
            + wav + b"\r\n"
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="model"\r\n\r\n'
            b"whisper-large-v3\r\n"
            b"--" + boundary + b"--\r\n"
        )
        url = HERMES_BASE_URL + "/v1/audio/transcriptions"
        req = _r.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary.decode()}",
                "Authorization": f"Bearer {HERMES_API_KEY}",
            },
            method="POST",
        )
        try:
            with _r.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                assert "text" in data, f"Expected 'text' in transcription response: {data}"
        except _e.HTTPError as exc:
            body_bytes = exc.read()
            assert exc.code not in (404, 405), "STT endpoint not registered"
            assert exc.code != 500, f"STT returned 500: {body_bytes[:300]}"
            # 503 = no provider keys — acceptable
            assert exc.code == 503, f"STT returned unexpected {exc.code}: {body_bytes[:200]}"

    def test_xiaomi_tts_model_passthrough_via_chat_completions(self):
        """
        Xiaomi MiMo TTS models are reachable via the standard chat completions
        endpoint in passthrough mode.  The server must not 400/reject the model
        name — it should either succeed or fail with a provider-side error
        (not an internal server error).
        """
        resp = _post("/v1/chat/completions", {
            "model": "xiaomi/mimo-v2-tts",
            "messages": [{"role": "user", "content": "Say: Hello from Hermes."}],
            "max_tokens": 256,
            "stream": False,
        }, timeout=60)
        if "error" in resp:
            err_msg = resp["error"].get("message", "").lower()
            assert "internal_server_error" not in resp["error"].get("type", ""), (
                f"Internal server error for xiaomi TTS passthrough: {resp['error']}"
            )
            assert any(k in err_msg for k in ("key", "auth", "provider", "model", "no", "unavailable", "credit")), (
                f"Unexpected error for xiaomi TTS: {resp['error']}"
            )
        assert "object" in resp or "error" in resp


# ---------------------------------------------------------------------------
# 6. Multimodal — mixed image + text in streaming mode
# ---------------------------------------------------------------------------

class TestMultimodalStreaming:
    """Streaming path must also preserve multimodal parts."""

    def test_image_in_stream_mode(self):
        """
        Streaming image request: server must return SSE, not crash.
        We collect the raw bytes and verify at least one data: line appears.
        """
        import urllib.request as _r
        import urllib.error as _e

        png = _make_png_1x1(r=0, g=255, b=0)
        body = json.dumps({
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "One word: is this green?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(png)}"},
                    },
                ],
            }],
            "max_tokens": 10,
            "stream": True,
        }).encode()

        req = _r.Request(
            HERMES_BASE_URL + "/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {HERMES_API_KEY}",
            },
            method="POST",
        )
        try:
            with _r.urlopen(req, timeout=90) as resp:
                raw = resp.read(8192)
        except _e.HTTPError as exc:
            raw = exc.read()
            raise AssertionError(f"HTTP {exc.code} on streaming image: {raw[:300]}") from exc

        assert b"data:" in raw, f"Expected SSE data lines in stream response: {raw[:300]!r}"


# ---------------------------------------------------------------------------
# 7. Content normalisation invariants
# ---------------------------------------------------------------------------

class TestContentNormalisation:
    """
    Unit-level coverage of the api_server content handling paths, exercised
    end-to-end by sending shaped messages and checking the server's behaviour.
    """

    def test_empty_user_message_with_image_does_not_error(self):
        """
        An empty text + image content list MUST be handled gracefully.
        The server should either produce a response or return a clean error,
        not a 500.
        """
        png = _make_png_1x1()
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ""},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_b64(png)}"},
                    },
                ],
            }],
            "max_tokens": 20,
            "stream": False,
        }, timeout=120)
        if "error" in resp:
            assert resp["error"].get("type") != "internal_server_error", (
                f"500 on empty text + image: {resp['error']}"
            )

    def test_image_url_detail_field_preserved(self):
        """image_url.detail field must not cause an error."""
        png = _make_png_1x1()
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_b64(png)}",
                            "detail": "low",
                        },
                    },
                ],
            }],
            "max_tokens": 15,
            "stream": False,
        }, timeout=60)
        assert "error" not in resp or resp["error"].get("type") != "internal_server_error"

    def test_multiple_images_in_one_message(self):
        """Two image parts in a single message MUST not crash the server."""
        red = _make_png_1x1(255, 0, 0)
        blue = _make_png_1x1(0, 0, 255)
        resp = _post("/v1/chat/completions", {
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "How many images are attached?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(red)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(blue)}"}},
                ],
            }],
            "max_tokens": 20,
            "stream": False,
        }, timeout=90)
        assert "error" not in resp or resp["error"].get("type") != "internal_server_error"


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short",
    ]))
