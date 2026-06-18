"""Xiaomi MiMo Free-tier authentication via client fingerprint bootstrap.

The free tier uses a machine-fingerprint-based auth flow:
1. Generate a fingerprint from hostname/OS/arch/CPU/username
2. POST to /api/free-ai/bootstrap with {"client": "<fingerprint>"} -> JWT
3. Use JWT as Bearer token for /api/free-ai/openai endpoint

JWTs are cached and refreshed before expiry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.xiaomimimo.com"
_BOOTSTRAP_URL = f"{_BASE_URL}/api/free-ai/bootstrap"
_CHAT_URL = f"{_BASE_URL}/api/free-ai/openai"
_FINGERPRINT_CACHE = Path.home() / ".hermes" / ".mimo_free_fingerprint"
_JWT_CACHE: Optional[dict] = None
_JWT_REFRESH_MARGIN = 300  # refresh 5 min before expiry


def _get_fingerprint() -> str:
    """Generate or load a persistent machine fingerprint."""
    if _FINGERPRINT_CACHE.is_file():
        try:
            return _FINGERPRINT_CACHE.read_text().strip()
        except Exception:
            pass

    hostname = os.uname().nodename
    cpu = platform.machine()
    os_name = os.uname().sysname
    try:
        username = os.getlogin()
    except Exception:
        username = os.getenv("USER", "unknown")

    raw = f"{hostname}|{os_name}|{cpu}|{cpu}|{username}"
    fp = hashlib.sha256(raw.encode()).hexdigest()

    try:
        _FINGERPRINT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _FINGERPRINT_CACHE.write_text(fp)
        _FINGERPRINT_CACHE.chmod(0o600)
    except Exception:
        pass

    return fp


def _decode_jwt_exp(jwt_str: str) -> Optional[float]:
    """Decode exp claim from a JWT without verification (for caching only)."""
    try:
        parts = jwt_str.split(".")
        if len(parts) < 2:
            return None
        import base64
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
    except Exception:
        pass
    return None


def bootstrap_jwt() -> str:
    """Get a valid JWT, bootstrapping if needed."""
    global _JWT_CACHE

    # Return cached JWT if still valid
    if _JWT_CACHE and _JWT_CACHE.get("exp", 0) > time.time() + _JWT_REFRESH_MARGIN:
        return _JWT_CACHE["jwt"]

    fingerprint = _get_fingerprint()

    try:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                _BOOTSTRAP_URL,
                json={"client": fingerprint},
                headers={"Content-Type": "application/json"},
            )
    except Exception as exc:
        raise RuntimeError(f"Mimo free bootstrap request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Mimo free bootstrap failed with status {resp.status_code}: "
            f"{resp.text[:200]}"
        )

    data = resp.json()
    jwt = data.get("jwt")
    if not jwt:
        raise RuntimeError("Mimo free bootstrap response missing jwt")

    exp = _decode_jwt_exp(jwt)
    if exp is None:
        exp = time.time() + 3600  # fallback 1 hour

    _JWT_CACHE = {"jwt": jwt, "exp": exp}
    logger.info("Mimo free JWT bootstrapped, expires in %ds", int(exp - time.time()))
    return jwt


def mimo_free_fetch(
    url: str,
    *,
    method: str = "POST",
    headers: Optional[dict] = None,
    content: Optional[bytes] = None,
    json_body: Optional[dict] = None,
    timeout: float = 120.0,
) -> httpx.Response:
    """Make an authenticated request to the MiMo free endpoint.

    Handles JWT bootstrap and auto-refresh on 401/403.
    """
    jwt = bootstrap_jwt()

    req_headers = dict(headers or {})
    req_headers["Authorization"] = f"Bearer {jwt}"
    req_headers["X-Mimo-Source"] = "hermes-gateway"

    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.request(
            method,
            url,
            headers=req_headers,
            content=content,
            json=json_body,
        )

    # If auth failed, refresh JWT and retry once
    if resp.status_code in (401, 403):
        global _JWT_CACHE
        _JWT_CACHE = None
        jwt = bootstrap_jwt()
        req_headers["Authorization"] = f"Bearer {jwt}"
        with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
            resp = client.request(
                method,
                url,
                headers=req_headers,
                content=content,
                json=json_body,
            )

    return resp


def get_chat_url() -> str:
    """Return the chat completions URL for MiMo free tier."""
    return _CHAT_URL


def is_mimo_auto_model(model: str) -> bool:
    """Check if a model name refers to MiMo Auto (free)."""
    return model.lower().replace("xiaomi/", "") == "mimo-auto"
