"""
OpenAI Codex device-code OAuth broker.

This module implements the headless device-code flow for authenticating with
OpenAI's Codex backend. It mirrors the pattern in
`omp-src/packages/ai/src/utils/oauth/openai-codex.ts`, and is designed for
server-side and k8s deployments where a browser redirect is not possible.

Usage:
    import agent.codex_oauth as codex_oauth
    tokens = codex_oauth.start_device_code_flow(label="my-account")
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Constants (from omp-src constants.ts)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
DEVICE_AUTH_URL = "https://auth.openai.com/codex/device"
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_POLL_INTERVAL_MS = 5_000
DEVICE_POLL_SAFETY_MARGIN_MS = 3_000
DEVICE_MAX_POLLS = 120
TOKEN_REQUEST_TIMEOUT_MS = 15_000


def start_device_code_flow(
    *,
    label: Optional[str] = None,
    on_progress: Optional[callable] = None,
    timeout: float = 900.0,
) -> Dict[str, Any]:
    """Run the OpenAI device-code login flow and return tokens.

    Args:
        label: Human-readable account label for logs/auth store.
        on_progress: Optional callback for status messages.
        timeout: Maximum wait time for user completion (default: 15 minutes).

    Returns:
        Dict with access_token, refresh_token, expires_at, and profile info.
    """
    logger.info("Starting OpenAI device-code OAuth flow")

    with httpx.Client(timeout=httpx.Timeout(TOKEN_REQUEST_TIMEOUT_MS / 1000)) as client:
        # Step 1: Request device code
        try:
            resp = client.post(
                DEVICE_USERCODE_URL,
                json={"client_id": CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to request device code: {exc}") from exc

        if resp.status_code != 200:
            raise RuntimeError(f"Device code request failed: {resp.status_code}")

        device_data = resp.json()
        device_auth_id = device_data.get("device_auth_id")
        user_code = device_data.get("user_code")
        poll_interval = max(3, int(device_data.get("interval", "5")))

        if not device_auth_id or not user_code:
            raise RuntimeError("Device code response missing required fields")

        if on_progress:
            on_progress(f"Visit {DEVICE_AUTH_URL} and enter code: {user_code}")

        logger.info(
            "Device authorization required: visit %s and enter code %s",
            DEVICE_AUTH_URL,
            user_code,
        )

        # Step 2: Poll for completion
        start_time = time.monotonic()
        poll_response = None

        while (time.monotonic() - start_time) < timeout:
            time.sleep(poll_interval if poll_response else min(poll_interval, 1))

            try:
                poll_resp = client.post(
                    DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    },
                    headers={"Content-Type": "application/json"},
                )
            except Exception as exc:
                logger.warning("Polling failed: %s", exc)
                continue

            if poll_resp.status_code == 200:
                poll_response = poll_resp.json()
                break
            elif poll_resp.status_code in {403, 404}:
                continue
            else:
                raise RuntimeError(f"Polling failed: {poll_resp.status_code}")

        if not poll_response:
            raise TimeoutError("Device authorization timed out")

        authorization_code = poll_response.get("authorization_code")
        code_verifier = poll_response.get("code_verifier")

        if not authorization_code or not code_verifier:
            raise RuntimeError("Device auth response missing authorization_code or code_verifier")

        # Step 3: Exchange code for tokens
        token_resp = client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": DEVICE_REDIRECT_URI,
                "client_id": CLIENT_ID,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if token_resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed: {token_resp.status_code}")

        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token or not refresh_token:
            raise RuntimeError("Token exchange response missing required fields")

        profile = _extract_token_profile(access_token)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": int(time.time()) + expires_in,
            "label": label or profile.get("email", "unknown"),
            "account_id": profile.get("account_id"),
            "email": profile.get("email"),
        }


def refresh_tokens(
    refresh_token: str,
    *,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Refresh existing Codex OAuth tokens.

    Args:
        refresh_token: The refresh token to exchange.
        timeout: HTTP timeout.

    Returns:
        Dict with updated access_token, refresh_token, and expires_at.
    """
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        response = client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Token refresh failed: {response.status_code}")

        token_data = response.json()
        access_token = token_data.get("access_token")
        new_refresh_token = token_data.get("refresh_token", refresh_token)
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            raise RuntimeError("Token refresh response missing access_token")

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "expires_at": int(time.time()) + expires_in,
        }


def _extract_token_profile(access_token: str) -> Dict[str, Any]:
    """Extract account profile from a Codex JWT."""
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return {}
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        auth_claim = claims.get("https://api.openai.com/auth", {})
        return {
            "account_id": auth_claim.get("chatgpt_account_id"),
            "email": claims.get("email"),
        }
    except Exception:
        return {}
