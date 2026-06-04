"""GitHub Copilot authentication utilities.

Implements the OAuth device code flow used by the Copilot CLI and handles
token validation/exchange for the Copilot API.

Token type support (per GitHub docs):
  gho_          OAuth token           ✓  (default via copilot login)
  github_pat_   Fine-grained PAT      ✓  (needs Copilot Requests permission)
  ghu_          GitHub App token      ✓  (via environment variable)
  ghp_          Classic PAT           ✗  NOT SUPPORTED

Credential search order (matching Copilot CLI behaviour):
  1. COPILOT_GITHUB_TOKEN env var
  2. GH_TOKEN env var
  3. GITHUB_TOKEN env var
  4. gh auth token  CLI fallback
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OAuth device code flow constants (same client ID as opencode/Copilot CLI)
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Validate that a token is usable with the Copilot API.

    Returns (valid, message).
    """
    token = token.strip()
    if not token:
        return False, "Empty token"

    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use one of:\n"
            "  → `copilot login` or `hermes model` to authenticate via OAuth\n"
            "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n"
            "  → `gh auth login` with the default device code flow (produces gho_* tokens)"
        )

    return True, "OK"


# ─── Cached Copilot Token Resolution ──────────────────────────────────────
# The `gh auth token` subprocess is expensive and the result is stable for
# the lifetime of a `gh` session.  Cache the resolved raw token in memory
# and only re-resolve when the exchanged Copilot API token expires.
_cached_copilot_raw_token: str = ""
_cached_copilot_raw_source: str = ""


def resolve_and_cache_copilot_token(
    *, base_url: str | None = None
) -> tuple[str, str]:
    """Return ``(exchanged_api_token, source)`` with in-process caching.

    On the first call, runs ``resolve_copilot_token()`` (which may spawn
    ``gh auth token``) and exchanges the result for a short-lived Copilot
    API token via ``get_copilot_api_token()``.  Subsequent calls reuse the
    cached exchanged token while it remains valid (``_jwt_cache`` inside
    ``exchange_copilot_token()`` handles expiry and refresh automatically).

    Only re-runs ``gh auth token`` when the cached raw token fails to
    exchange — this avoids the ~200ms subprocess on every pool load.
    """
    global _cached_copilot_raw_token, _cached_copilot_raw_source

    if _cached_copilot_raw_token:
        # Try the cached raw token first — exchange_copilot_token has its
        # own _jwt_cache with refresh-margin expiry checking, so this is
        # cheap when the token is still valid.
        api_token = get_copilot_api_token(_cached_copilot_raw_token, base_url=base_url)
        # get_copilot_api_token returns the raw token unchanged on failure.
        if api_token and api_token != _cached_copilot_raw_token:
            return api_token, _cached_copilot_raw_source
        # Exchange failed — raw token may be stale (e.g. revoked or
        # re-authenticated via `gh auth login`).  Fall through to re-resolve.

    _cached_copilot_raw_token, _cached_copilot_raw_source = resolve_copilot_token()
    api_token = get_copilot_api_token(_cached_copilot_raw_token, base_url=base_url)
    return api_token, _cached_copilot_raw_source


def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a GitHub token suitable for Copilot API use.

    Returns (token, source) where source describes where the token came from.
    Raises ValueError if only a classic PAT is available.
    """
    # 1. Check env vars in priority order
    for env_var in COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if val:
            valid, msg = validate_copilot_token(val)
            if not valid:
                logger.warning(
                    "Token from %s is not supported: %s", env_var, msg
                )
                continue
            return val, env_var

    # 2. Fall back to gh auth token
    token = _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            raise ValueError(
                f"Token from `gh auth token` is a classic PAT (ghp_*). {msg}"
            )
        return token, "gh auth token"

    return "", ""


def _gh_cli_candidates() -> list[str]:
    """Return candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = []

    resolved = shutil.which("gh")
    if resolved:
        candidates.append(resolved)

    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if candidate in candidates:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            candidates.append(candidate)

    return candidates


def _try_gh_cli_token() -> Optional[str]:
    """Return a token from ``gh auth token`` when the GitHub CLI is available.

    When COPILOT_GH_HOST is set, passes ``--hostname`` so gh returns the
    correct host's token.  Also strips GITHUB_TOKEN / GH_TOKEN from the
    subprocess environment so ``gh`` reads from its own credential store
    (hosts.yml) instead of just echoing the env var back.
    """
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()

    # Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}

    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                env=clean_env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


# ─── OAuth Device Code Flow ────────────────────────────────────────────────

def copilot_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot.

    Prints instructions for the user, polls for completion, and returns
    the OAuth access token on success, or None on failure/cancellation.

    This replicates the flow used by opencode and the Copilot CLI.
    """
    import urllib.request
    import urllib.parse

    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"

    # Step 1: Request device code
    data = urllib.parse.urlencode({
        "client_id": COPILOT_OAUTH_CLIENT_ID,
        "scope": "read:user",
    }).encode()

    req = urllib.request.Request(
        device_code_url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "HermesAgent/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            device_data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)

    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None

    # Step 2: Show instructions
    print()
    print(f"  Open this URL in your browser: {verification_uri}")
    print(f"  Enter this code: {user_code}")
    print()
    print("  Waiting for authorization...", end="", flush=True)

    # Step 3: Poll for completion
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)

        poll_data = urllib.parse.urlencode({
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }).encode()

        poll_req = urllib.request.Request(
            access_token_url,
            data=poll_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "HermesAgent/1.0",
            },
        )

        try:
            with urllib.request.urlopen(poll_req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception:
            print(".", end="", flush=True)
            continue

        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]

        error = result.get("error", "")
        if error == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif error == "slow_down":
            # RFC 8628: add 5 seconds to polling interval
            server_interval = result.get("interval")
            if isinstance(server_interval, (int, float)) and server_interval > 0:
                interval = int(server_interval)
            else:
                interval += 5
            print(".", end="", flush=True)
            continue
        elif error == "expired_token":
            print()
            print("  ✗ Device code expired. Please try again.")
            return None
        elif error == "access_denied":
            print()
            print("  ✗ Authorization was denied.")
            return None
        elif error:
            print()
            print(f"  ✗ Authorization failed: {error}")
            return None

    print()
    print("  ✗ Timed out waiting for authorization.")
    return None


# ─── Copilot Token Exchange ────────────────────────────────────────────────

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch).
_jwt_cache: dict[str, tuple[str, float]] = {}
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry

# Token exchange endpoint and headers (matching VS Code / Copilot CLI)
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_EDITOR_VERSION = "vscode/1.104.1"
_EXCHANGE_USER_AGENT = "GitHubCopilotChat/0.26.7"


def _derive_enterprise_exchange_url(base_url: str | None) -> str | None:
    """Derive a GHE token-exchange URL from a Copilot Enterprise base URL."""
    b = str(base_url or "").strip().rstrip("/")
    if not b:
        return None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(b)
        host = (parsed.hostname or "").lower()
        if not host:
            return None
        if host.startswith("copilot-api."):
            host = "api." + host[len("copilot-api."):]
        if host in {"api.githubcopilot.com", "api.github.com"}:
            return None
        if host.endswith(".ghe.com") or host.startswith("api."):
            return f"https://{host}/copilot_internal/v2/token"
    except Exception:
        return None
    return None


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def exchange_copilot_token(
    raw_token: str,
    *,
    timeout: float = 10.0,
    base_url: str | None = None,
) -> tuple[str, float]:
    """Exchange a raw GitHub token for a short-lived Copilot API token.

    Calls ``GET https://api.github.com/copilot_internal/v2/token`` with
    the raw GitHub token and returns ``(api_token, expires_at)``.

    The returned token is a semicolon-separated string (not a standard JWT)
    used as ``Authorization: Bearer <token>`` for Copilot API requests.

    Results are cached in-process and reused until close to expiry.
    Raises ``ValueError`` on failure.
    """
    import urllib.request

    enterprise_exchange = _derive_enterprise_exchange_url(base_url)
    exchange_candidates: list[str] = [_TOKEN_EXCHANGE_URL]
    if enterprise_exchange and enterprise_exchange not in exchange_candidates:
        exchange_candidates.append(enterprise_exchange)

    fp = _token_fingerprint(raw_token + "|" + (enterprise_exchange or "public"))

    # Check cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at

    last_exc: Exception | None = None
    for exchange_url in exchange_candidates:
        req = urllib.request.Request(
            exchange_url,
            method="GET",
            headers={
                "Authorization": f"token {raw_token}",
                "User-Agent": _EXCHANGE_USER_AGENT,
                "Accept": "application/json",
                "Editor-Version": _EDITOR_VERSION,
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())

            api_token = data.get("token", "")
            expires_at = data.get("expires_at", 0)
            if not api_token:
                raise ValueError("Copilot token exchange returned empty token")

            # Convert expires_at to float if needed
            expires_at = float(expires_at) if expires_at else time.time() + 1800

            _jwt_cache[fp] = (api_token, expires_at)
            logger.debug("Copilot token exchanged via %s, expires_at=%s", exchange_url, expires_at)
            return api_token, expires_at
        except Exception as exc:
            last_exc = exc
            logger.debug("Copilot token exchange failed via %s: %s", exchange_url, exc)
            continue

    raise ValueError(f"Copilot token exchange failed: {last_exc}")


def get_copilot_api_token(raw_token: str, *, base_url: str | None = None) -> str:
    """Exchange a raw GitHub token for a Copilot API token, with fallback.

    Convenience wrapper: returns the exchanged token on success, or the
    raw token unchanged if the exchange fails (e.g. network error, unsupported
    account type). This preserves existing behaviour for accounts that don't
    need exchange while enabling access to internal-only models for those that do.
    """
    if not raw_token:
        return raw_token
    try:
        api_token, _ = exchange_copilot_token(raw_token, base_url=base_url)
        return api_token
    except Exception as exc:
        logger.debug("Copilot token exchange failed, using raw token: %s", exc)
        return raw_token


# ─── Copilot API Headers ───────────────────────────────────────────────────

def copilot_request_headers(
    *,
    is_agent_turn: bool = True,
    is_vision: bool = False,
    base_url: str | None = None,
) -> dict[str, str]:
    """Build the standard headers for Copilot API requests.

    Replicates the header set used by opencode and the Copilot CLI.
    """
    normalized_base = str(base_url or "").strip().rstrip("/").lower()
    default_integration_id = "vscode-chat"
    if "copilot-api." in normalized_base or normalized_base.startswith("https://api.sita.ghe.com"):
        default_integration_id = "copilot-developer-cli"

    headers: dict[str, str] = {
        "Editor-Version": "vscode/1.104.1",
        "User-Agent": "HermesAgent/1.0",
        "Copilot-Integration-Id": os.getenv("GITHUB_COPILOT_INTEGRATION_ID", default_integration_id),
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers
