import json
import time

from agent import claude_code_client as ccc


def test_refresh_token_uses_shared_anthropic_adapter(monkeypatch):
    calls = []

    def fake_refresh(refresh_token, *, use_json=False):
        calls.append((refresh_token, use_json))
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_at_ms": int(time.time() * 1000) + 3600_000,
        }

    monkeypatch.setattr(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
        fake_refresh,
    )

    result = ccc._claude_oauth_refresh_token("stale-refresh")

    assert calls == [("stale-refresh", False)]
    assert result["access_token"] == "fresh-access"
    assert result["refresh_token"] == "fresh-refresh"
    assert result["expires_in"] > 0


def test_recovered_expired_credentials_are_refreshed_before_cli_use(tmp_path, monkeypatch):
    home = tmp_path / "home"
    backup_dir = home / ".claude_backup"
    backup_dir.mkdir(parents=True)
    expired_ms = int(time.time() * 1000) - 60_000
    backup_creds = {
        "claudeAiOauth": {
            "accessToken": "expired-access",
            "refreshToken": "backup-refresh",
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
            "subscriptionType": "max",
            "rateLimitTier": "standard",
        }
    }
    (backup_dir / ".credentials.json").write_text(json.dumps(backup_creds), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ccc, "_resolve_home_dir", lambda: str(home))
    monkeypatch.setattr(ccc, "_persist_claude_credentials_to_auth_json", lambda **_: None)
    monkeypatch.setattr(
        ccc,
        "_claude_oauth_refresh_token",
        lambda refresh_token: {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_in": 3600,
        },
    )

    assert ccc._maybe_refresh_claude_oauth() is True

    restored_path = home / ".claude" / ".credentials.json"
    restored = json.loads(restored_path.read_text(encoding="utf-8"))
    oauth = restored["claudeAiOauth"]
    assert oauth["accessToken"] == "fresh-access"
    assert oauth["refreshToken"] == "fresh-refresh"
    assert oauth["expiresAt"] > int(time.time() * 1000)
    assert oauth["scopes"] == ["user:inference"]
    assert oauth["subscriptionType"] == "max"
    assert oauth["rateLimitTier"] == "standard"


def test_mcp_config_uses_claude_code_shape_and_returns_path(tmp_path):
    client = ccc.ClaudeCodeClient(claude_cwd=str(tmp_path))
    manifest = tmp_path / "tools.json"
    manifest.write_text("[]", encoding="utf-8")

    config_path = client._build_mcp_config(str(manifest))

    assert config_path
    config = json.loads(open(config_path, encoding="utf-8").read())
    assert list(config) == ["mcpServers"]
    server = config["mcpServers"]["hermes-tools"]
    assert server["type"] == "stdio"
    assert server["command"] == "python3"
    assert server["args"] and server["args"][0].endswith("claude_mcp_bridge.py")
    assert server["env"]["HERMES_TOOLS_FILE"] == str(manifest)


def test_allowed_tool_names_use_claude_mcp_prefix(tmp_path):
    client = ccc.ClaudeCodeClient(claude_cwd=str(tmp_path))

    assert client._allowed_tool_names([
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {}},
    ]) == [
        "mcp__hermes-tools__read_file",
        "mcp__hermes-tools__web_search",
    ]
