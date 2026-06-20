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


def test_access_only_credentials_restore_refreshable_backup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)
    expired_ms = int(time.time() * 1000) - 60_000
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "expired-access-only",
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
        }
    }), encoding="utf-8")

    backup_dir = home / ".claude_backup"
    backup_dir.mkdir(parents=True)
    (backup_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "backup-access",
            "refreshToken": "backup-refresh",
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
        }
    }), encoding="utf-8")

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

    restored = json.loads((claude_dir / ".credentials.json").read_text(encoding="utf-8"))
    oauth = restored["claudeAiOauth"]
    assert oauth["accessToken"] == "fresh-access"
    assert oauth["refreshToken"] == "fresh-refresh"
    assert oauth["scopes"] == ["user:inference"]


def test_missing_credentials_prefer_auth_json_before_backup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    calls = []

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ccc, "_resolve_home_dir", lambda: str(home))

    def recover_auth_json():
        calls.append("auth_json")
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "auth-access",
                "refreshToken": "auth-refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
                "scopes": ["user:inference"],
            }
        }), encoding="utf-8")
        return True

    def recover_backup(_creds_path):
        calls.append("backup")
        return True

    monkeypatch.setattr(ccc, "_recover_claude_tokens_from_auth_json", recover_auth_json)
    monkeypatch.setattr(ccc, "_restore_claude_credentials_from_backup", recover_backup)

    assert ccc._maybe_refresh_claude_oauth() is True
    assert calls == ["auth_json"]


def test_access_only_credentials_prefer_auth_json_before_backup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)
    expired_ms = int(time.time() * 1000) - 60_000
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "expired-access-only",
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
        }
    }), encoding="utf-8")
    calls = []

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ccc, "_resolve_home_dir", lambda: str(home))

    def recover_auth_json():
        calls.append("auth_json")
        (claude_dir / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "auth-access",
                "refreshToken": "auth-refresh",
                "expiresAt": int(time.time() * 1000) + 3600_000,
                "scopes": ["user:inference"],
            }
        }), encoding="utf-8")
        return True

    def recover_backup(_creds_path):
        calls.append("backup")
        return True

    monkeypatch.setattr(ccc, "_recover_claude_tokens_from_auth_json", recover_auth_json)
    monkeypatch.setattr(ccc, "_restore_claude_credentials_from_backup", recover_backup)

    assert ccc._maybe_refresh_claude_oauth() is True
    assert calls == ["auth_json"]


def test_expired_credentials_are_replaced_by_newer_auth_json(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True)
    expired_ms = int(time.time() * 1000) - 60_000
    fresh_ms = int(time.time() * 1000) + 3600_000
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "backup-access",
            "refreshToken": "backup-refresh",
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
        }
    }), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ccc, "_resolve_home_dir", lambda: str(home))
    monkeypatch.setattr(
        ccc,
        "_auth_json_claude_credentials",
        lambda: ({"accessToken": "auth-access", "refreshToken": "auth-refresh"}, fresh_ms),
    )

    def recover_auth_json():
        (claude_dir / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "auth-access",
                "refreshToken": "auth-refresh",
                "expiresAt": fresh_ms,
                "scopes": ["user:inference"],
            }
        }), encoding="utf-8")
        return True

    monkeypatch.setattr(ccc, "_recover_claude_tokens_from_auth_json", recover_auth_json)

    assert ccc._maybe_refresh_claude_oauth() is True

    restored = json.loads((claude_dir / ".credentials.json").read_text(encoding="utf-8"))
    oauth = restored["claudeAiOauth"]
    assert oauth["accessToken"] == "auth-access"
    assert oauth["refreshToken"] == "auth-refresh"
    assert oauth["expiresAt"] == fresh_ms


def test_backup_restore_refuses_to_overwrite_newer_active_credentials(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    backup_dir = home / ".claude_backup"
    claude_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)
    older_ms = int(time.time() * 1000) + 60_000
    newer_ms = int(time.time() * 1000) + 3600_000
    active_path = claude_dir / ".credentials.json"
    active_path.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "active-access",
            "refreshToken": "active-refresh",
            "expiresAt": newer_ms,
        }
    }), encoding="utf-8")
    (backup_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "backup-access",
            "refreshToken": "backup-refresh",
            "expiresAt": older_ms,
        }
    }), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ccc, "_resolve_home_dir", lambda: str(home))
    monkeypatch.setattr(ccc, "_auth_json_claude_credentials", lambda: ({}, 0))

    assert ccc._restore_claude_credentials_from_backup(active_path) is False

    restored = json.loads(active_path.read_text(encoding="utf-8"))
    assert restored["claudeAiOauth"]["accessToken"] == "active-access"


def test_backup_restore_refuses_to_overwrite_newer_auth_json(tmp_path, monkeypatch):
    home = tmp_path / "home"
    claude_dir = home / ".claude"
    backup_dir = home / ".claude_backup"
    claude_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)
    older_ms = int(time.time() * 1000) + 60_000
    newer_ms = int(time.time() * 1000) + 3600_000
    active_path = claude_dir / ".credentials.json"
    (backup_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "backup-access",
            "refreshToken": "backup-refresh",
            "expiresAt": older_ms,
        }
    }), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(ccc, "_resolve_home_dir", lambda: str(home))
    monkeypatch.setattr(
        ccc,
        "_auth_json_claude_credentials",
        lambda: ({"accessToken": "auth-access", "refreshToken": "auth-refresh"}, newer_ms),
    )

    assert ccc._restore_claude_credentials_from_backup(active_path) is False
    assert not active_path.exists()


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
