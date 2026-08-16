import pytest


@pytest.fixture(autouse=True)
def isolated_cooldown_db(tmp_path, monkeypatch):
    import agent.model_cooldown_db as db

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MAX_COOLDOWN_SECONDS", "3600")
    if db._connection is not None:
        db._connection.close()
        db._connection = None
    yield
    if db._connection is not None:
        db._connection.close()
        db._connection = None


def test_health_retired_cooldown_bypasses_global_cap():
    from agent.model_cooldown_db import mark_model_cooldown, model_cooldown_remaining

    mark_model_cooldown(
        "ollama",
        "ollama/qwen3-coder-next",
        cooldown_seconds=86400,
        reason="hermes_code_health_retired_model_nonstream",
    )

    assert model_cooldown_remaining("ollama", "ollama/qwen3-coder-next") > 86000


def test_ordinary_cooldown_uses_global_cap():
    from agent.model_cooldown_db import mark_model_cooldown, model_cooldown_remaining

    mark_model_cooldown(
        "ollama",
        "ollama/qwen3-coder-next",
        cooldown_seconds=86400,
        reason="slow_response",
    )

    remaining = model_cooldown_remaining("ollama", "ollama/qwen3-coder-next")
    assert 3500 < remaining <= 3600


def test_provider_quota_cooldown_bypasses_global_cap():
    from agent.model_cooldown_db import mark_model_cooldown, model_cooldown_remaining

    mark_model_cooldown(
        "opencode-go",
        "opencode-go/qwen3.6-plus",
        cooldown_seconds=21600,
        reason="hermes_code_passthrough_provider_exhausted_provider_quota",
    )

    assert model_cooldown_remaining("opencode-go", "opencode-go/qwen3.6-plus") > 21500
