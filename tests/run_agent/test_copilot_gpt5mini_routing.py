import pytest
from run_agent import AIAgent


def test_copilot_gpt_5_mini_defaults_to_chat_completions(monkeypatch):
    # Minimal bootstrap patching used widely in tests
    monkeypatch.setenv("GITHUB_COPILOT_BASE_URL", "https://api.githubcopilot.com")
    monkeypatch.setenv("GITHUB_COPILOT_API_KEY", "copilot-token")

    agent = AIAgent(
        model="gpt-5-mini",
        provider="copilot",
        base_url="https://api.githubcopilot.com",
        api_key="gh-token",
        quiet_mode=True,
        max_iterations=1,
        skip_context_files=True,
        skip_memory=True,
    )
    assert agent.provider == "copilot"
    assert agent.api_mode == "chat_completions"
