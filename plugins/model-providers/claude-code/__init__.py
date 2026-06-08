"""Claude Code provider profile.

claude-code uses an external Claude Code CLI subprocess — NOT the standard
Anthropic API. api_mode="chat_completions" is handled separately in
auxiliary_client.py and gateway. The profile captures auth + endpoint metadata
for registry migration.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeProfile(ProviderProfile):
    """Claude Code — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the Claude Code CLI."""
        return None


claude_code = ClaudeCodeProfile(
    name="claude-code",
    aliases=("claude-code-cli", "claude"),
    api_mode="chat_completions",  # CLI uses chat_completions routing
    env_vars=(),  # Managed by Claude Code CLI
    base_url="claude://codex",  # Internal scheme
    auth_type="external_process",
)

register_provider(claude_code)