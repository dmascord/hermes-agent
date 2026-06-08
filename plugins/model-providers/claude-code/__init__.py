"""Claude Code CLI provider profile.

claude-code-cli uses an external Claude Code CLI subprocess — NOT the standard
Anthropic API. api_mode="chat_completions" is handled separately in
auxiliary_client.py and gateway. The profile captures auth + endpoint metadata
for registry migration.

Note: The name "claude-code" is already used as an alias for the "anthropic"
provider, so we use "claude-code-cli" as the canonical name to avoid conflict.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeCLIProfile(ProviderProfile):
    """Claude Code CLI — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the Claude Code CLI."""
        return None


claude_code_cli = ClaudeCodeCLIProfile(
    name="claude-code-cli",
    aliases=("claude-code-external",),
    api_mode="chat_completions",  # CLI uses chat_completions routing
    env_vars=(),  # Managed by Claude Code CLI
    base_url="claude://codex",  # Internal scheme
    auth_type="external_process",
)

register_provider(claude_code_cli)