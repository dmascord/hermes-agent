"""Cohere provider profile."""

from providers import register_provider
from providers.base import ProviderProfile

cohere = ProviderProfile(
    name="cohere",
    env_vars=("COHERE_API_KEY",),
    base_url_env_var="COHERE_BASE_URL",
    base_url="https://api.cohere.com/compatibility/v1",
    models_url="https://api.cohere.com/v2/models?endpoint=chat&page_size=1000",
    display_name="Cohere",
    description="Cohere Command models via the OpenAI-compatible API",
    signup_url="https://dashboard.cohere.com/api-keys",
    fallback_models=(
        "command-a-plus-05-2026",
        "command-a-03-2025",
        "command-r-plus",
        "command-r",
    ),
)

register_provider(cohere)
