"""Arliai provider profile.

Arliai provides open-weight models via an OpenAI-compatible API.
"""

from providers import register_provider
from providers.base import ProviderProfile

arliai = ProviderProfile(
    name="arliai",
    aliases=("arlee",),
    env_vars=("ARLIAI_API_KEY",),
    display_name="Arliai",
    description="Arliai — open-weight models",
    signup_url="https://arliai.com/",
    fallback_models=(
        "Gemma-4-31B-Claude-4.6-Opus-Reasoning-Distilled",
        "GLM-4.7",
        "Mistral-Medium-3.5-128B",
    ),
    base_url="https://api.arliai.com/v1",
)

register_provider(arliai)
