"""Optional smart-router shim for API server requests.

Default behavior leaves routing unchanged while preserving the public hook used
by the gateway.
"""


class _NoopRouter:
    def route(self, prompt: str) -> dict:
        return {
            "complexity": "normal",
            "model": "",
            "tier": "primary",
            "savings_vs_primary": 0,
        }


_GLOBAL = _NoopRouter()


def get_global_router():
    return _GLOBAL
