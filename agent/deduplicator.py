"""Optional response deduplication shim.

The API server treats this as an optimization only.  This default
implementation is intentionally inert so tests and minimal deployments do not
depend on an optional cache backend.
"""


class _NoopDeduplicator:
    def compute_key(self, prompt: str, model: str) -> str:
        return ""

    def get(self, prompt: str, model: str):
        return None, False


_GLOBAL = _NoopDeduplicator()


def get_global_deduplicator():
    return _GLOBAL
