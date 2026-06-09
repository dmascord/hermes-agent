"""One-off script to manually boost Codex quality scores for chain testing.

Manually sets quality_score on openai-codex models in the quality DB so they
rank higher in the hermes-code chain. This is for testing whether Codex
works in the rotation after long cooldowns — NOT a permanent quality
override.

Usage:
    python scripts/boost_codex_quality.py set [score]
    python scripts/boost_codex_quality.py clear
    python scripts/boost_codex_quality.py show
"""
import sqlite3
import sys
from pathlib import Path

# Add repo root to sys.path so we can import hermes_constants
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_constants import get_hermes_home

DB_PATH = get_hermes_home() / "model_quality.db"

# Boost target: openai-codex / gpt-5-mini — the freshest available Codex model.
# Score of 95 puts it near the top of the chain (minimax/gpt-5-mini is ~90-95).
TARGETS = [
    ("openai-codex", "gpt-5-mini"),
]


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def show() -> None:
    """Print current quality scores for the target models."""
    conn = _connect()
    try:
        for prov, model in TARGETS:
            row = conn.execute(
                "SELECT quality_score, total_calls, success_calls, failure_calls "
                "FROM model_metrics WHERE provider=? AND model=? ORDER BY quality_score DESC",
                (prov, model),
            ).fetchall()
            if not row:
                print(f"  {prov}/{model}: no entries")
            for r in row:
                print(f"  {prov}/{model}: score={r[0]} total={r[1]} ok={r[2]} fail={r[3]}")
    finally:
        conn.close()


def set_score(score: float) -> None:
    """Insert or update quality_score for the target models.

    If no entry exists, creates one with all counts = 0 (so the
    record gets a manual quality score without any call history).
    """
    conn = _connect()
    try:
        import time
        now = time.time()
        for prov, model in TARGETS:
            key = f"{prov}/{model}/"  # empty base_url for the manual boost
            conn.execute(
                """INSERT INTO model_metrics
                       (key, provider, model, base_url, total_calls, success_calls,
                        failure_calls, text_only_calls, avg_latency_ms, quality_score,
                        updated_at)
                   VALUES (?, ?, ?, '', 0, 0, 0, 0, 0, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                        quality_score=excluded.quality_score,
                        updated_at=excluded.updated_at""",
                (key, prov, model, score, now),
            )
            print(f"  boosted {prov}/{model} -> {score}")
        conn.commit()
    finally:
        conn.close()


def clear() -> None:
    """Remove the manual boost entries so the real score is used."""
    conn = _connect()
    try:
        for prov, model in TARGETS:
            key = f"{prov}/{model}/"
            cur = conn.execute("DELETE FROM model_metrics WHERE key=?", (key,))
            print(f"  cleared {prov}/{model} ({cur.rowcount} rows)")
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"set", "clear", "show"}:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "show":
        show()
    elif cmd == "set":
        score = float(sys.argv[2]) if len(sys.argv) > 2 else 95.0
        if not 0 <= score <= 100:
            print(f"Score must be 0-100, got {score}", file=sys.stderr)
            return 1
        set_score(score)
    elif cmd == "clear":
        clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
