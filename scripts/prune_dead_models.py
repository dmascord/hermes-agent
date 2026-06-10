"""Prune dead/chronically-bad models from model_quality.db.

Removes metric rows for models that match ALL of the following:
  - success rate is 0% AND total_calls >= 20
  - last_failure_at is older than 24h (i.e. the model has been broken for a while)
  - last_success_at is NULL or older than 7d (i.e. the model has not worked recently)
  - quality_score is < 50 (i.e. clearly below the picker floor of 60)

The picker already filters quality_score < 60, so removing these rows is
purely cosmetic — but it keeps the database tidy, removes false signals
from `get_all_quality_scores()`, and prevents tools like `hermes models`
from listing models that have been broken for days.

Usage:
    python scripts/prune_dead_models.py --dry-run   # show what would be deleted
    python scripts/prune_dead_models.py --yes        # actually delete
    python scripts/prune_dead_models.py --min-calls 10   # override threshold
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

# Add repo root to sys.path so we can import hermes_constants
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_constants import get_hermes_home

DB_PATH = get_hermes_home() / "model_quality.db"

# A failure older than this means the model has had a real chance to recover
MAX_FAILURE_AGE_S = 24 * 3600
# A model that hasn't succeeded in this long is dead
MAX_SUCCESS_AGE_S = 7 * 24 * 3600


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def find_dead_models(conn: sqlite3.Connection, min_calls: int) -> list:
    """Return rows for models that are demonstrably dead."""
    now = time.time()
    # A model is "dead" if:
    #   - total_calls >= min_calls (enough signal)
    #   - success_calls = 0 (never worked)
    #   - last_failure_at < now - MAX_FAILURE_AGE_S (failure was real, not a transient blip)
    #   - quality_score <= 50 (well below the 60 picker floor — and the
    #     scorer gives 0-success models exactly 50 by default, so this
    #     threshold is what actually catches the all-broken ones)
    rows = conn.execute(
        """
        SELECT provider, model, total_calls, success_calls, failure_calls,
               quality_score, last_failure_at, last_success_at
        FROM model_metrics
        WHERE total_calls >= ?
          AND success_calls = 0
          AND quality_score <= 50.0
          AND (
              last_failure_at IS NULL OR
              last_failure_at < ?
          )
          AND (
              last_success_at IS NULL OR
              last_success_at < ?
          )
        ORDER BY total_calls DESC
        """,
        (min_calls, now - MAX_FAILURE_AGE_S, now - MAX_SUCCESS_AGE_S),
    ).fetchall()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without modifying the DB",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    ap.add_argument(
        "--min-calls",
        type=int,
        default=20,
        help="Minimum total_calls before a zero-success model is considered dead "
             "(default: 20)",
    )
    args = ap.parse_args()

    if not args.dry_run and not args.yes:
        print("Refusing to delete rows without --yes. Use --dry-run to preview first.")
        return 2

    conn = _connect()
    rows = find_dead_models(conn, args.min_calls)
    if not rows:
        print("No dead models found.")
        return 0

    print(f"Found {len(rows)} dead model(s):")
    for prov, model, tot, ok, fail, q, lf, ls in rows:
        last_fail = (
            f"{(time.time() - lf) / 3600:.1f}h ago" if lf else "never"
        )
        last_succ = (
            f"{(time.time() - ls) / 3600:.1f}h ago" if ls else "never"
        )
        print(
            f"  {prov}/{model}: calls={tot} ok={ok} fail={fail} "
            f"qual={q:.0f} last_fail={last_fail} last_ok={last_succ}"
        )

    if args.dry_run:
        print("\n--dry-run specified; no rows were deleted.")
        return 0

    n = conn.execute(
        """
        DELETE FROM model_metrics
        WHERE total_calls >= ?
          AND success_calls = 0
          AND quality_score <= 50.0
          AND (
              last_failure_at IS NULL OR
              last_failure_at < ?
          )
          AND (
              last_success_at IS NULL OR
              last_success_at < ?
          )
        """,
        (args.min_calls, time.time() - MAX_FAILURE_AGE_S, time.time() - MAX_SUCCESS_AGE_S),
    ).rowcount
    # Also drop the related event rows so the events table doesn't accumulate
    # orphan entries pointing at a deleted metrics row.
    n_events = conn.execute(
        """
        DELETE FROM model_events
        WHERE (provider, model) NOT IN (SELECT provider, model FROM model_metrics)
        """
    ).rowcount
    conn.commit()
    print(f"\nDeleted {n} model_metrics row(s) and {n_events} orphan model_events row(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
