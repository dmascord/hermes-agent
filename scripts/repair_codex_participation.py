import sqlite3
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hermes_constants import get_hermes_home

QUALITY_DB = get_hermes_home() / "model_quality.db"
COOLDOWN_DB = get_hermes_home() / "model_cooldowns.db"

def repair():
    print("Repairing openai-codex participation...")
    
    # 1. Quality DB: Reset metrics for openai-codex
    if QUALITY_DB.exists():
        conn = sqlite3.connect(str(QUALITY_DB))
        print(f"  Resetting metrics in {QUALITY_DB}...")
        # We need to find the keys first
        cursor = conn.execute("SELECT key FROM model_metrics WHERE provider='openai-codex'")
        keys = [row[0] for row in cursor.fetchall()]
        for key in keys:
            print(f"    Deleting key: {key}")
            conn.execute("DELETE FROM model_metrics WHERE key=?", (key,))
            conn.execute("DELETE FROM model_events WHERE provider='openai-codex'")
        conn.commit()
        conn.close()
    else:
        print(f"  Quality DB not found at {QUALITY_DB}")

    # 2. Cooldown DB: Clear cooldowns for openai-codex
    if COOLDOWN_DB.exists():
        conn = sqlite3.connect(str(COOLDOWN_DB))
        print(f"  Clearing cooldowns in {COOLDOWN_DB}...")
        conn.execute("DELETE FROM cooldowns WHERE provider='openai-codex'")
        conn.execute("DELETE FROM circuit_breakers WHERE provider='openai-codex'")
        conn.commit()
        conn.close()
    else:
        print(f"  Cooldown DB not found at {COOLDOWN_DB}")
    
    print("Repair complete. Please restart the deployment pod.")

if __name__ == "__main__":
    repair()
