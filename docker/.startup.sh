#!/bin/bash
set -e

# This script runs inside the container as the `hermes` user.
# It starts the gateway in the background, waits for the API
# server to respond, primes credential pools, then foregrounds
# the gateway process.

export HERMES_HOME="${HERMES_HOME:-/opt/data}"

# The entrypoint.sh already sources the venv, but make sure
# it's available in case we're invoked directly
if [ -f /opt/hermes/.venv/bin/activate ]; then
    source /opt/hermes/.venv/bin/activate
fi

echo "=== Starting Hermes Gateway ==="
hermes gateway run &
GATEWAY_PID=$!

# Wait for the API server to be ready (up to 60s)
echo "Waiting for API server..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8642/v1/health > /dev/null 2>&1; then
        echo "API server ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "WARNING: API server did not become ready within 60s"
    fi
    sleep 2
done

# Prime all API-key credential pools so they are cached
# and don't slow down the first user request
echo "Priming credential pools..."
python3 -c "
from agent.credential_pool import load_pool
from hermes_cli.auth import PROVIDER_REGISTRY

ok = fail = 0
for pid, cfg in PROVIDER_REGISTRY.items():
    auth_type = getattr(cfg, 'auth_type', None)
    if auth_type == 'api_key':
        try:
            load_pool(pid)
            ok += 1
        except Exception as e:
            print('  WARN: ' + pid + ' - ' + str(e))
            fail += 1

# Also prime the Copilot credential pool (OAuth-based, not api_key)
# so load_pool('copilot') is fast when the gateway needs it.
try:
    load_pool('copilot')
    ok += 1
    print('  OK: copilot')
except Exception as e:
    print('  WARN: copilot - ' + str(e))
    fail += 1

# Prime the openai-codex credential pool (OAuth-based) so that after a
# container restart the restored codex entries from auth.json (restored
# by the entrypoint before gosu) are actually loaded into the runtime pool.
try:
    load_pool('openai-codex')
    ok += 1
    print('  OK: openai-codex')
except Exception as e:
    print('  WARN: openai-codex - ' + str(e))
    fail += 1

print('Primed ' + str(ok) + ' pools OK')
if fail > 0:
    print('  (' + str(fail) + ' had warnings)')
"

echo "=== Warm-up complete ==="

# Bring gateway to foreground so signals (SIGTERM etc.)
# are delivered correctly to the gateway process
wait $GATEWAY_PID
