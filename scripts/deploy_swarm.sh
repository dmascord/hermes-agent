#!/bin/bash
# deploy_swarm.sh — Build and deploy hermes-swarm via docker-compose
#
# Usage:
#   ./scripts/deploy_swarm.sh          # rebuild + restart (default)
#   ./scripts/deploy_swarm.sh --no-build  # just restart with existing image
#   ./scripts/deploy_swarm.sh --logs     # tail container logs
#
# Requirements:
#   - Docker and docker-compose installed
#   - .env file in the hermes-agent directory (copy from .env.swarm.example)
#   - Being on the remote server (or running with SSH agent forwarded)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export COMPOSE_BAKE=false

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

NO_BUILD=false
SHOW_LOGS=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      NO_BUILD=true
      shift
      ;;
    --logs)
      SHOW_LOGS=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--no-build] [--logs]"
      echo "  --no-build  Start with existing image (skip docker build)"
      echo "  --logs      Tail container logs after starting"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$SHOW_LOGS" == "true" ]]; then
  docker compose logs -f --tail 50
  exit $?
fi

echo "=== Building hermes-agent Docker image ==="
if [[ "$NO_BUILD" == "true" ]]; then
  echo "[skip] --no-build flag set"
else
  docker build -t hermes-agent:latest .
fi

echo ""
echo "=== Stopping existing hermes-swarm container ==="
docker compose down --remove-orphans 2>/dev/null || true

echo ""
echo "=== Starting hermes-swarm ==="
docker compose up -d

echo ""
echo "=== Waiting for health check ==="
for i in $(seq 1 15); do
  sleep 2
  HEALTH=$(curl -sf --max-time 5 "http://127.0.0.1:8642/health" \
    -H "Authorization: Bearer ${API_SERVER_KEY}" 2>/dev/null || echo "waiting...")
  if [[ "$HEALTH" == *"ok"* ]]; then
    echo "✓ hermes-swarm is healthy on localhost"
    break
  fi
  echo "  attempt $i/15 — $HEALTH"
done

echo ""
echo "=== Public health check ==="
curl -sf --max-time 5 "https://hermes.tusker.net.au/health" \
  -H "Authorization: Bearer ${API_SERVER_KEY}" \
  >/dev/null && echo "✓ public /health reachable" || echo "Public /health check failed — check logs with: $0 --logs"

echo ""
echo "=== Public models check ==="
curl -sf --max-time 5 "https://hermes.tusker.net.au/v1/models" \
  -H "Authorization: Bearer ${API_SERVER_KEY}" \
  >/dev/null && echo "✓ public /v1/models reachable" || echo "Public /v1/models check failed — check logs with: $0 --logs"

echo ""
echo "=== Container status ==="
docker compose ps
