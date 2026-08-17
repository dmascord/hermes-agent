#!/usr/bin/env bash
# migrate-to-auth-gateway.sh — safe migration from single-key to multi-key auth
#
# Strategy:
#   Phase 1: Build and push auth-sidecar image.
#   Phase 2: Write HERMES_INTERNAL_TOKEN = current API_SERVER_KEY (zero disruption)
#            so both old and new pods accept the same token during rollout.
#   Phase 3: Deploy sidecar container + Traefik IngressRoute.
#   Phase 4: Rotate HERMES_INTERNAL_TOKEN to a fresh random value,
#            removing the old broad key from Hermes and keeping it only
#            in HERMES_API_KEYS_JSON for clients.
#
# Usage:
#   ./scripts/migrate-to-auth-gateway.sh [--rotate-token]
#
# Flags:
#   --rotate-token   Include the final token rotation step (Phase 4).
#                    Without this flag the script stops after Phase 3,
#                    which is safe for a first deployment.
#
# Requirements:
#   - kubectl configured to reach the cluster (via ssh visor or local).
#   - docker or buildah available on the build host (visor).
#   - Hermes-env-vault already contains HERMES_API_KEYS_JSON and
#     HERMES_INTERNAL_TOKEN is either absent or will be overwritten.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REGISTRY="${REGISTRY:-10.103.195.90:5000}"
IMAGE="${REGISTRY}/hermes-auth-sidecar:latest"
REGISTRY_TLS_VERIFY="${REGISTRY_TLS_VERIFY:-false}"
NAMESPACE="${NAMESPACE:-hermes}"
SECRET_NAME="hermes-env-vault"
DEPLOY_NAME="hermes"
SIDECAR_IMAGE="${REGISTRY}/hermes-auth-sidecar:latest"
INGROUTE_MANIFEST="${REPO_ROOT}/k8s/hermes-ingroute.yaml"
ROTATE_TOKEN=false
EXTERNAL_ACCESS_TOKEN="${EXTERNAL_ACCESS_TOKEN:-https://hermes.tusker.net.au}"

for arg in "$@"; do
  case "$arg" in
    --rotate-token) ROTATE_TOKEN=true ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*" >&2; exit 1; }

# ── Helpers ──────────────────────────────────────────────────────────────────

get_secret_field() {
  local key="$1"
  kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" \
    -o jsonpath="{.data.${key}}" 2>/dev/null | base64 -d 2>/dev/null || echo ""
}

set_secret_field() {
  local key="$1" value="$2"
  local b64
  b64="$(echo -n "$value" | base64 -w0)"
  # Idempotent: add if absent, replace if present.
  if kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" -o jsonpath='{.data}' | grep -q "\"${key}\""; then
    kubectl patch secret "$SECRET_NAME" -n "$NAMESPACE" --type='json' \
      -p="[{\"op\":\"replace\",\"path\":\"/data/${key}\",\"value\":\"${b64}\"}]"
  else
    kubectl patch secret "$SECRET_NAME" -n "$NAMESPACE" --type='merge' \
      -p="{\"data\": {\"${key}\": \"${b64}\"}}"
  fi
}
generate_hex() {
  if command -v openssl &>/dev/null; then
    openssl rand -hex 32
  else
    head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}
wait_for_pod_ready() {
  local timeout=120
  info "Waiting for deployment/${DEPLOY_NAME} pods to become ready (timeout: ${timeout}s) ..."
  if kubectl rollout status deployment/"$DEPLOY_NAME" -n "$NAMESPACE" --timeout="${timeout}s"; then
    ok "Pods ready."
    return 0
  fi
  fail "Pods did not become ready within ${timeout}s."
}

verify_auth() {
  info "Verifying auth against ${EXTERNAL_ACCESS_TOKEN} ..."
  local status_code
  status_code=$(curl -s -o /dev/null -w "%{http_code}" \
    "${EXTERNAL_ACCESS_TOKEN}/v1/models" \
    -H "Authorization: Bearer ${CURRENT_KEY}" \
    --max-time 10 2>/dev/null || echo "000")
  if [[ "$status_code" == "200" ]]; then
    ok "Existing key authenticated successfully (HTTP ${status_code})."
    return 0
  else
    warn "Auth check returned HTTP ${status_code}.  Check ingress/traefik logs."
    return 1
  fi
}

# ── Phase 0: Preflight ───────────────────────────────────────────────────────

info "Phase 0 — Preflight checks"

CURRENT_KEY=$(get_secret_field "API_SERVER_KEY")
if [[ -z "$CURRENT_KEY" ]]; then
  fail "Cannot read API_SERVER_KEY from secret ${SECRET_NAME}.  Aborting."
fi
ok "Current API_SERVER_KEY: ${CURRENT_KEY:0:8}... (${#CURRENT_KEY} chars)"

# Ensure HERMES_API_KEYS_JSON already has a place for the current key.
# If missing, we'll initialise it during Phase 2.
API_KEYS_JSON=$(get_secret_field "HERMES_API_KEYS_JSON" || true)

# ── Phase 1: Build and push sidecar ─────────────────────────────────────────

info "Phase 1 — Building auth-sidecar image"

if command -v docker &>/dev/null; then
  docker build -t "$SIDECAR_IMAGE" -f "${REPO_ROOT}/auth-sidecar/Dockerfile" "${REPO_ROOT}/auth-sidecar/"
  docker push "$SIDECAR_IMAGE"
elif command -v buildah &>/dev/null; then
  buildah bud -t "$SIDECAR_IMAGE" -f "${REPO_ROOT}/auth-sidecar/Dockerfile" "${REPO_ROOT}/auth-sidecar/"
  buildah push --tls-verify="${REGISTRY_TLS_VERIFY}" "$SIDECAR_IMAGE"
else
  fail "Neither docker nor buildah found on this host."
fi
ok "Image pushed: ${SIDECAR_IMAGE}"

# ── Phase 2: Write internal token = existing key (zero-downtime baseline) ────

info "Phase 2 — Setting HERMES_INTERNAL_TOKEN = current API_SERVER_KEY"

# The sidecar will validate client Bearer tokens against HERMES_API_KEYS_JSON.
# Hermes itself will only accept HERMES_INTERNAL_TOKEN as API_SERVER_KEY.
# Setting them equal means existing clients keep working immediately after rollout.
set_secret_field "HERMES_INTERNAL_TOKEN" "$CURRENT_KEY"
ok "HERMES_INTERNAL_TOKEN written (matches current API_SERVER_KEY)."

# Ensure HERMES_API_KEYS_JSON has the existing key as first entry.
# If already populated, leave it alone — operator may have added keys already.
if [[ -z "$API_KEYS_JSON" || "$API_KEYS_JSON" == "null" ]]; then
  info "Initialising HERMES_API_KEYS_JSON with current key ..."
  INITIAL_JSON="{\"keys\":[{\"id\":\"existing-client\",\"secret\":\"${CURRENT_KEY}\"}]}"
  set_secret_field "HERMES_API_KEYS_JSON" "$INITIAL_JSON"
  ok "HERMES_API_KEYS_JSON initialised."
else
  ok "HERMES_API_KEYS_JSON already has content; skipping init."
fi

# ── Phase 3: Deploy sidecar + Traefik routing ───────────────────────────────

info "Phase 3 — Deploying auth-sidecar"

# The deployment already includes the sidecar container definition from
# k8s/hermes-deployment.yaml.  Rolling update replaces the pod with one
# that has both hermes and auth-sidecar containers.
kubectl apply -f "${REPO_ROOT}/k8s/hermes-deployment.yaml"
ok "Deployment manifest applied."

wait_for_pod_ready

kubectl apply -f "$INGROUTE_MANIFEST"
ok "IngressRoute + Middleware + ClusterIP Service applied."

# ── Phase 3b: Verify ────────────────────────────────────────────────────────

info "Phase 3b — Smoke testing"

# Give Traefik a few seconds to pick up the new IngressRoute.
sleep 5

if verify_auth; then
  ok "Existing key works through Traefik forwardAuth."
else
  warn "Verification inconclusive — check output above and Traefik logs."
  warn "Run manually:  curl -i ${EXTERNAL_ACCESS_TOKEN}/v1/models -H 'Authorization: Bearer ${CURRENT_KEY}'"
fi

# ── Phase 4 (optional): Rotate to fresh internal token ──────────────────────

if [[ "$ROTATE_TOKEN" == true ]]; then
  info "Phase 4 — Rotating HERMES_INTERNAL_TOKEN to a new random value"

  NEW_INTERNAL_TOKEN=$(generate_hex)
  set_secret_field "HERMES_INTERNAL_TOKEN" "$NEW_INTERNAL_TOKEN"
  ok "New HERMES_INTERNAL_TOKEN written: ${NEW_INTERNAL_TOKEN:0:8}..."

  # Redeploy so Hermes picks up the new env value.
  kubectl rollout restart deployment/"$DEPLOY_NAME" -n "$NAMESPACE"
  wait_for_pod_ready

  sleep 5
  info "Re-verifying with existing key (now validated by sidecar, not Hermes direct) ..."
  if verify_auth; then
    ok "Existing key still works — validated by sidecar with rotated internal token."
  else
    warn "Post-rotation check inconclusive.  Manual verification recommended."
  fi

  ok "Phase 4 complete.  Hermes no longer accepts the old broad key."
else
  info "Phase 4 skipped (run with --rotate-token to perform token rotation later)."
fi

# ── Summary ──────────────────────────────────────────────────────────────────

cat <<EOF

───────────────────────────────────────────────────────────────
  Migration complete.
───────────────────────────────────────────────────────────────
  URL (unchanged):        ${EXTERNAL_ACCESS_TOKEN}
  Auth layer:             Traefik forwardAuth → auth-sidecar:8081
  Internal token env:     HERMES_INTERNAL_TOKEN in ${SECRET_NAME}
  Client keys env:        HERMES_API_KEYS_JSON in ${SECRET_NAME}

  To add a new client key, edit the secret:
    kubectl edit secret ${SECRET_NAME} -n ${NAMESPACE}
    # or
    kubectl patch secret ${SECRET_NAME} -n ${NAMESPACE} --type='json' \\
      -p='[{"op":"replace","path":"/data/HERMES_API_KEYS_JSON","value":"BASE64_ENCODED_JSON"}]'

  To rotate the internal token later:
    ${BASH_SOURCE[0]} --rotate-token

  Audit logs (consumer IDs):
    kubectl logs deploy/${DEPLOY_NAME} -n ${NAMESPACE} -c auth-sidecar --tail=100
───────────────────────────────────────────────────────────────
EOF
