#!/usr/bin/env bash
# manage-api-keys.sh — helper for managing HERMES_API_KEYS_JSON secret

set -euo pipefail

NAMESPACE="${NAMESPACE:-hermes}"
SECRET_NAME="hermes-env-vault"

info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m  %s\n" "$*" >&2; exit 1; }

get_keys() {
  kubectl get secret "$SECRET_NAME" -n "$NAMESPACE" \
    -o jsonpath="{.data.HERMES_API_KEYS_JSON}" 2>/dev/null | base64 -d 2>/dev/null || echo '{"keys":[]}'
}

save_keys() {
  local json="$1"
  local b64
  b64="$(echo -n "$json" | base64 -w0)"
  kubectl patch secret "$SECRET_NAME" -n "$NAMESPACE" --type='json' \
    -p="[{\"op\":\"replace\",\"path\":\"/data/HERMES_API_KEYS_JSON\",\"value\":\"${b64}\"}]"
}

usage() {
  echo "Usage: $0 [list|add|revoke] ..."
  echo "  list                  List all API keys"
  echo "  add <id>              Add a new key with ID <id>"
  echo "  revoke <id>           Revoke key with ID <id>"
  exit 1
}

case "${1:-}" in
  list)
    get_keys | jq .
    ;;
  add)
    [ -z "${2:-}" ] && usage
    ID="$2"
    SECRET=$(openssl rand -hex 32)
    KEYS=$(get_keys)
    if echo "$KEYS" | jq -e --arg id "$ID" '.keys[] | select(.id == $id)' >/dev/null; then
      fail "Key ID '$ID' already exists."
    fi
    NEW_KEYS=$(echo "$KEYS" | jq --arg id "$ID" --arg sec "$SECRET" '.keys += [{"id": $id, "secret": $sec}]')
    save_keys "$NEW_KEYS"
    if [ "${NORESTART:-0}" != "1" ]; then
      kubectl rollout restart deployment/hermes -n "$NAMESPACE"
    fi
    ok "Added key for '$ID': $SECRET"
    ;;
  revoke)
    [ -z "${2:-}" ] && usage
    ID="$2"
    KEYS=$(get_keys)
    NEW_KEYS=$(echo "$KEYS" | jq --arg id "$ID" 'del(.keys[] | select(.id == $id))')
    save_keys "$NEW_KEYS"
    if [ "${NORESTART:-0}" != "1" ]; then
      kubectl rollout restart deployment/hermes -n "$NAMESPACE"
    fi
    ok "Revoked key for '$ID'"
    ;;
  *)
    usage
    ;;
esac
