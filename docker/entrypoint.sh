#!/bin/bash
# Docker/Podman entrypoint: bootstrap config files into the mounted volume, then run hermes.
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"

# --- Privilege dropping via gosu ---
# When started as root (the default for Docker, or fakeroot in rootless Podman),
# optionally remap the hermes user/group to match host-side ownership, fix volume
# permissions, then re-exec as hermes.
if [ "$(id -u)" = "0" ]; then
    if [ -n "$HERMES_UID" ] && [ "$HERMES_UID" != "$(id -u hermes)" ]; then
        echo "Changing hermes UID to $HERMES_UID"
        usermod -u "$HERMES_UID" hermes
    fi

    if [ -n "$HERMES_GID" ] && [ "$HERMES_GID" != "$(id -g hermes)" ]; then
        echo "Changing hermes GID to $HERMES_GID"
        # -o allows non-unique GID (e.g. macOS GID 20 "staff" may already exist
        # as "dialout" in the Debian-based container image)
        groupmod -o -g "$HERMES_GID" hermes 2>/dev/null || true
    fi

    # Fix ownership of the data volume. When HERMES_UID remaps the hermes user,
    # files created by previous runs (under the old UID) become inaccessible.
    # Always chown -R when UID was remapped; otherwise only if top-level is wrong.
    actual_hermes_uid=$(id -u hermes)
    needs_chown=false
    if [ -n "$HERMES_UID" ] && [ "$HERMES_UID" != "10000" ]; then
        needs_chown=true
    elif [ "$(stat -c %u "$HERMES_HOME" 2>/dev/null)" != "$actual_hermes_uid" ]; then
        needs_chown=true
    fi
    if [ "$needs_chown" = true ]; then
        echo "Fixing ownership of $HERMES_HOME to hermes ($actual_hermes_uid)"
        # In rootless Podman the container's "root" is mapped to an unprivileged
        # host UID — chown will fail.  That's fine: the volume is already owned
        # by the mapped user on the host side.
        chown -R hermes:hermes "$HERMES_HOME" 2>/dev/null || \
            echo "Warning: chown failed (rootless container?) — continuing anyway"
    fi

    # Ensure config.yaml is readable by the hermes runtime user even if it was
    # edited on the host after initial ownership setup. Must run here (as root)
    # rather than after the gosu drop, otherwise a non-root caller like
    # `docker run -u $(id -u):$(id -g)` hits "Operation not permitted" (#15865).
    if [ -f "$HERMES_HOME/config.yaml" ]; then
        chown hermes:hermes "$HERMES_HOME/config.yaml" 2>/dev/null || true
        chmod 640 "$HERMES_HOME/config.yaml" 2>/dev/null || true
    fi

    # auth.json is now root-owned (written by Python above as root).
    # chown it to hermes so the drop-privilege step doesn't break the
    # hermes runtime's ability to read/write it.
    chown hermes:hermes "$HERMES_HOME/auth.json" 2>/dev/null || true

    # Restore Claude Code CLI credentials from PVC backup.
    # The PVC mount persists across pod restarts. Restore into the hermes-owned
    # subprocess HOME so ClaudeCodeClient (running as hermes) can access them.
    if [ -d "${HERMES_HOME}/.claude_backup" ]; then
        _claude_backup_restored=0
        mkdir -p "${HERMES_HOME}/home/.claude" 2>/dev/null || true
        if [ -f "${HERMES_HOME}/.claude_backup/.credentials.json" ]; then
            if HERMES_HOME="${HERMES_HOME}" python3 - <<'PY'
import json
import os
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
backup = home / ".claude_backup" / ".credentials.json"
active = home / "home" / ".claude" / ".credentials.json"
auth_json = home / "auth.json"

def expires_from_credentials(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth", {}) if isinstance(data, dict) else {}
        return int(oauth.get("expiresAt") or 0)
    except Exception:
        return 0

def expires_from_auth_json(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        providers = data.get("providers", {}) if isinstance(data, dict) else {}
        state = providers.get("claude-code-cli", {}) if isinstance(providers, dict) else {}
        full = state.get("full_credentials", {}) if isinstance(state, dict) else {}
        if not full.get("accessToken") or not full.get("refreshToken"):
            return 0
        return int(state.get("expires_at_ms") or full.get("expiresAt") or 0)
    except Exception:
        return 0

backup_expires = expires_from_credentials(backup)
active_expires = expires_from_credentials(active)
auth_expires = expires_from_auth_json(auth_json)

if (active_expires and active_expires > backup_expires) or (auth_expires and auth_expires > backup_expires):
    raise SystemExit(1)
PY
            then
                cp -f "${HERMES_HOME}/.claude_backup/.credentials.json" "${HERMES_HOME}/home/.claude/.credentials.json" 2>/dev/null || true
                chown hermes:hermes "${HERMES_HOME}/home/.claude/.credentials.json" 2>/dev/null || true
                chmod 600 "${HERMES_HOME}/home/.claude/.credentials.json" 2>/dev/null || true
                _claude_backup_restored=1
            else
                echo "Skipped Claude Code CLI PVC backup restore because newer credentials exist"
            fi
        fi
        if [ -f "${HERMES_HOME}/.claude_backup/claude.json" ]; then
            cp -f "${HERMES_HOME}/.claude_backup/claude.json" "${HERMES_HOME}/home/.claude.json" 2>/dev/null || true
            chown hermes:hermes "${HERMES_HOME}/home/.claude.json" 2>/dev/null || true
            chmod 600 "${HERMES_HOME}/home/.claude.json" 2>/dev/null || true
        fi

        if [ "${_claude_backup_restored}" = "1" ]; then
            echo "Restored Claude Code CLI credentials from PVC backup"
        fi
    fi
    # Symlink ~/.claude → hermes subprocess home so Claude CLI finds credentials
    # for direct shell invocations. The gateway path resolves HOME via
    # _resolve_home_dir(), but CLI invocations outside the gateway use $HOME.
    # Try several likely home paths (HERMES_UID user, then fall back to tusker,
    # then hermes) so the symlink lands wherever the running process is homed.
    HERMES_UID_NAME="$(stat -c %U "$HERMES_HOME" 2>/dev/null || echo hermes)"
    for _candidate in "/home/${HERMES_UID_NAME}" "/home/tusker" "/home/hermes" "/root"; do
        if [ -d "${HERMES_HOME}/home/.claude" ] && [ ! -e "${_candidate}/.claude" ] && [ -d "${_candidate}" ]; then
            ln -s "${HERMES_HOME}/home/.claude" "${_candidate}/.claude" 2>/dev/null && \
                echo "Symlinked ${_candidate}/.claude → ${HERMES_HOME}/home/.claude" && break
        fi
    done

    # Ensure all Claude credential files are owned by hermes.
    # Files written as root (e.g. manual credential pushes via kubectl exec,
    # entrypoint PVC restore, or _write_claude_credentials_file running
    # before the gosu drop) block the hermes runtime from reading them.
    # Checking by name rather than by owner so this also handles files
    # that were chown'd to a non-hermes user by the host.
    find "${HERMES_HOME}" /home/tusker -name '.credentials.json' \
        -exec chown hermes:hermes {} \; \
        -exec chmod 600 {} \; \
        2>/dev/null || true
    find "${HERMES_HOME}" /home/tusker -name '.claude.json' \
        -exec chown hermes:hermes {} \; \
        -exec chmod 600 {} \; \
        2>/dev/null || true

    # Register hermes-tools MCP server for mimo CLI.
    # The mimo CLI discovers MCP servers via .mcp.json in the working directory.
    # This registers a bridge that lets the gateway intercept and proxy tool calls.
    if command -v mimo >/dev/null 2>&1; then
        _mcp_dir="${HERMES_HOME}/home"
        mkdir -p "$_mcp_dir"
        cat > "${_mcp_dir}/.mcp.json" <<'MCPJSON'
{
  "mcpServers": {
    "hermes-tools": {
      "type": "stdio",
      "command": "python3",
      "args": ["/opt/hermes/agent/mimocode_mcp_bridge.py"],
      "env": {
        "HERMES_TOOLS_FILE": "/tmp/hermes_tools.json",
        "HERMES_QUEUE_IN": "/tmp/hermes_queue.in",
        "HERMES_QUEUE_OUT_DIR": "/tmp/hermes_result"
      }
    }
  }
}
MCPJSON
        chown hermes:hermes "${_mcp_dir}/.mcp.json" 2>/dev/null || true
        echo "Registered hermes-tools MCP server for mimo CLI"
    fi

    # Pre-accept mimo free-tier agreement so non-interactive `mimo run --pure`
    # works. Without this, the interactive agreement dialog silently blocks
    # requests in non-TTY mode (the binary can't show the dialog, so it exits
    # with no stdout).
    if command -v mimo >/dev/null 2>&1; then
        _state_dir="${HERMES_HOME}/.local/state/mimocode"
        mkdir -p "$_state_dir"
        kv_file="$_state_dir/kv.json"
        if [ ! -f "$kv_file" ] || ! grep -q "free_agreement_accepted" "$kv_file" 2>/dev/null; then
            cat > "$kv_file" <<'KVJSON'
{
  "locale": "auto",
  "free_agreement_accepted": true
}
KVJSON
            chown hermes:hermes "$kv_file" 2>/dev/null || true
            chmod 600 "$kv_file" 2>/dev/null || true
            echo "Pre-accepted mimo free-tier agreement"
        fi
    fi

    echo "Dropping root privileges"
    exec gosu hermes "$0" "$@"
fi

# --- Running as hermes from here ---
source "${INSTALL_DIR}/.venv/bin/activate"

# Create essential directory structure.  Cache and platform directories
# (cache/images, cache/audio, platforms/whatsapp, etc.) are created on
# demand by the application — don't pre-create them here so new installs
# get the consolidated layout from get_hermes_dir().
# The "home/" subdirectory is a per-profile HOME for subprocesses (git,
# ssh, gh, npm …).  Without it those tools write to /root which is
# ephemeral and shared across profiles.  See issue #4426.
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

# .env
if [ ! -f "$HERMES_HOME/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
fi

# config.yaml
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
    cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
fi

# Enable observability/langfuse plugin if credentials are configured.
# This is idempotent — re-running only adds the entry if absent.
if [ -n "${HERMES_LANGFUSE_PUBLIC_KEY:-}" ] && [ -n "${HERMES_LANGFUSE_SECRET_KEY:-}" ]; then
    python3 - <<'PYEOF'
import yaml, sys
from pathlib import Path
config_path = Path(f"{__import__('os').environ['HERMES_HOME']}/config.yaml")
try:
    data = yaml.safe_load(config_path.read_text()) or {}
    plugins = data.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])
    if "observability/langfuse" not in enabled:
        enabled.append("observability/langfuse")
        config_path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
        print("[entrypoint] observability/langfuse plugin enabled", flush=True)
except Exception as e:
    print(f"[entrypoint] warning: could not enable langfuse plugin: {e}", file=sys.stderr, flush=True)
PYEOF
fi

# SOUL.md
if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"
fi

# Sync bundled skills (manifest-based so user edits are preserved)
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

# Optionally start `hermes dashboard` as a side-process.
#
# Toggled by HERMES_DASHBOARD=1 (also accepts "true"/"yes", case-insensitive).
# Host/port/TUI can be overridden via:
#   HERMES_DASHBOARD_HOST  (default 0.0.0.0 — exposed outside the container)
#   HERMES_DASHBOARD_PORT  (default 9119, matches `hermes dashboard` default)
#   HERMES_DASHBOARD_TUI   (already honored by `hermes dashboard` itself)
#
# The dashboard is a long-lived server.  We background it *before* the final
# `exec hermes "$@"` so the user's chosen foreground command (chat, gateway,
# sleep infinity, …) remains PID-of-interest for the container runtime.  When
# the container stops the whole process tree is torn down, so no explicit
# cleanup is needed.
case "${HERMES_DASHBOARD:-}" in
    1|true|TRUE|True|yes|YES|Yes)
        dash_host="${HERMES_DASHBOARD_HOST:-0.0.0.0}"
        dash_port="${HERMES_DASHBOARD_PORT:-9119}"
        dash_args=(--host "$dash_host" --port "$dash_port" --no-open)
        # Binding to anything other than localhost requires --insecure — the
        # dashboard refuses otherwise because it exposes API keys.  Inside a
        # container this is the expected deployment (host reaches it via
        # published port), so opt in automatically.
        if [ "$dash_host" != "127.0.0.1" ] && [ "$dash_host" != "localhost" ]; then
            dash_args+=(--insecure)
        fi
        echo "Starting hermes dashboard on ${dash_host}:${dash_port} (background)"
        # Prefix dashboard output so it's distinguishable from the main
        # process in `docker logs`.  stdbuf keeps the pipe line-buffered.
        (
            stdbuf -oL -eL hermes dashboard "${dash_args[@]}" 2>&1 \
                | sed -u 's/^/[dashboard] /'
        ) &
        ;;
esac

# Final exec: two supported invocation patterns.
#
#   docker run <image>                 -> exec `hermes` with no args (legacy default)
#   docker run <image> chat -q "..."   -> exec `hermes chat -q "..."` (legacy wrap)
#   docker run <image> sleep infinity  -> exec `sleep infinity` directly
#   docker run <image> bash            -> exec `bash` directly
#
# If the first positional arg resolves to an executable on PATH, we assume the
# caller wants to run it directly (needed by the launcher which runs long-lived
# `sleep infinity` sandbox containers — see tools/environments/docker.py).
# Otherwise we treat the args as a hermes subcommand and wrap with `hermes`,
# preserving the documented `docker run <image> <subcommand>` behavior.
if [ $# -gt 0 ] && command -v "$1" >/dev/null 2>&1; then
    exec "$@"
fi
exec hermes "$@"
