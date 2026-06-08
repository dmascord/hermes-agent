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
    # The PVC mount persists across pod restarts, but /root/.claude is
    # ephemeral (and the hermes user's $HOME is the PVC, so $HOME/.claude
    # is also persisted).  Run this as root BEFORE dropping privileges
    # so we can populate /root/.claude for any root-level tool, and also
    # populate the hermes user's $HOME/.claude (= /opt/data/.claude) for
    # tools invoked as the dropped user.  See issue: CrashLoopBackOff
    # after the credential-restore commit moved this block below the
    # gosu drop — the non-root `hermes` user can't write to /root, and
    # `set -e` killed the container on mkdir failure.
    if [ -d "${HERMES_HOME}/.claude_backup" ]; then
        # /root/.claude — for tools that hardcode this path while running as root
        mkdir -p /root/.claude
        if [ -f "${HERMES_HOME}/.claude_backup/.credentials.json" ]; then
            cp -f "${HERMES_HOME}/.claude_backup/.credentials.json" /root/.claude/.credentials.json
            chmod 600 /root/.claude/.credentials.json
        fi
        if [ -f "${HERMES_HOME}/.claude_backup/claude.json" ]; then
            cp -f "${HERMES_HOME}/.claude_backup/claude.json" /root/.claude.json
            chmod 600 /root/.claude.json
        fi
        # $HOME/.claude — for tools that resolve the user-home config dir.
        # We resolve via getent passwd (avoids hardcoding /opt/data), and
        # fall back to the literal path so this still works in stripped
        # images where getent is missing.  Skip if hermes home is unknown.
        _hermes_home=""
        if command -v getent >/dev/null 2>&1; then
            _hermes_home="$(getent passwd hermes 2>/dev/null | cut -d: -f6)"
        fi
        if [ -z "$_hermes_home" ]; then
            _hermes_home="$HERMES_HOME"
        fi
        if [ -n "$_hermes_home" ] && [ "$_hermes_home" != "/" ]; then
            mkdir -p "$_hermes_home/.claude"
            chown hermes:hermes "$_hermes_home/.claude" 2>/dev/null || true
            if [ -f "${HERMES_HOME}/.claude_backup/.credentials.json" ]; then
                cp -f "${HERMES_HOME}/.claude_backup/.credentials.json" \
                    "$_hermes_home/.claude/.credentials.json"
                chown hermes:hermes "$_hermes_home/.claude/.credentials.json" 2>/dev/null || true
                chmod 600 "$_hermes_home/.claude/.credentials.json" 2>/dev/null || true
            fi
            if [ -f "${HERMES_HOME}/.claude_backup/claude.json" ]; then
                cp -f "${HERMES_HOME}/.claude_backup/claude.json" "$_hermes_home/.claude.json"
                chown hermes:hermes "$_hermes_home/.claude.json" 2>/dev/null || true
                chmod 600 "$_hermes_home/.claude.json" 2>/dev/null || true
            fi
        fi
        echo "Restored Claude Code CLI credentials from PVC backup"
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
