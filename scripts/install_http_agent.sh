#!/usr/bin/env bash
# Install the optional HTTP LaunchAgent, or remove it with --uninstall.
#
# Only needed if you want local-memory reachable over TCP. The stdio server
# that OpenCode and Claude Code use is spawned by the client on demand, so
# there is nothing to daemonise for normal use.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
LABEL="com.local-memory.http"
SRC="$ROOT/scripts/com.local-memory.http.plist"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
BINARY="$ROOT/.venv/bin/local-memory-http"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  if [[ -e "$DEST" ]]; then
    rm "$DEST"
    echo "removed $DEST"
  else
    echo "nothing installed at $DEST"
  fi
  exit 0
fi

if [[ ! -x "$BINARY" ]]; then
  echo "error: $BINARY not found." >&2
  echo "Run 'uv sync' in $ROOT first." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"

if [[ -e "$DEST" ]]; then
  echo "replacing existing agent at $DEST"
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
fi

sed -e "s|__LABEL__|$LABEL|" -e "s|__BINARY__|$BINARY|" "$SRC" > "$DEST"
plutil -lint "$DEST" >/dev/null
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL" 2>/dev/null || true

sleep 1
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  echo "installed and loaded: $LABEL"
  echo "  listen: [server] host/port from config.toml, path /mcp"
  echo "  logs:   /tmp/local-memory-http.err.log"
  echo "  stop:   launchctl bootout gui/\$(id -u)/$LABEL"
else
  echo "error: agent failed to load. Check /tmp/local-memory-http.err.log" >&2
  exit 1
fi