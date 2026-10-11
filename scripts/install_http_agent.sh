#!/usr/bin/env bash
# Install the optional HTTP LaunchAgent.
#
# Only needed if you want local-memory reachable over TCP. The stdio server
# that OpenCode and Claude Code use is spawned by the client on demand, so
# there is nothing to daemonise for normal use.
set -euo pipefail

LABEL="dev.gonczarek.local-memory.http"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/com.local-memory.http.plist"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
BINARY="/Users/nsjg/Desktop/priv-projects/local_memory/.venv/bin/local-memory-http"

if [[ ! -x "$BINARY" ]]; then
  echo "error: $BINARY not found." >&2
  echo "Run 'uv sync' in /Users/nsjg/Desktop/priv-projects/local_memory first." >&2
  exit 1
fi

plutil -lint "$SRC" >/dev/null

mkdir -p "$HOME/Library/LaunchAgents"

if [[ -e "$DEST" ]]; then
  echo "replacing existing agent at $DEST"
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
fi

cp "$SRC" "$DEST"
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL" 2>/dev/null || true

sleep 1
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  echo "installed and loaded: $LABEL"
  echo "  listen: 127.0.0.1:8787/mcp"
  echo "  logs:   /tmp/local-memory-http.err.log"
  echo "  stop:   launchctl bootout gui/\$(id -u)/$LABEL"
else
  echo "error: agent failed to load. Check /tmp/local-memory-http.err.log" >&2
  exit 1
fi