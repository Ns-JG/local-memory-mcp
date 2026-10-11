#!/usr/bin/env bash
# Stop and remove the optional HTTP LaunchAgent.
set -euo pipefail

LABEL="dev.gonczarek.local-memory.http"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

if [[ -e "$DEST" ]]; then
  rm "$DEST"
  echo "removed $DEST"
else
  echo "nothing installed at $DEST"
fi