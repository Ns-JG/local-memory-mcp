#!/usr/bin/env bash
# Install the config template and the MCP instructions to ~/.config/local-memory.
#
#   ./scripts/install_config.sh              install (never overwrites existing files)
#   ./scripts/install_config.sh --uninstall  back up config.toml to backups/, then remove both
#
# The installed config.toml is the persistent store for your choices and is
# re-read on every call, so an existing copy is left alone.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
DEST="$HOME/.config/local-memory"
FILES=(config.toml local_memory_instructions.md)

if [[ "${1:-}" == "--uninstall" ]]; then
  if [[ -f "$DEST/config.toml" ]]; then
    mkdir -p "$ROOT/backups"
    cp "$DEST/config.toml" "$ROOT/backups/config.toml.bak"
    echo "backed up config.toml to $ROOT/backups/config.toml.bak"
  fi
  rm -f "${FILES[@]/#/$DEST/}"
  rmdir "$DEST" 2>/dev/null || true
  echo "uninstalled from $DEST"
  exit 0
fi

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  if [[ -e "$DEST/$f" ]]; then
    echo "kept existing $DEST/$f"
  else
    cp "$ROOT/$f" "$DEST/$f"
    echo "installed $DEST/$f"
  fi
done
