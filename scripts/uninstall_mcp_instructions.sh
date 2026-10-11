#!/usr/bin/env bash
# Uninstall only this MCP's instructions; leaves the instructions/mcp/ tree intact.

set -euo pipefail

DEST_DIR="${OPENCODE_INSTRUCTIONS_DIR:-}"

# Auto-detect opencode config directory (same logic as installer)
if [[ -z "$DEST_DIR" ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ -d "$HOME/Library/Application Support/opencode" ]]; then
            DEST_DIR="$HOME/Library/Application Support/opencode/instructions/mcp"
        else
            DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/instructions/mcp"
        fi
    else
        DEST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/instructions/mcp"
    fi
fi

TARGET="$DEST_DIR/local_memory.md"

if [[ -f "$TARGET" ]]; then
    rm "$TARGET"
    echo "Removed: $TARGET"
else
    echo "Not found: $TARGET"
fi

# Do NOT remove parent directories — other MCPs may use them