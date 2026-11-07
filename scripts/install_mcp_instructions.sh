#!/usr/bin/env bash

# Install local-memory MCP guide (remove it with --uninstall). The templates in
# installs/ are filled in from the active config.toml; an existing installed copy
# is overwritten after being saved to backups/.
# - Global instructions registered in '~/.config/opencode/opencode.jsonc'
# - Detailed instructions installed as a skill at ~/.opencode/skills/local-memory-mcp

set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
source "$ROOT/scripts/helpers.sh"

# --- Targets (override with env vars for a dry run) ---
INSTRUCTIONS_DIR="${OPENCODE_INSTRUCTIONS_DIR:-$HOME/.config/opencode/instructions}"
SKILL_DIR="${OPENCODE_SKILLS_DIR:-$HOME/.opencode/skills}"
CONFIG_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"
GLOBAL_INSTRUCTIONS_FILE="$INSTRUCTIONS_DIR/local_memory_mcp_instructions.md"
GLOBAL_SKILL_FILE="$SKILL_DIR/local-memory-mcp/SKILL.md"

CONFIG_FILE=""
for candidate in "$CONFIG_DIR/opencode.jsonc" "$CONFIG_DIR/opencode.json"; do
    if [[ -f "$candidate" ]]; then
        CONFIG_FILE="$candidate"
        break
    fi
done

# --- Uninstall: remove only this MCP's files and its config entry ---
if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$GLOBAL_INSTRUCTIONS_FILE" "$GLOBAL_SKILL_FILE"
    rmdir "$SKILL_DIR/local-memory-mcp" 2>/dev/null || true
    echo "Removed: $GLOBAL_INSTRUCTIONS_FILE"
    echo "Removed: $GLOBAL_SKILL_FILE"
    if [[ -n "$CONFIG_FILE" ]]; then
        python3 "$ROOT/scripts/opencode_config.py" remove "$CONFIG_FILE" "$GLOBAL_INSTRUCTIONS_FILE"
    fi
    # Parent directories stay: other MCPs may use them.
    exit 0
fi

# --- Sources (relative to repo root, not $PWD) ---
global_memory_instructions="$ROOT/installs/mcp_instructions.md"
local_memory_skill="$ROOT/installs/SKILL.md"

for src in "$global_memory_instructions" "$local_memory_skill"; do
    if [[ ! -f "$src" ]]; then
        echo "Error: source file not found: $src" >&2
        exit 1
    fi
done

# --- Install files (rendered with the scopes and kinds from the active config.toml) ---
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Error: $PYTHON not found. Run 'uv sync' in $ROOT first." >&2
    exit 1
fi

mkdir -p "$INSTRUCTIONS_DIR"
mkdir -p "$SKILL_DIR/local-memory-mcp"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

install_rendered() {
    local template="$1" dest="$2" label="$3"
    "$PYTHON" "$ROOT/scripts/render_template.py" "$template" > "$TMP/rendered"
    backup_if_changed "$dest" "$TMP/rendered" "$ROOT/backups"
    cp "$TMP/rendered" "$dest"
    echo "Installed $label: $dest"
}

install_rendered "$global_memory_instructions" "$GLOBAL_INSTRUCTIONS_FILE" "instructions"
install_rendered "$local_memory_skill" "$GLOBAL_SKILL_FILE" "skill"

# --- Link instructions to config ---
# Skills are picked up from the skills dir automatically and are NOT registered in
# the config. Only the global instructions path is added to the "instructions"
# array. We parse the file as text (not JSON) so comments and formatting survive.
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Warning: No opencode config found. Checked:"
    echo "  $CONFIG_DIR/opencode.jsonc"
    echo "  $CONFIG_DIR/opencode.json"
    exit 0
fi

echo "Found config: $CONFIG_FILE"

python3 "$ROOT/scripts/opencode_config.py" add "$CONFIG_FILE" "$GLOBAL_INSTRUCTIONS_FILE"
