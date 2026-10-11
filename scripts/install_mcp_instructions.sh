#!/usr/bin/env bash

# Install local-memory MCP guide:
# - Global instructions registered in '~/.config/opencode/opencode.jsonc'
# - Detailed instructions installed as a skill at ~/.opencode/skills/local-memory-mcp

set -euo pipefail

# --- Sources (relative to repo root, not $PWD) ---
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
global_memory_instructions="$ROOT/installs/mcp_instructions.md"
local_memory_skill="$ROOT/installs/SKILL.md"

for src in "$global_memory_instructions" "$local_memory_skill"; do
    if [[ ! -f "$src" ]]; then
        echo "Error: source file not found: $src" >&2
        exit 1
    fi
done

# --- Targets (override with env vars for a dry run) ---
INSTRUCTIONS_DIR="${OPENCODE_INSTRUCTIONS_DIR:-$HOME/.config/opencode/instructions}"
SKILL_DIR="${OPENCODE_SKILLS_DIR:-$HOME/.opencode/skills}"
CONFIG_DIR="${OPENCODE_CONFIG_DIR:-$HOME/.config/opencode}"

# --- Install files ---
mkdir -p "$INSTRUCTIONS_DIR"
mkdir -p "$SKILL_DIR/local-memory-mcp"

GLOBAL_INSTRUCTIONS_FILE="$INSTRUCTIONS_DIR/local_memory_mcp_instructions.md"
GLOBAL_SKILL_FILE="$SKILL_DIR/local-memory-mcp/SKILL.md"

cp "$global_memory_instructions" "$GLOBAL_INSTRUCTIONS_FILE"
echo "Installed instructions: $GLOBAL_INSTRUCTIONS_FILE"

cp "$local_memory_skill" "$GLOBAL_SKILL_FILE"
echo "Installed skill: $GLOBAL_SKILL_FILE"

# --- Link instructions to config ---
# Skills are picked up from the skills dir automatically and are NOT registered in
# the config. Only the global instructions path is added to the "instructions"
# array. We parse the file as text (not JSON) so comments and formatting survive.
CONFIG_FILE=""
for candidate in "$CONFIG_DIR/opencode.jsonc" "$CONFIG_DIR/opencode.json"; do
    if [[ -f "$candidate" ]]; then
        CONFIG_FILE="$candidate"
        break
    fi
done

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Warning: No opencode config found. Checked:"
    echo "  $CONFIG_DIR/opencode.jsonc"
    echo "  $CONFIG_DIR/opencode.json"
    exit 0
fi

echo "Found config: $CONFIG_FILE"

python3 - "$CONFIG_FILE" "$GLOBAL_INSTRUCTIONS_FILE" <<'PYEOF'
import re
import sys

config_file, instr_path = sys.argv[1], sys.argv[2]

with open(config_file) as f:
    text = f.read()


def find_matching_bracket(s, open_idx):
    """Index of the ']' matching the '[' at open_idx, ignoring brackets in strings."""
    depth = 0
    in_str = False
    esc = False
    for i in range(open_idx, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


key = re.search(r'"instructions"\s*:\s*\[', text)

if key:
    arr_start = key.end() - 1
    arr_end = find_matching_bracket(text, arr_start)
    if arr_end == -1:
        print("Error: unbalanced 'instructions' array.", file=sys.stderr)
        sys.exit(1)

    inner = text[arr_start + 1:arr_end]
    if f'"{instr_path}"' in inner:
        print(f"Instruction path already present: {instr_path}")
    else:
        if inner.strip() == "":
            new_inner = f' "{instr_path}" '
        else:
            line_start = text.rfind("\n", 0, key.start()) + 1
            key_indent = text[line_start:key.start()]
            entry_indent = key_indent + "  "
            body = inner.rstrip()
            trailing = inner[len(body):]
            new_inner = f'{body},\n{entry_indent}"{instr_path}"{trailing}'
        text = text[:arr_start + 1] + new_inner + text[arr_end:]
        print(f"Added instruction path: {instr_path}")
else:
    brace = text.find("{")
    if brace == -1:
        print("Error: no top-level object in config.", file=sys.stderr)
        sys.exit(1)
    block = (
        f'\n  // --- Global Instructions ---'
        f'\n  "instructions": ['
        f'\n    "{instr_path}"'
        f'\n  ],'
    )
    text = text[:brace + 1] + block + text[brace + 1:]
    print(f"Created 'instructions' array with: {instr_path}")

with open(config_file, "w") as f:
    f.write(text)

print(f"Updated config: {config_file}")
PYEOF
