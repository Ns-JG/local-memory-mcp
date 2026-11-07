"""Add or remove one entry in the "instructions" array of an opencode.json(c) file.

    opencode_config.py add    CONFIG_FILE INSTRUCTION_PATH
    opencode_config.py remove CONFIG_FILE INSTRUCTION_PATH

The file is edited as text, not parsed as JSON, so comments and formatting survive.
"""

import re
import sys
from pathlib import Path


def _matching_bracket(text: str, open_idx: int) -> int:
    """Index of the ']' matching the '[' at open_idx, ignoring brackets in strings."""
    depth, in_str, esc = 0, False, False
    for i in range(open_idx, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def add(config_file: Path, instr_path: str) -> int:
    text = config_file.read_text()
    key = re.search(r'"instructions"\s*:\s*\[', text)

    if key:
        start = key.end() - 1
        end = _matching_bracket(text, start)
        if end == -1:
            print("Error: unbalanced 'instructions' array.", file=sys.stderr)
            return 1
        inner = text[start + 1 : end]
        if f'"{instr_path}"' in inner:
            print(f"Instruction path already present: {instr_path}")
            return 0
        if inner.strip() == "":
            new_inner = f' "{instr_path}" '
        else:
            indent = text[text.rfind("\n", 0, key.start()) + 1 : key.start()] + "  "
            body = inner.rstrip()
            new_inner = f'{body},\n{indent}"{instr_path}"{inner[len(body) :]}'
        text = text[: start + 1] + new_inner + text[end:]
        print(f"Added instruction path: {instr_path}")
    else:
        brace = text.find("{")
        if brace == -1:
            print("Error: no top-level object in config.", file=sys.stderr)
            return 1
        block = f'\n  // --- Global Instructions ---\n  "instructions": [\n    "{instr_path}"\n  ],'
        text = text[: brace + 1] + block + text[brace + 1 :]
        print(f"Created 'instructions' array with: {instr_path}")

    config_file.write_text(text)
    print(f"Updated config: {config_file}")
    return 0


def remove(config_file: Path, instr_path: str) -> int:
    text = config_file.read_text()
    entry = re.escape(f'"{instr_path}"')
    for pattern in (entry + r"\s*,\s*", r",\s*" + entry, entry):
        new, n = re.subn(pattern, "", text, count=1)
        if n:
            config_file.write_text(new)
            print(f"Removed instruction path from: {config_file}")
            break
    return 0


def main(argv: list[str]) -> int:
    commands = {"add": add, "remove": remove}
    if len(argv) != 4 or argv[1] not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    return commands[argv[1]](Path(argv[2]), argv[3])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
