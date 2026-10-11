"""stdio transport - the default for OpenCode and Claude Code.

The client spawns this process per session and speaks MCP over stdin/stdout.
Anything written to stdout that is not a protocol frame corrupts the stream, so
diagnostics go to stderr.
"""

from __future__ import annotations

import sys


def main() -> int:
    from .server import build_server

    try:
        server = build_server()
    except Exception as exc:  # noqa: BLE001 - must not raise into the client
        print(f"local-memory: failed to start: {exc}", file=sys.stderr)
        return 1

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
