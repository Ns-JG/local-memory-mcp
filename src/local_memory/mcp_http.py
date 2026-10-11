"""HTTP transport.

The same service and the same tools, reachable over the network for agents that
cannot spawn a subprocess. Bind to loopback by default: this is a personal
memory store with no authentication, and it must not be reachable off-host.
"""

from __future__ import annotations

import os

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


def main() -> int:
    from .server import build_server

    host = os.environ.get("LOCAL_MEMORY_HOST", DEFAULT_HOST)
    port = int(os.environ.get("LOCAL_MEMORY_PORT", DEFAULT_PORT))
    if host not in {"127.0.0.1", "::1", "localhost"}:
        print(
            f"local-memory: refusing to bind {host!r}; this server has no "
            "authentication and must stay on loopback",
            flush=True,
        )
        return 1

    build_server().run(
        transport="http",
        host=host,
        port=port,
        # Binding to loopback stops remote hosts, but not a web page the user
        # happens to have open: any origin can POST to 127.0.0.1:<port>/mcp
        # and read the response. FastMCP's Host/Origin check rejects those.
        # It is off by default in 4.0.10, so it is requested explicitly here.
        host_origin_protection=True,
        allowed_hosts=[f"{host}:{port}", f"localhost:{port}"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
