"""Transport adapters.

The HTTP tests bind a real socket. Loopback only, and skipped when the
environment forbids binding, so the suite still runs in a locked-down sandbox.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_memory.config import ConfigManager
from local_memory.mcp import http as mcp_http
from local_memory.mcp import stdio as mcp_stdio

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
TESTS_DIR = Path(__file__).parent
TEMPLATE = TESTS_DIR.parent / "config.toml"

# Runs the real transport in a subprocess, with the model swapped for the fake.
DRIVER = f"""
import sys
sys.path.insert(0, {str(TESTS_DIR)!r})
from conftest import FakeEmbedder
import local_memory.app as app
app.build_embedder = lambda *args, **kwargs: FakeEmbedder()
from local_memory.mcp.http import main
raise SystemExit(main())
"""


def _write_config(path: Path, *, host: str = "127.0.0.1", port: int = 8787) -> Path:
    """The template with db, model cache and [server] pointed at tmp."""
    text = TEMPLATE.read_text()
    for key, value in {
        "db_path": f'"{path.parent / "memory.db"}"',
        "model_cache": f'"{path.parent / "models"}"',
        "host": f'"{host}"',
        "port": port,
    }.items():
        text = re.sub(rf"(?m)^{key} = .*$", f"{key} = {value}", text)
    path.write_text(text)
    shutil.copy(TEMPLATE.with_name("local_memory_instructions.md"), path.parent)
    return path


@pytest.fixture
def http_config(tmp_path, monkeypatch):
    """Call with host/port; ConfigManager() then resolves to that file."""

    def write(host: str = "127.0.0.1", port: int = 8787) -> None:
        path = _write_config(tmp_path / "config.toml", host=host, port=port)
        monkeypatch.setattr(ConfigManager, "GLOBAL_PATH", path)

    return write


def _server_env(tmp_path: Path, port: int) -> dict[str, str]:
    """HOME points at tmp, so ConfigManager finds our config as the global one."""
    config = tmp_path / ".config" / "local-memory" / "config.toml"
    config.parent.mkdir(parents=True)
    _write_config(config, port=port)
    return {**os.environ, "HOME": str(tmp_path), "FASTMCP_SHOW_SERVER_BANNER": "0"}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


# -- stdio ------------------------------------------------------------------


def test_stdio_module_exposes_main() -> None:
    assert callable(mcp_stdio.main)


# -- http bind guard --------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("192.168.1.5", False),
    ],
)
def test_http_refuses_non_loopback(monkeypatch, capsys, http_config, host: str, ok: bool) -> None:
    """This server has no authentication; it must never be reachable off-host."""
    http_config(host=host)

    started: list[str] = []

    class FakeServer:
        def run(self, **kwargs):
            started.append(kwargs["host"])

    monkeypatch.setattr("local_memory.mcp.server.build_server", lambda: FakeServer())

    code = mcp_http.main()
    assert (code == 0) is ok
    if not ok:
        assert "refusing to bind" in capsys.readouterr().out
        assert not started
    else:
        assert started == [host]


def test_template_server_binds_loopback() -> None:
    assert ConfigManager(TEMPLATE).load().server.host in LOOPBACK


# -- http end to end --------------------------------------------------------


def test_http_transport_serves_tools(tmp_path) -> None:
    """Boot the real server over TCP and talk to it as an MCP client."""
    from fastmcp import Client

    port = _free_port()
    env = _server_env(tmp_path, port)
    proc = subprocess.Popen(
        [sys.executable, "-c", DRIVER],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_for_port(port):
            output = proc.stdout.read().decode() if proc.stdout else ""
            pytest.skip(f"could not bind loopback port {port}: {output[-500:]}")

        import asyncio
        import json

        async def exercise() -> dict:
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                names = sorted(t.name for t in await client.list_tools())
                stored = await client.call_tool(
                    "memory_store",
                    {"title": "HTTP works", "body": "served over streamable http"},
                )
                found = await client.call_tool("memory_search", {"query": "streamable http"})
                return {
                    "tools": names,
                    "outcome": json.loads(stored.content[0].text)["outcome"],
                    "count": json.loads(found.content[0].text)["count"],
                }

        result = asyncio.run(exercise())
        assert result["tools"] == [
            "memory_confirm_delete",
            "memory_forget",
            "memory_get",
            "memory_search",
            "memory_stats",
            "memory_store",
        ]
        assert result["outcome"] == "created"
        assert result["count"] == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_http_enables_origin_protection(monkeypatch, http_config) -> None:
    """A web page the user has open must not be able to reach loopback MCP.

    Binding to 127.0.0.1 stops remote hosts but not the user's own browser:
    any origin can POST to 127.0.0.1:<port>/mcp. FastMCP's Host/Origin check
    is off by default in 4.0.10, so the adapter must request it explicitly.
    """
    captured: dict = {}

    class FakeServer:
        def run(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("local_memory.mcp.server.build_server", lambda: FakeServer())
    http_config(port=9999)

    assert mcp_http.main() == 0
    assert captured["transport"] == "http"
    assert captured["host_origin_protection"] is True
    assert "127.0.0.1:9999" in captured["allowed_hosts"]
    assert "localhost:9999" in captured["allowed_hosts"]


def test_http_origin_protection_blocks_a_foreign_origin(tmp_path) -> None:
    """Integration: a browser-supplied Origin must be refused with 403."""
    import json
    import urllib.error
    import urllib.request

    from fastmcp import Client

    port = _free_port()
    env = _server_env(tmp_path, port)
    proc = subprocess.Popen(
        [sys.executable, "-c", DRIVER],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_port(port):
            pytest.skip(f"could not bind loopback port {port}")

        import asyncio

        async def still_works() -> int:
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                return len(await client.list_tools())

        assert asyncio.run(still_works()) == 6

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/mcp",
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
            headers={
                "Host": f"127.0.0.1:{port}",
                "Origin": "https://evil.example",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=5)
        assert excinfo.value.code == 403
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
