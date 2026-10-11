"""MCP surface: tool registration, schemas, and error translation.

Uses an in-memory FastMCP client against a fake-embedder service, so this
exercises the tool schemas and the JSON an agent actually receives without
downloading a model or touching the real database.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client, FastMCP

from local_memory.models import Status
from local_memory.server import DETAIL_TIERS, SERVER_INSTRUCTIONS, register_tools
from local_memory.service import MemoryService
from local_memory.store import Store

EXPECTED_TOOLS = {
    "memory_search",
    "memory_get",
    "memory_store",
    "memory_forget",
    "memory_confirm_delete",
    "memory_stats",
}


@pytest.fixture
def service(store: Store, embedder) -> MemoryService:
    return MemoryService(store, embedder)


@pytest.fixture
async def client(service: MemoryService):
    mcp = FastMCP(name="local-memory", instructions=SERVER_INSTRUCTIONS)
    register_tools(mcp, service)
    async with Client(mcp) as connected:
        yield connected


async def payload(result) -> dict:
    """Unwrap an MCP text result into JSON."""
    return json.loads(result.content[0].text)


def text(result) -> str:
    return result.content[0].text


# -- registration -----------------------------------------------------------


async def test_all_tools_are_registered(client: Client) -> None:
    names = {t.name for t in await client.list_tools()}
    assert names == EXPECTED_TOOLS


async def test_every_tool_is_documented(client: Client) -> None:
    for tool in await client.list_tools():
        assert tool.description, f"{tool.name} has no description"
        assert tool.description.strip()


async def test_server_instructions_teach_the_delete_semantics() -> None:
    """The agent must not learn that memory_forget deletes immediately."""
    lowered = SERVER_INSTRUCTIONS.lower()
    assert "supersedes_id" in SERVER_INSTRUCTIONS
    assert "confirm" in lowered
    assert "reinforce" in lowered
    assert "never tell the user" in lowered


# -- store round trip -------------------------------------------------------


async def test_store_then_search_returns_the_record(client: Client, service) -> None:
    stored = await payload(
        await client.call_tool(
            "memory_store",
            {
                "title": "Use uv for dependencies",
                "body": "This project uses uv, never pip install.",
                "kind": "decision",
                "scope": "work",
                "tags": ["python"],
            },
        )
    )
    assert stored["outcome"] == "created"
    assert stored["record"]["rev"] == 1
    assert stored["record"]["tags"] == ["python"]
    assert service.index.contains(stored["record"]["id"])

    found = await payload(await client.call_tool("memory_search", {"query": "uv dependencies"}))
    assert found["count"] == 1
    assert found["results"][0]["id"] == stored["record"]["id"]
    assert "hint" in found


async def test_store_reinforcement_is_explained(client: Client) -> None:
    args = {"title": "Same", "body": "identical body text"}
    await client.call_tool("memory_store", args)
    second = await payload(await client.call_tool("memory_store", args))
    assert second["outcome"] == "reinforced"
    assert "reinforced" in second["note"]


async def test_store_rejects_empty_title_as_a_plain_string(client: Client) -> None:
    result = await client.call_tool("memory_store", {"title": "  ", "body": "b"})
    assert text(result) == "store failed: title must not be empty"


async def test_pending_delete_is_not_a_dedup_target(client: Client, service) -> None:
    """A record awaiting delete confirmation must be free to be re-stored."""
    stored, _ = service.store_record(title="T", body="shared body text")
    service.store.set_status(stored.id, Status.PENDING_DELETE, expected_rev=stored.rev)

    again = await payload(
        await client.call_tool("memory_store", {"title": "T", "body": "shared body text"})
    )
    assert again["outcome"] == "created"
    assert again["record"]["id"] != stored.id


# -- search -----------------------------------------------------------------


@pytest.fixture
async def seeded(client: Client):
    await client.call_tool(
        "memory_store",
        {
            "title": "Postgres connection pooling",
            "body": "Put pgbouncer in front of postgres in production.",
            "kind": "decision",
        },
    )
    await client.call_tool(
        "memory_store",
        {
            "title": "Editor tooling",
            "body": "NsJG uses neovim with lazy.nvim.",
            "kind": "preference",
            "scope": "personal",
        },
    )
    return client


async def test_search_snippets_omit_the_body(seeded: Client) -> None:
    found = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    assert "body" not in found["results"][0]


async def test_search_full_includes_the_body(seeded: Client) -> None:
    found = await payload(
        await seeded.call_tool("memory_search", {"query": "pgbouncer", "detail": "full"})
    )
    assert "pgbouncer" in found["results"][0]["body"]


async def test_search_titles_suppresses_the_gist(seeded: Client) -> None:
    found = await payload(
        await seeded.call_tool("memory_search", {"query": "pgbouncer", "detail": "titles"})
    )
    assert found["results"][0]["gist"] == ""


async def test_search_rejects_bad_detail(seeded: Client) -> None:
    result = await seeded.call_tool("memory_search", {"query": "x", "detail": "everything"})
    assert text(result).startswith("search failed: detail must be one of")


async def test_search_filters_by_scope(seeded: Client) -> None:
    found = await payload(
        await seeded.call_tool("memory_search", {"query": "what do you know", "scope": "personal"})
    )
    assert {r["scope"] for r in found["results"]} == {"personal"}


async def test_search_exposes_per_channel_ranks(seeded: Client) -> None:
    found = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    hit = found["results"][0]
    assert "dense" in hit["found_by"]
    assert hit["dense_rank"] is not None


# -- get --------------------------------------------------------------------


async def test_get_returns_the_full_record(seeded: Client) -> None:
    found = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    record_id = found["results"][0]["id"]
    full = await payload(await seeded.call_tool("memory_get", {"record_id": record_id}))
    assert full["id"] == record_id
    assert "pgbouncer" in full["body"]
    assert "rev" in full


async def test_get_unknown_id_is_a_string_not_a_crash(seeded: Client) -> None:
    result = await seeded.call_tool("memory_get", {"record_id": "no-such-id"})
    assert text(result) == "no record with id 'no-such-id'"


# -- delete handshake -------------------------------------------------------


async def test_forget_reports_pending_not_deleted(seeded: Client) -> None:
    found = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    record_id = found["results"][0]["id"]

    out = await payload(await seeded.call_tool("memory_forget", {"record_id": record_id}))
    assert out["status"] == Status.PENDING_DELETE
    assert out["pending_confirmation"] is True
    assert "not" in out["note"] and "report it as deleted" in out["note"]

    gone = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    assert record_id not in [r["id"] for r in gone["results"]]


async def test_confirm_delete_approve_writes_the_tombstone(seeded: Client) -> None:
    found = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    record_id = found["results"][0]["id"]
    await seeded.call_tool("memory_forget", {"record_id": record_id})

    out = await payload(
        await seeded.call_tool(
            "memory_confirm_delete", {"record_ids": [record_id], "approve": True}
        )
    )
    assert out["results"] == [{"id": record_id, "outcome": "deleted"}]


async def test_confirm_delete_dismiss_restores(seeded: Client) -> None:
    found = await payload(await seeded.call_tool("memory_search", {"query": "pgbouncer"}))
    record_id = found["results"][0]["id"]
    await seeded.call_tool("memory_forget", {"record_id": record_id})

    out = await payload(
        await seeded.call_tool(
            "memory_confirm_delete", {"record_ids": [record_id], "approve": False}
        )
    )
    assert out["results"][0]["outcome"] == "restored"
    back = await payload(await seeded.call_tool("memory_get", {"record_id": record_id}))
    assert back["status"] == Status.ACTIVE


# -- stats ------------------------------------------------------------------


async def test_stats_reports_the_model(seeded: Client) -> None:
    stats = await payload(await seeded.call_tool("memory_stats"))
    assert stats["model"] == "fake/hashed-bow"
    assert stats["total"] == 2
    assert "scope_paths" in stats


def test_detail_tiers_match_the_service() -> None:
    from local_memory.service import DETAIL_TIERS as service_tiers

    assert service_tiers == DETAIL_TIERS
