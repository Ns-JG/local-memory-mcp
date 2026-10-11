"""Tests for the read-only browse viewer.

The important one is :func:`test_browse_issues_no_writes`: it wraps the live
SQLite connection and fails if any non-SELECT statement runs. A read-only
guarantee asserted by reading the code is not a guarantee, and this is the only
cheap way to hold the line as the module grows.
"""

from __future__ import annotations

import io
import re
import sys

import pytest
from conftest import FAKE_DIM, FakeEmbedder
from rich.console import Console

from local_memory import browse
from local_memory.models import Config, Status
from local_memory.service import MemoryService
from local_memory.store import Store

# -- fixtures ---------------------------------------------------------------


def _record_body(index: int) -> str:
    return f"Record number {index} about topic{index % 3} with distinct wording {index * 7}."


#: Scope and kind vary on *different* periods, so every (scope, kind) pairing
#: occurs and filter tests cannot accidentally pass by correlation.
_SCOPES = ("work", "uni", "personal")
_KINDS = ("pattern", "decision", "fact")


@pytest.fixture
def populated(store, embedder):
    """A store with 12 active records spanning every scope and kind."""
    service = MemoryService(store, embedder, dedup_threshold=0.92, rrf_k=60)
    ids = []
    for i in range(12):
        stored, outcome = service.store_record(
            title=f"record {i}",
            body=_record_body(i),
            # Scope cycles every 3, kind every 9, so all nine (scope, kind)
            # pairs occur and no two filters are correlated.
            kind=_KINDS[(i // 3) % len(_KINDS)],
            scope=_SCOPES[i % len(_SCOPES)],
            tags=[f"tag{i % 4}"],
            source="test",
        )
        assert outcome == "created"
        ids.append(stored.id)
    return store, service, ids


#: A real registry entry whose dimension matches ``FAKE_DIM``, so the viewer
#: builds a ``MemoryService`` against vectors the fixture actually wrote.
#: Without this the index raises "stored vectors have dim 256, index expects 384".
_TEST_MODEL = "minishlab/potion-base-8M"


@pytest.fixture(scope="session")
def _test_model_dim_ok():
    """Ensure the test model has the expected dimension.

    Runs once per session at fixture setup time, not at module import.
    """
    from local_memory.models import MODEL_REGISTRY

    assert MODEL_REGISTRY[_TEST_MODEL].dim == FAKE_DIM


@pytest.fixture
def config(tmp_path, _test_model_dim_ok) -> Config:
    return Config(
        db_path=tmp_path / "memory.db",
        model_cache=tmp_path / "models",
        default_model=_TEST_MODEL,
        threads=1,
        scope_paths={},
        dedup_threshold=0.92,
        rrf_k=60,
        tombstone_retention_days=730,
    )


@pytest.fixture
def fake_build(monkeypatch):
    """Make the viewer's lazy embedder resolve to the offline fake.

    ``_LazySearcher`` builds a real ONNX embedder through
    ``build_embedder``; tests must not download a model to exercise search.
    """
    calls: list[str] = []
    embedder = FakeEmbedder()

    def counting(name, cache, threads=None):
        calls.append(name)
        return embedder

    monkeypatch.setattr("local_memory.embedder.build_embedder", counting)
    return calls


def _console(width: int = 120) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=width, force_terminal=False, no_color=True), buf


# -- listing ----------------------------------------------------------------


def test_list_records_returns_newest_first(populated):
    store, _service, ids = populated
    rows = browse.list_records(store, limit=100)
    assert len(rows) == 12
    assert {r.id for r in rows} == set(ids)
    assert rows == sorted(rows, key=lambda r: (r.updated_at, r.id), reverse=True)


def test_list_records_filters_by_scope(populated):
    store, _service, _ids = populated
    rows = browse.list_records(store, scope="work", limit=100)
    assert rows
    assert {r.scope for r in rows} == {"work"}


def test_list_records_filters_by_kind(populated):
    store, _service, _ids = populated
    rows = browse.list_records(store, kind="decision", limit=100)
    assert rows
    assert {r.kind for r in rows} == {"decision"}


def test_list_records_filters_by_status(populated):
    """``statuses=None`` means *no* status filter, not "active only".

    That is deliberate: the point of a browser is to inspect what search hides,
    so the viewer passes statuses explicitly (the CLI defaults to active).
    """
    store, service, ids = populated
    assert len(browse.list_records(store, limit=100)) == 12

    service.forget(ids[0], reason="test")
    assert len(browse.list_records(store, limit=100)) == 12, "unfiltered sees tombstones"

    active = browse.list_records(store, statuses=("active",), limit=100)
    assert len(active) == 11

    pending = browse.list_records(store, statuses=("pending_delete",), limit=100)
    assert [r.id for r in pending] == [ids[0]]

    # Multiple statuses at once.
    both = browse.list_records(store, statuses=("active", "pending_delete"), limit=100)
    assert len(both) == 12


def test_list_records_paginates(populated):
    store, _service, ids = populated
    first = browse.list_records(store, limit=5)
    second = browse.list_records(store, limit=5, offset=5)
    assert len(first) == len(second) == 5
    assert {r.id for r in first}.isdisjoint({r.id for r in second})


def test_count_records_respects_filters(populated):
    store, _service, _ids = populated
    assert browse.count_records(store) == 12
    assert browse.count_records(store, scope="work") == 4
    assert browse.count_records(store, kind="fact") == 3
    # kind = (i // 3) % 3, scope = i % 3, so:
    #   work   -> i = 0, 3, 6, 9  -> kinds pattern, decision, fact, pattern
    #   uni    -> i = 1, 4, 7, 10 -> kinds pattern, decision, fact, pattern
    assert browse.count_records(store, scope="work", kind="pattern") == 2
    assert browse.count_records(store, scope="uni", kind="decision") == 1
    assert browse.count_records(store, scope="uni", kind="fact") == 1


def test_list_records_rejects_bad_limits(populated):
    store, _service, _ids = populated
    with pytest.raises(browse.BrowseError):
        browse.list_records(store, limit=0)
    with pytest.raises(browse.BrowseError):
        browse.list_records(store, offset=-1)


# -- rendering --------------------------------------------------------------


def test_build_table_shows_expected_columns(populated):
    store, _service, _ids = populated
    rows = browse.list_records(store, limit=3)
    console, buf = _console()
    console.print(browse.build_table(rows, total=12, width=120))
    out = buf.getvalue()
    for header in ("TITLE", "KIND", "SCOPE", "CONF", "STATUS", "UPDATED"):
        assert header in out
    for row in rows:
        assert row.title in out
    assert "3 of 12" in out


def test_build_table_empty_state(store):
    console, buf = _console()
    console.print(browse.build_table([], total=0, width=100))
    assert "0 of 0" in buf.getvalue()


def test_build_table_marks_cursor(populated):
    store, _service, _ids = populated
    rows = browse.list_records(store, limit=3)
    console, buf = _console()
    console.print(browse.build_table(rows, cursor=1, total=12, width=120))
    out = buf.getvalue()
    # The marker is a block, and the caption reports the 1-based position.
    assert "▸" in out
    assert "cursor 2" in out


def test_build_table_drops_columns_when_narrow(populated):
    """A narrow terminal must lose the least informative columns, not wrap."""
    store, _service, _ids = populated
    rows = browse.list_records(store, limit=2)
    console, buf = _console(width=46)
    console.print(browse.build_table(rows, total=12, width=46))
    out = buf.getvalue()
    assert "TITLE" in out
    assert "UPDATED" not in out
    assert len(out.splitlines()) <= 8


def test_build_detail_includes_body_and_metadata(populated):
    store, _service, _ids = populated
    record = browse.list_records(store, limit=1)[0]
    console, buf = _console()
    console.print(browse.build_detail(record))
    out = buf.getvalue()
    assert record.title in out
    assert record.body[:30] in out
    assert record.id in out
    assert "confidence" in out
    assert "rev" in out


def test_build_detail_flags_pending_delete(populated):
    store, service, ids = populated
    service.forget(ids[0], reason="because")
    record = store.get(ids[0])
    console, buf = _console()
    console.print(browse.build_detail(record))
    out = buf.getvalue()
    assert "pending_delete" in out
    assert "because" in out
    assert "supersedes" not in out


def test_render_browse_shows_filters_and_help(populated):
    store, _service, _ids = populated
    state = browse.BrowseState(
        records=browse.list_records(store, limit=3),
        total=12,
        scope="work",
        kind="pattern",
        statuses=("active",),
    )
    console, buf = _console()
    for renderable in browse.render_browse(state, width=120):
        console.print(renderable)
    out = buf.getvalue()
    assert "scope=work" in out
    assert "kind=pattern" in out

    console, buf = _console()
    for renderable in browse.render_browse(state, width=120, show_help=True):
        console.print(renderable)
    assert "page_down" in buf.getvalue() or "PgDn" in buf.getvalue()


# -- state machine ----------------------------------------------------------


def test_state_clamp_keeps_cursor_in_range(populated):
    """The cursor is bounded by the result set, not by the loaded window."""
    store, _service, _ids = populated
    state = browse.BrowseState(records=browse.list_records(store, limit=5), total=12, page=5)
    state.move(-10)
    assert state.cursor == 0
    state.move(100)
    assert state.cursor == 11
    state.move(-1)
    assert state.cursor == 10


def test_state_empty_is_safe():
    state = browse.BrowseState()
    state.move(1)
    state.seek_page(3)
    state.clamp()
    assert state.cursor == 0
    assert state.current is None


def test_state_goto_last_uses_total(populated):
    """Home/End address the whole set, so the caller knows to reload a page."""
    store, _service, _ids = populated
    state = browse.BrowseState(records=browse.list_records(store, limit=5), total=12, page=5)
    state.goto(11)
    assert state.cursor == 11
    assert state.cursor > len(state.records) - 1
    state.goto(999)
    assert state.cursor == 11


def test_state_seek_page_moves_by_page(populated):
    """Seeking past the loaded window must be allowed; the caller reloads."""
    store, _service, _ids = populated
    state = browse.BrowseState(records=browse.list_records(store, limit=5), total=12, page=5)
    state.seek_page(1)
    assert state.cursor == 5, "clamp must not pull the cursor back into the window"
    state.seek_page(1)
    assert state.cursor == 10
    state.seek_page(1)
    assert state.cursor == 11, "cannot move past the last record"
    state.seek_page(-3)
    assert state.cursor == 0


# -- key handling -----------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"j", "down"),
        (b"k", "up"),
        (b"\n", "open"),
        (b"/", "search"),
        (b"c", "clear"),
        (b"?", "help"),
        (b"q", "quit"),
        (b"\x1b", "back"),
        (b"\x1b[A", "up"),
        (b"\x1b[B", "down"),
        (b"\x1b[5~", "page_up"),
        (b"\x1b[6~", "page_down"),
        (b"\x1b[H", "home"),
        (b"\x1b[F", "end"),
        (b"g", "home"),
        (b"G", "end"),
        (b"b", "page_down"),
        (b"w", "page_up"),
    ],
)
def test_read_key_maps_bytes_to_actions(data: bytes, expected: str):
    assert browse.read_key(io.BytesIO(data)) == expected


def test_read_key_returns_none_at_eof():
    assert browse.read_key(io.BytesIO(b"")) is None


def test_read_key_treats_unknown_sequence_as_other():
    """An unrecognised CSI must not be mistaken for a bare Escape."""
    assert browse.read_key(io.BytesIO(b"\x1b[200~")) == "other"


def test_read_line_submits_typed_text():
    assert browse.read_line("", io.BytesIO(b"pooling\r")) == ("submit", "pooling")


def test_read_line_edits_and_backspaces():
    assert browse.read_line("", io.BytesIO(b"poolx\x7fing\n")) == ("submit", "pooling")


def test_read_line_starts_from_initial():
    assert browse.read_line("old", io.BytesIO(b"\n")) == ("submit", "old")


def test_read_line_cancels_on_escape():
    assert browse.read_line("", io.BytesIO(b"abc\x1b"))[0] == "cancel"


def test_read_line_aborts_on_ctrl_c():
    assert browse.read_line("", io.BytesIO(b"\x03"))[0] == "abort"


def test_read_line_handles_arrow_keys_without_inserting_them():
    """Arrows are navigation elsewhere; while typing they must be swallowed."""
    action, text = browse.read_line("", io.BytesIO(b"ab\x1b[Dcd\n"))
    assert action == "submit"
    assert text == "abcd"


def test_read_line_preserves_non_ascii():
    action, text = browse.read_line("", io.BytesIO("łączenie\n".encode()))
    assert action == "submit"
    assert text == "łączenie"


def test_read_line_ignores_other_control_characters():
    action, text = browse.read_line("", io.BytesIO(b"a\x07b\n"))
    assert (action, text) == ("submit", "ab")


def test_read_key_accepts_a_text_wrapped_stream():
    """Regression: sys.stdin is text-mode, so reads must go through .buffer.

    Feeding BytesIO hides this completely - ``b'j'`` matches the key table while
    ``'j'`` does not. Only a TextIOWrapper like the real sys.stdin reproduces it.
    """
    # Precondition on a separate instance: TextIOWrapper reads ahead, so consuming
    # from one wrapper would empty the buffer of another over the same BytesIO.
    probe = io.TextIOWrapper(io.BytesIO(b"j"), encoding="utf-8")
    assert probe.read(1) == "j", "precondition: text mode yields str"

    wrapped = io.TextIOWrapper(io.BytesIO(b"j"), encoding="utf-8")
    assert browse._binary(wrapped).read(1) == b"j"

    wrapped = io.TextIOWrapper(io.BytesIO(b"j"), encoding="utf-8")
    assert browse.read_key(browse._binary(wrapped)) == "down"


def test_run_browse_resolves_text_stdin_to_bytes(populated, config, fake_build):
    """A text-wrapped stream must still drive navigation and quit."""
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    # Two downs (jj), then quit (q). Avoids weird CSI sequences.
    wrapped = io.TextIOWrapper(io.BytesIO(b"jjq"), encoding="utf-8")
    code = browse.run_browse(store, config, interactive=True, renderer=rec, stream=wrapped)
    assert code == 0
    # Two downs moved the cursor from 0 to 2 (1-based = 3).
    assert "cursor 3" in rec.text()


def test_read_line_accepts_a_text_wrapped_stream():
    wrapped = io.TextIOWrapper(io.BytesIO(b"query\r"), encoding="utf-8")
    assert browse.read_line("", browse._binary(wrapped)) == ("submit", "query")


# -- interactive loop -------------------------------------------------------


def test_run_browse_quits_on_q(populated, config):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    code = browse.run_browse(store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"q"))
    assert code == 0
    assert "local-memory" in rec.text()


def test_run_browse_opens_detail_and_returns(populated, config):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    browse.run_browse(store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"\nj\x1bq"))
    out = rec.text()
    assert "detail" in out
    assert "confidence" in out


def test_run_browse_navigates_without_crashing(populated, config):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    keys = b"jjkG\x1b[B\x1b[5\x1b[6ghG?j?cq"
    code = browse.run_browse(store, config, interactive=True, renderer=rec, stream=io.BytesIO(keys))
    assert code == 0
    assert len(rec.frames) > 5


def test_run_browse_search_filters_and_clears(populated, config, fake_build):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    browse.run_browse(
        store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"/topic1\nq")
    )
    assert "search:" in rec.text()

    rec = browse._RecordingRenderer()
    browse.run_browse(
        store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"/topic1\ncq")
    )
    # After clearing, the query is gone from the filter summary.
    assert "topic1" not in rec.frames[-1][1].plain


def test_run_browse_search_empty_query_is_noticed(populated, config, fake_build):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    browse.run_browse(store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"/\nq"))
    assert "query empty" in rec.text()


def test_run_browse_search_can_be_cancelled(populated, config, fake_build):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    code = browse.run_browse(
        store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"/abc\x1bq")
    )
    assert code == 0
    assert "abc" in rec.text(), "a cancelled query must not be applied"


def test_run_browse_search_reports_no_matches(populated, config, fake_build):
    """Reachable only when nothing is searchable.

    A nonsense query does *not* produce zero hits: cosine always has a nearest
    neighbour, so the dense channel returns the corpus (see DESIGN.md). The
    empty path is reached when every record is unsearchable.
    """
    store, service, ids = populated
    for record_id in ids:
        service.forget(record_id)
    rec = browse._RecordingRenderer()
    browse.run_browse(
        store, config, interactive=True, renderer=rec, stream=io.BytesIO(b"/topic1\nq")
    )
    assert "no matches" in rec.text()


def test_nonsense_query_returns_the_corpus(populated, config, fake_build):
    """Pin the surprising behaviour so it cannot change unnoticed."""
    store, _service, _ids = populated
    hits = browse._LazySearcher(config, store).search("zzzzqqqq", k=5, scope=None, kind=None)
    assert len(hits) == 5


def test_run_browse_with_initial_search(populated, config, fake_build):
    store, _service, _ids = populated
    rec = browse._RecordingRenderer()
    browse.run_browse(
        store, config, search="topic1", interactive=True, renderer=rec, stream=io.BytesIO(b"q")
    )
    assert "topic1" in rec.text()


def test_run_browse_pages_through_results(config, embedder, fake_build):
    """More records than one page: paging must load the next window."""
    with Store(config.db_path) as store:
        service = MemoryService(store, embedder)
        for i in range(12):
            service.store_record(title=f"paged {i}", body=_record_body(i), source="test")
        rec = browse._RecordingRenderer()
        browse.run_browse(
            store,
            config,
            limit=5,
            interactive=True,
            renderer=rec,
            stream=io.BytesIO(b"b" * 4 + b"q"),
        )
        out = rec.text()
        # Every record must have been reachable, so no page was skipped.
        for i in range(12):
            assert f"paged {i}" in out


# -- static fallback --------------------------------------------------------


def test_run_static_prints_table(populated, config):
    store, _service, _ids = populated
    console, buf = _console()
    browse.run_static(store, config, limit=5, console=console)
    out = buf.getvalue()
    assert "1-5 of 12" in out, "the caption reports the window that was loaded"
    # Exactly the five rows the store returned. Substring matching would be
    # wrong here: "record 1" is a prefix of "record 11".
    rows = browse.list_records(store, limit=5)
    assert len(rows) == 5
    for row in rows:
        assert f" {row.title} " in out


def test_run_static_with_search_prints_detail(populated, config, fake_build):
    store, _service, _ids = populated
    console, buf = _console()
    browse.run_static(store, config, search="topic1", limit=5, console=console)
    assert "detail" in buf.getvalue()


def test_run_static_search_with_nothing_searchable(populated, config, fake_build):
    store, service, ids = populated
    for record_id in ids:
        service.forget(record_id)
    console, buf = _console()
    browse.run_static(store, config, search="topic1", console=console)
    assert "0 of 0" in buf.getvalue()


def test_run_browse_falls_back_when_not_a_tty(populated, config):
    """A non-tty stream must degrade to the static table, not error."""
    store, _service, _ids = populated
    console, buf = _console()
    code = browse.run_browse(
        store, config, interactive=True, console=console, stream=io.BytesIO(b"jjjq")
    )
    assert code == 0
    # Degraded to one static table rather than consuming the keys.
    assert "of 12" in buf.getvalue()


def test_is_interactive_rejects_non_tty(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert browse._is_interactive(io.BytesIO(b"")) is False


def test_is_interactive_rejects_dumb_terminal(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.delenv("CI", raising=False)

    class Tty(io.StringIO):
        def isatty(self):
            return True

    assert browse._is_interactive(Tty()) is False


def test_is_interactive_rejects_ci(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("CI", "true")

    class Tty(io.StringIO):
        def isatty(self):
            return True

    assert browse._is_interactive(Tty()) is False


# -- lazy embedder ----------------------------------------------------------


def test_lazy_searcher_defers_the_model(populated, config, fake_build):
    """Listing must not pay for an ONNX session; only searching may."""
    store, _service, _ids = populated
    searcher = browse._LazySearcher(config, store)
    assert searcher.loaded is False

    browse.list_records(store, limit=5)
    browse.count_records(store)
    assert fake_build == [], "listing must not construct an embedder"

    hits = searcher.search("topic1", k=3, scope=None, kind=None)
    assert fake_build == [config.default_model]
    assert searcher.loaded is True
    assert len(hits) <= 3

    # A second search reuses the cached service rather than rebuilding it.
    searcher.search("topic2", k=3, scope=None, kind=None)
    assert fake_build == [config.default_model]


def test_lazy_searcher_rejects_empty_query(populated, config):
    store, _service, _ids = populated
    with pytest.raises(browse.BrowseError):
        browse._LazySearcher(config, store).search("  ", k=3, scope=None, kind=None)


def test_lazy_searcher_sees_only_searchable(populated, config, fake_build):
    """Search must never surface a pending delete, whatever the listing shows."""
    store, service, ids = populated
    service.forget(ids[0], reason="hidden")
    searcher = browse._LazySearcher(config, store)
    hits = []
    for i in range(12):
        hits.extend(searcher.search(f"topic{i % 3}", k=20, scope=None, kind=None))
    assert ids[0] not in {hit.id for hit in hits}


# -- the read-only guarantee -----------------------------------------------

_WRITE = re.compile(
    r"^\s*(insert|update|delete|replace|create|drop|alter|attach|detach|"
    r"pragma\s+\w+\s*=|vacuum|reindex|begin|commit|rollback|savepoint|release)\b",
    re.IGNORECASE,
)


class _WriteWatch:
    """Records every SQL statement the store runs, via SQLite's trace callback.

    Monkeypatching ``Connection.execute`` is not possible; the attribute is
    read-only. ``set_trace_callback`` is the supported hook and has the
    advantage of seeing statements issued internally by ``executescript`` and
    by triggers, not just the Python-level calls.
    """

    def __init__(self, store):
        self.store = store
        self.statements: list[str] = []
        store._conn.set_trace_callback(self.statements.append)

    def detach(self) -> None:
        self.store._conn.set_trace_callback(None)

    def writes(self) -> list[str]:
        return [s for s in self.statements if _WRITE.match(s.strip())]


@pytest.fixture
def watched(populated):
    store, _service, _ids = populated
    watcher = _WriteWatch(store)
    before = store.epoch()
    try:
        yield store, watcher, before
    finally:
        watcher.detach()


def test_browse_issues_no_writes(watched, config, fake_build):
    """The core invariant: every statement browse runs is a SELECT."""
    store, watcher, epoch_before = watched
    console, _buf = _console()

    # Exercise every read path the viewer has.
    browse.list_records(store, limit=5)
    browse.list_records(store, scope="work", kind="pattern", statuses=("active",), offset=2)
    browse.count_records(store)
    browse.count_records(store, scope="uni")
    browse.list_records(store, limit=3) and browse.build_detail(browse.list_records(store)[0])
    browse.run_static(store, config, limit=5, console=console)
    browse.run_browse(store, config, console=console, stream=io.BytesIO(b"jj/\n\x1bq"))
    browse._LazySearcher(config, store).search("topic1", k=3, scope=None, kind=None)

    assert watcher.writes() == [], f"browse issued writes: {watcher.writes()}"
    assert store.epoch() == epoch_before, "browse bumped the store epoch"


def test_browse_leaves_records_untouched(watched, config):
    store, watcher, _before = watched
    ids = {r.id for r in store.all_records()}
    console, _buf = _console()
    browse.run_browse(store, config, console=console, stream=io.BytesIO(b"jj\nq"))
    assert {r.id for r in store.all_records()} == ids


def test_browse_does_not_deduplicate(populated, config, fake_build):
    """Reading must never reinforce records. Only store() may touch confidence."""
    store, _service, _ids = populated
    watch = _WriteWatch(store)
    before = {r.id: (r.confidence, r.times_confirmed) for r in store.all_records()}
    console, _buf = _console()
    browse._LazySearcher(config, store).search("topic0 topic1 topic2", k=12, scope=None, kind=None)
    assert watch.writes() == []
    after = {r.id: (r.confidence, r.times_confirmed) for r in store.all_records()}
    assert after == before


# -- cli wiring -------------------------------------------------------------


def test_cli_browse_static(populated, config, monkeypatch):
    """End-to-end through cmd_browse, with the store and config stubbed."""
    store, _service, _ids = populated
    monkeypatch.setattr("local_memory.cli.load_config", lambda: config)
    monkeypatch.setattr("local_memory.cli.Store", lambda path: store)
    monkeypatch.setattr("sys.argv", ["local-memory", "browse", "-n", "5"])
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # not a tty -> static path

    from local_memory.cli import main

    assert main() == 0


def test_cli_rejects_unknown_status(config, monkeypatch):
    """argparse rejects it before the command body; assert that contract holds."""
    from local_memory.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["browse", "--status", "not_a_status"])


def test_cli_browse_accepts_every_browsable_status():
    from local_memory.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["browse", "--status", "pending_delete", "--status", "deleted"])
    assert args.status == ["pending_delete", "deleted"]
    assert args.limit == 50
    assert args.no_interactive is False


def test_browsable_statuses_matches_status_enum():
    assert set(browse.BROWSABLE_STATUSES) == {s.value for s in Status}


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO records (id) VALUES ('x')",
        "  update records set title = 'y'",
        "DELETE FROM records WHERE id = 'z'",
        "CREATE TABLE t (a int)",
        "DROP TABLE records",
        "pragma journal_mode=WAL",
        "PRAGMA user_version=2",
        "BEGIN",
        "commit",
        "VACUUM",
        "reindex records_fts",
    ],
)
def test_write_watch_regex_flags_writes(sql: str):
    assert _WRITE.match(sql), f"write regex missed {sql!r}"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM records",
        "  select count(*) from records",
        "PRAGMA user_version",
        "pragma wal_checkpoint(truncate)",
        "WITH x AS (SELECT 1) SELECT * FROM x",
    ],
)
def test_write_watch_regex_allows_reads(sql: str):
    assert not _WRITE.match(sql), f"write regex false-positived on {sql!r}"


def test_browse_does_not_checkpoint_wal(populated, config):
    """A checkpoint is a write disguised as maintenance; browse must not do it."""
    store, _service, _ids = populated
    watch = _WriteWatch(store)
    console, _buf = _console()
    browse.list_records(store, limit=5)
    browse.count_records(store)
    browse.run_static(store, config, console=console)
    assert not any("checkpoint" in s.lower() for s in watch.statements)
