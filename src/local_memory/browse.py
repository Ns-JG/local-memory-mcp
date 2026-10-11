"""Interactive read-only viewer for the memory store.

Deliberately a **read path**, not a database browser. It goes through
:class:`~local_memory.store.Store` read methods rather than handing you raw
tables, so it shows decoded tags, the status lifecycle and confirmation counts
instead of five ``records_fts_*`` shadow tables and a float32 blob.

Two invariants this module must not break:

1. **No writes.** Every statement issued here is a ``SELECT``. Editing through a
   viewer would bypass ``rev``, ``store_epoch`` and vector recomputation, which
   is how optimistic concurrency silently stops working. ``tests/test_browse.py``
   asserts this by counting non-SELECT statements executed against a real store.
2. **No model load on startup.** Building the ONNX session costs ~0.2s, which a
   listing command has no business paying. The embedder is constructed only when
   a search is actually requested, and cached for later ones.
"""

from __future__ import annotations

import io
import os
import sys
import termios
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, cast

from rich.box import SIMPLE_HEAD
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import Config, MemoryRecord, SearchHit, Status

if TYPE_CHECKING:
    from .store import Store


class _Renderer(Protocol):
    def __enter__(self) -> _Renderer: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None: ...
    def update(self, renderables: tuple) -> None: ...
    def refresh(self) -> None: ...


#: Default page size. Matches the viewer's own scrolling step.
DEFAULT_PAGE = 50

#: Column widths for the table, in order. The console total is checked against
#: ``--width`` and degrades by dropping trailing columns.
_COLUMNS: tuple[tuple[str, str, int, bool], ...] = (
    # (header, key, min_width, flexible)
    ("", "#", 3, False),
    ("TITLE", "title", 24, True),
    ("KIND", "kind", 12, False),
    ("SCOPE", "scope", 9, False),
    ("CONF", "conf", 5, False),
    ("×", "confirmed", 3, False),
    ("UPDATED", "updated", 16, False),
    ("STATUS", "status", 14, False),
)

#: Statuses selectable on the command line. Not just ``SEARCHABLE``: the whole
#: point of a browser is to inspect what search deliberately hides.
BROWSABLE_STATUSES: tuple[str, ...] = tuple(s.value for s in Status)

_STATUS_STYLE: dict[str, str] = {
    Status.ACTIVE.value: "bold green",
    Status.PENDING_DELETE.value: "yellow",
    Status.SUPERSEDED.value: "dim red",
    Status.DELETED.value: "dim red",
}

_HELP = """[bold]keys[/bold]
  j / Down      next record
  k / Up        previous record
  g / Home      first record
  G / End       last record
  PgDn / b      next page
  PgUp / w      previous page
  Enter         show full record
  Esc / q      back to list, then quit
  /             hybrid search over the current filters
  c             clear search, back to the plain listing
  ?             this help
  q             quit
"""


class BrowseError(Exception):
    """Viewer-level failure, rendered without a traceback."""


# -- data access ------------------------------------------------------------


def list_records(
    store: Store,
    *,
    scope: str | None = None,
    kind: str | None = None,
    statuses: tuple[str, ...] | None = None,
    limit: int = DEFAULT_PAGE,
    offset: int = 0,
) -> list[MemoryRecord]:
    """Newest-first page of records matching the filters.

    A thin pass-through to :meth:`Store.list_records`; the filtering, ordering
    and pagination already live there and are tested there.
    """
    if limit < 1:
        raise BrowseError(f"limit must be >= 1, got {limit}")
    if offset < 0:
        raise BrowseError(f"offset must be >= 0, got {offset}")
    return store.list_records(
        scope=scope,
        kind=kind,
        statuses=statuses,
        limit=limit,
        offset=offset,
    )


def count_records(
    store: Store,
    *,
    scope: str | None = None,
    kind: str | None = None,
    statuses: tuple[str, ...] | None = None,
) -> int:
    """Total matching the filters, ignoring pagination.

    ``Store.count`` takes no scope/kind today, so scope and kind are applied by
    fetching the candidate ids and counting them. That is O(n) in records, which
    is irrelevant at the 10^2-10^4 scale this store targets; if it ever is not,
    extend ``Store.count`` rather than optimising here.
    """
    if scope is None and kind is None:
        return store.count(statuses=statuses)
    if statuses is not None:
        return len(store.candidate_ids(scope=scope, kind=kind, statuses=statuses))
    return len(store.candidate_ids(scope=scope, kind=kind))


class _LazySearcher:
    """Builds a :class:`MemoryService` on first search, then reuses it.

    Constructing the embedder and index is what costs ~0.2s and a model cache
    stat, so it is deferred until a query is actually typed.
    """

    def __init__(self, config: Config, store: Store) -> None:
        self._config = config
        self._store = store
        self._service: object | None = None

    @property
    def loaded(self) -> bool:
        return self._service is not None

    def _ensure(self) -> Any:
        if self._service is None:
            from .embedder import build_embedder
            from .models import resolve_model
            from .service import MemoryService

            spec = resolve_model(self._config.default_model)
            embedder = build_embedder(
                spec.name, self._config.model_cache, threads=self._config.threads
            )
            self._service = MemoryService(
                self._store,
                embedder,
                dedup_threshold=self._config.dedup_threshold,
                rrf_k=self._config.rrf_k,
            )
        return self._service

    def search(
        self,
        query: str,
        *,
        k: int,
        scope: str | None,
        kind: str | None,
    ) -> list[SearchHit]:
        """Run a hybrid search through the real service path.

        Note that ``statuses`` is not a parameter here: search only ever sees
        ``SEARCHABLE``. A search result set therefore cannot show pending deletes
        or tombstones, which is correct behaviour and slightly surprising if you
        came from a listing that did show them.
        """
        if not query.strip():
            raise BrowseError("search query is empty")
        service = self._ensure()
        hits = service.search(query, k=k, scope=scope, kind=kind, detail="snippets")
        return cast(list[SearchHit], hits)


# -- rendering --------------------------------------------------------------


def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _cell(record: MemoryRecord, key: str, width: int) -> Text:
    if key == "title":
        style = "dim" if record.status != Status.ACTIVE else ""
        return Text(_truncate(record.title, width), style=style)
    if key == "kind":
        return Text(_truncate(record.kind, width), style="cyan")
    if key == "scope":
        return Text(_truncate(record.scope, width), style="magenta")
    if key == "conf":
        return Text(f"{record.confidence:.2f}")
    if key == "confirmed":
        return Text(str(record.times_confirmed) if record.times_confirmed else "", style="bold")
    if key == "updated":
        return Text(record.updated_at.replace("T", " ")[:16], style="dim")
    if key == "status":
        if record.status == Status.ACTIVE:
            return Text("")
        return Text(record.status, style=_STATUS_STYLE.get(record.status, ""))
    return Text("")


def build_table(
    records: list[MemoryRecord],
    *,
    cursor: int = 0,
    total: int | None = None,
    offset: int = 0,
    page: int = DEFAULT_PAGE,
    width: int = 100,
    title: str | None = None,
) -> Table:
    """Build the record table.

    Columns are dropped from the right when the console is too narrow, so a
    narrow terminal loses the least informative fields first rather than
    wrapping every column into unreadable stacks.
    """
    # Each column costs its width plus one space of padding on each side,
    # roughly. "#" and "TITLE" are the two columns never dropped: the cursor
    # marker needs somewhere to live, and a title is the point of the table.
    keep_keys = {"#", "title"}
    budget = width - 4
    columns = list(_COLUMNS)
    while len(columns) > len(keep_keys):
        needed = sum(w for _, _, w, _ in columns) + 2 * len(columns)
        if needed <= budget:
            break
        for i in range(len(columns) - 1, -1, -1):
            if columns[i][1] not in keep_keys:
                del columns[i]
                break
        else:
            break

    table = Table(
        box=SIMPLE_HEAD,
        expand=True,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        title=title,
        title_style="bold",
        header_style="dim bold",
    )
    for header, _key, col_width, flexible in columns:
        table.add_column(header, width=None if flexible else col_width, no_wrap=not flexible)

    for i, record in enumerate(records):
        index = offset + i
        row = [_cell(record, key, w if not flex else max(w, 12)) for _h, key, w, flex in columns]
        if columns[0][1] == "#":
            marker = Text("▸" if index == cursor else " ")
            marker.style = "bold reverse" if index == cursor else ""
            row[0] = marker
        table.add_row(*row, style="reverse" if index == cursor else "")

    shown = f"{offset + 1}-{offset + len(records)}" if records else "0"
    caption = f"{shown} of {total if total is not None else '?'}  ·  cursor {cursor + 1}"
    table.caption = Text(caption, style="dim")
    table.caption_justify = "left"
    return table


def build_detail(record: MemoryRecord) -> Panel:
    """Full single-record view: metadata block, then the body verbatim."""
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim", justify="right")
    meta.add_column()

    meta.add_row("id", record.id)
    if record.supersedes_id:
        meta.add_row("supersedes", Text(record.supersedes_id, style="yellow"))
    meta.add_row("status", Text(record.status, style=_STATUS_STYLE.get(record.status, "")))
    meta.add_row("kind", Text(record.kind, style="cyan"))
    meta.add_row("scope", Text(record.scope, style="magenta"))
    meta.add_row("confidence", f"{record.confidence:.2f}")
    meta.add_row("confirmed", str(record.times_confirmed))
    meta.add_row("source", record.source)
    if record.tags:
        meta.add_row("tags", Text(", ".join(record.tags)))
    meta.add_row("created", record.created_at.replace("T", " "))
    meta.add_row("updated", record.updated_at.replace("T", " "))
    if record.deleted_at:
        meta.add_row("deleted", record.deleted_at.replace("T", " "))
    if record.delete_reason:
        meta.add_row("reason", record.delete_reason)
    meta.add_row("rev", Text(str(record.rev), style="dim"))
    if record.embedding_model:
        model = f"{record.embedding_model} ({record.embedding_dim}d)"
        if record.embedded_at:
            model += f" at {record.embedded_at.replace('T', ' ')}"
        meta.add_row("embedded", model)
    if record.origin_device:
        meta.add_row("device", record.origin_device)

    body = Text(record.body or "(empty body)")
    return Panel(
        Group(Panel(meta, title=Text(record.title, style="bold"), border_style="dim"), body),
        border_style="dim",
        padding=(0, 1),
        title="detail",
        title_align="left",
    )


# -- key input --------------------------------------------------------------

#: Escape sequences mapped to logical actions.
_KEYS: dict[bytes, str] = {
    b"j": "down",
    b"k": "up",
    b"\n": "open",
    b"\r": "open",
    b"/": "search",
    b"c": "clear",
    b"?": "help",
    b"q": "quit",
    b"\x1b": "back",
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "page_down",
    b"\x1b[D": "page_up",
    b"\x1b[H": "home",
    b"\x1b[F": "end",
    b"\x1b[5~": "page_up",
    b"\x1b[6~": "page_down",
    b"g": "home",
    b"G": "end",
    b"b": "page_down",
    b"w": "page_up",
    b" ": "open",
}


#: Control bytes meaningful when editing a query line.
_LINE_CTRL: dict[bytes, str] = {
    b"\n": "submit",
    b"\r": "submit",
    b"\x1b": "cancel",
    b"\x7f": "backspace",
    b"\x08": "backspace",
    b"\x03": "abort",
}


def read_key(stream: BinaryIO | None = None) -> str | None:
    """Read one logical keypress, or ``None`` at EOF.

    Raw bytes rather than :mod:`readline` or :mod:`termios`: arrow keys arrive as
    multi-byte escape sequences, and this maps the common ones without depending
    on terminal state. Unknown sequences resolve to ``"other"`` so an
    unrecognised CSI does not get mistaken for a bare Escape.

    The stream is converted to binary via ``_binary()`` so callers may pass
    ``sys.stdin`` (text-mode) or an already-binary stream.
    """
    stream = _binary(stream or sys.stdin.buffer)
    first = stream.read(1)
    if not first:
        return None
    if first != b"\x1b":
        return _KEYS.get(first, "other")

    # Escape: either a bare Esc, or the start of a CSI sequence.
    second = stream.read(1)
    if not second:
        return "back"
    if second != b"[":
        return "other"
    seq = b""
    while True:
        ch = stream.read(1)
        if not ch:
            break
        seq += ch
        if ch.isalpha() or ch == b"~":
            break
    return _KEYS.get(b"\x1b[" + seq, "other")


def read_line(
    initial: str = "",
    stream: BinaryIO | None = None,
    on_change: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Read an edited query line one character at a time.

    Returns ``(action, text)`` where action is ``submit``, ``cancel`` or
    ``abort``. This is separate from :func:`read_key` because the two need
    different things: navigation maps bytes to actions, while text entry needs
    the bytes themselves. A UTF-8 continuation byte is preserved so
    non-ASCII queries survive, and a multi-byte character is reassembled before
    being appended.

    A raw Ctrl-C arrives as ``0x03``; it is returned as ``abort`` rather than
    raised, so the viewer can exit cleanly and let the shell print ``^C``
    instead of dumping a traceback.

    ``on_change`` is invoked with the buffer after every edit. Without it the
    prompt would block until submission and the user would be typing blind,
    since this loop owns the thread while it runs.

    The stream is converted to binary via ``_binary()`` so callers may pass
    ``sys.stdin`` (text-mode) or an already-binary stream.
    """
    stream = _binary(stream or sys.stdin.buffer)
    buffer = initial
    pending = b""
    if on_change is not None:
        on_change(buffer)
    while True:
        byte = stream.read(1)
        if not byte:
            return ("submit", buffer)

        if byte == b"\x1b":
            following = stream.read(1)
            if not following:
                return ("cancel", buffer)
            if following != b"[":
                # ESC followed by another key: that key was itself a cancel or
                # quit request, which this reader does not model. Treat the
                # pair as cancel rather than losing the second keypress.
                return ("cancel", buffer)
            # A CSI sequence: consume exactly the rest of it. Arrows are
            # navigation elsewhere and meaningless while typing, so they are
            # discarded rather than cancelling the query.
            while True:
                ch = stream.read(1)
                if not ch or ch.isalpha() or ch == b"~":
                    break
            continue

        control = _LINE_CTRL.get(byte)
        if control == "submit":
            return ("submit", buffer)
        if control == "abort":
            return ("abort", buffer)
        if control == "backspace":
            buffer = buffer[:-1]
            if on_change is not None:
                on_change(buffer)
            continue

        raw = pending + byte
        pending = b""
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Incomplete multibyte sequence, wait for the rest.
            pending = raw
            continue
        if len(decoded) > 1 or ord(decoded) >= 32:
            buffer += decoded
            if on_change is not None:
                on_change(buffer)


# -- viewer -----------------------------------------------------------------


@dataclass
class BrowseState:
    """Everything the key handler needs. Kept data-only so it is testable."""

    records: list[MemoryRecord] = field(default_factory=list)
    cursor: int = 0
    offset: int = 0
    page: int = DEFAULT_PAGE
    total: int | None = None
    scope: str | None = None
    kind: str | None = None
    statuses: tuple[str, ...] | None = None
    query: str = ""
    notice: str = ""

    def clamp(self) -> None:
        """Keep the cursor inside the *result set*, not inside the loaded window.

        Clamping to ``len(self.records)`` would make paging impossible: seeking
        one page past the end would be pulled straight back to the last loaded
        row. The cursor is therefore allowed past the window, and the caller
        reloads when it lands outside.
        """
        upper = max(0, (self.total if self.total is not None else len(self.records)) - 1)
        self.cursor = max(0, min(self.cursor, upper))
        if not self.records and not self.total:
            self.cursor = 0
        self.offset = max(0, min(self.offset, self.cursor))
        self.offset = min(self.offset, max(0, upper))

    @property
    def current(self) -> MemoryRecord | None:
        if not self.records:
            return None
        return self.records[self.cursor]

    def move(self, delta: int) -> None:
        self.cursor += delta
        self.clamp()

    def goto(self, position: int) -> None:
        """Absolute cursor position in the *whole* result set.

        ``Home``/``End``/``G`` need to reach past the loaded window, which is
        what triggers a reload.
        """
        if self.total is None:
            return
        self.cursor = max(0, min(position, max(0, self.total - 1)))
        self.clamp()

    def seek_page(self, delta: int) -> None:
        self.cursor += delta * self.page
        self.clamp()


def _filter_text(state: BrowseState) -> str:
    bits = []
    if state.scope:
        bits.append(f"scope={state.scope}")
    if state.kind:
        bits.append(f"kind={state.kind}")
    if state.statuses is not None:
        bits.append("status=" + ",".join(state.statuses))
    if state.query:
        bits.append(f"/{state.query}")
    return "  ".join(bits) if bits else "(no filters)"


def _status_bar(state: BrowseState) -> Text:
    """Filter summary plus transient notice, shown under every view."""
    bar = Text()
    bar.append(_filter_text(state), style="bold")
    if state.notice:
        bar.append("   ")
        bar.append(state.notice, style="bold yellow")
    bar.append("   ")
    bar.append("[? help] [/ search] [q quit]", style="dim")
    return bar


def render_browse(
    state: BrowseState,
    *,
    width: int = 100,
    show_help: bool = False,
) -> tuple[Any, ...]:
    """Renderables for the list view: the help panel or the table, plus the bar."""
    if show_help:
        return (Panel(_HELP, title="keys", border_style="dim"), _status_bar(state))
    table = build_table(
        state.records,
        cursor=state.cursor,
        total=state.total,
        offset=state.offset,
        page=state.page,
        width=width,
        title=f"local-memory  ·  {_filter_text(state)}",
    )
    return (table, _status_bar(state))


def _reload(store: Store, state: BrowseState) -> None:
    """Re-run the listing for the current filters, keeping the cursor stable."""
    state.records = list_records(
        store,
        scope=state.scope,
        kind=state.kind,
        statuses=state.statuses,
        limit=state.page,
        offset=state.offset,
    )
    state.total = count_records(store, scope=state.scope, kind=state.kind, statuses=state.statuses)
    state.clamp()


def run_static(
    store: Store,
    config: Config,
    *,
    scope: str | None = None,
    kind: str | None = None,
    statuses: tuple[str, ...] | None = None,
    limit: int = DEFAULT_PAGE,
    search: str | None = None,
    console: Console | None = None,
) -> int:
    """Non-interactive listing. Used when stdout is not a terminal.

    Piped or redirected output must not attempt cursor control, so this path
    prints one table and the selected record and exits.
    """
    console = console or Console()
    state = BrowseState(page=limit, scope=scope, kind=kind, statuses=statuses)
    if search:
        searcher = _LazySearcher(config, store)
        state.query = search
        state.records = _hits_to_records(
            store, searcher.search(search, k=limit, scope=scope, kind=kind)
        )
        state.total = len(state.records)
        console.print(
            build_table(
                state.records,
                cursor=state.cursor,
                total=state.total,
                page=limit,
                width=console.width,
                title=f"local-memory  ·  /{search}",
            )
        )
        if state.records and state.current is not None:
            console.print(build_detail(state.current))
        else:
            console.print(_status_bar(state))
    else:
        _reload(store, state)
        for renderable in render_browse(state, width=console.width):
            console.print(renderable)
    return 0


def _hits_to_records(store: Store, hits: list[SearchHit]) -> list[MemoryRecord]:
    """Hydrate search hits into records, preserving rank order.

    Search returns previews for context economy; the viewer wants the real rows
    so the table columns are uniform.
    """
    if not hits:
        return []
    return store.get_many([hit.id for hit in hits])


def run_browse(
    store: Store,
    config: Config,
    *,
    scope: str | None = None,
    kind: str | None = None,
    statuses: tuple[str, ...] | None = None,
    limit: int = DEFAULT_PAGE,
    search: str | None = None,
    interactive: bool = True,
    console: Console | None = None,
    stream: BinaryIO | None = None,
    renderer: _Renderer | None = None,
) -> int:
    """Run the viewer.

    Degrades to :func:`run_static` when ``interactive`` is false or the terminal
    cannot do raw key reads, rather than failing: a viewer that exits with an
    error is worse than one that prints a table.
    """
    stream = _binary(stream)
    # An injected renderer is the test double for the terminal, so the tty probe
    # is skipped: the caller has already declared this an interactive session.
    if not interactive or (renderer is None and not _is_interactive(stream)):
        return run_static(
            store,
            config,
            scope=scope,
            kind=kind,
            statuses=statuses,
            limit=limit,
            search=search,
            console=console,
        )

    console = console or Console()
    searcher = _LazySearcher(config, store)
    state = BrowseState(page=limit, scope=scope, kind=kind, statuses=statuses)
    state.query = search or ""
    if state.query:
        state.records = _hits_to_records(
            store, searcher.search(state.query, k=limit, scope=scope, kind=kind)
        )
        state.total = len(state.records)
    else:
        _reload(store, state)

    detail: MemoryRecord | None = None
    show_help = False
    prompting = False
    buffer = ""
    width = console.width

    def _echo(text: str) -> None:
        """Repaint the prompt with the query typed so far."""
        nonlocal buffer, prompting
        buffer = text
        prompting = True
        draw()

    def draw() -> None:
        """One repaint. Shared by the key loop and the query prompt."""
        if prompting:
            line = Text()
            line.append("search: ", style="bold cyan")
            line.append(buffer, style="")
            line.append("▏", style="reverse")
            live.update(render_browse(state, width=width, show_help=show_help) + (line,))
        elif detail is not None:
            live.update((build_detail(detail), _status_bar(state)))
        else:
            live.update(render_browse(state, width=width, show_help=show_help))
        live.refresh()

    # Cbreak only when drawing for real: a BytesIO has no fd, and the
    # recording renderer used by tests must not touch the test runner's tty.
    terminal = _NULL_TERMINAL() if renderer is not None else _CbreakTerminal(stream)
    with terminal, renderer or _LiveRenderer(console) as live:
        while True:
            draw()

            if prompting:
                # Text entry uses a separate reader, so a query can contain
                # j, k, / or q without them being read as navigation. on_change
                # repaints as characters arrive, otherwise the user types blind.
                action, text = read_line(buffer, stream, on_change=lambda b: _echo(b))
                if action == "abort":
                    return 0
                if action == "cancel":
                    prompting = False
                    buffer = ""
                    continue
                buffer = text
                prompting = False
                if not buffer.strip():
                    state.notice = "query empty"
                    continue
                state.query = buffer.strip()
                state.offset = 0
                state.cursor = 0
                try:
                    state.records = _hits_to_records(
                        store,
                        searcher.search(
                            state.query, k=state.page, scope=state.scope, kind=state.kind
                        ),
                    )
                except BrowseError as exc:
                    state.records = []
                    state.total = 0
                    state.notice = str(exc)
                else:
                    state.total = len(state.records)
                    state.notice = "" if state.records else "no matches"
                continue

            key = read_key(stream)
            if key is None:
                break

            if detail is not None:
                if key == "quit":
                    return 0
                if key in ("back", "open", "other", "help"):
                    detail = None
                continue

            if key == "quit":
                return 0
            if key == "help":
                show_help = not show_help
            elif key == "open":
                if state.current is not None:
                    detail = state.current
            elif key == "down":
                state.move(1)
            elif key == "up":
                state.move(-1)
            elif key == "page_down":
                state.seek_page(1)
            elif key == "page_up":
                state.seek_page(-1)
            elif key == "home":
                state.cursor = 0
                state.clamp()
            elif key == "end":
                state.goto((state.total or len(state.records)) - 1)
            elif key == "search":
                prompting = True
                buffer = state.query
            elif key == "clear":
                state.query = ""
                state.offset = 0
                state.cursor = 0
                _reload(store, state)
                state.notice = ""
            elif key == "other":
                continue

            # A cursor move past the loaded window needs a new page.
            if state.cursor < state.offset or state.cursor >= state.offset + state.page:
                state.offset = (state.cursor // state.page) * state.page
                _reload(store, state)

    return 0


def _binary(stream: BinaryIO | None) -> BinaryIO:
    """Return a byte-oriented view of ``stream``.

    ``sys.stdin`` is a :class:`io.TextIOWrapper`: ``read(1)`` returns ``'j'``,
    not ``b'j'``, so every lookup in the byte-key tables misses and every
    keystroke degrades to ``"other"``. Tests that feed a raw ``BytesIO`` pass
    happily and never catch this; only a real terminal does.
    """
    if stream is None:
        # Handle case where sys.stdin is monkeypatched (e.g., in tests) to a StringIO
        if hasattr(sys.stdin, "buffer"):
            return sys.stdin.buffer
        # If no buffer, fall back to a BytesIO over the text stream
        return sys.stdin.buffer if hasattr(sys.stdin, "buffer") else io.BytesIO()
    # If stream already has a buffer attribute, use it. Otherwise assume it's already binary.
    return getattr(stream, "buffer", stream)


class _NULL_TERMINAL:
    """Stand-in used when there is no real terminal to reconfigure."""

    def __enter__(self) -> _NULL_TERMINAL:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _CbreakTerminal:
    """Put the terminal in cbreak mode for the duration of the viewer.

    Without this the tty buffers input a line at a time and echoes it, so `jj`
    would not be delivered until Enter and every keystroke would be printed in
    the middle of the table. ``tty.setcbreak`` clears ``ICANON`` and ``ECHO``
    but deliberately leaves ``ISIG`` set, so Ctrl-C still works as the
    interpreter expects rather than arriving as byte ``0x03``.

    Restoring is not conditional on a normal exit: a crash must not leave the
    user's shell without an echo.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self._fd = stream.fileno()
        self._saved: list | None = None

    def __enter__(self) -> _CbreakTerminal:
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd, termios.TCSANOW)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)


class _LiveRenderer:
    """Draws frames to the terminal via :class:`rich.live.Live`."""

    def __init__(self, console: Console) -> None:
        self._live = Live(console=console, screen=False, auto_refresh=False)

    def __enter__(self) -> _LiveRenderer:
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, renderables: tuple) -> None:
        # Group, not the raw tuple: Live renders exactly one renderable and a
        # tuple raises NotRenderableError on a real terminal.
        self._live.update(Group(*renderables))

    def refresh(self) -> None:
        self._live.refresh()


class _RecordingRenderer:
    """Keeps every frame instead of drawing it.

    ``Live`` does not re-render to a non-terminal console, so asserting on a
    captured buffer only ever sees the first frame. Recording the frames is what
    makes the interactive path testable at all.
    """

    def __init__(self) -> None:
        self.frames: list[tuple] = []

    def __enter__(self) -> _RecordingRenderer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, renderables: tuple) -> None:
        self.frames.append(renderables)

    def refresh(self) -> None:
        return None

    def text(self, width: int = 120) -> str:
        """All frames flattened into one string, for substring assertions."""
        out = []
        for frame in self.frames:
            buf = io.StringIO()
            console = Console(file=buf, width=width, force_terminal=False, no_color=True)
            for renderable in frame:
                console.print(renderable)
            out.append(buf.getvalue())
        return "\n".join(out)


def _is_interactive(stream: BinaryIO) -> bool:
    """Can this stream support raw single-key reads and a cursor-addressed UI?

    Checks: it is a tty, ``TERM`` is not ``dumb``, and it is not a dumb terminal
    on either side. ``CI`` is treated as non-interactive because most CI
    terminals have no usable raw mode.
    """
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return os.environ.get("CI", "").lower() not in ("1", "true", "yes")


__all__ = [
    "BROWSABLE_STATUSES",
    "BrowseError",
    "BrowseState",
    "DEFAULT_PAGE",
    "build_detail",
    "build_table",
    "count_records",
    "list_records",
    "read_key",
    "render_browse",
    "run_browse",
    "run_static",
]
