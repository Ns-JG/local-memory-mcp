"""The viewer itself: static listing and the interactive key loop."""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO

from rich.console import Console
from rich.text import Text

from local_memory.browse.data import (
    DEFAULT_PAGE,
    BrowseError,
    _hits_to_records,
    _LazySearcher,
    count_records,
    list_records,
)
from local_memory.browse.render import (
    BrowseState,
    _status_bar,
    build_detail,
    build_table,
    render_browse,
)
from local_memory.browse.terminal import (
    _NULL_TERMINAL,
    _binary,
    _CbreakTerminal,
    _is_interactive,
    _LiveRenderer,
    _Renderer,
    read_key,
    read_line,
)
from local_memory.config import Config
from local_memory.models import MemoryRecord

if TYPE_CHECKING:
    from local_memory.storage import Store


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
