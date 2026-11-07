"""Pure rendering and view state: no I/O, no terminal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.box import SIMPLE_HEAD
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from local_memory.browse.data import DEFAULT_PAGE
from local_memory.models import MemoryRecord, Status

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
