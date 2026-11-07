"""Interactive read-only viewer for the memory store.

Deliberately a **read path**, not a database browser. It goes through
:class:`~local_memory.storage.Store` read methods rather than handing you raw
tables, so it shows decoded tags, the status lifecycle and confirmation counts
instead of five ``records_fts_*`` shadow tables and a float32 blob.

Two invariants this package must not break:

1. **No writes.** Every statement issued here is a ``SELECT``. Editing through a
   viewer would bypass ``rev``, ``store_epoch`` and vector recomputation, which
   is how optimistic concurrency silently stops working. ``tests/test_browse.py``
   asserts this by counting non-SELECT statements executed against a real store.
2. **No model load on startup.** Building the ONNX session costs ~0.2s, which a
   listing command has no business paying. The embedder is constructed only when
   a search is actually requested, and cached for later ones.

Layout: ``data`` (store reads, lazy search), ``render`` (tables, detail view,
state), ``terminal`` (the only place that touches termios/tty), ``app`` (the
static listing and the key loop).
"""

from local_memory.browse.app import run_browse, run_static
from local_memory.browse.data import (
    BROWSABLE_STATUSES,
    DEFAULT_PAGE,
    BrowseError,
    _LazySearcher,
    count_records,
    list_records,
)
from local_memory.browse.render import BrowseState, build_detail, build_table, render_browse
from local_memory.browse.terminal import (
    _binary,
    _is_interactive,
    _RecordingRenderer,
    read_key,
    read_line,
)

__all__ = [
    "BROWSABLE_STATUSES",
    "DEFAULT_PAGE",
    "BrowseError",
    "BrowseState",
    "_LazySearcher",
    "_RecordingRenderer",
    "_binary",
    "_is_interactive",
    "build_detail",
    "build_table",
    "count_records",
    "list_records",
    "read_key",
    "read_line",
    "render_browse",
    "run_browse",
    "run_static",
]
