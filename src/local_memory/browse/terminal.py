"""Everything that touches the real terminal: key reads, cbreak mode, drawing.

This is the only module allowed to import ``termios``/``tty``.
"""

from __future__ import annotations

import io
import os
import sys
import termios
import tty
from collections.abc import Callable
from typing import Any, BinaryIO, Protocol

from rich.console import Console, Group
from rich.live import Live


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
