import secrets
import time
import uuid
from typing import Self


class Timer:
    """Context manager for timing a block of code."""

    def __init__(self, task_name: str = "") -> None:
        self.start: float = -1.0
        self.end: float = -1.0
        self.elapsed: float | None = None

    def __enter__(self) -> Self:
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

    def elapsed_ms(self) -> float:
        return (self.end - self.start) * 1000

    @staticmethod
    def utcnow() -> str:
        """UTC ISO-8601 with a Z suffix, second resolution."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id() -> str:
    """UUIDv7: 48-bit millisecond timestamp, then random.

    Sorting by id orders by millisecond first, which makes ``ORDER BY id`` an
    approximate chronological index. Within a single millisecond the tail is
    random, as the UUIDv7 spec requires, so the ordering is not strictly
    monotonic - use ``created_at`` when exact order matters. Collision-free
    across devices without coordination. Python 3.13 has no ``uuid.uuid7``.

    The ``random-a`` and ``random-b`` fields are interleaved per RFC 9562
    rather than packed adjacently, so the layout is a valid v7 UUID.
    """
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    value = (ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return str(uuid.UUID(int=value))


utcnow = Timer.utcnow
