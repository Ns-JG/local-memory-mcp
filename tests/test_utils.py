import time
import uuid

from local_memory.utils import new_id, utcnow


def _ms(value: str) -> int:
    """The 48-bit millisecond timestamp a UUIDv7 encodes at bits 127..80."""
    return uuid.UUID(value).int >> 80


def test_new_id_is_uuid7_with_a_time_ordered_prefix() -> None:
    ids = [new_id() for _ in range(200)]
    assert ids[0][14] == "7"  # version nibble
    assert ids[0][19] in "89ab"  # RFC 9562 variant
    assert len(set(ids)) == 200
    assert [_ms(i) for i in ids] == sorted(_ms(i) for i in ids)


def test_new_id_timestamp_advances_with_the_clock() -> None:
    """Time order lives in the timestamp bits, not in the id's string prefix."""
    first = _ms(new_id())
    time.sleep(0.02)
    assert _ms(new_id()) > first


def test_utcnow_format() -> None:
    stamp = utcnow()
    assert stamp.endswith("Z")
    assert time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
