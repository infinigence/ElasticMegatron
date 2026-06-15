"""Pure byte-level chunk scheduling for BatchedTransfer's packed path.

Deliberately torch-free so it unit-tests on any machine
(tests/test_chunk_schedule.py). Correctness premise: the sender's slice list
to a peer and the receiver's landing list from that peer describe the SAME
bytes in the SAME order (already required by the packed path today), so both
ranks derive identical chunk boundaries from (slice sizes, cap) with no
collective exchange.
"""


def chunk_ranges(total_nbytes: int, cap: int | None) -> list[tuple[int, int]]:
    """Split [0, total_nbytes) into consecutive [start, end) ranges of at most
    ``cap`` bytes. ``cap`` None or <= 0 means no split (single chunk — the
    legacy-residency fallback). 0 total bytes -> no chunks."""
    if total_nbytes == 0:
        return []
    if cap is None or cap <= 0 or cap >= total_nbytes:
        return [(0, total_nbytes)]
    return [(s, min(s + cap, total_nbytes)) for s in range(0, total_nbytes, cap)]


def slice_spans(
    slice_nbytes: list[int], start: int, end: int
) -> list[tuple[int, int, int]]:
    """Map byte range [start, end) of the logical concatenation of slices onto
    per-slice spans: ``[(slice_index, offset_in_slice, nbytes), ...]``.
    Zero-length slices never produce spans."""
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for idx, n in enumerate(slice_nbytes):
        lo = max(start, cursor)
        hi = min(end, cursor + n)
        if lo < hi:
            spans.append((idx, lo - cursor, hi - lo))
        cursor += n
        if cursor >= end:
            break
    return spans


STAGING_CAP_FLOOR = 1 << 30  # 1 GiB
STAGING_CAP_CEIL = 8 << 30  # 8 GiB
STAGING_CAP_DEFAULT = 2 << 30  # 2 GiB (fixed-mode cap)


def derive_staging_cap(
    available_bytes: int | None,
    floor: int = STAGING_CAP_FLOOR,
    ceil: int = STAGING_CAP_CEIL,
) -> int:
    """Staging-chunk cap = ``available_bytes`` // 2, clamped to [floor, ceil].

    Halved because the send and recv staging buffers coexist. ``available_bytes``
    is the memory known free for staging (free mode: the smallest current free
    GPU memory across the union ranks). None/<=0 falls back to the floor."""
    if not available_bytes or available_bytes <= 0:
        return floor
    return min(ceil, max(floor, available_bytes // 2))
