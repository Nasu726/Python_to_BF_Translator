"""Partition minimum that shrinks its active answer width at runtime.

The compile-time bounded-answer lowering compares only the hexadecimal lanes
needed by the initial nonnegative answer.  In many partition workloads the
first useful candidate is much smaller than that initial bound.  Once ANS fits
in two low nibbles (<= 0xff), later comparisons can use the two-lane bounded
kernel instead of continuing to pay for every lane in the original extent.

This prototype keeps candidate construction and absolute value common.  Before
minimum propagation it preserves/tests ANS lanes 2..initial_extent-1 and
branches once: a zero high part uses the two-lane kernel; otherwise the normal
compile-time extent is retained.  INT64_MIN remains correct because the
bounded kernel independently checks the sign nibble and treats that wrapped
negative answer as sticky.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _set_hex_const
from bfhexpartition_boundedans import (
    MAX_BOUNDED_NIBBLES,
    _min_and_move_ans_bounded,
    answer_extent,
)
from bfhexpartition_totalcandidate import _abs_total_inplace
from bfhexseq import (
    ANS,
    BACK,
    DATA,
    MARKER,
    RECORD_STRIDE,
    RuntimeHexIntSequence,
    _RelativeBuilder,
)


LOW_EXTENT = 2
CHECK_TMP = DATA + 12
CHECK_RESTORE = DATA + 13
WIDE_GATE = DATA + 14
LOW_GATE = DATA + 15


def _set_wide_gate_from_answer(r: _RelativeBuilder, *, extent: int) -> None:
    """Set WIDE_GATE iff preserved ANS needs more than LOW_EXTENT nibbles."""
    r.clear(WIDE_GATE)
    for i in range(LOW_EXTENT, extent):
        r.copy_preserved(ANS + i, CHECK_TMP, CHECK_RESTORE)
        r.move(CHECK_TMP)
        r.emit("[")
        # Only the fact of nonzero matters; consume the temporary in one pass.
        r.clear(CHECK_TMP)
        r.set_const(WIDE_GATE, 1)
        r.move(CHECK_TMP)
        r.emit("]")


def _min_and_move_ans_adaptive(r: _RelativeBuilder, *, extent: int) -> None:
    if not LOW_EXTENT < extent <= MAX_BOUNDED_NIBBLES:
        raise ValueError("adaptive answer extent must exceed the low tier")

    _set_wide_gate_from_answer(r, extent=extent)

    # LOW_GATE = !WIDE_GATE while preserving WIDE_GATE for its own branch.
    r.set_const(LOW_GATE, 1)
    r.copy_preserved(WIDE_GATE, CHECK_TMP, CHECK_RESTORE)
    r.move(CHECK_TMP)
    r.emit("[")
    r.clear(CHECK_TMP)
    r.clear(LOW_GATE)
    r.move(CHECK_TMP)
    r.emit("]")

    # The bounded kernels use DATA low lanes plus LEFT high scratch only; the
    # four DATA cells above are deliberately outside either branch's footprint.
    r.move(LOW_GATE)
    r.emit("[")
    r.add(LOW_GATE, -1)
    _min_and_move_ans_bounded(r, extent=LOW_EXTENT)
    r.move(LOW_GATE)
    r.emit("]")

    r.move(WIDE_GATE)
    r.emit("[")
    r.add(WIDE_GATE, -1)
    _min_and_move_ans_bounded(r, extent=extent)
    r.move(WIDE_GATE)
    r.emit("]")

    for cell in (CHECK_TMP, CHECK_RESTORE, WIDE_GATE, LOW_GATE):
        r.clear(cell)


def _partition_body(extent: int) -> str:
    r = _RelativeBuilder()
    add_data_and_move_state_total_minus_double_left_into_total(r)
    _abs_total_inplace(r)
    if extent <= LOW_EXTENT:
        _min_and_move_ans_bounded(r, extent=extent)
    else:
        _min_and_move_ans_adaptive(r, extent=extent)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=None)
def partition_body(extent: int) -> str:
    if not 1 <= extent <= MAX_BOUNDED_NIBBLES:
        raise ValueError("adaptive answer extent outside supported scratch layout")
    return _partition_body(extent)


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    extent = answer_extent(initial_ans)
    if extent > MAX_BOUNDED_NIBBLES:
        raise ValueError("initial_ans is too wide for adaptive-answer prototype")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body(extent) + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["LOW_EXTENT", "partition_body", "run_partition_min_pass"]
