"""Zero-aware partition pass consuming prefix sums precomputed during input."""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexcandidate_prefix import move_total_minus_double_prefix_into_total
from bfhexpartition import MASK64, _set_hex_const
from bfhexpartition_adaptiveans import _min_and_move_ans_adaptive
from bfhexpartition_boundedans import (
    MAX_BOUNDED_NIBBLES,
    _min_and_move_ans_bounded,
    answer_extent,
)
from bfhexpartition_totalcandidate import _abs_total_inplace
from bfhexpartition_zeroans import (
    NONZERO_GATE,
    ZERO_GATE,
    _set_answer_zero_gates,
    _zero_answer_step,
)
from bfhexseq import ANS, BACK, MARKER, RECORD_STRIDE, RuntimeHexIntSequence, _RelativeBuilder


def _partition_body(extent: int) -> str:
    r = _RelativeBuilder()
    move_total_minus_double_prefix_into_total(r)
    _set_answer_zero_gates(r, extent=extent)

    r.move(ZERO_GATE)
    r.emit("[")
    r.add(ZERO_GATE, -1)
    _zero_answer_step(r)
    r.move(ZERO_GATE)
    r.emit("]")

    r.move(NONZERO_GATE)
    r.emit("[")
    r.add(NONZERO_GATE, -1)
    _abs_total_inplace(r)
    if extent <= 2:
        _min_and_move_ans_bounded(r, extent=extent)
    else:
        _min_and_move_ans_adaptive(r, extent=extent)
    r.move(NONZERO_GATE)
    r.emit("]")

    r.clear(ZERO_GATE)
    r.clear(NONZERO_GATE)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=None)
def partition_body(extent: int) -> str:
    if not 1 <= extent <= MAX_BOUNDED_NIBBLES:
        raise ValueError("stored-prefix extent outside supported scratch layout")
    return _partition_body(extent)


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    extent = answer_extent(initial_ans)
    if extent > MAX_BOUNDED_NIBBLES:
        raise ValueError("initial_ans is too wide for stored-prefix prototype")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body(extent) + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
