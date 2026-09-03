"""Partition/min pass using the one-pass fused candidate arithmetic.

This module is a drop-in experimental replacement for ``bfhexpartition``.  It
keeps the already-tested LEFT accumulation, abs, signed-min, and carried-state
logic, but replaces the old two-stage

    DATA = 2 * next.LEFT
    DATA = next.TOTAL - DATA

sequence with ``bfhexcandidate.total_minus_double_next_left_into_data``.
Keeping this path isolated makes step-count comparison and rollback trivial.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexcandidate import total_minus_double_next_left_into_data
from bfhexpartition import (
    MASK64,
    _abs_data_inplace,
    _consume_data_into_left,
    _min_ans_with_data,
    _move_primary_state_right,
    _set_hex_const,
)
from bfhexseq import (
    ANS,
    BACK,
    MARKER,
    RECORD_STRIDE,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _transfer_word,
)


def _partition_body() -> str:
    r = _RelativeBuilder()
    _consume_data_into_left(r)
    _move_primary_state_right(r)
    total_minus_double_next_left_into_data(r)
    _abs_data_inplace(r)
    _min_ans_with_data(r)

    _transfer_word(r, ANS, RECORD_STRIDE + ANS)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=1)
def partition_body() -> str:
    return _partition_body()


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    """Run the partition minimum pass with fused candidate arithmetic."""
    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)

    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
