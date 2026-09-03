"""Partition experiment using the safe tiered add-candidate decoder."""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate_tiered import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _set_hex_const
from bfhexpartition_nonnegans import _min_and_move_ans_nonnegative
from bfhexpartition_totalcandidate import _abs_total_inplace
from bfhexseq import BACK, MARKER, RECORD_STRIDE, RuntimeHexIntSequence, _RelativeBuilder


def _partition_body() -> str:
    r = _RelativeBuilder()
    add_data_and_move_state_total_minus_double_left_into_total(r)
    _abs_total_inplace(r)
    _min_and_move_ans_nonnegative(r)
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
    if not 0 <= initial_ans < (1 << 63):
        raise ValueError("tiered partition requires signed nonnegative initial_ans")

    _set_hex_const(bf, seq.base + 50, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
