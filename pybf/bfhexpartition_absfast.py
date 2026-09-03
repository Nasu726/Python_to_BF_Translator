"""Partition pass combining destructive candidate/min fusion with fast abs."""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexabsfast import abs_data_inplace_fast
from bfhexcandidate_move import move_state_and_total_minus_double_left_into_data
from bfhexpartition import MASK64, _consume_data_into_left, _set_hex_const
from bfhexpartition_minfused import _min_and_move_ans_with_data
from bfhexseq import ANS, BACK, MARKER, RECORD_STRIDE, RuntimeHexIntSequence, _RelativeBuilder


def _partition_body() -> str:
    r = _RelativeBuilder()
    _consume_data_into_left(r)
    move_state_and_total_minus_double_left_into_data(r)
    abs_data_inplace_fast(r)
    _min_and_move_ans_with_data(r)
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
    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")
    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
