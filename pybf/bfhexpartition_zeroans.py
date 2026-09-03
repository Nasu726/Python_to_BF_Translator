"""Adaptive partition minimum with a fast path once ANS reaches zero.

With the project's fixed-width signed-int64 semantics, ``abs(x)`` is
nonnegative except for ``INT64_MIN``, whose negation wraps to itself. Therefore
when the carried ANS is exactly zero, a new candidate can improve it iff the
raw pre-abs candidate is exactly INT64_MIN. In that state we can skip the full
16-nibble absolute value and bounded minimum comparison.

The normal nonzero path delegates to the adaptive-width minimum lowering.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _set_hex_const
from bfhexpartition_adaptiveans import _min_and_move_ans_adaptive
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
    HEX_DIGITS,
    LEFT,
    MARKER,
    RECORD_STRIDE,
    TOTAL,
    RuntimeHexIntSequence,
    _RelativeBuilder,
)


# NONZERO_GATE must survive the normal branch, whose abs uses DATA[0..15] and
# whose adaptive min uses LEFT[0..14]. LEFT[15] is intentionally spare.
NONZERO_GATE = LEFT + 15
ZERO_GATE = DATA + 15
CHECK_TMP = DATA + 13
CHECK_RESTORE = DATA + 14
IS_MIN = DATA + 12
LOW_NONZERO = DATA + 11


def _mark_nonzero_preserved(r: _RelativeBuilder, src: int) -> None:
    """Set NONZERO_GATE to one when preserved src is nonzero."""
    r.copy_preserved(src, CHECK_TMP, CHECK_RESTORE)
    r.move(CHECK_TMP)
    r.emit("[")
    r.clear(CHECK_TMP)
    r.set_const(NONZERO_GATE, 1)
    r.move(CHECK_TMP)
    r.emit("]")


def _set_answer_zero_gates(r: _RelativeBuilder, *, extent: int) -> None:
    """Set complementary ZERO_GATE/NONZERO_GATE for bounded ANS."""
    r.clear(NONZERO_GATE)
    for i in range(extent):
        _mark_nonzero_preserved(r, ANS + i)
    # Bounded ANS has no high live lanes except wrapped INT64_MIN at nibble 15.
    _mark_nonzero_preserved(r, ANS + HEX_DIGITS - 1)

    r.set_const(ZERO_GATE, 1)
    r.copy_preserved(NONZERO_GATE, CHECK_TMP, CHECK_RESTORE)
    r.move(CHECK_TMP)
    r.emit("[")
    r.clear(CHECK_TMP)
    r.clear(ZERO_GATE)
    r.move(CHECK_TMP)
    r.emit("]")


def _raw_candidate_is_int64_min_destructive(r: _RelativeBuilder) -> None:
    """Set IS_MIN iff current raw TOTAL is exactly 0x8000...; consume TOTAL."""
    r.clear(LOW_NONZERO)
    for i in range(HEX_DIGITS - 1):
        cell = TOTAL + i
        r.move(cell)
        r.emit("[")
        r.clear(cell)
        r.set_const(LOW_NONZERO, 1)
        r.move(cell)
        r.emit("]")

    high = TOTAL + HEX_DIGITS - 1
    r.clear(IS_MIN)
    # Consume at most nine guarded units. Exactly the eighth unit marks nibble
    # value 8; reaching the ninth proves >8 and clears both the flag and the
    # remaining nibble so the enclosing guards close immediately.
    for step in range(1, 10):
        r.move(high)
        r.emit("[")
        r.add(high, -1)
        if step == 8:
            r.set_const(IS_MIN, 1)
        elif step == 9:
            r.clear(IS_MIN)
            r.clear(high)
    for _ in range(9):
        r.move(high)
        r.emit("]")

    r.move(LOW_NONZERO)
    r.emit("[")
    r.add(LOW_NONZERO, -1)
    r.clear(IS_MIN)
    r.move(LOW_NONZERO)
    r.emit("]")


def _zero_answer_step(r: _RelativeBuilder) -> None:
    """Propagate zero unless raw candidate is wrapped-negative INT64_MIN."""
    _raw_candidate_is_int64_min_destructive(r)

    for i in range(HEX_DIGITS):
        r.clear(RECORD_STRIDE + ANS + i)

    r.move(IS_MIN)
    r.emit("[")
    r.add(IS_MIN, -1)
    r.set_const(RECORD_STRIDE + ANS + HEX_DIGITS - 1, 8)
    r.move(IS_MIN)
    r.emit("]")

    for cell in (IS_MIN, LOW_NONZERO, CHECK_TMP, CHECK_RESTORE):
        r.clear(cell)


def _partition_body(extent: int) -> str:
    r = _RelativeBuilder()
    add_data_and_move_state_total_minus_double_left_into_total(r)
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
        raise ValueError("zero-answer extent outside supported scratch layout")
    return _partition_body(extent)


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    extent = answer_extent(initial_ans)
    if extent > MAX_BOUNDED_NIBBLES:
        raise ValueError("initial_ans is too wide for zero-answer prototype")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body(extent) + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
