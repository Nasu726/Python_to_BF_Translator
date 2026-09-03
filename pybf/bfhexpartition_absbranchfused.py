"""Partition prototype that fuses negative abs with min comparison.

The earlier abs/min fusion rebuilt positive/negative gates for every nibble and
was slower than the established two-pass path. This variant branches once per
record. Nonnegative raw candidates use the established minimum kernel. Negative
raw candidates build their two's-complement magnitude directly into the saved
candidate used by the minimum comparison, avoiding a separate abs materialize
pass.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _nibble_ge8, _set_hex_const
from bfhexpartition_nonnegans import _min_and_move_ans_nonnegative
from bfhexradixfast import map_total_base16_threshold
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


# Branch gates sit above the scratch cells used by both minimum kernels.
SIGN = DATA + 12
NEG_GATE = DATA + 13
POS_GATE = DATA + 14
GATE_TMP = DATA + 15


def _negative_abs_min_and_move_ans(r: _RelativeBuilder) -> None:
    """Compare abs(TOTAL) with ANS assuming raw TOTAL is signed-negative.

    Current TOTAL is consumed nibble-by-nibble. Its two's-complement magnitude
    is rebuilt in the same TOTAL cells while the unsigned candidate-ANS
    subtraction is accumulated. Current LEFT is free after candidate
    construction and is reused for the subtraction-result nibble.
    """
    acc = DATA
    negate_carry = DATA + 1

    # Two's-complement magnitude starts with +1. Unsigned candidate-ANS
    # subtraction independently starts with radix carry one in MARKER.
    r.set_const(negate_carry, 1)
    r.set_const(MARKER, 1)

    for i in range(HEX_DIGITS):
        raw_and_saved = TOTAL + i
        count_and_diff = LEFT + i
        ans = ANS + i
        next_ans = RECORD_STRIDE + ANS + i

        # acc = subtraction carry + 15 + magnitude nibble.
        r.clear(acc)
        r.transfer(MARKER, acc)
        r.add(acc, 15)

        # count = 15 - raw_nibble + incoming negate carry. Consuming raw makes
        # TOTAL[i] available to hold the magnitude nibble directly.
        r.set_const(count_and_diff, 15)
        r.move(raw_and_saved)
        r.emit("[")
        r.add(raw_and_saved, -1)
        r.add(count_and_diff, -1)
        r.move(raw_and_saved)
        r.emit("]")
        r.transfer(negate_carry, count_and_diff)

        # count is in 0..16. Emit low radix-16 magnitude into TOTAL[i] and add
        # the same nibble to acc. At exactly 16, low digit wraps to zero and
        # the outgoing two's-complement carry is one.
        for step in range(1, 17):
            r.move(count_and_diff)
            r.emit("[")
            r.add(count_and_diff, -1)
            if step == 16:
                r.clear(raw_and_saved)
                r.add(acc, -15)
                r.set_const(negate_carry, 1)
            else:
                r.add(raw_and_saved, 1)
                r.add(acc, 1)
        for _ in range(16):
            r.move(count_and_diff)
            r.emit("]")

        # Carry old ANS right while subtracting it from the same accumulator.
        r.move(ans)
        r.emit("[")
        r.add(ans, -1)
        r.add(next_ans, 1)
        r.add(acc, -1)
        r.move(ans)
        r.emit("]")

        # LEFT[i] is zero after the bounded magnitude decoder and can hold the
        # dead subtraction-result digit. MARKER receives the next radix carry.
        map_total_base16_threshold(
            r,
            acc,
            count_and_diff,
            MARKER,
        )

    # Unsigned magnitude < old answer is !final carry.
    ult = DATA
    r.set_const(ult, 1)
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    r.clear(ult)
    r.move(MARKER)
    r.emit("]")

    sign_candidate = DATA + 1
    sign_ans = DATA + 2
    choose = DATA + 3
    tmp = DATA + 4
    restore = DATA + 5

    _nibble_ge8(r, sign_candidate, TOTAL + HEX_DIGITS - 1, tmp, restore)
    _nibble_ge8(
        r,
        sign_ans,
        RECORD_STRIDE + ANS + HEX_DIGITS - 1,
        tmp,
        restore,
    )

    r.copy_preserved(ult, choose, tmp)

    # abs_s64(INT64_MIN) wraps to INT64_MIN, which is signed-negative and must
    # beat any nonnegative carried answer.
    r.move(sign_candidate)
    r.emit("[")
    r.add(sign_candidate, -1)
    r.set_const(choose, 1)
    r.move(sign_candidate)
    r.emit("]")

    # Once INT64_MIN is the carried answer, no later abs candidate is smaller.
    r.move(sign_ans)
    r.emit("[")
    r.add(sign_ans, -1)
    r.clear(choose)
    r.move(sign_ans)
    r.emit("]")

    r.move(choose)
    r.emit("[")
    r.add(choose, -1)
    for i in range(HEX_DIGITS):
        r.clear(RECORD_STRIDE + ANS + i)
        r.transfer(TOTAL + i, RECORD_STRIDE + ANS + i)
    r.move(choose)
    r.emit("]")

    r.clear(negate_carry)


def _abs_min_branch_once(r: _RelativeBuilder) -> None:
    """Dispatch once on raw candidate sign, then execute the matching kernel."""
    _nibble_ge8(r, SIGN, TOTAL + HEX_DIGITS - 1, GATE_TMP, DATA + 11)
    r.copy_preserved(SIGN, NEG_GATE, GATE_TMP)

    r.set_const(POS_GATE, 1)
    r.move(SIGN)
    r.emit("[")
    r.add(SIGN, -1)
    r.clear(POS_GATE)
    r.move(SIGN)
    r.emit("]")

    r.move(POS_GATE)
    r.emit("[")
    r.add(POS_GATE, -1)
    _min_and_move_ans_nonnegative(r)
    r.move(POS_GATE)
    r.emit("]")

    r.move(NEG_GATE)
    r.emit("[")
    r.add(NEG_GATE, -1)
    _negative_abs_min_and_move_ans(r)
    r.move(NEG_GATE)
    r.emit("]")

    for cell in (SIGN, NEG_GATE, POS_GATE, GATE_TMP, DATA + 11):
        r.clear(cell)


def _partition_body() -> str:
    r = _RelativeBuilder()
    add_data_and_move_state_total_minus_double_left_into_total(r)
    _abs_min_branch_once(r)
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
        raise ValueError("branch-fused partition requires signed nonnegative initial_ans")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
