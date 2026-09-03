"""Partition pass with destructive candidate comparison and ANS transport.

After the candidate is made absolute, current DATA is dead after the minimum
operation and current ANS must move to the next record regardless of the
comparison outcome.  This variant exploits both facts: while computing the
unsigned comparison it consumes DATA into current TOTAL scratch and consumes
ANS directly into next.ANS.  If the signed comparison chooses the candidate,
next.ANS is replaced from the saved candidate.  This removes both preserved
operand copies and the separate final ANS transport.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexcandidate_move import move_state_and_total_minus_double_left_into_data
from bfhexpartition import MASK64, _nibble_ge8, _set_hex_const
from bfhexpartition_fastops import abs_data_inplace, consume_data_into_left
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


def _copy_flag(r: _RelativeBuilder, dst: int, src: int, tmp: int) -> None:
    r.copy_preserved(src, dst, tmp)


def _min_and_move_ans_with_data(r: _RelativeBuilder) -> None:
    """Set next.ANS=min_signed(ANS, DATA), destructively consuming both words.

    Current TOTAL/LEFT are scratch after the live state has already moved to the
    next record.  TOTAL saves the candidate while LEFT holds per-nibble
    subtraction totals.  The radix-complement subtraction DATA-ANS leaves
    MARKER=1 iff candidate >= ans in unsigned order.
    """
    r.set_const(MARKER, 1)

    for i in range(HEX_DIGITS):
        candidate = DATA + i
        saved = TOTAL + i
        total = LEFT + i
        ans = ANS + i
        next_ans = RECORD_STRIDE + ANS + i

        r.clear(saved)
        r.clear(total)
        r.transfer(MARKER, total)

        # Candidate is dead after min: consume it once while saving exactly one
        # copy for the possible replacement of next.ANS.
        r.move(candidate)
        r.emit("[")
        r.add(candidate, -1)
        r.add(saved, 1)
        r.add(total, 1)
        r.move(candidate)
        r.emit("]")

        # ANS always moves right.  Consume it directly into next.ANS while
        # subtracting it from the comparison total; no preserve/restore loop.
        r.add(total, 15)
        r.move(ans)
        r.emit("[")
        r.add(ans, -1)
        r.add(next_ans, 1)
        r.add(total, -1)
        r.move(ans)
        r.emit("]")

        # DATA is zero now and can hold the dead difference digit.
        map_total_base16_threshold(r, total, candidate, MARKER)

    # MARKER==1 means candidate >= ans.  Convert to unsigned candidate < ans.
    ult = LEFT
    r.set_const(ult, 1)
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    r.clear(ult)
    r.move(MARKER)
    r.emit("]")

    # Signed comparison: when signs differ, candidate is smaller exactly when
    # it is negative.  Otherwise the unsigned comparison is already correct.
    sign_data = LEFT + 1
    sign_ans = LEFT + 2
    signs_differ = LEFT + 3
    gate = LEFT + 4
    tmp = LEFT + 5
    restore = LEFT + 6
    choose = LEFT + 7

    _nibble_ge8(r, sign_data, TOTAL + HEX_DIGITS - 1, tmp, restore)
    _nibble_ge8(
        r,
        sign_ans,
        RECORD_STRIDE + ANS + HEX_DIGITS - 1,
        tmp,
        restore,
    )

    _copy_flag(r, signs_differ, sign_data, tmp)
    _copy_flag(r, gate, sign_ans, tmp)
    r.move(gate)
    r.emit("[")
    r.add(gate, -1)
    # Toggle signs_differ.  It is Boolean, so one guarded decrement turns
    # tmp=1 into zero iff the old value was one.
    r.set_const(tmp, 1)
    r.move(signs_differ)
    r.emit("[")
    r.add(signs_differ, -1)
    r.add(tmp, -1)
    r.move(signs_differ)
    r.emit("]")
    r.transfer(tmp, signs_differ)
    r.move(gate)
    r.emit("]")

    _copy_flag(r, choose, ult, tmp)
    r.move(signs_differ)
    r.emit("[")
    r.add(signs_differ, -1)
    r.clear(choose)
    _copy_flag(r, choose, sign_data, tmp)
    r.move(signs_differ)
    r.emit("]")

    r.move(choose)
    r.emit("[")
    r.add(choose, -1)
    for i in range(HEX_DIGITS):
        r.clear(RECORD_STRIDE + ANS + i)
        r.transfer(TOTAL + i, RECORD_STRIDE + ANS + i)
    r.move(choose)
    r.emit("]")

    for cell in (
        MARKER,
        ult,
        sign_data,
        sign_ans,
        signs_differ,
        gate,
        tmp,
        restore,
        choose,
    ):
        r.clear(cell)


def _partition_body() -> str:
    r = _RelativeBuilder()
    consume_data_into_left(r)
    move_state_and_total_minus_double_left_into_data(r)
    abs_data_inplace(r)
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
    """Run partition minimum pass with fused minimum/ANS transport."""
    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)

    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
