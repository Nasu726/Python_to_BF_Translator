"""Partition experiment that keeps the candidate in current TOTAL scratch.

This variant removes the candidate TOTAL->DATA->TOTAL round trip between fused
candidate construction and minimum comparison.  Live TOTAL/LEFT still move to
the next record; only dead current-record scratch placement changes.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexcandidate_total import move_state_and_total_minus_double_left_into_total
from bfhexpartition import MASK64, _nibble_ge8, _set_hex_const
from bfhexpartition_fastops import consume_data_into_left
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


def _abs_total_inplace(r: _RelativeBuilder) -> None:
    """Two's-complement absolute value of current TOTAL candidate in place."""
    sign = LEFT
    tmp = LEFT + 1
    restore = LEFT + 2
    _nibble_ge8(r, sign, TOTAL + HEX_DIGITS - 1, tmp, restore)

    r.move(sign)
    r.emit("[")
    r.add(sign, -1)
    r.set_const(MARKER, 1)
    for i in range(HEX_DIGITS):
        acc = DATA + i
        r.set_const(acc, 15)
        r.move(TOTAL + i)
        r.emit("[")
        r.add(TOTAL + i, -1)
        r.add(acc, -1)
        r.move(TOTAL + i)
        r.emit("]")
        r.transfer(MARKER, acc)
        map_total_base16_threshold(r, acc, TOTAL + i, MARKER)
    r.clear(MARKER)
    r.move(sign)
    r.emit("]")

    for cell in (sign, tmp, restore):
        r.clear(cell)


def _min_and_move_ans_with_total(r: _RelativeBuilder) -> None:
    """Set next.ANS=min_signed(ANS,TOTAL), consuming current operands.

    Candidate nibbles are saved one lane to the right in current LEFT while the
    unsigned subtraction uses DATA scratch.  Current TOTAL then becomes dead
    difference output.  This needs one local candidate move instead of the old
    TOTAL->DATA move followed by DATA->TOTAL save.
    """
    r.set_const(MARKER, 1)

    for i in range(HEX_DIGITS):
        candidate = TOTAL + i
        saved = LEFT + i
        acc = DATA + i
        ans = ANS + i
        next_ans = RECORD_STRIDE + ANS + i

        r.clear(saved)
        r.clear(acc)
        r.transfer(MARKER, acc)

        r.move(candidate)
        r.emit("[")
        r.add(candidate, -1)
        r.add(saved, 1)
        r.add(acc, 1)
        r.move(candidate)
        r.emit("]")

        r.add(acc, 15)
        r.move(ans)
        r.emit("[")
        r.add(ans, -1)
        r.add(next_ans, 1)
        r.add(acc, -1)
        r.move(ans)
        r.emit("]")

        map_total_base16_threshold(r, acc, candidate, MARKER)

    # Unsigned candidate < ans is !final-carry.
    ult = DATA
    r.set_const(ult, 1)
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    r.clear(ult)
    r.move(MARKER)
    r.emit("]")

    # Signed ordering. Saved LEFT is the absolute-value candidate; next.ANS is
    # the old ans after destructive transport.  DATA is now free flag scratch.
    sign_data = DATA + 1
    sign_ans = DATA + 2
    signs_differ = DATA + 3
    gate = DATA + 4
    tmp = DATA + 5
    restore = DATA + 6
    choose = DATA + 7

    _nibble_ge8(r, sign_data, LEFT + HEX_DIGITS - 1, tmp, restore)
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
        r.transfer(LEFT + i, RECORD_STRIDE + ANS + i)
    r.move(choose)
    r.emit("]")

    # Current DATA/TOTAL/LEFT/ANS are dead once we advance. MARKER was consumed
    # above and must remain zero for the outer record loop.


def _partition_body() -> str:
    r = _RelativeBuilder()
    consume_data_into_left(r)
    move_state_and_total_minus_double_left_into_total(r)
    _abs_total_inplace(r)
    _min_and_move_ans_with_total(r)
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
