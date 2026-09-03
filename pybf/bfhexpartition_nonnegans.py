"""Partition variant with a cheaper min path for nonnegative initial answers.

After two's-complement abs, the candidate is nonnegative except for INT64_MIN,
whose wrapped absolute value is still INT64_MIN.  If the initial answer is a
nonnegative signed int64, the carried answer can therefore only be nonnegative
or INT64_MIN.  This lets signed min avoid the general sign-XOR machinery:

* negative carried ans -> keep it;
* otherwise negative candidate -> choose it;
* otherwise use the unsigned comparison result.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _nibble_ge8, _set_hex_const
from bfhexpartition_totalcandidate import _abs_total_inplace
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


def _min_and_move_ans_nonnegative(r: _RelativeBuilder) -> None:
    """Set next.ANS=min_signed(ANS,TOTAL) under the nonnegative-ans invariant."""
    r.set_const(MARKER, 1)

    # Destructively compare candidate against ans while carrying the old ans
    # right and saving the candidate in dead current LEFT scratch.
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

    # Unsigned candidate < ans is !final carry.
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

    _nibble_ge8(r, sign_candidate, LEFT + HEX_DIGITS - 1, tmp, restore)
    _nibble_ge8(
        r,
        sign_ans,
        RECORD_STRIDE + ANS + HEX_DIGITS - 1,
        tmp,
        restore,
    )

    r.copy_preserved(ult, choose, tmp)

    # Wrapped abs can be negative only for INT64_MIN.  Against a nonnegative
    # carried answer it must win.
    r.move(sign_candidate)
    r.emit("[")
    r.add(sign_candidate, -1)
    r.set_const(choose, 1)
    r.move(sign_candidate)
    r.emit("]")

    # Once INT64_MIN has become the carried answer, no later abs candidate can
    # be smaller; keeping the existing answer is also correct for equality.
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
        r.transfer(LEFT + i, RECORD_STRIDE + ANS + i)
    r.move(choose)
    r.emit("]")


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
        raise ValueError("nonnegative-answer partition requires signed nonnegative initial_ans")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
