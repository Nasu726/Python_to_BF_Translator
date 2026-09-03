"""Counted lexical reader that retains each prefix sum in DATA.

The canonical reader computes a running TOTAL and destructively transfers it to
the next record, leaving the just-consumed record's DATA as the original input
value.  The partition specialization immediately recomputes prefix sums on its
second pass.

This variant uses the same parser but, after each record, destructively splits
running TOTAL into two destinations in one loop: current DATA receives the
prefix sum and next.TOTAL receives the carried running sum.  Current TOTAL is
still zero afterwards, so the existing final-TOTAL backward propagation remains
valid.  The second pass can consume the stored prefix directly.
"""

from __future__ import annotations

from functools import lru_cache

from bfhexcounted import _drain_remaining_line_body
from bfhexcounted_direct import (
    _decrement_count_with_extent,
    _move_extent_and_set_next_marker,
    _prepare_count_from_first_line,
)
from bfhexcounted_lexfast import _classify_int_char
from bfhexdecimal_compact import decimal_digit_kernel, negate_data_kernel
from bfhexradixfast import add_data_to_total_kernel
from bfhexseq import (
    ACTIVE,
    ANS,
    BACK,
    CH,
    CONT,
    DATA,
    DELIMITER,
    END_LINE,
    GATE,
    HAS_TOKEN,
    HEX_DIGITS,
    IS_MINUS,
    LEFT,
    MARKER,
    RECORD_STRIDE,
    RESTORE,
    SIGN,
    SKIP,
    TOTAL,
    WORKSPACE_END,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _flag_not,
    _transfer_word,
)


def _split_total_into_prefix_and_next(r: _RelativeBuilder) -> None:
    """Consume TOTAL into current DATA and next.TOTAL simultaneously."""
    for i in range(HEX_DIGITS):
        src = TOTAL + i
        prefix = DATA + i
        nxt = RECORD_STRIDE + TOTAL + i
        r.clear(prefix)
        r.clear(nxt)
        r.move(src)
        r.emit("[")
        r.add(src, -1)
        r.add(prefix, 1)
        r.add(nxt, 1)
        r.move(src)
        r.emit("]")


@lru_cache(maxsize=1)
def _counted_record_body() -> str:
    r = _RelativeBuilder()

    # LEFT[0] retains the canonical reader's early-line-end flag. Prefix data
    # is stored in DATA only after parsing, so this control ABI is unchanged.
    r.clear(CH)
    _flag_not(r, SKIP, LEFT)
    r.move(SKIP)
    r.emit("[")
    r.add(SKIP, -1)
    r.move(CH)
    r.emit(",")
    r.move(SKIP)
    r.emit("]")

    r.clear(SIGN)
    _classify_int_char(r)

    r.move(SKIP)
    r.emit("[")
    r.add(SKIP, -1)
    r.move(CH)
    r.emit(",")
    _classify_int_char(r)
    r.move(SKIP)
    r.emit("]")

    _flag_not(r, HAS_TOKEN, END_LINE)
    r.copy_preserved(HAS_TOKEN, GATE, RESTORE)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)

    for i in range(HEX_DIGITS):
        r.clear(DATA + i)

    r.move(IS_MINUS)
    r.emit("[")
    r.add(IS_MINUS, -1)
    r.set_const(SIGN, 1)
    r.move(CH)
    r.emit(",")
    _classify_int_char(r, detect_minus=False)
    r.move(IS_MINUS)
    r.emit("]")

    _flag_not(r, ACTIVE, DELIMITER)
    r.move(ACTIVE)
    r.emit("[")
    r.add(ACTIVE, -1)
    r.add(CH, -ord("0"))
    r.move(0)
    r.emit(decimal_digit_kernel())
    r.pos = 0
    r.move(CH)
    r.emit(",")
    _classify_int_char(r, detect_minus=False)
    _flag_not(r, ACTIVE, DELIMITER)
    r.move(ACTIVE)
    r.emit("]")

    r.move(SIGN)
    r.emit("[")
    r.add(SIGN, -1)
    r.move(0)
    r.emit(negate_data_kernel())
    r.pos = 0
    r.move(SIGN)
    r.emit("]")

    r.move(0)
    r.emit(add_data_to_total_kernel())
    r.pos = 0
    r.move(GATE)
    r.emit("]")

    r.copy_preserved(END_LINE, CONT, RESTORE)
    for cell in range(CH, WORKSPACE_END):
        if cell != CONT:
            r.clear(cell)

    # Replace the canonical TOTAL->next.TOTAL transfer. This also overwrites
    # DATA with the inclusive prefix sum, including synthesized short-line zeros.
    _split_total_into_prefix_and_next(r)
    r.set_const(RECORD_STRIDE + BACK, 1)
    _decrement_count_with_extent(r, ANS)
    _transfer_word(r, ANS, RECORD_STRIDE + ANS)
    _move_extent_and_set_next_marker(r)

    r.transfer(CONT, RECORD_STRIDE + LEFT)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


def read_counted_two_line_s64s_and_sum(
    bf,
    seq: RuntimeHexIntSequence,
) -> None:
    """Read N values while storing every inclusive prefix sum in DATA."""
    if seq.base < 0:
        raise ValueError("sequence base must be non-negative")

    _prepare_count_from_first_line(bf, seq)
    bf.move(seq.base + MARKER)
    bf.emit("[" + _counted_record_body() + "]")

    bf.emit(_drain_remaining_line_body())
    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["read_counted_two_line_s64s_and_sum"]
