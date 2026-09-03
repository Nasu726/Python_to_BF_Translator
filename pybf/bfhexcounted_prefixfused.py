"""Counted lexical reader with fused prefix-sum emission.

This is the prefix-retaining reader with its two arithmetic stages fused. The
parsed DATA value and incoming running TOTAL are consumed once and the resulting
prefix sum is emitted directly to both current DATA and next.TOTAL.
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
from bfhexprefixsum import prefix_sum_kernel
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
    WORKSPACE_END,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _flag_not,
    _transfer_word,
)


@lru_cache(maxsize=1)
def _counted_record_body() -> str:
    r = _RelativeBuilder()

    # Canonical early-line-end state lives in LEFT[0] until parsing starts.
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

    # DATA must be zero even when the physical line ended early; the fused sum
    # kernel then naturally emits an unchanged running prefix for zero-fill.
    for i in range(HEX_DIGITS):
        r.clear(DATA + i)

    r.copy_preserved(HAS_TOKEN, GATE, RESTORE)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)

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
    r.move(GATE)
    r.emit("]")

    # Parser scratch occupies future-record cells. Scrub it before emitting the
    # next carried TOTAL, otherwise the scrub would erase that state.
    r.copy_preserved(END_LINE, CONT, RESTORE)
    for cell in range(CH, WORKSPACE_END):
        if cell != CONT:
            r.clear(cell)

    r.move(0)
    r.emit(prefix_sum_kernel())
    r.pos = 0

    r.set_const(RECORD_STRIDE + BACK, 1)
    _decrement_count_with_extent(r, ANS)
    _transfer_word(r, ANS, RECORD_STRIDE + ANS)
    _move_extent_and_set_next_marker(r)

    r.transfer(CONT, RECORD_STRIDE + LEFT)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


def read_counted_two_line_s64s_and_sum(bf, seq: RuntimeHexIntSequence) -> None:
    """Read N values and retain prefixes with a fused sum/transport kernel."""
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
