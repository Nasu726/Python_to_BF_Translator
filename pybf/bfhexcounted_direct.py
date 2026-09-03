"""N-counted direct-hex reader with compact count extent metadata.

This combines the count-extent invariant with direct decimal accumulation into
DATA nibbles for the N values. The first-line count intentionally keeps the
source-compact radix-4 parser so the direct-decimal kernel is emitted only once.
The direct kernel itself uses operation-specific radix-16 bounds.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexcounted import _drain_remaining_line_body, _nibble_ge8
from bfhexdecimal_compact import decimal_digit_kernel, negate_data_kernel
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
    LINE_TMP,
    MARKER,
    RECORD_STRIDE,
    RESTORE,
    SIGN,
    SKIP,
    TOTAL,
    WORKSPACE_END,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _add_data_to_total_kernel,
    _eq_const,
    _flag_not,
    _is_hspace,
    _is_line_end,
    _transfer_word,
)


COUNT_EXTENT = LEFT + HEX_DIGITS - 1


def _set_count_extent_and_marker(
    r: _RelativeBuilder,
    word_base: int,
    *,
    tmp: int,
    restore: int,
) -> None:
    r.clear(COUNT_EXTENT)
    r.clear(MARKER)
    for i in range(HEX_DIGITS):
        r.copy_preserved(word_base + i, tmp, restore)
        r.move(tmp)
        r.emit("[")
        r.clear(tmp)
        r.set_const(COUNT_EXTENT, i + 1)
        r.set_const(MARKER, 1)
        r.move(tmp)
        r.emit("]")
    r.clear(tmp)
    r.clear(restore)


def _decrement_count_with_extent(r: _RelativeBuilder, word_base: int) -> None:
    borrow = LEFT
    is_zero = LEFT + 1
    tmp = LEFT + 2
    restore = LEFT + 3
    gate = LEFT + 4
    became_zero = LEFT + 5
    extent_match = LEFT + 6
    r.set_const(borrow, 1)

    for i in range(HEX_DIGITS):
        r.copy_preserved(borrow, gate, tmp)
        r.move(gate)
        r.emit("[")
        r.add(gate, -1)
        r.clear(borrow)

        _eq_const(r, is_zero, word_base + i, 0, tmp, restore)
        r.add(word_base + i, -1)
        r.move(is_zero)
        r.emit("[")
        r.add(is_zero, -1)
        r.set_const(word_base + i, 15)
        r.set_const(borrow, 1)
        r.move(is_zero)
        r.emit("]")

        _eq_const(r, became_zero, word_base + i, 0, tmp, restore)
        r.move(became_zero)
        r.emit("[")
        r.add(became_zero, -1)
        _eq_const(r, extent_match, COUNT_EXTENT, i + 1, tmp, restore)
        r.move(extent_match)
        r.emit("[")
        r.add(extent_match, -1)
        r.add(COUNT_EXTENT, -1)
        r.move(extent_match)
        r.emit("]")
        r.move(became_zero)
        r.emit("]")

        r.move(gate)
        r.emit("]")

    for cell in (
        borrow,
        is_zero,
        tmp,
        restore,
        gate,
        became_zero,
        extent_match,
    ):
        r.clear(cell)


def _move_extent_and_set_next_marker(r: _RelativeBuilder) -> None:
    next_extent = RECORD_STRIDE + COUNT_EXTENT
    next_marker = RECORD_STRIDE + MARKER
    r.clear(next_extent)
    r.clear(next_marker)
    r.move(COUNT_EXTENT)
    r.emit("[")
    r.add(COUNT_EXTENT, -1)
    r.add(next_extent, 1)
    r.set_const(next_marker, 1)
    r.move(COUNT_EXTENT)
    r.emit("]")


def _prepare_count_from_first_line(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
) -> None:
    seq.read_lf_terminated_s64s_and_sum(bf)

    r = _RelativeBuilder()
    _transfer_word(r, DATA, ANS)

    sign = LEFT
    tmp = LEFT + 1
    restore = LEFT + 2
    _nibble_ge8(r, sign, ANS + HEX_DIGITS - 1, tmp, restore)
    r.move(sign)
    r.emit("[")
    r.add(sign, -1)
    for i in range(HEX_DIGITS):
        r.clear(ANS + i)
    r.move(sign)
    r.emit("]")

    _set_count_extent_and_marker(r, ANS, tmp=tmp, restore=restore)
    r.clear(BACK)
    for cell in (sign, tmp, restore):
        r.clear(cell)
    r.move(MARKER)

    bf.move(seq.base + MARKER)
    bf.emit(r.code())
    bf.ptr = seq.base + MARKER


@lru_cache(maxsize=1)
def _counted_record_body() -> str:
    r = _RelativeBuilder()

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
    _is_hspace(r, SKIP, CH)
    r.move(SKIP)
    r.emit("[")
    r.add(SKIP, -1)
    r.move(CH)
    r.emit(",")
    _is_hspace(r, SKIP, CH)
    r.move(SKIP)
    r.emit("]")

    _is_line_end(r, END_LINE, CH)
    _flag_not(r, HAS_TOKEN, END_LINE)
    r.copy_preserved(HAS_TOKEN, GATE, RESTORE)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)

    for i in range(HEX_DIGITS):
        r.clear(DATA + i)

    _eq_const(r, IS_MINUS, CH, ord("-"))
    r.move(IS_MINUS)
    r.emit("[")
    r.add(IS_MINUS, -1)
    r.set_const(SIGN, 1)
    r.move(CH)
    r.emit(",")
    r.move(IS_MINUS)
    r.emit("]")

    _is_line_end(r, END_LINE, CH)
    _is_hspace(r, DELIMITER, CH)
    r.copy_preserved(END_LINE, LINE_TMP, RESTORE)
    r.move(LINE_TMP)
    r.emit("[")
    r.add(LINE_TMP, -1)
    r.set_const(DELIMITER, 1)
    r.move(LINE_TMP)
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
    _is_line_end(r, END_LINE, CH)
    _is_hspace(r, DELIMITER, CH)
    r.copy_preserved(END_LINE, LINE_TMP, RESTORE)
    r.move(LINE_TMP)
    r.emit("[")
    r.add(LINE_TMP, -1)
    r.set_const(DELIMITER, 1)
    r.move(LINE_TMP)
    r.emit("]")
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
    r.emit(_add_data_to_total_kernel())
    r.pos = 0
    r.move(GATE)
    r.emit("]")

    r.copy_preserved(END_LINE, CONT, RESTORE)
    for cell in range(CH, WORKSPACE_END):
        if cell != CONT:
            r.clear(cell)

    _transfer_word(r, TOTAL, RECORD_STRIDE + TOTAL)
    r.set_const(RECORD_STRIDE + BACK, 1)
    _decrement_count_with_extent(r, ANS)
    _transfer_word(r, ANS, RECORD_STRIDE + ANS)
    _move_extent_and_set_next_marker(r)

    r.transfer(CONT, RECORD_STRIDE + LEFT)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


def read_counted_two_line_s64s_and_sum(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
) -> None:
    """Read N and N integers using compact direct-hex parsing plus count extent."""
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


__all__ = ["COUNT_EXTENT", "read_counted_two_line_s64s_and_sum"]
