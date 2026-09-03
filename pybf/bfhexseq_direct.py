"""Direct-hex decimal reader for ``RuntimeHexIntSequence``.

This is an experimental drop-in input path.  It preserves the established
line/token, sentinel, BACK-chain, and running-TOTAL semantics from ``bfhexseq``
but accumulates decimal digits directly into the sixteen DATA nibbles.  It
therefore avoids the 32-lane radix-4 decimal accumulator and the subsequent
radix-4 -> hex packing pass.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexdecimal import decimal_digit_kernel, negate_data_kernel
from bfhexseq import (
    ACTIVE,
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


@lru_cache(maxsize=1)
def _read_record_body_direct() -> str:
    r = _RelativeBuilder()
    r.move(CH)
    r.emit(",")
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

    # DATA is the decimal accumulator itself on this path.  Clear it once per
    # token instead of building a separate radix-4 word and packing later.
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
    _flag_not(r, CONT, END_LINE)
    r.move(GATE)
    r.emit("]")

    _flag_not(r, GATE, HAS_TOKEN)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)
    r.clear(MARKER)
    r.move(GATE)
    r.emit("]")

    # The direct kernel uses only cells already inside the established future
    # parser workspace.  Scrub it under the same contract as the old reader.
    for cell in range(CH, WORKSPACE_END):
        if cell not in (CONT, HAS_TOKEN):
            r.clear(cell)

    r.move(HAS_TOKEN)
    r.emit("[")
    r.add(HAS_TOKEN, -1)
    _transfer_word(r, TOTAL, RECORD_STRIDE + TOTAL)
    r.set_const(RECORD_STRIDE + BACK, 1)
    r.move(HAS_TOKEN)
    r.emit("]")

    r.move(CONT)
    r.emit("[")
    r.add(CONT, -1)
    r.set_const(RECORD_STRIDE + MARKER, 1)
    r.move(CONT)
    r.emit("]")
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


def read_lf_terminated_s64s_and_sum_direct(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
) -> None:
    """Read one LF-terminated signed-int64 line directly into hex records."""
    if seq.base < 0:
        raise ValueError("sequence base must be non-negative")

    bf.move(seq.base)
    bf.set_const(seq.base + MARKER, 1)
    bf.clear(seq.base + BACK)
    bf.move(seq.base + MARKER)
    bf.emit("[" + _read_record_body_direct() + "]")

    bf.emit("<" * (RECORD_STRIDE - BACK))
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["read_lf_terminated_s64s_and_sum_direct"]
