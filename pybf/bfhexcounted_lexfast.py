"""Counted direct-hex reader with a single-pass ASCII token classifier.

The established parser repeatedly preserves and compares CH for horizontal
space, line end, and minus.  For valid signed-decimal ``split`` input all
relevant bytes are <= '9'.  This variant preserves CH once through a bounded
57-step decoder and changes classification flags only at category boundaries.
It keeps the count-extent and direct-hex arithmetic from bfhexcounted_direct.
"""

from __future__ import annotations

from functools import lru_cache

from bfhexcounted import _drain_remaining_line_body
from bfhexcounted_direct import (
    _decrement_count_with_extent,
    _move_extent_and_set_next_marker,
    _prepare_count_from_first_line,
)
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
    TMP,
    TOTAL,
    WORKSPACE_END,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _add_data_to_total_kernel,
    _flag_not,
    _transfer_word,
)


def _classify_int_char(r: _RelativeBuilder, *, detect_minus: bool = True) -> None:
    """Classify CH while preserving it with one bounded ASCII pass.

    Outputs:
      SKIP       horizontal space (space/tab/CR)
      END_LINE   LF or NUL
      DELIMITER  horizontal space or line end
      IS_MINUS   '-' when detect_minus is true

    The decoder only needs exact transitions through ASCII '9' (57).  Values
    above that are outside the signed-decimal input language recognized by this
    runtime slice, just as the older parser did not validate arbitrary text.
    """
    r.clear(SKIP)
    r.set_const(END_LINE, 1)   # NUL
    r.set_const(DELIMITER, 1)  # NUL
    r.clear(IS_MINUS)
    r.clear(TMP)

    for step in range(1, ord("9") + 1):
        r.move(CH)
        r.emit("[")
        r.add(CH, -1)
        r.add(TMP, 1)

        if step == 1:
            r.clear(END_LINE)
            r.clear(DELIMITER)
        elif step == ord("\t"):
            r.set_const(SKIP, 1)
            r.set_const(DELIMITER, 1)
        elif step == ord("\n"):
            r.clear(SKIP)
            r.set_const(END_LINE, 1)
            r.set_const(DELIMITER, 1)
        elif step == ord("\n") + 1:
            r.clear(END_LINE)
            r.clear(DELIMITER)
        elif step == ord("\r"):
            r.set_const(SKIP, 1)
            r.set_const(DELIMITER, 1)
        elif step == ord("\r") + 1:
            r.clear(SKIP)
            r.clear(DELIMITER)
        elif step == ord(" "):
            r.set_const(SKIP, 1)
            r.set_const(DELIMITER, 1)
        elif step == ord(" ") + 1:
            r.clear(SKIP)
            r.clear(DELIMITER)
        elif detect_minus and step == ord("-"):
            r.set_const(IS_MINUS, 1)
        elif detect_minus and step == ord("-") + 1:
            r.clear(IS_MINUS)

    for _ in range(ord("9")):
        r.move(CH)
        r.emit("]")

    # Restore the byte. TMP is only four cells from CH, much closer than the
    # generic equality helper's repeated save/restore workspaces.
    r.move(TMP)
    r.emit("[")
    r.add(TMP, -1)
    r.add(CH, 1)
    r.move(TMP)
    r.emit("]")


@lru_cache(maxsize=1)
def _counted_record_body() -> str:
    r = _RelativeBuilder()

    # LEFT[0] carries the previous early-line-end state.  When it is set, skip
    # physical input and classify synthetic NUL instead.
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

    # Consume any additional horizontal whitespace using the same classifier.
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
    # Do not re-arm the loop control if malformed input contains another '-'.
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
    bf,
    seq: RuntimeHexIntSequence,
) -> None:
    """Read N and N values with count extent, direct hex, and fused lexing."""
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
