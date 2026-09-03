"""N-counted input for the runtime hexadecimal integer sequence.

The initial scalable partition slice inferred the record count from the number
of tokens on the second line.  This module instead carries the parsed first-line
N through the record chain so the emitted runtime loop itself decides how many
values participate in the recognized ``range(n)`` passes.

Extra tokens after N are drained to preserve one-line ``input()`` semantics. If
the line ends before N values, remaining records are materialized as zeroes,
matching the generic runtime's zero-initialized out-of-length list loads rather
than reading into the following input line or looping forever at EOF.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
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
    _decimal_digit_kernel,
    _eq_const,
    _flag_not,
    _is_hspace,
    _is_line_end,
    _negate_kernel,
    _pack_hex_kernel,
    _transfer_word,
)


def _nibble_ge8(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """result = (src >= 8), preserving a 0..15 nibble."""
    r.clear(result)
    r.copy_preserved(src, tmp, restore)
    for step in range(1, 9):
        r.move(tmp)
        r.emit("[")
        r.add(tmp, -1)
        if step == 8:
            r.set_const(result, 1)
            r.clear(tmp)
    for _ in range(8):
        r.move(tmp)
        r.emit("]")


def _set_flag_if_word_nonzero(
    r: _RelativeBuilder,
    flag: int,
    word_base: int,
    tmp: int,
    restore: int,
) -> None:
    """flag = any(word nibble != 0), preserving the word."""
    r.clear(flag)
    for i in range(HEX_DIGITS):
        r.copy_preserved(word_base + i, tmp, restore)
        r.move(tmp)
        r.emit("[")
        r.clear(tmp)
        r.set_const(flag, 1)
        r.move(tmp)
        r.emit("]")
    r.clear(tmp)
    r.clear(restore)


def _decrement_hex_word(r: _RelativeBuilder, word_base: int) -> None:
    """Subtract one modulo 2**64 from a known-positive hex word."""
    borrow = LEFT
    is_zero = LEFT + 1
    tmp = LEFT + 2
    restore = LEFT + 3
    gate = LEFT + 4
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
        r.move(gate)
        r.emit("]")

    for cell in (borrow, is_zero, tmp, restore, gate):
        r.clear(cell)


def _prepare_count_from_first_line(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
) -> None:
    """Read first-line N, move it to record-zero ANS, and seed its marker.

    Reusing the existing line parser gives identical signed-int64 behavior.
    Negative N is normalized to zero because both recognized ``range(n)`` loops
    are empty for negative values.
    """
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

    _set_flag_if_word_nonzero(r, MARKER, ANS, tmp, restore)
    r.clear(BACK)
    for cell in (sign, tmp, restore):
        r.clear(cell)
    r.move(MARKER)

    bf.move(seq.base + MARKER)
    bf.emit(r.code())
    bf.ptr = seq.base + MARKER


@lru_cache(maxsize=1)
def _counted_record_body() -> str:
    """Materialize one record, decrement carried count, and advance."""
    r = _RelativeBuilder()

    # LEFT[0] is a carried Boolean saying the source line has already ended.
    # Conditional input prevents missing values from consuming the next line or
    # repeatedly reading EOF; the normal parser then sees a synthetic NUL and
    # leaves DATA at its zero-initialized value.
    r.clear(CH)
    _flag_not(r, SKIP, LEFT)
    r.move(SKIP)
    r.emit("[")
    r.add(SKIP, -1)
    r.move(CH)
    r.emit(",")
    r.move(SKIP)
    r.emit("]")

    # SIGN aliases the following record's BACK cell.  The one-value first-line
    # parse leaves that sentinel BACK set, so every second-line iteration must
    # explicitly zero SIGN rather than relying on virgin future tape.
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
    r.emit(_decimal_digit_kernel())
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
    r.emit(_negate_kernel())
    r.pos = 0
    r.move(SIGN)
    r.emit("]")

    r.move(0)
    r.emit(_pack_hex_kernel())
    r.pos = 0
    r.emit(_add_data_to_total_kernel())
    r.pos = 0
    r.move(GATE)
    r.emit("]")

    # END_LINE is true both for a real LF/EOF and for the synthetic NUL used
    # after a previous early line end. Carry it to the next record before the
    # future-record parser workspace is scrubbed.
    r.copy_preserved(END_LINE, CONT, RESTORE)
    for cell in range(CH, WORKSPACE_END):
        if cell != CONT:
            r.clear(cell)

    # Every counted iteration is one logical range element.  Missing source
    # tokens therefore leave DATA=0 but still consume one unit of N.
    _transfer_word(r, TOTAL, RECORD_STRIDE + TOTAL)
    r.set_const(RECORD_STRIDE + BACK, 1)
    _decrement_hex_word(r, ANS)
    _transfer_word(r, ANS, RECORD_STRIDE + ANS)

    _set_flag_if_word_nonzero(
        r,
        RECORD_STRIDE + MARKER,
        RECORD_STRIDE + ANS,
        LEFT,
        LEFT + 1,
    )

    r.transfer(CONT, RECORD_STRIDE + LEFT)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=1)
def _drain_remaining_line_body() -> str:
    """At the counted-loop sentinel, consume through LF/EOF only if needed."""
    r = _RelativeBuilder()
    need = DATA
    ch = DATA + 1
    is_end = DATA + 2
    tmp = DATA + 3
    restore = DATA + 4

    r.set_const(need, 1)
    r.copy_preserved(LEFT, tmp, restore)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(need)
    r.move(tmp)
    r.emit("]")

    r.move(need)
    r.emit("[")
    r.add(need, -1)
    r.move(ch)
    r.emit(",")
    _is_line_end(r, is_end, ch)
    r.set_const(need, 1)
    r.copy_preserved(is_end, tmp, restore)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(need)
    r.move(tmp)
    r.emit("]")
    r.move(need)
    r.emit("]")

    for cell in (LEFT, need, ch, is_end, tmp, restore):
        r.clear(cell)
    r.move(MARKER)
    return r.code()


def read_counted_two_line_s64s_and_sum(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
) -> None:
    """Read ``N`` then materialize exactly max(N, 0) integer records.

    Extra second-line tokens are drained. If the line supplies fewer than N
    tokens, the remaining DATA records are zero-filled without crossing into a
    following line. The end sentinel receives the carried TOTAL.
    """
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
