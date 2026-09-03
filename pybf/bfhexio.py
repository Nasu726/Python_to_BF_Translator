"""Final-output helpers for runtime hexadecimal carried-state records.

The sequential hex passes deliberately consume record markers.  The BACK lane
remains intact, so a final state word can still be transported from the sentinel
back to record zero without knowing runtime N.

For compact final decimal output, the dead runtime records are reused as a
20-step carried-state division chain.  Each step divides one hexadecimal int64
word by ten, leaves one decimal remainder in the current record, moves the
quotient one record right, and carries a bounded iteration marker.  A reverse
BACK walk then prints those remainders most-significant first.  This avoids the
large fixed Boolean-BCD printer used by the general scalar backend.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bfhexseq import (
    ANS,
    BACK,
    DATA,
    HEX_DIGITS,
    MARKER,
    RECORD_STRIDE,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _map_total_base16,
    _transfer_word,
)


DEC_REM = DATA
DEC_COMBINED = DATA + 1
DEC_QUOT = DATA + 2
DEC_COUNTDOWN = DATA + 3
DEC_ZERO = DATA + 4
DEC_TMP = DATA + 5
DEC_RESTORE = DATA + 6
DEC_STARTED = DATA + 7
DEC_ASCII = DATA + 8
DEC_GATE = DATA + 9
DEC_SIGN = DATA + 10
DEC_CARRY = DATA + 11
DEC_TOTAL = DATA + 12
DECIMAL_DIGITS = 20


def propagate_field_back_after_consumed_markers(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    field_base: int,
) -> None:
    """Move a sixteen-nibble sentinel field back to record zero.

    This is intended for final carried state after a pass that consumed all
    materialized MARKER cells.  Record zero has BACK=0; every later materialized
    record and the sentinel have BACK=1.  We first walk one record beyond the
    sentinel to the first zero BACK, step left once, and then move the field
    backward until record zero is reached.
    """
    if not 0 <= field_base <= RECORD_STRIDE - HEX_DIGITS:
        raise ValueError("field must fit inside one runtime hex record")

    # Record 1 is either the first successor (BACK=1) or, for an empty
    # sequence, already the first unmaterialized record (BACK=0).
    bf.move(seq.base + RECORD_STRIDE + BACK)
    bf.emit("[" + ">" * RECORD_STRIDE + "]")
    bf.emit("<" * RECORD_STRIDE)

    # We are at sentinel BACK (or record-zero BACK for an empty sequence).
    r = _RelativeBuilder(initial_pos=BACK)
    _transfer_word(r, field_base, field_base - RECORD_STRIDE)
    r.move(BACK - RECORD_STRIDE)
    bf.emit("[" + r.code() + "]")

    bf.emit("<" * BACK)
    bf.ptr = seq.base


def consume_hex_word_to_binary64(
    bf: BFEmitter,
    *,
    hex_base: int,
    dst: Int64Ref,
    scratch_base: int,
) -> None:
    """Consume sixteen 0..15 nibbles into a Boolean-bit int64 word.

    Each nibble drives at most fifteen iterations of a four-bit incrementer, so
    runtime is bounded independently of the represented 64-bit magnitude.  This
    helper remains useful at representation boundaries; the compact final
    printer below no longer needs that conversion.
    """
    core = Binary64Core(bf, scratch_base=scratch_base)

    for bit in range(WORD_BITS):
        bf.clear(dst.bit(bit))
    core._clear_scratch()

    for nibble in range(HEX_DIGITS):
        src = hex_base + nibble
        low_bit = nibble * 4

        bf.begin_while(src)
        bf.add_const(src, -1)
        bf.set_const(core.carry0, 1)
        bf.clear(core.carry1)

        carry_in, carry_out = core.carry0, core.carry1
        for offset in range(4):
            bf.clear(carry_out)
            bf.begin_while(carry_in)
            bf.add_const(carry_in, -1)
            core._add_one(dst.bit(low_bit + offset), carry_out, core.s2)
            bf.end_while(carry_in)
            carry_in, carry_out = carry_out, carry_in

        bf.clear(core.carry0)
        bf.clear(core.carry1)
        bf.end_while(src)

    core._clear_scratch()


def _flag_zero(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """Set result to one iff src is zero, preserving src."""
    r.set_const(result, 1)
    r.copy_preserved(src, tmp, restore)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(result)
    r.move(tmp)
    r.emit("]")


def _nibble_ge8(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """Set result to one iff a preserved 0..15 nibble is at least eight."""
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


def _negate_hex_word(r: _RelativeBuilder, word_base: int) -> None:
    """Two's-complement negate a sixteen-nibble word in place."""
    r.set_const(DEC_CARRY, 1)
    for i in range(HEX_DIGITS):
        cell = word_base + i
        r.set_const(DEC_TOTAL, 15)
        r.move(cell)
        r.emit("[")
        r.add(cell, -1)
        r.add(DEC_TOTAL, -1)
        r.move(cell)
        r.emit("]")
        r.transfer(DEC_CARRY, DEC_TOTAL)
        _map_total_base16(r, DEC_TOTAL, cell, DEC_CARRY)
    r.clear(DEC_CARRY)
    r.clear(DEC_TOTAL)


def _emit_signed_magnitude_prefix(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    field_base: int,
) -> None:
    """Print a leading minus if needed and leave field_base as magnitude."""
    r = _RelativeBuilder()
    _nibble_ge8(
        r,
        DEC_SIGN,
        field_base + HEX_DIGITS - 1,
        DEC_TMP,
        DEC_RESTORE,
    )
    r.move(DEC_SIGN)
    r.emit("[")
    r.add(DEC_SIGN, -1)
    r.set_const(DEC_ASCII, ord("-"))
    r.move(DEC_ASCII)
    r.emit(".")
    _negate_hex_word(r, field_base)
    r.move(DEC_SIGN)
    r.emit("]")
    r.clear(DEC_SIGN)
    r.clear(DEC_TMP)
    r.clear(DEC_RESTORE)
    r.move(MARKER)

    bf.move(seq.base + MARKER)
    bf.emit(r.code())
    bf.ptr = seq.base + MARKER


@lru_cache(maxsize=None)
def _div10_carried_body(field_base: int) -> str:
    """Divide current hex word by ten and carry quotient one record right."""
    r = _RelativeBuilder()

    # The marker itself is the number of decimal rounds remaining.  Consume one
    # round now and move the remainder to the next record at the end.
    r.add(MARKER, -1)
    r.clear(DEC_REM)

    for i in range(HEX_DIGITS - 1, -1, -1):
        src = field_base + i
        dst = RECORD_STRIDE + field_base + i

        r.clear(DEC_COMBINED)
        r.clear(DEC_QUOT)
        r.set_const(DEC_COUNTDOWN, 10)

        # combined = previous_remainder * 16 + current_nibble.
        r.transfer(src, DEC_COMBINED)
        r.move(DEC_REM)
        r.emit("[")
        r.add(DEC_REM, -1)
        r.add(DEC_COMBINED, 16)
        r.move(DEC_REM)
        r.emit("]")

        # Divide a bounded 0..159 byte by ten. COUNTDOWN cycles 10..1; each
        # zero crossing emits one quotient unit and resets the remainder.
        r.move(DEC_COMBINED)
        r.emit("[")
        r.add(DEC_COMBINED, -1)
        r.add(DEC_REM, 1)
        r.add(DEC_COUNTDOWN, -1)
        _flag_zero(r, DEC_ZERO, DEC_COUNTDOWN, DEC_TMP, DEC_RESTORE)
        r.move(DEC_ZERO)
        r.emit("[")
        r.add(DEC_ZERO, -1)
        r.add(DEC_QUOT, 1)
        r.clear(DEC_REM)
        r.set_const(DEC_COUNTDOWN, 10)
        r.move(DEC_ZERO)
        r.emit("]")
        r.move(DEC_COMBINED)
        r.emit("]")

        r.clear(dst)
        r.transfer(DEC_QUOT, dst)
        r.clear(DEC_COUNTDOWN)
        r.clear(DEC_ZERO)
        r.clear(DEC_TMP)
        r.clear(DEC_RESTORE)

    # DEC_REM is this round's least-significant decimal digit.  The quotient is
    # already resident in the next record's field word.
    r.set_const(RECORD_STRIDE + BACK, 1)
    r.clear(RECORD_STRIDE + MARKER)
    r.transfer(MARKER, RECORD_STRIDE + MARKER)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=1)
def _reverse_decimal_print_body() -> str:
    """Carry leading-zero state left and print the previous record's digit."""
    r = _RelativeBuilder(initial_pos=BACK)

    prev_started = DEC_STARTED - RECORD_STRIDE
    prev_ascii = DEC_ASCII - RECORD_STRIDE
    prev_digit = DEC_REM - RECORD_STRIDE
    prev_gate = DEC_GATE - RECORD_STRIDE
    prev_tmp = DEC_TMP - RECORD_STRIDE
    prev_restore = DEC_RESTORE - RECORD_STRIDE
    prev_back = BACK - RECORD_STRIDE

    r.clear(prev_started)
    r.transfer(DEC_STARTED, prev_started)
    r.set_const(prev_ascii, ord("0"))

    # Consuming the decimal digit builds its ASCII byte and marks output as
    # started if the digit was nonzero.
    r.move(prev_digit)
    r.emit("[")
    r.add(prev_digit, -1)
    r.add(prev_ascii, 1)
    r.set_const(prev_started, 1)
    r.move(prev_digit)
    r.emit("]")

    # Record zero is the final digit position; force one printed zero when all
    # twenty remainders were zero.
    _flag_zero(r, prev_gate, prev_back, prev_tmp, prev_restore)
    r.move(prev_gate)
    r.emit("[")
    r.add(prev_gate, -1)
    r.set_const(prev_started, 1)
    r.move(prev_gate)
    r.emit("]")

    r.copy_preserved(prev_started, prev_gate, prev_tmp)
    r.move(prev_gate)
    r.emit("[")
    r.add(prev_gate, -1)
    r.move(prev_ascii)
    r.emit(".")
    r.move(prev_gate)
    r.emit("]")

    r.clear(prev_tmp)
    r.clear(prev_restore)
    r.move(prev_back)
    return r.code()


def print_record_hex_s64_compact(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    field_base: int = ANS,
) -> None:
    """Print a record-zero signed int64 field in decimal with a trailing LF.

    Preconditions:
    * the final word is on record zero;
    * MARKER cells are zero (as after the final sequential pass);
    * the runtime record region is dead except for the word being printed.

    Twenty carried divide-by-ten rounds cover every unsigned 64-bit magnitude.
    The reverse BACK walk suppresses leading zeroes while always printing at
    least one digit.
    """
    if field_base < DEC_TOTAL + 1 or field_base + HEX_DIGITS > RECORD_STRIDE:
        raise ValueError("compact decimal output field overlaps local scratch")

    _emit_signed_magnitude_prefix(bf, seq, field_base)

    bf.set_const(seq.base + MARKER, DECIMAL_DIGITS)
    bf.move(seq.base + MARKER)
    bf.emit("[" + _div10_carried_body(field_base) + "]")

    # The carried loop exits on record 20's zero marker. Initialize the reverse
    # leading-zero state there, then walk/print records 19..0.
    bf.emit(">" * DEC_STARTED + "[-]" + "<" * (DEC_STARTED - BACK))
    bf.emit("[" + _reverse_decimal_print_body() + "]")

    # Reverse loop exits at record-zero BACK. Reuse its ASCII cell for LF.
    bf.emit(">" * (DEC_ASCII - BACK) + "[-]++++++++++." + "<" * DEC_ASCII)
    bf.ptr = seq.base


__all__ = [
    "consume_hex_word_to_binary64",
    "print_record_hex_s64_compact",
    "propagate_field_back_after_consumed_markers",
]
