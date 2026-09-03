"""Fused hexadecimal candidate arithmetic for the partition runtime.

After the partition cursor has moved live TOTAL/LEFT into the following record,
the current record is scratch.  Compute::

    DATA = next.TOTAL - 2 * next.LEFT   (mod 2**64)

in one radix-16 pass.  The earlier vertical slice first materialized
``2*LEFT`` and then ran a second subtraction pass, requiring two radix maps per
nibble.  Here the doubled-value carry and subtraction carry are combined into a
single bounded total in 0..31, so each nibble needs only one radix map.
"""

from __future__ import annotations

from bfhexseq import (
    DATA,
    HEX_DIGITS,
    LEFT,
    MARKER,
    RECORD_STRIDE,
    TOTAL,
    _RelativeBuilder,
    _add_preserved_small,
    _map_total_base16,
)


def _nibble_ge8(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """result = (src >= 8), preserving one 0..15 nibble."""
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


def _subtract_double_preserved(
    r: _RelativeBuilder,
    src: int,
    total: int,
    tmp: int,
) -> None:
    """total -= 2*src while preserving a 0..15 source nibble."""
    r.clear(tmp)
    r.move(src)
    r.emit("[")
    r.add(src, -1)
    r.add(total, -2)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.add(tmp, -1)
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


def total_minus_double_next_left_into_data(r: _RelativeBuilder) -> None:
    """Compute next.TOTAL - 2*next.LEFT into current DATA in one nibble pass.

    Let ``dc`` be the carry from doubling the previous LEFT nibble and ``sc``
    the radix-complement carry for subtraction.  For LEFT nibble ``l`` and TOTAL
    nibble ``t`` we map the bounded value::

        t + 15 + sc - 2*l - dc + 16*(l >= 8)

    The expression is always in 0..31.  Its low hexadecimal digit is the result
    nibble and its radix carry is the next subtraction carry.  ``l >= 8`` is
    exactly the next carry of ``2*l + dc`` because ``dc`` is only 0 or 1.
    """
    sub_carry = LEFT
    next_sub_carry = LEFT + 1
    next_double_carry = LEFT + 2
    preserve_tmp = LEFT + 3
    restore = LEFT + 4
    gate = LEFT + 5

    r.clear(MARKER)  # incoming doubling carry
    r.set_const(sub_carry, 1)  # radix-complement +1 for subtraction

    for i in range(HEX_DIGITS):
        total = TOTAL + i
        next_total = RECORD_STRIDE + TOTAL + i
        next_left = RECORD_STRIDE + LEFT + i

        r.clear(total)
        r.clear(next_sub_carry)
        r.clear(next_double_carry)
        r.transfer(sub_carry, total)
        _add_preserved_small(r, next_total, total, preserve_tmp)
        r.add(total, 15)

        # For l>=8 add the compensating radix before subtracting 2*l, keeping
        # the temporary non-negative and therefore inside ordinary byte range.
        _nibble_ge8(
            r,
            next_double_carry,
            next_left,
            preserve_tmp,
            restore,
        )
        r.copy_preserved(next_double_carry, gate, preserve_tmp)
        r.move(gate)
        r.emit("[")
        r.add(gate, -1)
        r.add(total, 16)
        r.move(gate)
        r.emit("]")

        _subtract_double_preserved(r, next_left, total, preserve_tmp)

        # Consume the incoming doubling carry as the final -dc term.
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        r.add(total, -1)
        r.move(MARKER)
        r.emit("]")

        _map_total_base16(r, total, DATA + i, next_sub_carry)
        r.transfer(next_sub_carry, sub_carry)
        r.transfer(next_double_carry, MARKER)

    # Both final carries represent fixed-width overflow/no-borrow and are dead.
    for cell in (
        MARKER,
        sub_carry,
        next_sub_carry,
        next_double_carry,
        preserve_tmp,
        restore,
        gate,
    ):
        r.clear(cell)


__all__ = ["total_minus_double_next_left_into_data"]
