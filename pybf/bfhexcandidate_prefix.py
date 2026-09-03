"""Build TOTAL-2*prefix from prefix sums stored directly in DATA.

The prefix-retaining reader leaves each record's inclusive prefix sum in DATA.
After the final sum is propagated back to record zero, the partition pass only
needs to carry the full TOTAL right and subtract twice the local stored prefix.
No runtime LEFT update or LEFT transport is required.
"""

from __future__ import annotations

from bfhexradixfast import map_total_base16_threshold
from bfhexseq import (
    DATA,
    HEX_DIGITS,
    LEFT,
    MARKER,
    RECORD_STRIDE,
    TOTAL,
    _RelativeBuilder,
)


def _consume_prefix_double(r: _RelativeBuilder, *, src: int, acc: int) -> None:
    """Consume one 0..15 prefix nibble while subtracting twice its value."""
    for step in range(1, 16):
        r.move(src)
        r.emit("[")
        r.add(src, -1)
        if step == 8:
            # Crossing 16 in 2*src: compensate the current radix and carry the
            # high bit into the next nibble through MARKER.
            r.add(acc, 16)
            r.set_const(MARKER, 1)
        r.add(acc, -2)
    for _ in range(15):
        r.move(src)
        r.emit("]")


def move_total_minus_double_prefix_into_total(r: _RelativeBuilder) -> None:
    """Carry full TOTAL right and leave the raw candidate in current TOTAL.

    Preconditions:
    - current marker is one;
    - current DATA contains the inclusive prefix sum;
    - current TOTAL contains the full array total;
    - next.TOTAL is zero.

    Postconditions:
    - next.TOTAL contains the unchanged full array total;
    - current TOTAL contains ``full_total - 2*prefix`` modulo 2**64;
    - current DATA is consumed;
    - LEFT remains scratch/dead state.
    """
    subtraction_carry = DATA  # DATA[0] is free after prefix nibble zero is consumed.
    r.add(MARKER, -1)

    for i in range(HEX_DIGITS):
        prefix = DATA + i
        src_total = TOTAL + i
        next_total = RECORD_STRIDE + TOTAL + i
        acc = LEFT + i
        candidate = TOTAL + i

        r.clear(acc)
        r.clear(next_total)
        r.move(src_total)
        r.emit("[")
        r.add(src_total, -1)
        r.add(next_total, 1)
        r.add(acc, 1)
        r.move(src_total)
        r.emit("]")

        # Radix-complement subtraction of 2*prefix.
        r.add(acc, 15)
        if i == 0:
            r.add(acc, 1)
        else:
            r.transfer(subtraction_carry, acc)

        # Subtract the incoming high bit of twice the previous prefix nibble.
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        r.add(acc, -1)
        r.move(MARKER)
        r.emit("]")

        _consume_prefix_double(r, src=prefix, acc=acc)

        # acc is bounded to 0..31. Current TOTAL[i] is zero after transport and
        # can hold the candidate digit; DATA[0] stores subtraction carry.
        map_total_base16_threshold(r, acc, candidate, subtraction_carry)

    r.clear(MARKER)
    r.clear(subtraction_carry)


__all__ = ["move_total_minus_double_prefix_into_total"]
