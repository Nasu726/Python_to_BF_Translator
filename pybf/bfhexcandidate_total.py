"""Fused candidate/state transport that leaves the candidate in TOTAL scratch.

The public move-fused kernel historically maps each candidate nibble into the
now-dead current TOTAL cell and then immediately transfers it back to DATA.
The minimum pass later consumes DATA and saves that same candidate into TOTAL
again.  This experimental kernel removes the first round trip: after live TOTAL
has moved to the next record, current TOTAL becomes the candidate word.
"""

from __future__ import annotations

from bfhexcandidate_move import _consume_left_nibble_into_next_and_acc
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


def move_state_and_total_minus_double_left_into_total(
    r: _RelativeBuilder,
) -> None:
    """Move live TOTAL/LEFT right and leave abs-input candidate in current TOTAL.

    Preconditions match the established move-fused kernel: DATA is zero after
    ``LEFT += DATA`` and current TOTAL/LEFT hold the live state.  On return,
    next.TOTAL/next.LEFT hold that state and current TOTAL equals
    ``TOTAL - 2*LEFT`` modulo 2**64.
    """
    # DATA[0] is free after the consumed input value and carries subtraction
    # radix state between nibbles. LEFT[0] becomes free after nibble zero and is
    # the mapper's temporary carry output.
    sub_carry = DATA
    carry_tmp = LEFT
    r.clear(MARKER)  # incoming doubling carry for nibble zero
    r.clear(sub_carry)

    for i in range(HEX_DIGITS):
        src_total = TOTAL + i
        src_left = LEFT + i
        next_total = RECORD_STRIDE + TOTAL + i
        next_left = RECORD_STRIDE + LEFT + i
        acc = DATA + i
        mapped = TOTAL + i

        r.clear(acc)
        r.clear(next_total)
        r.move(src_total)
        r.emit("[")
        r.add(src_total, -1)
        r.add(next_total, 1)
        r.add(acc, 1)
        r.move(src_total)
        r.emit("]")

        # Radix-complement subtraction t + 15 + subtraction-carry.
        r.add(acc, 15)
        if i == 0:
            r.add(acc, 1)
        else:
            r.transfer(sub_carry, acc)

        # Incoming carry from doubling the previous LEFT nibble.
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        r.add(acc, -1)
        r.move(MARKER)
        r.emit("]")

        _consume_left_nibble_into_next_and_acc(
            r,
            src=src_left,
            dst=next_left,
            acc=acc,
        )

        # src_total is now dead. Keep the mapped candidate there instead of
        # transferring it back into DATA only to save it again in the min pass.
        map_total_base16_threshold(r, acc, mapped, carry_tmp)
        r.transfer(carry_tmp, sub_carry)

    r.clear(MARKER)
    r.clear(sub_carry)
    r.clear(carry_tmp)


__all__ = ["move_state_and_total_minus_double_left_into_total"]
