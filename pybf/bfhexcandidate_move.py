"""Destructive fused candidate/state transport for the hex partition runtime.

After ``LEFT += DATA`` the current record owns the live TOTAL and LEFT words,
while the following record has empty TOTAL/LEFT lanes.  This kernel consumes
those current words once, reconstructs them in the following record, and at the
same time computes::

    DATA = TOTAL - 2 * LEFT   (mod 2**64)

The earlier fused experiment first moved state right and then had to preserve
``next.LEFT``/``next.TOTAL`` while reading them back.  Those save/restore loops
made runtime slower despite reducing source size.  Here transport itself is the
read, so no full-nibble preserved copy is needed.
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


def _consume_left_nibble_into_next_and_acc(
    r: _RelativeBuilder,
    *,
    src: int,
    dst: int,
    acc: int,
) -> None:
    """Consume one 0..15 LEFT nibble, rebuilding dst and subtracting 2*value.

    Fifteen nested tests consume at most the original nibble value once.  At
    the eighth consumed unit the doubled value crosses radix 16, so the
    compensation +16 is applied and MARKER becomes the carry for the next
    nibble.  The +16 happens before that iteration's -2 to keep acc nonnegative.
    """
    r.clear(dst)
    for step in range(1, 16):
        r.move(src)
        r.emit("[")
        r.add(src, -1)
        r.add(dst, 1)
        if step == 8:
            r.add(acc, 16)
            r.set_const(MARKER, 1)
        r.add(acc, -2)
    for _ in range(15):
        r.move(src)
        r.emit("]")


def move_state_and_total_minus_double_left_into_data(r: _RelativeBuilder) -> None:
    """Move TOTAL/LEFT right while computing TOTAL-2*LEFT into current DATA.

    ``MARKER`` carries the high bit of the previous doubled LEFT nibble.  The
    subtraction carry is kept in current TOTAL[0] between nibble iterations;
    current LEFT[0] is a temporary carry output after nibble zero has itself
    been consumed.  Thus no extra record cells are required.
    """
    sub_carry = TOTAL
    carry_tmp = LEFT
    r.clear(MARKER)  # incoming carry of 2*LEFT for nibble zero

    for i in range(HEX_DIGITS):
        src_total = TOTAL + i
        src_left = LEFT + i
        next_total = RECORD_STRIDE + TOTAL + i
        next_left = RECORD_STRIDE + LEFT + i
        acc = DATA + i
        mapped = TOTAL + i

        # Consume t once: rebuild next.TOTAL and accumulate t locally.
        r.clear(acc)
        r.clear(next_total)
        r.move(src_total)
        r.emit("[")
        r.add(src_total, -1)
        r.add(next_total, 1)
        r.add(acc, 1)
        r.move(src_total)
        r.emit("]")

        # Radix-complement subtraction starts with +15+carry.  The first carry
        # is one; later carries live in TOTAL[0] and are consumed here.
        r.add(acc, 15)
        if i == 0:
            r.add(acc, 1)
        else:
            r.transfer(sub_carry, acc)

        # Subtract the incoming carry from doubling the previous LEFT nibble.
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        r.add(acc, -1)
        r.move(MARKER)
        r.emit("]")

        # Consume l once, reconstructing next.LEFT.  This also sets MARKER to
        # the outgoing doubling carry when l >= 8.
        _consume_left_nibble_into_next_and_acc(
            r,
            src=src_left,
            dst=next_left,
            acc=acc,
        )

        # acc is guaranteed in 0..31. Map it once to the candidate nibble.
        # src_total is now zero and can hold the mapped digit; LEFT[0] is zero
        # once nibble zero has been consumed and serves as the carry temporary.
        map_total_base16_threshold(r, acc, mapped, carry_tmp)
        r.transfer(mapped, DATA + i)
        r.transfer(carry_tmp, sub_carry)

    # Final fixed-width carries are overflow/no-borrow state and are dead.
    r.clear(MARKER)
    r.clear(sub_carry)
    r.clear(carry_tmp)


__all__ = ["move_state_and_total_minus_double_left_into_data"]
