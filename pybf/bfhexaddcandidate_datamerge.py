"""Candidate kernel that forms LEFT+DATA by moving DATA into LEFT.

The canonical add-candidate kernel moves the old LEFT nibble into DATA before
consuming the combined 0..31 sum.  That makes the preliminary merge cost
proportional to the prefix-state nibble.  This variant reverses only that step:
DATA is destructively transferred into the old LEFT cell, so the preliminary
merge cost is proportional to the input nibble instead.  The resulting sum and
all carry semantics are otherwise identical.
"""

from __future__ import annotations

from bfhexaddcandidate import (
    _apply_combined_incoming_carries,
    _consume_sum_into_next_left_and_candidate,
)
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


def add_data_and_move_state_total_minus_double_left_into_total(
    r: _RelativeBuilder,
) -> None:
    """Move DATA into LEFT before the fused prefix/candidate update.

    On return the contract matches ``bfhexaddcandidate`` exactly: next.TOTAL is
    the full sum, next.LEFT is the updated prefix, and current TOTAL is the raw
    signed candidate ``TOTAL - 2*next.LEFT`` modulo 2**64.
    """
    # LEFT[0] becomes free after the first combined sum is consumed and then
    # carries radix subtraction state across all later nibbles.
    subtraction_carry = LEFT
    r.add(MARKER, -1)

    for i in range(HEX_DIGITS):
        data = DATA + i
        combined_sum = LEFT + i
        candidate_acc = DATA + i
        src_total = TOTAL + i
        next_total = RECORD_STRIDE + TOTAL + i
        next_left = RECORD_STRIDE + LEFT + i
        candidate_out = TOTAL + i

        # Reverse the canonical merge direction. transfer() leaves DATA zero,
        # so that same cell is immediately reusable as candidate accumulator.
        r.transfer(data, combined_sum)

        r.clear(next_total)
        r.move(src_total)
        r.emit("[")
        r.add(src_total, -1)
        r.add(next_total, 1)
        r.add(candidate_acc, 1)
        r.move(src_total)
        r.emit("]")
        r.add(candidate_acc, 15)

        if i == 0:
            r.add(candidate_acc, 1)
        else:
            r.transfer(subtraction_carry, candidate_acc)
            _apply_combined_incoming_carries(
                r,
                x=combined_sum,
                candidate_acc=candidate_acc,
            )

        _consume_sum_into_next_left_and_candidate(
            r,
            x=combined_sum,
            next_left=next_left,
            candidate_acc=candidate_acc,
        )

        # combined_sum is now zero. For nibble zero this is LEFT[0], exactly
        # the persistent subtraction-carry cell needed by later nibbles.
        map_total_base16_threshold(
            r,
            candidate_acc,
            candidate_out,
            subtraction_carry,
        )

    r.clear(MARKER)
    r.clear(subtraction_carry)


__all__ = ["add_data_and_move_state_total_minus_double_left_into_total"]
