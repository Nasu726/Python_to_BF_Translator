"""Fuse LEFT += DATA with partition candidate/state transport.

For each hexadecimal digit the old implementation first materializes
``LEFT = LEFT + DATA`` and a second pass immediately consumes that LEFT again
to form ``TOTAL - 2*LEFT``.  This kernel combines both operations.

The outgoing addition carry and doubling carry are encoded together in MARKER:
0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).  The subtraction carry lives in DATA[0]
after nibble zero has been consumed.  No extra record cells are required.
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


def _apply_combined_incoming_carries(
    r: _RelativeBuilder,
    *,
    x: int,
    candidate_acc: int,
) -> None:
    """Consume MARKER's encoded (add, double) carries into this nibble.

    State transitions are arranged so a three-level bounded decoder implements
    the four states without an extra temporary cell:

    * state 1: doubling carry only -> candidate -= 1
    * state 2: addition carry only -> x += 1
    * state 3: both effects
    """
    for step in range(1, 4):
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        if step == 1:
            r.add(candidate_acc, -1)
        elif step == 2:
            # Transition from encoded state 1 to state 2: undo the doubling
            # effect and apply the addition carry.
            r.add(candidate_acc, 1)
            r.add(x, 1)
        else:
            # State 3 has both carries, so re-apply the doubling effect.
            r.add(candidate_acc, -1)
    for _ in range(3):
        r.move(MARKER)
        r.emit("]")


def _consume_sum_into_next_left_and_candidate(
    r: _RelativeBuilder,
    *,
    x: int,
    next_left: int,
    candidate_acc: int,
) -> None:
    """Consume x in 0..31, producing x%16 and candidate correction.

    Only thresholds 8, 16 and 24 change carry state.  The first sixteen units
    are decoded with bounded guards.  If a wrapped residual remains, another
    eight guards reach threshold 24, after which at most seven ordinary units
    remain and can use one tiny runtime loop.  This is equivalent to the older
    31-level decoder but emits and executes fewer nested guards.
    """
    r.clear(next_left)

    # First radix: detect the 8 doubling threshold and 16 addition wrap.
    for step in range(1, 17):
        r.move(x)
        r.emit("[")
        r.add(x, -1)
        if step == 8:
            r.add(next_left, 1)
            r.add(candidate_acc, 14)
            r.set_const(MARKER, 1)
        elif step == 16:
            r.clear(next_left)
            r.add(candidate_acc, 14)
            r.set_const(MARKER, 2)
        else:
            r.add(next_left, 1)
            r.add(candidate_acc, -2)
    for _ in range(16):
        r.move(x)
        r.emit("]")

    # Wrapped low digit: threshold eight corresponds to original total 24.
    for step in range(1, 9):
        r.move(x)
        r.emit("[")
        r.add(x, -1)
        r.add(next_left, 1)
        if step == 8:
            r.add(candidate_acc, 14)
            r.set_const(MARKER, 3)
        else:
            r.add(candidate_acc, -2)
    for _ in range(8):
        r.move(x)
        r.emit("]")

    # After threshold 24 the residual is at most seven and has no further
    # state transition, so a simple bounded-by-contract loop is sufficient.
    r.move(x)
    r.emit("[")
    r.add(x, -1)
    r.add(next_left, 1)
    r.add(candidate_acc, -2)
    r.move(x)
    r.emit("]")


def add_data_and_move_state_total_minus_double_left_into_total(
    r: _RelativeBuilder,
) -> None:
    """Consume DATA/old LEFT while moving state and producing candidate TOTAL.

    On entry the record marker is one, DATA is this element, TOTAL is the full
    sum, and LEFT is the previous prefix.  On return next.TOTAL holds the full
    sum, next.LEFT holds the updated prefix, and current TOTAL holds
    ``TOTAL - 2*next.LEFT`` modulo 2**64.  DATA/current LEFT are dead scratch.
    """
    subtraction_carry = DATA  # DATA[0], free after nibble zero is consumed

    # Consume the active record marker.  There are no incoming add/double
    # carries for the least-significant digit.
    r.add(MARKER, -1)

    for i in range(HEX_DIGITS):
        x = DATA + i
        candidate_acc = LEFT + i
        src_total = TOTAL + i
        next_total = RECORD_STRIDE + TOTAL + i
        next_left = RECORD_STRIDE + LEFT + i
        candidate_out = TOTAL + i

        # x = old LEFT digit + DATA digit.  This also frees candidate_acc.
        r.transfer(candidate_acc, x)

        # Move the live total right while building the local candidate base.
        r.clear(candidate_acc)
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
            # Radix-complement subtraction starts with carry one.
            r.add(candidate_acc, 1)
        else:
            r.transfer(subtraction_carry, candidate_acc)
            _apply_combined_incoming_carries(
                r,
                x=x,
                candidate_acc=candidate_acc,
            )

        _consume_sum_into_next_left_and_candidate(
            r,
            x=x,
            next_left=next_left,
            candidate_acc=candidate_acc,
        )

        # candidate_acc is guaranteed in 0..31 by the compensation schedule.
        # DATA[0] is free after nibble zero and carries the subtraction radix
        # bit across all later digits.
        map_total_base16_threshold(
            r,
            candidate_acc,
            candidate_out,
            subtraction_carry,
        )

    # Fixed-width overflow/carry state is dead at the 64-bit boundary.
    r.clear(MARKER)
    r.clear(subtraction_carry)


__all__ = ["add_data_and_move_state_total_minus_double_left_into_total"]
