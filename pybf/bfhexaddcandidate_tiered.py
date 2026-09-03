"""Tiered-decoder experiment for fused LEFT/candidate arithmetic.

The established add-candidate kernel uses one 31-level nested decoder for the
bounded sum ``old_left_digit + data_digit + carry``.  State changes occur only
at totals 8, 16 and 24.  This variant preserves the exact one-unit accounting
but nests three eight-unit groups and, after threshold 24, consumes the final
0..7 residual in a simple loop *inside* the deepest active guard.

Keeping the residual loop inside the active threshold-24 branch is essential:
unlike the rejected sequential split, no nonzero control value is allowed to
reach a closing bracket that would jump back into the deepest bounded guard.
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
    for step in range(1, 4):
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        if step == 1:
            r.add(candidate_acc, -1)
        elif step == 2:
            r.add(candidate_acc, 1)
            r.add(x, 1)
        else:
            r.add(candidate_acc, -1)
    for _ in range(3):
        r.move(MARKER)
        r.emit("]")


def _ordinary_sum_unit(r: _RelativeBuilder, *, next_left: int, candidate_acc: int) -> None:
    r.add(next_left, 1)
    r.add(candidate_acc, -2)


def _threshold_sum_unit(
    r: _RelativeBuilder,
    *,
    step: int,
    next_left: int,
    candidate_acc: int,
) -> None:
    if step == 8:
        r.add(next_left, 1)
        r.add(candidate_acc, 14)
        r.set_const(MARKER, 1)
    elif step == 16:
        r.clear(next_left)
        r.add(candidate_acc, 14)
        r.set_const(MARKER, 2)
    elif step == 24:
        r.add(next_left, 1)
        r.add(candidate_acc, 14)
        r.set_const(MARKER, 3)
    else:
        raise ValueError("unexpected threshold")


def _consume_sum_into_next_left_and_candidate(
    r: _RelativeBuilder,
    *,
    x: int,
    next_left: int,
    candidate_acc: int,
) -> None:
    """Consume x in 0..31 with nested 8/16/24 threshold tiers."""
    r.clear(next_left)

    def emit_group(start: int) -> None:
        # start is 1, 9 or 17; each group accounts for exactly eight units.
        for offset in range(8):
            step = start + offset
            r.move(x)
            r.emit("[")
            r.add(x, -1)
            if step in (8, 16, 24):
                _threshold_sum_unit(
                    r,
                    step=step,
                    next_left=next_left,
                    candidate_acc=candidate_acc,
                )
                if step < 24:
                    # The next bounded group is emitted *inside* this active
                    # deepest guard.  If x is already zero, its first '[' skips
                    # the whole nested group and all outer closings see zero.
                    emit_group(step + 1)
                else:
                    # After consuming unit 24 the residual is 0..7 and there
                    # are no further state thresholds.  A plain loop is safe
                    # here because it completes before the surrounding bounded
                    # guards close.
                    r.move(x)
                    r.emit("[")
                    r.add(x, -1)
                    _ordinary_sum_unit(
                        r,
                        next_left=next_left,
                        candidate_acc=candidate_acc,
                    )
                    r.move(x)
                    r.emit("]")
            else:
                _ordinary_sum_unit(
                    r,
                    next_left=next_left,
                    candidate_acc=candidate_acc,
                )
        for _ in range(8):
            r.move(x)
            r.emit("]")

    emit_group(1)


def add_data_and_move_state_total_minus_double_left_into_total(
    r: _RelativeBuilder,
) -> None:
    subtraction_carry = DATA
    r.add(MARKER, -1)

    for i in range(HEX_DIGITS):
        x = DATA + i
        candidate_acc = LEFT + i
        src_total = TOTAL + i
        next_total = RECORD_STRIDE + TOTAL + i
        next_left = RECORD_STRIDE + LEFT + i
        candidate_out = TOTAL + i

        r.transfer(candidate_acc, x)

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

        map_total_base16_threshold(
            r,
            candidate_acc,
            candidate_out,
            subtraction_carry,
        )

    r.clear(MARKER)
    r.clear(subtraction_carry)


__all__ = ["add_data_and_move_state_total_minus_double_left_into_total"]
