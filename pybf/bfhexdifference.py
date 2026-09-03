"""Carry the partition difference directly instead of TOTAL and LEFT.

For the partition recurrence

    left_i = left_{i-1} + a_i
    candidate_i = total - 2*left_i

we only need the derived state

    D_0 = total
    D_i = D_{i-1} - 2*a_i  (mod 2**64).

After TOTAL has been propagated back to record zero, TOTAL can therefore become
this running D word.  Each record consumes DATA and current TOTAL once, emits
raw D_i both locally (for abs/min) and into next.TOTAL (for the next record),
and does not carry LEFT at all.
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
)


def _consume_data_double_into_acc(
    r: _RelativeBuilder,
    *,
    src: int,
    acc: int,
) -> None:
    """Consume one 0..15 DATA nibble while subtracting twice its value.

    MARKER is the outgoing high bit of ``2*src``.  At the eighth consumed unit
    the doubled nibble crosses radix 16, so +16 compensation keeps ``acc`` in
    the bounded 0..31 mapper range.
    """
    for step in range(1, 16):
        r.move(src)
        r.emit("[")
        r.add(src, -1)
        if step == 8:
            r.add(acc, 16)
            r.set_const(MARKER, 1)
        r.add(acc, -2)
    for _ in range(15):
        r.move(src)
        r.emit("]")


def _map_total_base16_dual(
    r: _RelativeBuilder,
    *,
    total: int,
    local_out: int,
    next_out: int,
    carry: int,
) -> None:
    """Consume 0..31 into two identical low nibbles plus one radix carry."""
    r.clear(local_out)
    r.clear(next_out)
    r.clear(carry)

    for step in range(1, 17):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step == 16:
            # The first fifteen units were emitted to both outputs.  At radix
            # sixteen both low digits wrap to zero; residual is 0..15 and can
            # be consumed in one inner loop before the bounded guards close.
            r.clear(local_out)
            r.clear(next_out)
            r.add(carry, 1)
            r.move(total)
            r.emit("[")
            r.add(total, -1)
            r.add(local_out, 1)
            r.add(next_out, 1)
            r.move(total)
            r.emit("]")
        else:
            r.add(local_out, 1)
            r.add(next_out, 1)

    for _ in range(16):
        r.move(total)
        r.emit("]")


def update_difference_and_duplicate_candidate(r: _RelativeBuilder) -> None:
    """Advance ``D := D - 2*DATA`` and expose the new D as local candidate.

    Preconditions:
    - current MARKER is one;
    - current TOTAL is D from before this element;
    - current DATA is the element;
    - next.TOTAL is zero.

    Postconditions:
    - current TOTAL is D after this element, for immediate abs/min use;
    - next.TOTAL is the same raw signed D, for the next record;
    - DATA and current LEFT are dead/zero scratch;
    - current MARKER is consumed; next MARKER is untouched.
    """
    sub_carry = DATA  # DATA[0], free after nibble zero is consumed.

    # Consume active-record marker; MARKER is then the doubling carry.
    r.add(MARKER, -1)

    for i in range(HEX_DIGITS):
        src_d = TOTAL + i
        src_data = DATA + i
        acc = LEFT + i
        local_d = TOTAL + i
        next_d = RECORD_STRIDE + TOTAL + i

        # Radix-complement subtraction: d + 15 + subtraction carry.
        r.clear(acc)
        r.transfer(src_d, acc)
        r.add(acc, 15)
        if i == 0:
            r.add(acc, 1)
        else:
            r.transfer(sub_carry, acc)

        # Subtract the high bit carried from doubling the previous DATA nibble.
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        r.add(acc, -1)
        r.move(MARKER)
        r.emit("]")

        # Consume this input nibble and subtract twice its value. For i==0 this
        # also frees DATA[0] before it is reused as the subtraction carry.
        _consume_data_double_into_acc(r, src=src_data, acc=acc)

        # acc is guaranteed in 0..31. Emit the new D digit twice in one map:
        # current TOTAL for abs/min, and next.TOTAL for recurrence state.
        _map_total_base16_dual(
            r,
            total=acc,
            local_out=local_d,
            next_out=next_d,
            carry=sub_carry,
        )

    # Fixed-width overflow/no-borrow and final doubling carry are dead.
    r.clear(MARKER)
    r.clear(sub_carry)


__all__ = ["update_difference_and_duplicate_candidate"]
