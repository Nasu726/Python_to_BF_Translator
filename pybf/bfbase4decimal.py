"""Decimal accumulation helpers for the experimental radix-4 int64 backend.

This module is separate from ``bfbase4`` so decimal-I/O experiments cannot
silently destabilize the already-tested arithmetic core.

For one decimal input digit ``d`` it computes::

    value = value * 10 + d  (mod 2**64)

as two fixed 32-lane passes:

1. ``scratch = 2 * value``;
2. ``value = scratch + 4*scratch + d``.

The decimal digit is seeded as the radix carry of lane zero, so no third full
word-add pass is required.  All per-lane totals are bounded by 12; there is no
loop proportional to an arbitrary byte value.
"""

from __future__ import annotations

from bfbase4 import (
    DIGITS,
    STRIDE,
    Base4I64Core,
    Base4I64Ref,
    _RelativeBuilder,
    _add_preserved,
)
from bfcore import BFEmitter


def _map_total_wide(
    r: _RelativeBuilder,
    total: int,
    out: int,
    carry: int,
    *,
    max_total: int,
) -> None:
    """Map bounded total to radix-4 digit/carry, consuming total."""
    if not 0 <= max_total <= 15:
        raise ValueError("wide radix-4 mapper is intended only for tiny totals")

    r.clear(out)
    r.clear(carry)
    for step in range(1, max_total + 1):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step % 4 == 0:
            r.clear(out)
            r.add(carry, 1)
        else:
            r.add(out, 1)
    for _ in range(max_total):
        r.move(total)
        r.emit("]")


class Base4DecimalCore:
    """Source-compact unsigned decimal accumulation into radix-4 words."""

    def __init__(self, bf: BFEmitter) -> None:
        self.bf = bf
        self.base4 = Base4I64Core(bf)

    def _move_digit_to_initial_carry(
        self,
        digit_cell: int,
        dst: Base4I64Ref,
    ) -> None:
        """Consume a numeric digit 0..9 into dst.marker(0)."""
        bf = self.bf
        carry = dst.marker(0)
        bf.clear(carry)
        bf.move(digit_cell)
        bf.emit("[")
        bf.add_const(digit_cell, -1)
        bf.add_const(carry, 1)
        bf.move(digit_cell)
        bf.emit("]")

    def mul10_add_digit_inplace(
        self,
        dst: Base4I64Ref,
        scratch: Base4I64Ref,
        digit_cell: int,
    ) -> None:
        """Consume digit_cell (0..9) and perform dst = dst*10 + digit."""
        if dst.base == scratch.base:
            raise ValueError("decimal accumulation requires a distinct scratch word")
        if digit_cell in range(dst.base, dst.base + dst.cells):
            raise ValueError("digit cell must not alias destination word")
        if digit_cell in range(scratch.base, scratch.base + scratch.cells):
            raise ValueError("digit cell must not alias scratch word")

        bf = self.bf
        self.base4.double64(scratch, dst)
        self._move_digit_to_initial_carry(digit_cell, dst)

        delta = dst.base - scratch.base

        # Lane zero: 2*x low digit + decimal digit.  The total is <= 12.
        r0 = _RelativeBuilder()
        marker = 0
        current = 1
        carry_in = delta
        out = delta + 1
        carry_out = delta + STRIDE

        r0.clear(marker)
        r0.clear(out)
        r0.clear(carry_out)
        r0.transfer(carry_in, marker)
        _add_preserved(r0, current, marker, out)
        _map_total_wide(r0, marker, out, carry_out, max_total=12)
        r0.move(STRIDE)

        bf.move(scratch.marker(0))
        bf.emit(r0.code())
        bf.ptr = scratch.marker(1)

        # Remaining lanes add the current and one-digit-shifted doubled value
        # plus carry. Maximum total is 3 + 3 + 3 = 9; using the same 12-step
        # mapper keeps one simple, audited body.
        for digit in range(1, DIGITS):
            bf.set_const(scratch.marker(digit), 1)
        bf.clear(scratch.marker(DIGITS))

        r = _RelativeBuilder()
        marker = 0
        current = 1
        previous = -STRIDE + 1
        carry_in = delta
        out = delta + 1
        carry_out = delta + STRIDE

        r.clear(marker)
        r.clear(out)
        r.clear(carry_out)
        r.transfer(carry_in, marker)
        _add_preserved(r, current, marker, out)
        _add_preserved(r, previous, marker, out)
        _map_total_wide(r, marker, out, carry_out, max_total=12)
        r.move(STRIDE)

        bf.move(scratch.marker(1))
        bf.emit("[" + r.code() + "]")
        bf.ptr = scratch.marker(DIGITS)

        # Modulo-2**64 overflow is intentionally discarded.
        bf.clear(dst.marker(DIGITS))
        bf.clear(dst.value(DIGITS))


__all__ = ["Base4DecimalCore"]
