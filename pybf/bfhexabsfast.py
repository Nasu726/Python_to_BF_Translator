"""Fast in-place absolute value for the partition hex candidate.

For a negative fixed-width candidate, two's-complement negation per nibble is

    out = 15 - x + carry  (mod 16)

and the carry survives exactly when ``x == 0`` and the incoming carry is one.
The established implementation routed this 0..16 total through the generic
radix-16 mapper.  This kernel instead remembers whether the consumed nibble was
zero and handles the sole overflow case directly.
"""

from __future__ import annotations

from bfhexpartition import _nibble_ge8
from bfhexseq import DATA, HEX_DIGITS, LEFT, MARKER, TOTAL, _RelativeBuilder


def abs_data_inplace_fast(r: _RelativeBuilder) -> None:
    """Two's-complement absolute value of DATA with direct carry handling."""
    sign = LEFT
    tmp = LEFT + 1
    restore = LEFT + 2
    gate_zero = LEFT + 3
    next_carry = LEFT + 4

    _nibble_ge8(r, sign, DATA + HEX_DIGITS - 1, tmp, restore)

    r.move(sign)
    r.emit("[")
    r.add(sign, -1)
    r.set_const(MARKER, 1)

    for i in range(HEX_DIGITS):
        src = DATA + i
        out = TOTAL + i
        r.set_const(out, 15)
        r.set_const(gate_zero, 1)
        r.clear(next_carry)

        # Consume x while forming 15-x. gate_zero remains one only for x==0;
        # for nonzero x its one-shot loop is consumed on the first iteration.
        r.move(src)
        r.emit("[")
        r.add(src, -1)
        r.add(out, -1)
        r.move(gate_zero)
        r.emit("[")
        r.add(gate_zero, -1)
        r.move(gate_zero)
        r.emit("]")
        r.move(src)
        r.emit("]")

        # Incoming carry adds one. Only x==0 overflows 15+1 to digit zero and
        # emits the next carry. next_carry is separate from MARKER so the outer
        # carry loop cannot be re-entered after setting the outgoing carry.
        r.move(MARKER)
        r.emit("[")
        r.add(MARKER, -1)
        r.add(out, 1)
        r.move(gate_zero)
        r.emit("[")
        r.add(gate_zero, -1)
        r.clear(out)
        r.set_const(next_carry, 1)
        r.move(gate_zero)
        r.emit("]")
        r.move(MARKER)
        r.emit("]")

        r.clear(gate_zero)
        r.transfer(next_carry, MARKER)
        r.transfer(out, src)

    r.clear(MARKER)
    r.move(sign)
    r.emit("]")

    for cell in (sign, tmp, restore, gate_zero, next_carry):
        r.clear(cell)


__all__ = ["abs_data_inplace_fast"]
