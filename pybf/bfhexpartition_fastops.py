"""Source-compact partition primitives using the threshold radix-16 mapper.

These helpers preserve the fixed-int64 semantics of the established partition
runtime while replacing only the generic 0..31 radix map.  They live outside
``bfhexpartition`` so the proven baseline remains available for differential
regression and telemetry.
"""

from __future__ import annotations

from bfhexradixfast import map_total_base16_threshold
from bfhexseq import (
    DATA,
    HEX_DIGITS,
    LEFT,
    MARKER,
    TOTAL,
    _RelativeBuilder,
)


def _nibble_ge8(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """Set result=(src>=8), preserving one hexadecimal digit."""
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


def consume_data_into_left(r: _RelativeBuilder) -> None:
    """LEFT += DATA modulo 2**64, consuming DATA."""
    # The active record marker is one on entry.  Reuse it as radix carry.
    r.add(MARKER, -1)
    for i in range(HEX_DIGITS):
        r.transfer(LEFT + i, DATA + i)
        r.transfer(MARKER, DATA + i)
        map_total_base16_threshold(r, DATA + i, LEFT + i, MARKER)
    r.clear(MARKER)


def abs_data_inplace(r: _RelativeBuilder) -> None:
    """Two's-complement absolute value of DATA under fixed int64 wraparound."""
    sign = LEFT
    tmp = LEFT + 1
    restore = LEFT + 2
    _nibble_ge8(r, sign, DATA + HEX_DIGITS - 1, tmp, restore)

    r.move(sign)
    r.emit("[")
    r.add(sign, -1)
    r.set_const(MARKER, 1)
    for i in range(HEX_DIGITS):
        total = TOTAL + i
        r.set_const(total, 15)
        r.move(DATA + i)
        r.emit("[")
        r.add(DATA + i, -1)
        r.add(total, -1)
        r.move(DATA + i)
        r.emit("]")
        r.transfer(MARKER, total)
        map_total_base16_threshold(r, total, DATA + i, MARKER)
    r.clear(MARKER)
    r.move(sign)
    r.emit("]")

    for cell in (sign, tmp, restore):
        r.clear(cell)


__all__ = ["abs_data_inplace", "consume_data_into_left"]
