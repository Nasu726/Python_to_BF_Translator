"""Low-overhead radix-16 helpers for bounded hexadecimal lane arithmetic.

The established mapper opens one guarded level for every possible total up to
31.  Once the sixteenth unit is consumed, however, the carry is known to be
one and the remaining value is already the low hexadecimal digit (0..15).
This helper therefore needs only sixteen guarded levels and transfers the
residual directly into the output cell.
"""

from __future__ import annotations

from functools import lru_cache

from bfhexseq import (
    DATA,
    HEX_DIGITS,
    LEFT,
    TOTAL,
    _RelativeBuilder,
    _add_preserved_small,
)


def map_total_base16_threshold(
    r: _RelativeBuilder,
    total: int,
    out: int,
    carry: int,
) -> None:
    """Consume a value in 0..31 into ``out=value%16`` and ``carry=value//16``."""
    r.clear(out)
    r.clear(carry)

    for step in range(1, 17):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step == 16:
            # The high radix bit is now fixed.  The residual total is in
            # 0..15 by contract, so move it directly instead of opening a
            # further fifteen nested guards.
            r.clear(out)
            r.add(carry, 1)
            r.transfer(total, out)
        else:
            r.add(out, 1)

    for _ in range(16):
        r.move(total)
        r.emit("]")


@lru_cache(maxsize=1)
def add_data_to_total_kernel() -> str:
    """TOTAL += DATA modulo 2**64, preserving DATA, with threshold radix maps."""
    r = _RelativeBuilder()
    total_tmp = LEFT
    copy_tmp = LEFT + 1
    carry = LEFT + 2
    next_carry = LEFT + 3
    r.clear(carry)

    for i in range(HEX_DIGITS):
        a = TOTAL + i
        b = DATA + i
        r.clear(total_tmp)
        r.clear(next_carry)
        r.transfer(carry, total_tmp)
        r.transfer(a, total_tmp)
        _add_preserved_small(r, b, total_tmp, copy_tmp)
        map_total_base16_threshold(r, total_tmp, a, next_carry)
        r.transfer(next_carry, carry)

    for cell in (total_tmp, copy_tmp, carry, next_carry):
        r.clear(cell)
    r.move(0)
    return r.code()


__all__ = ["add_data_to_total_kernel", "map_total_base16_threshold"]
