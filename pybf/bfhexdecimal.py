"""Direct decimal accumulation into the runtime hexadecimal int64 lanes.

The original hex-sequence reader parses every decimal digit through a 32-digit
radix-4 accumulator and packs that word into sixteen hex nibbles afterwards.
That path is source-compact but expensive at runtime.  This module performs

    value = value * 10 + digit  (mod 2**64)

directly on the sixteen little-endian hex nibbles.

For one old nibble x and incoming radix carry c (0..9)::

    10*x = 16*q + r,  q <= 9, r < 16
    10*x + c = 16*(q + extra) + out

Because x is only 0..15, a bounded nested walk consumes it and incrementally
maintains r/q.  Then the existing <=31 radix-16 mapper adds the small incoming
carry.  No loop is proportional to an arbitrary byte or int64 magnitude.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexseq import (
    ACC_BASE,
    CH,
    DATA,
    HEX_DIGITS,
    _RelativeBuilder,
    _map_total_base16,
)


HEX_TOTAL = ACC_BASE
HEX_CARRY_A = ACC_BASE + 1
HEX_CARRY_B = ACC_BASE + 2
HEX_EXTRA = ACC_BASE + 3


def _consume_nibble_times10(
    r: _RelativeBuilder,
    src: int,
    total: int,
    quotient: int,
) -> None:
    """Consume src=x and leave total=(10*x)%16, quotient=floor(10*x/16)."""
    r.clear(total)
    r.clear(quotient)
    remainder = 0
    for _step in range(1, 16):
        r.move(src)
        r.emit("[")
        r.add(src, -1)
        if remainder < 6:
            r.add(total, 10)
            remainder += 10
        else:
            r.add(total, -6)
            r.add(quotient, 1)
            remainder -= 6
    for _ in range(15):
        r.move(src)
        r.emit("]")


def _decimal_digit_body() -> str:
    r = _RelativeBuilder()
    r.clear(HEX_CARRY_A)
    r.clear(HEX_CARRY_B)
    r.clear(HEX_TOTAL)
    r.clear(HEX_EXTRA)
    r.transfer(CH, HEX_CARRY_A)

    carry_in = HEX_CARRY_A
    carry_out = HEX_CARRY_B
    for i in range(HEX_DIGITS):
        _consume_nibble_times10(r, DATA + i, HEX_TOTAL, carry_out)
        r.transfer(carry_in, HEX_TOTAL)
        _map_total_base16(r, HEX_TOTAL, DATA + i, HEX_EXTRA)
        r.transfer(HEX_EXTRA, carry_out)
        carry_in, carry_out = carry_out, carry_in

    for cell in (HEX_TOTAL, HEX_CARRY_A, HEX_CARRY_B, HEX_EXTRA):
        r.clear(cell)
    r.move(0)
    return r.code()


@lru_cache(maxsize=1)
def decimal_digit_kernel() -> str:
    """Consume numeric CH (0..9) and update DATA as unsigned decimal int64."""
    return _decimal_digit_body()


def _negate_data_body() -> str:
    r = _RelativeBuilder()
    r.set_const(HEX_CARRY_A, 1)
    r.clear(HEX_TOTAL)

    for i in range(HEX_DIGITS):
        r.set_const(HEX_TOTAL, 15)
        r.move(DATA + i)
        r.emit("[")
        r.add(DATA + i, -1)
        r.add(HEX_TOTAL, -1)
        r.move(DATA + i)
        r.emit("]")
        r.transfer(HEX_CARRY_A, HEX_TOTAL)
        _map_total_base16(r, HEX_TOTAL, DATA + i, HEX_CARRY_A)

    for cell in (HEX_TOTAL, HEX_CARRY_A, HEX_CARRY_B, HEX_EXTRA):
        r.clear(cell)
    r.move(0)
    return r.code()


@lru_cache(maxsize=1)
def negate_data_kernel() -> str:
    """Two's-complement negate DATA in place modulo 2**64."""
    return _negate_data_body()


def emit_decimal_digit(bf: BFEmitter) -> None:
    """Emit one direct decimal accumulation step from numeric CH into DATA."""
    bf.move(0)
    bf.emit(decimal_digit_kernel())
    bf.ptr = 0


__all__ = [
    "decimal_digit_kernel",
    "emit_decimal_digit",
    "negate_data_kernel",
]
