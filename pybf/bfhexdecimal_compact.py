"""Source-compact direct decimal accumulation into hexadecimal int64 lanes.

This keeps the runtime algorithm from ``bfhexdecimal`` but specializes the
radix-16 mapper to the actual bounds of each operation.  Decimal accumulation
maps at most 23 (= 14 remainder + 9 carry), while two's-complement negation maps
at most 16.  The generic runtime helper is deliberately sized for 0..31 and
therefore emits unnecessary nested guards for these narrower domains.
"""

from __future__ import annotations

from functools import lru_cache

from bfhexseq import ACC_BASE, CH, DATA, HEX_DIGITS, _RelativeBuilder


HEX_TOTAL = ACC_BASE
HEX_CARRY_A = ACC_BASE + 1
HEX_CARRY_B = ACC_BASE + 2
HEX_EXTRA = ACC_BASE + 3


def _map_total_base16_bounded(
    r: _RelativeBuilder,
    total: int,
    out: int,
    carry: int,
    *,
    max_total: int,
) -> None:
    """Consume 0..max_total into a hex digit and 0/1 carry.

    ``max_total`` must stay below 32 so one carry bit is sufficient.
    """
    if not 0 <= max_total <= 31:
        raise ValueError("bounded base16 mapper requires max_total <= 31")
    r.clear(out)
    r.clear(carry)
    for step in range(1, max_total + 1):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step == 16:
            r.clear(out)
            r.add(carry, 1)
        else:
            r.add(out, 1)
    for _ in range(max_total):
        r.move(total)
        r.emit("]")


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
        # (10*x)%16 <= 14 and the incoming decimal radix carry <= 9.
        _map_total_base16_bounded(
            r,
            HEX_TOTAL,
            DATA + i,
            HEX_EXTRA,
            max_total=23,
        )
        r.transfer(HEX_EXTRA, carry_out)
        carry_in, carry_out = carry_out, carry_in

    for cell in (HEX_TOTAL, HEX_CARRY_A, HEX_CARRY_B, HEX_EXTRA):
        r.clear(cell)
    r.move(0)
    return r.code()


@lru_cache(maxsize=1)
def decimal_digit_kernel() -> str:
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
        # 15-x + carry <= 16.
        _map_total_base16_bounded(
            r,
            HEX_TOTAL,
            DATA + i,
            HEX_CARRY_A,
            max_total=16,
        )

    for cell in (HEX_TOTAL, HEX_CARRY_A, HEX_CARRY_B, HEX_EXTRA):
        r.clear(cell)
    r.move(0)
    return r.code()


@lru_cache(maxsize=1)
def negate_data_kernel() -> str:
    return _negate_data_body()


__all__ = ["decimal_digit_kernel", "negate_data_kernel"]
