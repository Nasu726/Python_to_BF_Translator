"""Fused running-sum update that emits local prefix and next carried total."""

from __future__ import annotations

from functools import lru_cache

from bfhexseq import DATA, HEX_DIGITS, LEFT, RECORD_STRIDE, TOTAL, _RelativeBuilder


def _map_total_base16_dual(
    r: _RelativeBuilder,
    *,
    total: int,
    local_out: int,
    next_out: int,
    carry: int,
) -> None:
    """Consume 0..31 into two identical radix-16 digits and one carry."""
    r.clear(local_out)
    r.clear(next_out)
    r.clear(carry)

    for step in range(1, 17):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step == 16:
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


def add_data_to_prefix_and_next_total(r: _RelativeBuilder) -> None:
    """Consume current TOTAL+DATA and emit the sum to DATA and next.TOTAL.

    Current TOTAL is the running sum before this element and DATA is the parsed
    signed-int64 element (or zero for a synthesized short-line record). Both are
    consumed. The inclusive prefix sum is emitted into current DATA while an
    identical word is emitted into next.TOTAL for the following input record.

    The counted-reader ABI reserves LEFT[15] for the live count extent. A single
    accumulator cell is sufficient because each radix digit is fully consumed
    before the next digit begins, so LEFT[1] is reused for all sixteen lanes.
    """
    carry = TOTAL  # TOTAL[0] becomes free during nibble zero.
    acc = LEFT + 1

    for i in range(HEX_DIGITS):
        a = TOTAL + i
        b = DATA + i
        prefix = DATA + i
        next_total = RECORD_STRIDE + TOTAL + i

        r.clear(acc)
        if i:
            r.transfer(carry, acc)
        r.transfer(a, acc)
        r.transfer(b, acc)
        _map_total_base16_dual(
            r,
            total=acc,
            local_out=prefix,
            next_out=next_total,
            carry=carry,
        )

    r.clear(acc)
    r.clear(carry)


@lru_cache(maxsize=1)
def prefix_sum_kernel() -> str:
    r = _RelativeBuilder()
    add_data_to_prefix_and_next_total(r)
    r.move(0)
    return r.code()


__all__ = ["add_data_to_prefix_and_next_total", "prefix_sum_kernel"]
