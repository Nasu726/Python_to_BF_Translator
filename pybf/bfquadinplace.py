"""In-place Quad arithmetic helpers used by the compact compiler path.

These helpers are deliberately separate from the correctness-first Quad core:
they exploit the compiler's two reserved Quad workspace words to eliminate an
otherwise unnecessary result word plus full-word copy for augmented assignment.
"""

from __future__ import annotations

from bfquad import (
    DIGITS,
    STRIDE,
    Quad64Ref,
    _RelativeBuilder,
    _add_preserved,
    _map_total,
)


def add64_inplace(backend, dst: Quad64Ref, rhs: Quad64Ref) -> bool:
    """Compute ``dst += rhs`` modulo 2**64 with one runtime lane body.

    Returns False when the two operands alias; callers can then use the normal
    out-of-place path.  A reserved Quad workspace word carries the ripple carry
    in its marker cells, so no full result word or trailing copy is emitted.
    """
    if dst.base == rhs.base:
        return False

    q0 = backend._qtmp(0)
    q1 = backend._qtmp(1)
    if q0.base not in (dst.base, rhs.base):
        carry_word = q0
    elif q1.base not in (dst.base, rhs.base):
        carry_word = q1
    else:
        return False

    bf = backend.bf
    for digit in range(DIGITS):
        bf.set_const(dst.marker(digit), 1)
    bf.clear(dst.marker(DIGITS))
    bf.clear(carry_word.marker(0))

    r = _RelativeBuilder()
    marker = 0
    a0, a1 = 1, 2
    b_delta = rhs.base - dst.base
    b0, b1 = b_delta + 1, b_delta + 2
    t_delta = carry_word.base - dst.base
    carry_in = t_delta
    total = t_delta + 1
    inner_carry = t_delta + 2
    carry_out = t_delta + STRIDE

    # The traversal marker is consumed first and is then free restoration
    # scratch for preserved RHS bits.
    r.clear(marker)

    # Low bit of the two-bit lane.
    r.clear(total)
    r.clear(inner_carry)
    r.transfer(carry_in, total)
    r.transfer(a0, total)
    _add_preserved(r, b0, total, marker)
    _map_total(r, total, a0, inner_carry)

    # High bit; its carry becomes the next lane's carry marker.
    r.clear(total)
    r.transfer(inner_carry, total)
    r.transfer(a1, total)
    _add_preserved(r, b1, total, marker)
    r.clear(carry_out)
    _map_total(r, total, a1, carry_out)

    r.move(STRIDE)

    bf.move(dst.marker(0))
    bf.emit("[" + r.code() + "]")
    bf.ptr = dst.marker(DIGITS)
    # Overflow beyond bit 63 is discarded by the fixed-width ABI.
    bf.clear(carry_word.marker(DIGITS))
    return True


__all__ = ["add64_inplace"]
