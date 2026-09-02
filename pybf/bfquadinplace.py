"""In-place Quad arithmetic helpers used by the compact compiler path.

These helpers are deliberately separate from the correctness-first Quad core:
they exploit the compiler's two reserved Quad workspace words to eliminate
otherwise unnecessary result words and full-word copies in hot scalar paths.
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


def _workspace_word(backend, *refs: Quad64Ref) -> Quad64Ref | None:
    """Return one reserved Quad word that does not alias any operand."""
    occupied = {ref.base for ref in refs}
    for index in (0, 1):
        candidate = backend._qtmp(index)
        if candidate.base not in occupied:
            return candidate
    return None


def add64_inplace(backend, dst: Quad64Ref, rhs: Quad64Ref) -> bool:
    """Compute ``dst += rhs`` modulo 2**64 with one runtime lane body."""
    if dst.base == rhs.base:
        return False

    carry_word = _workspace_word(backend, dst, rhs)
    if carry_word is None:
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

    r.clear(total)
    r.clear(inner_carry)
    r.transfer(carry_in, total)
    r.transfer(a0, total)
    _add_preserved(r, b0, total, marker)
    _map_total(r, total, a0, inner_carry)

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
    bf.clear(carry_word.marker(DIGITS))
    return True


def inc64_inplace(backend, dst: Quad64Ref) -> bool:
    """Compute ``dst += 1`` with one repeated two-bit-lane body.

    This is substantially smaller than materializing a Quad constant one,
    performing an out-of-place add, and copying the result back.  The reserved
    workspace word carries only the ripple carry; no user-visible value lives
    there.
    """
    carry_word = _workspace_word(backend, dst)
    if carry_word is None:
        return False

    bf = backend.bf
    for digit in range(DIGITS):
        bf.set_const(dst.marker(digit), 1)
    bf.clear(dst.marker(DIGITS))
    bf.set_const(carry_word.marker(0), 1)

    r = _RelativeBuilder()
    marker = 0
    a0, a1 = 1, 2
    t_delta = carry_word.base - dst.base
    carry_in = t_delta
    total = t_delta + 1
    inner_carry = t_delta + 2
    carry_out = t_delta + STRIDE

    r.clear(marker)

    r.clear(total)
    r.clear(inner_carry)
    r.transfer(carry_in, total)
    r.transfer(a0, total)
    _map_total(r, total, a0, inner_carry)

    r.clear(total)
    r.transfer(inner_carry, total)
    r.transfer(a1, total)
    r.clear(carry_out)
    _map_total(r, total, a1, carry_out)

    r.move(STRIDE)

    bf.move(dst.marker(0))
    bf.emit("[" + r.code() + "]")
    bf.ptr = dst.marker(DIGITS)
    # Fixed-width overflow is discarded.
    bf.clear(carry_word.marker(DIGITS))
    return True


def _toggle_local(r: _RelativeBuilder, bit: int, helper: int) -> None:
    """Toggle one Boolean lane bit using a lane-local helper cell."""
    r.clear(helper)
    r.add(helper, 1)
    r.move(bit)
    r.emit("[")
    r.emit("-")
    r.add(helper, -1)
    r.move(bit)
    r.emit("]")
    r.move(helper)
    r.emit("[")
    r.emit("-")
    r.add(bit, 1)
    r.move(helper)
    r.emit("]")


def neg64_inplace(backend, dst: Quad64Ref) -> bool:
    """Two's-complement negate ``dst`` using a compact lane walk plus +1."""
    bf = backend.bf
    for digit in range(DIGITS):
        bf.set_const(dst.marker(digit), 1)
    bf.clear(dst.marker(DIGITS))

    r = _RelativeBuilder()
    marker = 0
    _toggle_local(r, 1, marker)
    _toggle_local(r, 2, marker)
    r.move(STRIDE)

    bf.move(dst.marker(0))
    bf.emit("[" + r.code() + "]")
    bf.ptr = dst.marker(DIGITS)
    return inc64_inplace(backend, dst)


__all__ = ["add64_inplace", "inc64_inplace", "neg64_inplace"]
