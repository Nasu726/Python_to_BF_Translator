"""Compact base-4 lane representation for scalable int64 experiments.

``Quad64Ref`` stores every two-bit digit as two Boolean cells plus a traversal
marker.  This experimental representation stores the same radix-4 digit in one
cell (0..3) plus one marker, reducing a word from 99 cells to 66 while retaining
a runtime-walkable lane structure.

The module is intentionally isolated from the public scalar ABI until its
arithmetic/runtime trade-offs are validated.  All values remain exact unsigned
64-bit bit patterns; signed interpretation is a frontend/backend concern.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter


DIGITS = 32
STRIDE = 2
WORD_CELLS = (DIGITS + 1) * STRIDE
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Base4I64Ref:
    base: int

    def marker(self, digit: int) -> int:
        if not 0 <= digit <= DIGITS:
            raise IndexError(digit)
        return self.base + STRIDE * digit

    def value(self, digit: int) -> int:
        if not 0 <= digit <= DIGITS:
            raise IndexError(digit)
        return self.base + STRIDE * digit + 1

    @property
    def cells(self) -> int:
        return WORD_CELLS


class _RelativeBuilder:
    def __init__(self) -> None:
        self.pos = 0
        self.parts: list[str] = []

    def move(self, target: int) -> None:
        delta = target - self.pos
        if delta > 0:
            self.parts.append(">" * delta)
        elif delta < 0:
            self.parts.append("<" * -delta)
        self.pos = target

    def emit(self, code: str) -> None:
        self.parts.append(code)

    def add(self, target: int, amount: int) -> None:
        self.move(target)
        amount %= 256
        if amount <= 128:
            self.parts.append("+" * amount)
        else:
            self.parts.append("-" * (256 - amount))

    def clear(self, target: int) -> None:
        self.move(target)
        self.emit("[-]")

    def transfer(self, src: int, dst: int) -> None:
        self.move(src)
        self.emit("[")
        self.add(src, -1)
        self.add(dst, 1)
        self.move(src)
        self.emit("]")

    def code(self) -> str:
        return "".join(self.parts)


def _add_preserved(
    r: _RelativeBuilder,
    src: int,
    total: int,
    tmp: int,
) -> None:
    """``total += src`` for a radix-4 digit while restoring ``src``."""
    r.move(src)
    r.emit("[")
    r.add(src, -1)
    r.add(total, 1)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.add(tmp, -1)
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


def _add_complement_preserved(
    r: _RelativeBuilder,
    src: int,
    total: int,
    tmp: int,
) -> None:
    """``total += 3-src`` for one radix-4 digit, preserving ``src``.

    ``total`` is non-negative and receives three before ``src`` is subtracted,
    so the temporary arithmetic never relies on wrapping-byte underflow.
    """
    r.add(total, 3)
    r.move(src)
    r.emit("[")
    r.add(src, -1)
    r.add(total, -1)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.add(tmp, -1)
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


def _map_total_base4(
    r: _RelativeBuilder,
    total: int,
    out: int,
    carry: int,
) -> None:
    """Consume total in 0..7 into ``out = total % 4`` and carry 0/1.

    The seven nested BF tests are a fixed source body.  Runtime work is bounded
    by the small radix rather than by a byte value or word width.
    """
    r.clear(out)
    r.clear(carry)

    # Emit: [step1 [step2 [... [step7] ... ]]]
    for step in range(1, 8):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step == 4:
            r.clear(out)
            r.add(carry, 1)
        else:
            r.add(out, 1)
    for _ in range(7):
        r.move(total)
        r.emit("]")


class Base4I64Core:
    """Source-compact copy/add/sub/compare primitives over radix-4 int64 lanes."""

    def __init__(self, bf: BFEmitter) -> None:
        self.bf = bf

    def set_u64(self, dst: Base4I64Ref, value: int) -> None:
        value &= MASK64
        for digit in range(DIGITS):
            self.bf.clear(dst.marker(digit))
            self.bf.set_const(dst.value(digit), (value >> (2 * digit)) & 0x3)
        self.bf.clear(dst.marker(DIGITS))
        self.bf.clear(dst.value(DIGITS))

    def copy64(self, dst: Base4I64Ref, src: Base4I64Ref) -> None:
        if dst.base == src.base:
            return

        bf = self.bf
        delta = dst.base - src.base
        for digit in range(DIGITS):
            bf.set_const(src.marker(digit), 1)
        bf.clear(src.marker(DIGITS))

        r = _RelativeBuilder()
        marker = 0
        src_value = 1
        dst_marker = delta
        dst_value = delta + 1

        r.clear(marker)
        r.clear(dst_marker)
        r.clear(dst_value)
        _add_preserved(r, src_value, dst_value, marker)
        r.move(STRIDE)

        bf.move(src.marker(0))
        bf.emit("[" + r.code() + "]")
        bf.ptr = src.marker(DIGITS)
        bf.clear(dst.marker(DIGITS))
        bf.clear(dst.value(DIGITS))

    def add64(
        self,
        dst: Base4I64Ref,
        a: Base4I64Ref,
        b: Base4I64Ref,
    ) -> None:
        """``dst = a + b (mod 2**64)`` preserving a/b value digits."""
        if len({dst.base, a.base, b.base}) != 3:
            raise ValueError("base-4 add64 currently requires distinct operands")

        bf = self.bf
        b_delta = b.base - a.base
        d_delta = dst.base - a.base

        for digit in range(DIGITS):
            bf.set_const(a.marker(digit), 1)
        bf.clear(a.marker(DIGITS))
        bf.clear(dst.marker(0))

        r = _RelativeBuilder()
        marker = 0
        a_value = 1
        total = b_delta
        b_value = b_delta + 1
        carry_in = d_delta
        out = d_delta + 1
        carry_out = d_delta + STRIDE

        # a's consumed traversal marker is lane-local restoration scratch.
        r.clear(marker)
        r.clear(total)
        r.clear(out)
        r.clear(carry_out)
        r.transfer(carry_in, total)
        _add_preserved(r, a_value, total, marker)
        _add_preserved(r, b_value, total, marker)
        _map_total_base4(r, total, out, carry_out)
        r.move(STRIDE)

        bf.move(a.marker(0))
        bf.emit("[" + r.code() + "]")
        bf.ptr = a.marker(DIGITS)
        # Overflow beyond digit 31 is modulo-2**64 and therefore discarded.
        bf.clear(dst.marker(DIGITS))
        bf.clear(dst.value(DIGITS))

    def _sub64_with_carry(
        self,
        dst: Base4I64Ref,
        a: Base4I64Ref,
        b: Base4I64Ref,
    ) -> None:
        """Compute a-b and leave unsigned no-borrow in dst's sentinel marker.

        Radix-4 two's-complement subtraction is ``a + (3-b) + 1`` per digit.
        The final carry is one exactly when unsigned ``a >= b``.
        """
        if len({dst.base, a.base, b.base}) != 3:
            raise ValueError("base-4 sub64 currently requires distinct operands")

        bf = self.bf
        b_delta = b.base - a.base
        d_delta = dst.base - a.base

        for digit in range(DIGITS):
            bf.set_const(a.marker(digit), 1)
        bf.clear(a.marker(DIGITS))
        # Initial +1 of radix complement subtraction.
        bf.set_const(dst.marker(0), 1)

        r = _RelativeBuilder()
        marker = 0
        a_value = 1
        total = b_delta
        b_value = b_delta + 1
        carry_in = d_delta
        out = d_delta + 1
        carry_out = d_delta + STRIDE

        r.clear(marker)
        r.clear(total)
        r.clear(out)
        r.clear(carry_out)
        r.transfer(carry_in, total)
        _add_preserved(r, a_value, total, marker)
        _add_complement_preserved(r, b_value, total, marker)
        _map_total_base4(r, total, out, carry_out)
        r.move(STRIDE)

        bf.move(a.marker(0))
        bf.emit("[" + r.code() + "]")
        bf.ptr = a.marker(DIGITS)
        bf.clear(dst.value(DIGITS))
        # Deliberately retain dst.marker(DIGITS) as the final no-borrow carry.

    def sub64(
        self,
        dst: Base4I64Ref,
        a: Base4I64Ref,
        b: Base4I64Ref,
    ) -> None:
        """``dst = a - b (mod 2**64)`` preserving a/b value digits."""
        self._sub64_with_carry(dst, a, b)
        self.bf.clear(dst.marker(DIGITS))

    def uge64(
        self,
        result: int,
        a: Base4I64Ref,
        b: Base4I64Ref,
        tmp: Base4I64Ref,
    ) -> None:
        """Set one byte result to 1 iff unsigned ``a >= b``."""
        if result in range(tmp.base, tmp.base + tmp.cells):
            raise ValueError("result cell must not alias temporary word")

        self._sub64_with_carry(tmp, a, b)
        carry = tmp.marker(DIGITS)
        self.bf.clear(result)
        self.bf.move(carry)
        self.bf.emit("[-")
        self.bf.move(result)
        self.bf.emit("+")
        self.bf.move(carry)
        self.bf.emit("]")
        self.bf.clear(tmp.value(DIGITS))


__all__ = [
    "DIGITS",
    "STRIDE",
    "WORD_CELLS",
    "MASK64",
    "Base4I64Ref",
    "Base4I64Core",
]
