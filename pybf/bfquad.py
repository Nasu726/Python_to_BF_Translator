"""Source-compact bit-pair backend for 64-bit compiler scalars.

Each word uses 32 lanes plus a sentinel::

    [marker, bit0, bit1] * 32 + [sentinel, 0, 0]

The value bits still represent the exact same two's-complement int64 ABI as
``Int64Ref``.  ``Quad64Ref`` subclasses ``Int64Ref`` and only changes physical
bit addressing, so correctness-first backend operations can continue to work
through the normal ``bit(i)`` interface while hot copy/add/sub/compare
operations use one runtime lane-walking Brainfuck body instead of Python-
unrolled bit bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter, Int64Ref


DIGITS = 32
STRIDE = 3
WORD_CELLS = (DIGITS + 1) * STRIDE
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Quad64Ref(Int64Ref):
    """Int64-compatible reference with interleaved runtime-walk markers."""

    def marker(self, digit: int) -> int:
        if not 0 <= digit <= DIGITS:
            raise IndexError(digit)
        return self.base + STRIDE * digit

    def bit0(self, digit: int) -> int:
        if not 0 <= digit <= DIGITS:
            raise IndexError(digit)
        return self.base + STRIDE * digit + 1

    def bit1(self, digit: int) -> int:
        if not 0 <= digit <= DIGITS:
            raise IndexError(digit)
        return self.base + STRIDE * digit + 2

    def bit(self, bit_index: int) -> int:
        if not 0 <= bit_index < 64:
            raise IndexError(bit_index)
        digit, within = divmod(bit_index, 2)
        return self.bit0(digit) if within == 0 else self.bit1(digit)

    @property
    def cells(self) -> int:
        return WORD_CELLS


class _RelativeBuilder:
    """Build one loop body using offsets relative to the current A marker."""

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

    def clear(self, cell: int) -> None:
        self.move(cell)
        self.emit("[-]")

    def add(self, cell: int, amount: int) -> None:
        self.move(cell)
        self.emit("+" * amount if amount >= 0 else "-" * -amount)

    def transfer(self, src: int, dst: int) -> None:
        """Destructively add a Boolean source into dst."""
        self.move(src)
        self.emit("[-")
        self.move(dst)
        self.emit("+")
        self.move(src)
        self.emit("]")

    def code(self) -> str:
        return "".join(self.parts)


def _map_total(builder: _RelativeBuilder, total: int, out: int, carry: int) -> None:
    """Map total in 0..3 to parity and floor(total/2), consuming total."""
    r = builder
    r.move(total)
    r.emit("[")
    r.emit("-")
    r.add(out, 1)
    r.move(total)
    r.emit("[")
    r.emit("-")
    r.add(out, -1)
    r.add(carry, 1)
    r.move(total)
    r.emit("[")
    r.emit("-")
    r.add(out, 1)
    r.move(total)
    r.emit("]")
    r.move(total)
    r.emit("]")
    r.move(total)
    r.emit("]")


def _add_preserved(builder: _RelativeBuilder, src: int, total: int, tmp: int) -> None:
    """total += src for a Boolean src, restoring src and leaving tmp zero."""
    r = builder
    r.move(src)
    r.emit("[")
    r.emit("-")
    r.add(total, 1)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.emit("-")
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


def _add_not_preserved(
    builder: _RelativeBuilder,
    src: int,
    total: int,
    tmp: int,
    helper: int,
) -> None:
    """total += (1-src), preserving Boolean src; tmp/helper finish zero."""
    r = builder

    r.move(src)
    r.emit("[")
    r.emit("-")
    r.add(tmp, 1)
    r.add(helper, 1)
    r.move(src)
    r.emit("]")
    r.move(helper)
    r.emit("[")
    r.emit("-")
    r.add(src, 1)
    r.move(helper)
    r.emit("]")

    r.add(helper, 1)
    r.move(tmp)
    r.emit("[")
    r.emit("-")
    r.clear(helper)
    r.move(tmp)
    r.emit("]")
    r.move(helper)
    r.emit("[")
    r.emit("-")
    r.add(tmp, 1)
    r.move(helper)
    r.emit("]")

    r.transfer(tmp, total)


class Quad64Core:
    """Runtime-lane copy/add/sub/unsigned-compare over ``Quad64Ref`` values."""

    def __init__(self, bf: BFEmitter) -> None:
        self.bf = bf

    def set_u64(self, dst: Quad64Ref, value: int) -> None:
        value &= MASK64
        for digit in range(DIGITS):
            self.bf.clear(dst.marker(digit))
            self.bf.set_const(dst.bit0(digit), (value >> (2 * digit)) & 1)
            self.bf.set_const(dst.bit1(digit), (value >> (2 * digit + 1)) & 1)
        self.bf.clear(dst.marker(DIGITS))
        self.bf.clear(dst.bit0(DIGITS))
        self.bf.clear(dst.bit1(DIGITS))

    def copy64(self, dst: Quad64Ref, src: Quad64Ref) -> None:
        """Copy all value bits with one emitted 32-lane runtime loop."""
        if dst.base == src.base:
            return

        bf = self.bf
        delta = dst.base - src.base

        # Source markers drive the loop.  The current source marker is consumed
        # at the start of every lane and then reused as the restoration scratch
        # for the two Boolean payload bits.
        for digit in range(DIGITS):
            bf.set_const(src.marker(digit), 1)
        bf.clear(src.marker(DIGITS))

        r = _RelativeBuilder()
        marker = 0
        s0, s1 = 1, 2
        dmark, d0, d1 = delta, delta + 1, delta + 2

        r.clear(marker)
        r.clear(dmark)
        r.clear(d0)
        r.clear(d1)
        _add_preserved(r, s0, d0, marker)
        _add_preserved(r, s1, d1, marker)
        r.move(STRIDE)

        bf.move(src.marker(0))
        bf.emit("[" + r.code() + "]")
        bf.ptr = src.marker(DIGITS)
        bf.clear(dst.marker(DIGITS))
        bf.clear(dst.bit0(DIGITS))
        bf.clear(dst.bit1(DIGITS))

    def _prepare_binary_op(
        self, dst: Quad64Ref, a: Quad64Ref, b: Quad64Ref
    ) -> None:
        for digit in range(DIGITS):
            self.bf.set_const(a.marker(digit), 1)
            self.bf.clear(b.marker(digit))
            self.bf.clear(dst.marker(digit))
            self.bf.clear(dst.bit0(digit))
            self.bf.clear(dst.bit1(digit))
        self.bf.clear(a.marker(DIGITS))
        self.bf.clear(b.marker(DIGITS))
        self.bf.clear(dst.marker(DIGITS))

    def _emit_add_body(self, b_delta: int, d_delta: int) -> str:
        r = _RelativeBuilder()
        total = 0
        a0, a1 = 1, 2
        tmp = b_delta
        b0, b1 = b_delta + 1, b_delta + 2
        carry_in = d_delta
        d0, d1 = d_delta + 1, d_delta + 2
        carry_out = d_delta + STRIDE

        r.clear(total)
        r.transfer(carry_in, total)
        _add_preserved(r, a0, total, tmp)
        _add_preserved(r, b0, total, tmp)
        _map_total(r, total, d0, tmp)

        r.transfer(tmp, total)
        _add_preserved(r, a1, total, tmp)
        _add_preserved(r, b1, total, tmp)
        _map_total(r, total, d1, carry_out)

        r.move(STRIDE)
        return r.code()

    def _emit_sub_body(self, b_delta: int, d_delta: int) -> str:
        r = _RelativeBuilder()
        total = 0
        a0, a1 = 1, 2
        tmp = b_delta
        b0, b1 = b_delta + 1, b_delta + 2
        carry_in = d_delta
        d0, d1 = d_delta + 1, d_delta + 2
        carry_out = d_delta + STRIDE

        r.clear(total)
        r.transfer(carry_in, total)
        _add_preserved(r, a0, total, tmp)
        _add_not_preserved(r, b0, total, tmp, d0)
        _map_total(r, total, d0, tmp)

        r.transfer(tmp, total)
        _add_preserved(r, a1, total, tmp)
        _add_not_preserved(r, b1, total, tmp, d1)
        _map_total(r, total, d1, carry_out)

        r.move(STRIDE)
        return r.code()

    def add64(self, dst: Quad64Ref, a: Quad64Ref, b: Quad64Ref) -> None:
        """dst = a+b modulo 2**64, preserving value bits of a and b."""
        self._prepare_binary_op(dst, a, b)
        self.bf.move(a.marker(0))
        body = self._emit_add_body(b.base - a.base, dst.base - a.base)
        self.bf.emit("[" + body + "]")
        self.bf.ptr = a.marker(DIGITS)
        self.bf.clear(dst.marker(DIGITS))

    def sub64(self, dst: Quad64Ref, a: Quad64Ref, b: Quad64Ref) -> None:
        """dst = a-b modulo 2**64 via a + ~b + 1."""
        self._prepare_binary_op(dst, a, b)
        self.bf.set_const(dst.marker(0), 1)
        self.bf.move(a.marker(0))
        body = self._emit_sub_body(b.base - a.base, dst.base - a.base)
        self.bf.emit("[" + body + "]")
        self.bf.ptr = a.marker(DIGITS)
        self.bf.clear(dst.marker(DIGITS))

    def uge64(
        self,
        result: int,
        a: Quad64Ref,
        b: Quad64Ref,
        tmp: Quad64Ref,
    ) -> None:
        """Set result=1 iff unsigned a>=b using the subtraction carry-out."""
        self._prepare_binary_op(tmp, a, b)
        self.bf.set_const(tmp.marker(0), 1)
        self.bf.move(a.marker(0))
        body = self._emit_sub_body(b.base - a.base, tmp.base - a.base)
        self.bf.emit("[" + body + "]")
        self.bf.ptr = a.marker(DIGITS)

        carry = tmp.marker(DIGITS)
        self.bf.clear(result)
        self.bf.move(carry)
        self.bf.emit("[-")
        self.bf.move(result)
        self.bf.emit("+")
        self.bf.move(carry)
        self.bf.emit("]")


__all__ = ["DIGITS", "STRIDE", "WORD_CELLS", "Quad64Ref", "Quad64Core"]
