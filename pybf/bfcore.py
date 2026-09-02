"""Low-level Brainfuck code generation primitives.

The v1 transpiler emits Brainfuck directly from AST visitors.  This module
adds a deliberately small backend layer so arithmetic can be tested without
involving the Python frontend.

Integer convention
------------------
* unsigned 64-bit values are 64 consecutive cells, little-endian by bit;
* every bit cell is always 0 or 1;
* arithmetic is modulo 2**64;
* scratch cells are zero on entry and restored to zero on exit;
* operands and destination must not overlap.

Signed two's-complement values can use exactly the same add/sub primitives.
"""

from __future__ import annotations

from dataclasses import dataclass


WORD_BITS = 64


class BFEmitter:
    """Brainfuck emitter that tracks the physical data pointer."""

    def __init__(self) -> None:
        self.ptr = 0
        self.parts: list[str] = []

    def emit(self, code: str) -> None:
        self.parts.append(code)

    def move(self, cell: int) -> None:
        delta = cell - self.ptr
        if delta > 0:
            self.parts.append(">" * delta)
        elif delta < 0:
            self.parts.append("<" * -delta)
        self.ptr = cell

    def clear(self, cell: int) -> None:
        self.move(cell)
        self.emit("[-]")

    def set_const(self, cell: int, value: int) -> None:
        self.clear(cell)
        value %= 256
        if value <= 128:
            self.emit("+" * value)
        else:
            self.emit("-" * (256 - value))

    def add_const(self, cell: int, value: int) -> None:
        self.move(cell)
        value %= 256
        if value <= 128:
            self.emit("+" * value)
        else:
            self.emit("-" * (256 - value))

    def begin_while(self, cell: int) -> None:
        self.move(cell)
        self.emit("[")

    def end_while(self, cell: int) -> None:
        self.move(cell)
        self.emit("]")

    def code(self) -> str:
        return "".join(self.parts)


@dataclass(frozen=True)
class Int64Ref:
    base: int

    def bit(self, i: int) -> int:
        if not 0 <= i < WORD_BITS:
            raise IndexError(i)
        return self.base + i


class Binary64Core:
    """Correctness-first 64-bit arithmetic backend.

    Five scratch cells are sufficient for the current primitives.  They may be
    anywhere outside all live operands/destinations.
    """

    SCRATCH_CELLS = 5

    def __init__(self, bf: BFEmitter, scratch_base: int) -> None:
        self.bf = bf
        self.s0 = scratch_base
        self.s1 = scratch_base + 1
        self.s2 = scratch_base + 2
        self.carry0 = scratch_base + 3
        self.carry1 = scratch_base + 4

    def _clear_scratch(self) -> None:
        for c in (self.s0, self.s1, self.s2, self.carry0, self.carry1):
            self.bf.clear(c)

    def copy_cell(self, src: int, dst: int, tmp: int) -> None:
        """dst = src, preserving src.  dst/tmp must be distinct from src."""
        b = self.bf
        b.clear(dst)
        b.clear(tmp)
        b.begin_while(src)
        b.add_const(src, -1)
        b.add_const(dst, 1)
        b.add_const(tmp, 1)
        b.end_while(src)
        b.begin_while(tmp)
        b.add_const(tmp, -1)
        b.add_const(src, 1)
        b.end_while(tmp)

    def copy64(self, dst: Int64Ref, src: Int64Ref) -> None:
        for i in range(WORD_BITS):
            self.copy_cell(src.bit(i), dst.bit(i), self.s0)
        self.bf.clear(self.s0)

    def set_u64(self, dst: Int64Ref, value: int) -> None:
        value &= (1 << WORD_BITS) - 1
        for i in range(WORD_BITS):
            self.bf.set_const(dst.bit(i), (value >> i) & 1)

    def _toggle_bit(self, bit: int, flag: int) -> None:
        """bit ^= 1 for a 0/1 bit; flag is zero before/after."""
        b = self.bf
        b.set_const(flag, 1)
        b.begin_while(bit)
        b.add_const(bit, -1)
        b.clear(flag)
        b.end_while(bit)
        b.begin_while(flag)
        b.add_const(flag, -1)
        b.add_const(bit, 1)
        b.end_while(flag)

    def _add_one(self, out_bit: int, carry_out: int, flag: int) -> None:
        """Add one to a one-bit accumulator, emitting overflow to carry_out."""
        b = self.bf
        b.set_const(flag, 1)
        b.begin_while(out_bit)
        b.add_const(out_bit, -1)
        b.add_const(carry_out, 1)
        b.clear(flag)
        b.end_while(out_bit)
        b.begin_while(flag)
        b.add_const(flag, -1)
        b.add_const(out_bit, 1)
        b.end_while(flag)

    def _add_preserved_source_bit(self, src: int, out: int, carry_out: int) -> None:
        """out += src (mod 2), preserving src; carry_out receives overflow."""
        self.copy_cell(src, self.s0, self.s1)
        b = self.bf
        b.begin_while(self.s0)
        b.add_const(self.s0, -1)
        self._add_one(out, carry_out, self.s2)
        b.end_while(self.s0)

    def _add64_impl(
        self,
        dst: Int64Ref,
        a: Int64Ref,
        b_ref: Int64Ref,
        *,
        invert_b: bool,
        initial_carry: int,
    ) -> None:
        """Ripple-carry adder used by both addition and subtraction."""
        b = self.bf
        self._clear_scratch()
        b.set_const(self.carry0, initial_carry)
        carry_in, carry_out = self.carry0, self.carry1

        for i in range(WORD_BITS):
            out = dst.bit(i)
            b.clear(out)
            b.clear(carry_out)

            self._add_preserved_source_bit(a.bit(i), out, carry_out)

            if not invert_b:
                self._add_preserved_source_bit(b_ref.bit(i), out, carry_out)
            else:
                # Add (1 - b_i) without mutating b_i.
                self.copy_cell(b_ref.bit(i), self.s0, self.s1)
                self._toggle_bit(self.s0, self.s2)
                b.begin_while(self.s0)
                b.add_const(self.s0, -1)
                self._add_one(out, carry_out, self.s2)
                b.end_while(self.s0)

            # carry_in is dead after this bit, so consume it directly.
            b.begin_while(carry_in)
            b.add_const(carry_in, -1)
            self._add_one(out, carry_out, self.s2)
            b.end_while(carry_in)

            carry_in, carry_out = carry_out, carry_in

        self._clear_scratch()

    def add64(self, dst: Int64Ref, a: Int64Ref, b: Int64Ref) -> None:
        """dst = (a + b) mod 2**64, preserving a and b."""
        self._add64_impl(dst, a, b, invert_b=False, initial_carry=0)

    def sub64(self, dst: Int64Ref, a: Int64Ref, b: Int64Ref) -> None:
        """dst = (a - b) mod 2**64, preserving a and b."""
        # Two's complement: a - b == a + (~b) + 1.
        self._add64_impl(dst, a, b, invert_b=True, initial_carry=1)

    def eq64(self, result: int, a: Int64Ref, b_ref: Int64Ref) -> None:
        """result = 1 iff a == b, preserving operands."""
        b = self.bf
        self._clear_scratch()
        b.set_const(result, 1)

        for i in range(WORD_BITS):
            # s0 = a_i xor b_i
            self.copy_cell(a.bit(i), self.s0, self.s1)
            self.copy_cell(b_ref.bit(i), self.s1, self.s2)
            b.begin_while(self.s1)
            b.add_const(self.s1, -1)
            self._toggle_bit(self.s0, self.s2)
            b.end_while(self.s1)

            # Any differing bit clears result.  Consume s0.
            b.begin_while(self.s0)
            b.add_const(self.s0, -1)
            b.clear(result)
            b.end_while(self.s0)

        self._clear_scratch()
