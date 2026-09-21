"""Source-compact arithmetic/comparison directly on packed int64 bytes.

These primitives keep the project's exact modulo-2**64 / signed two's-complement
ABI while avoiding expansion into the 99-cell Quad representation. Runtime work
is intentionally byte-value-dependent (bounded by eight 0..255 lanes); the goal
is a compact intermediate representation for short-lived contest scalars.

Operands are preserved unless the method name explicitly says ``inplace``.
Scratch is zero on return.
"""

from __future__ import annotations

from bfcore import BFEmitter
from bfpacked64 import I64_BYTES, MASK64, PackedI64Ref


class PackedI64Ops:
    SCRATCH_CELLS = 16

    def __init__(self, bf: BFEmitter, scratch_base: int) -> None:
        self.bf = bf
        self.base = scratch_base

    def _s(self, index: int) -> int:
        return self.base + index

    def _clear_scratch(self) -> None:
        for index in range(self.SCRATCH_CELLS):
            self.bf.clear(self._s(index))

    def _copy_cell(self, src: int, dst: int, tmp: int) -> None:
        bf = self.bf
        bf.clear(dst)
        bf.clear(tmp)
        bf.begin_while(src)
        bf.add_const(src, -1)
        bf.add_const(dst, 1)
        bf.add_const(tmp, 1)
        bf.end_while(src)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        bf.add_const(src, 1)
        bf.end_while(tmp)

    def _zero_flag(self, result: int, src: int, tmp: int, helper: int) -> None:
        bf = self.bf
        bf.set_const(result, 1)
        self._copy_cell(src, tmp, helper)
        bf.begin_while(tmp)
        bf.clear(tmp)
        bf.clear(result)
        bf.end_while(tmp)

    def clear(self, ref: PackedI64Ref) -> None:
        for index in range(I64_BYTES):
            self.bf.clear(ref.byte(index))

    def set_u64(self, ref: PackedI64Ref, value: int) -> None:
        value &= MASK64
        for index in range(I64_BYTES):
            self.bf.set_const(ref.byte(index), (value >> (8 * index)) & 0xFF)

    def copy(self, dst: PackedI64Ref, src: PackedI64Ref) -> None:
        if dst.base == src.base:
            return
        for index in range(I64_BYTES):
            self._copy_cell(src.byte(index), dst.byte(index), self._s(14))
        self.bf.clear(self._s(14))

    def _split_parity(self, src: int, quotient: int, parity: int, gate: int) -> None:
        """Consume src into quotient/parity; outputs and gate must be zero."""
        bf = self.bf
        bf.begin_while(src)
        bf.add_const(src, -1)
        bf.set_const(gate, 1)
        bf.begin_while(parity)
        bf.add_const(parity, -1)
        bf.add_const(quotient, 1)
        bf.clear(gate)
        bf.end_while(parity)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.add_const(parity, 1)
        bf.end_while(gate)
        bf.end_while(src)

    def _move_cell(self, src: int, dst: int, scale: int = 1) -> None:
        """Consume src, adding scale*src to dst."""
        self.bf.begin_while(src)
        self.bf.add_const(src, -1)
        self.bf.add_const(dst, scale)
        self.bf.end_while(src)

    def _copy_into_zero_cell(self, src: int, dst: int, tmp: int) -> None:
        """Preserving copy when dst/tmp are already known zero."""
        self.bf.begin_while(src)
        self.bf.add_const(src, -1)
        self.bf.add_const(dst, 1)
        self.bf.add_const(tmp, 1)
        self.bf.end_while(src)
        self._move_cell(tmp, src)

    def add_inplace(self, dst: PackedI64Ref, rhs: PackedI64Ref) -> None:
        """Add modulo 2**64, preserving a disjoint rhs; exact alias doubles.

        Extract bits by repeated halving, so each input byte is scanned less
        than twice. The eight bit positions share one emitted runtime body.
        This avoids checking a full accumulated byte for every unit added.
        Partial overlap and scratch overlap are unsupported.
        """
        bf = self.bf
        a, b, q, pa, pb, carry, gate, tmp, weight, restore, bit = (
            self._s(i) for i in range(11))
        self._clear_scratch()
        active, scan, scan_restore = self._s(15), self._s(13), self._s(14)
        for byte_index in range(I64_BYTES):
            self._copy_into_zero_cell(rhs.byte(byte_index), scan, scan_restore)
            bf.begin_while(scan)
            bf.clear(scan)
            bf.set_const(active, 1)
            bf.end_while(scan)
        bf.begin_while(active)
        bf.clear(active)
        for byte_index in range(I64_BYTES):
            out = dst.byte(byte_index)
            self._copy_into_zero_cell(out, a, restore)
            self._copy_into_zero_cell(rhs.byte(byte_index), b, restore)
            bf.clear(out)
            bf.set_const(weight, 1)
            bf.begin_while(weight)
            self._split_parity(a, q, pa, gate)
            self._move_cell(q, a)
            self._split_parity(b, q, pb, gate)
            self._move_cell(q, b)
            self._move_cell(pa, tmp)
            self._move_cell(pb, tmp)
            self._move_cell(carry, tmp)
            self._split_parity(tmp, carry, bit, gate)
            bf.begin_while(bit)
            bf.add_const(bit, -1)
            # Add the current power of two without consuming the loop weight.
            bf.begin_while(weight)
            bf.add_const(weight, -1)
            bf.add_const(out, 1)
            bf.add_const(restore, 1)
            bf.end_while(weight)
            self._move_cell(restore, weight)
            bf.end_while(bit)
            self._move_cell(weight, tmp, 2)
            self._move_cell(tmp, weight)
            # 128*2 wraps to zero, ending exactly after eight iterations.
            bf.end_while(weight)
        bf.end_while(active)
        self._clear_scratch()

    def sub_inplace(self, dst: PackedI64Ref, rhs: PackedI64Ref) -> None:
        """``dst = dst - rhs (mod 2**64)``, preserving ``rhs``."""
        bf = self.bf
        count, borrow, next_borrow = self._s(0), self._s(1), self._s(2)
        zero, tmp, helper, gate = self._s(3), self._s(4), self._s(5), self._s(6)
        bf.clear(borrow)

        for byte_index in range(I64_BYTES):
            byte = dst.byte(byte_index)
            bf.clear(next_borrow)
            self._copy_cell(borrow, gate, helper)
            bf.clear(borrow)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            self._zero_flag(zero, byte, tmp, helper)
            bf.add_const(byte, -1)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.set_const(next_borrow, 1)
            bf.end_while(zero)
            bf.end_while(gate)

            self._copy_cell(rhs.byte(byte_index), count, helper)
            bf.begin_while(count)
            bf.add_const(count, -1)
            self._zero_flag(zero, byte, tmp, helper)
            bf.add_const(byte, -1)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.set_const(next_borrow, 1)
            bf.end_while(zero)
            bf.end_while(count)

            bf.begin_while(next_borrow)
            bf.add_const(next_borrow, -1)
            bf.add_const(borrow, 1)
            bf.end_while(next_borrow)
        self._clear_scratch()

    def equal(self, result: int, a: PackedI64Ref, b: PackedI64Ref) -> None:
        """Set ``result`` to one iff the packed words are bit-identical."""
        bf = self.bf
        work, count = self._s(0), self._s(1)
        zero, tmp, helper, gate = self._s(2), self._s(3), self._s(4), self._s(5)
        bf.set_const(result, 1)
        for byte_index in range(I64_BYTES):
            self._copy_cell(a.byte(byte_index), work, helper)
            self._copy_cell(b.byte(byte_index), count, helper)
            bf.begin_while(count)
            bf.add_const(count, -1)
            bf.add_const(work, -1)
            bf.end_while(count)
            self._zero_flag(zero, work, tmp, helper)
            bf.set_const(gate, 1)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.clear(gate)
            bf.end_while(zero)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.clear(result)
            bf.end_while(gate)
        self._clear_scratch()

    def _unsigned_lt(self, result: int, a: PackedI64Ref, b: PackedI64Ref) -> None:
        bf = self.bf
        work, count = self._s(0), self._s(1)
        borrow, next_borrow = self._s(2), self._s(3)
        zero, tmp, helper, gate = self._s(4), self._s(5), self._s(6), self._s(7)
        bf.clear(borrow)
        for byte_index in range(I64_BYTES):
            self._copy_cell(a.byte(byte_index), work, helper)
            bf.clear(next_borrow)
            self._copy_cell(borrow, gate, helper)
            bf.clear(borrow)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            self._zero_flag(zero, work, tmp, helper)
            bf.add_const(work, -1)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.set_const(next_borrow, 1)
            bf.end_while(zero)
            bf.end_while(gate)

            self._copy_cell(b.byte(byte_index), count, helper)
            bf.begin_while(count)
            bf.add_const(count, -1)
            self._zero_flag(zero, work, tmp, helper)
            bf.add_const(work, -1)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.set_const(next_borrow, 1)
            bf.end_while(zero)
            bf.end_while(count)
            bf.begin_while(next_borrow)
            bf.add_const(next_borrow, -1)
            bf.add_const(borrow, 1)
            bf.end_while(next_borrow)

        bf.clear(result)
        bf.begin_while(borrow)
        bf.add_const(borrow, -1)
        bf.add_const(result, 1)
        bf.end_while(borrow)
        self._clear_scratch()

    def _extract_sign(self, dst: int, src: int) -> None:
        """Extract bit 7 of one preserved byte into ``dst``."""
        bf = self.bf
        value, quotient, parity, gate = self._s(8), self._s(9), self._s(10), self._s(11)
        helper = self._s(7)
        self._copy_cell(src, value, helper)
        for _ in range(7):
            bf.clear(quotient)
            bf.clear(parity)
            bf.begin_while(value)
            bf.add_const(value, -1)
            bf.set_const(gate, 1)
            bf.begin_while(parity)
            bf.add_const(parity, -1)
            bf.clear(gate)
            bf.add_const(quotient, 1)
            bf.end_while(parity)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.add_const(parity, 1)
            bf.end_while(gate)
            bf.end_while(value)
            bf.begin_while(quotient)
            bf.add_const(quotient, -1)
            bf.add_const(value, 1)
            bf.end_while(quotient)
        bf.clear(dst)
        bf.begin_while(value)
        bf.add_const(value, -1)
        bf.add_const(dst, 1)
        bf.end_while(value)
        for cell in (quotient, parity, gate, helper):
            bf.clear(cell)

    def signed_lt(self, result: int, a: PackedI64Ref, b: PackedI64Ref) -> None:
        """Set ``result`` to one iff signed two's-complement ``a < b``."""
        bf = self.bf
        sign_a, sign_b, diff, gate = self._s(12), self._s(13), self._s(14), self._s(15)
        self._unsigned_lt(result, a, b)
        self._extract_sign(sign_a, a.byte(7))
        self._extract_sign(sign_b, b.byte(7))

        self._copy_cell(sign_a, diff, gate)
        self._copy_cell(sign_b, gate, self._s(11))
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.set_const(self._s(11), 1)
        bf.begin_while(diff)
        bf.add_const(diff, -1)
        bf.clear(self._s(11))
        bf.end_while(diff)
        bf.begin_while(self._s(11))
        bf.add_const(self._s(11), -1)
        bf.add_const(diff, 1)
        bf.end_while(self._s(11))
        bf.end_while(gate)

        bf.begin_while(diff)
        bf.add_const(diff, -1)
        bf.clear(result)
        self._copy_cell(sign_a, result, self._s(11))
        bf.end_while(diff)
        self._clear_scratch()


__all__ = ["PackedI64Ops"]
