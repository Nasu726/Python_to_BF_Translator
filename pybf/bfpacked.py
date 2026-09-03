"""Compact packed unsigned-32 primitives for heap/list metadata.

Ordinary Python integers currently use the correctness-first 64 bit-per-cell
representation.  Heap metadata does not need that cost: object handles, list
lengths, capacities and block counters are naturally non-negative and fit in
four ordinary 8-bit Brainfuck cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter


U32_BYTES = 4
U32_MASK = (1 << 32) - 1


@dataclass(frozen=True)
class PackedU32Ref:
    base: int

    def byte(self, index: int) -> int:
        if not 0 <= index < U32_BYTES:
            raise IndexError(index)
        return self.base + index

    @property
    def cells(self) -> int:
        return U32_BYTES


class PackedU32Core:
    """Source-compact operations on little-endian packed u32 values."""

    SCRATCH_CELLS = 4

    def __init__(self, bf: BFEmitter, scratch_base: int) -> None:
        self.bf = bf
        self.s0 = scratch_base
        self.s1 = scratch_base + 1
        self.s2 = scratch_base + 2
        self.s3 = scratch_base + 3

    def _clear_scratch(self) -> None:
        for cell in (self.s0, self.s1, self.s2, self.s3):
            self.bf.clear(cell)

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

    def _is_zero_byte(self, result: int, src: int, tmp: int) -> None:
        bf = self.bf
        bf.set_const(result, 1)
        self._copy_cell(src, tmp, self.s3)
        bf.begin_while(tmp)
        bf.clear(tmp)
        bf.clear(result)
        bf.end_while(tmp)

    def clear(self, ref: PackedU32Ref) -> None:
        for i in range(U32_BYTES):
            self.bf.clear(ref.byte(i))
        self._clear_scratch()

    def set_u32(self, ref: PackedU32Ref, value: int) -> None:
        value &= U32_MASK
        for i in range(U32_BYTES):
            self.bf.set_const(ref.byte(i), (value >> (8 * i)) & 0xFF)
        self._clear_scratch()

    def copy(self, dst: PackedU32Ref, src: PackedU32Ref) -> None:
        if dst == src:
            return
        for i in range(U32_BYTES):
            self._copy_cell(src.byte(i), dst.byte(i), self.s0)
        self._clear_scratch()

    def is_zero(self, result: int, ref: PackedU32Ref) -> None:
        """result = 1 iff all four bytes are zero."""
        bf = self.bf
        bf.set_const(result, 1)
        for i in range(U32_BYTES):
            self._copy_cell(ref.byte(i), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.clear(self.s0)
            bf.clear(result)
            bf.end_while(self.s0)
        self._clear_scratch()

    def equal(self, result: int, a: PackedU32Ref, b: PackedU32Ref) -> None:
        bf = self.bf
        bf.set_const(result, 1)
        for i in range(U32_BYTES):
            self._copy_cell(a.byte(i), self.s0, self.s2)
            self._copy_cell(b.byte(i), self.s1, self.s2)
            bf.begin_while(self.s1)
            bf.add_const(self.s1, -1)
            bf.add_const(self.s0, -1)
            bf.end_while(self.s1)
            bf.begin_while(self.s0)
            bf.clear(self.s0)
            bf.clear(result)
            bf.end_while(self.s0)
        self._clear_scratch()

    def increment(self, ref: PackedU32Ref) -> None:
        """ref = ref + 1 modulo 2**32."""
        bf = self.bf
        carry = self.s0
        gate = self.s1
        tmp = self.s2
        bf.set_const(carry, 1)
        for i in range(U32_BYTES):
            self._copy_cell(carry, gate, self.s3)
            bf.clear(carry)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.add_const(ref.byte(i), 1)
            self._is_zero_byte(carry, ref.byte(i), tmp)
            bf.end_while(gate)
        self._clear_scratch()

    def decrement(self, ref: PackedU32Ref) -> None:
        """ref = ref - 1 modulo 2**32."""
        bf = self.bf
        borrow = self.s0
        gate = self.s1
        tmp = self.s2
        bf.set_const(borrow, 1)
        for i in range(U32_BYTES):
            self._copy_cell(borrow, gate, self.s3)
            bf.clear(borrow)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            # Borrow continues exactly when this byte was zero before --.
            self._is_zero_byte(borrow, ref.byte(i), tmp)
            bf.add_const(ref.byte(i), -1)
            bf.end_while(gate)
        self._clear_scratch()


__all__ = ["U32_BYTES", "PackedU32Ref", "PackedU32Core"]
