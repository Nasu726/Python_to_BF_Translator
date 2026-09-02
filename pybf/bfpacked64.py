"""Eight-byte heap representation for signed/int64 values.

Python-facing arithmetic still uses the existing 64 Boolean-cell two's
complement representation.  Heap/list storage uses eight ordinary bytes to
keep object blocks compact.  These primitives convert losslessly between the
two representations without Python runtime help.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter, Int64Ref, WORD_BITS


I64_BYTES = 8


@dataclass(frozen=True)
class PackedI64Ref:
    base: int

    def byte(self, index: int) -> int:
        if not 0 <= index < I64_BYTES:
            raise IndexError(index)
        return self.base + index

    @property
    def cells(self) -> int:
        return I64_BYTES


class PackedI64Core:
    SCRATCH_CELLS = 4

    def __init__(self, bf: BFEmitter, scratch_base: int) -> None:
        self.bf = bf
        self.s0 = scratch_base
        self.s1 = scratch_base + 1
        self.s2 = scratch_base + 2
        self.s3 = scratch_base + 3

    def _clear_scratch(self) -> None:
        for c in (self.s0, self.s1, self.s2, self.s3):
            self.bf.clear(c)

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

    def clear(self, ref: PackedI64Ref) -> None:
        for i in range(I64_BYTES):
            self.bf.clear(ref.byte(i))
        self._clear_scratch()

    def copy(self, dst: PackedI64Ref, src: PackedI64Ref) -> None:
        if dst == src:
            return
        for i in range(I64_BYTES):
            self._copy_cell(src.byte(i), dst.byte(i), self.s0)
        self._clear_scratch()

    def from_int64(self, dst: PackedI64Ref, src: Int64Ref) -> None:
        """Pack 64 Boolean bit-cells into eight little-endian bytes."""
        bf = self.bf
        self.clear(dst)
        for byte_index in range(I64_BYTES):
            out = dst.byte(byte_index)
            for within in range(8):
                bit = src.bit(byte_index * 8 + within)
                self._copy_cell(bit, self.s0, self.s1)
                bf.begin_while(self.s0)
                bf.add_const(self.s0, -1)
                bf.add_const(out, 1 << within)
                bf.end_while(self.s0)
        self._clear_scratch()

    def _increment_bit_byte(self, dst: Int64Ref, bit_offset: int) -> None:
        """Increment one 8-bit Boolean slice; overflow is discarded."""
        bf = self.bf
        carry = self.s1
        gate = self.s2
        helper = self.s3
        bf.set_const(carry, 1)

        for within in range(8):
            bit = dst.bit(bit_offset + within)
            self._copy_cell(carry, gate, helper)
            bf.clear(carry)
            bf.begin_while(gate)
            bf.add_const(gate, -1)

            # Boolean bit + carry.  helper selects the zero-bit path.
            bf.set_const(helper, 1)
            bf.begin_while(bit)
            bf.add_const(bit, -1)
            bf.add_const(carry, 1)
            bf.clear(helper)
            bf.end_while(bit)
            bf.begin_while(helper)
            bf.add_const(helper, -1)
            bf.add_const(bit, 1)
            bf.end_while(helper)

            bf.end_while(gate)

    def to_int64(self, dst: Int64Ref, src: PackedI64Ref) -> None:
        """Expand eight bytes into the normal 64 Boolean-cell int64 layout."""
        bf = self.bf
        for i in range(WORD_BITS):
            bf.clear(dst.bit(i))

        for byte_index in range(I64_BYTES):
            self._copy_cell(src.byte(byte_index), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            self._increment_bit_byte(dst, byte_index * 8)
            bf.end_while(self.s0)

        self._clear_scratch()


__all__ = ["I64_BYTES", "PackedI64Ref", "PackedI64Core"]
