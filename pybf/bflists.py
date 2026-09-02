"""Fixed-capacity int64 list primitives for the Brainfuck runtime.

Storage is deliberately compact:

    [length: one byte][element 0: 8 bytes]...[element N-1: 8 bytes]

Python-facing arithmetic still uses the existing 64 Boolean-cell int64
representation.  List storage uses packed little-endian bytes and only expands
one selected element when it is actually consumed.  This avoids emitting 64
copies of the full 64-bit runtime for dynamic indexing/append.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import Int64Ref, WORD_BITS
from bfpacked64 import I64_BYTES, PackedI64Core, PackedI64Ref
from bftokens import BinaryTokenIO


@dataclass(frozen=True)
class IntListRef:
    base: int
    capacity: int

    @property
    def length_cell(self) -> int:
        return self.base

    def item(self, index: int) -> PackedI64Ref:
        if not 0 <= index < self.capacity:
            raise IndexError(index)
        return PackedI64Ref(self.base + 1 + index * I64_BYTES)

    @property
    def cells(self) -> int:
        return 1 + self.capacity * I64_BYTES


class BinaryListIO(BinaryTokenIO):
    """Integer-list operations over compact packed element storage."""

    def __init__(self, bf, scratch_base: int) -> None:
        super().__init__(bf, scratch_base=scratch_base)
        self.packed64 = PackedI64Core(bf, scratch_base)

    def clear_list(self, ref: IntListRef) -> None:
        # Logical clearing only needs to reset length.  Stale bytes past length
        # are unreachable and are overwritten before append exposes them.
        self.bf.clear(ref.length_cell)

    def set_list_literal(self, ref: IntListRef, values: list[int]) -> None:
        if len(values) > ref.capacity:
            raise ValueError("list literal exceeds allocated list capacity")
        self.clear_list(ref)
        self.bf.set_const(ref.length_cell, len(values))
        for i, value in enumerate(values):
            self.packed64.set_u64(ref.item(i), value)

    def copy_list(self, dst: IntListRef, src: IntListRef) -> None:
        if dst.capacity < src.capacity:
            raise ValueError("destination list capacity is smaller than source")
        self.clear_list(dst)
        self.copy_cell(src.length_cell, dst.length_cell, self.s0)
        # Current value-copy semantics copy all physical slots.  Each slot is
        # only eight bytes now, so this remains compact until alias semantics
        # move lists to the heap-backed object model.
        for i in range(src.capacity):
            self.packed64.copy(dst.item(i), src.item(i))
        self._clear_scratch()

    def list_length(self, dst: Int64Ref, ref: IntListRef, tmp: int) -> None:
        """Convert the compact byte length to a normal uint64 word."""
        bf = self.bf
        self._clear_word(dst)
        self.copy_cell(ref.length_cell, tmp, self.s0)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        self._inc64_inplace(dst)
        bf.end_while(tmp)
        self._clear_scratch()

    def get_const(self, dst: Int64Ref, ref: IntListRef, index: int) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.packed64.to_int64(dst, ref.item(index))

    def set_const(self, ref: IntListRef, index: int, value: Int64Ref) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.packed64.from_int64(ref.item(index), value)

    def _int64_low_byte(self, dst: int, src: Int64Ref) -> None:
        """Pack src bits 0..7 into one byte while preserving src."""
        bf = self.bf
        bf.clear(dst)
        for bit_index in range(8):
            self.copy_cell(src.bit(bit_index), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            bf.add_const(dst, 1 << bit_index)
            bf.end_while(self.s0)
        self._clear_scratch()

    def get_dynamic(
        self,
        dst: Int64Ref,
        ref: IntListRef,
        index: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Load a runtime index without duplicating 64-bit unpack per slot.

        A single signed bounds comparison validates the int64 index.  Selection
        is then performed on its low byte using compact eight-byte copies; only
        the selected packed value is expanded to the normal int64 layout.
        """
        bf = self.bf
        selected = PackedI64Ref(workspace_word.base)
        index_byte = workspace_word.base + I64_BYTES
        slot_match = workspace_word.base + I64_BYTES + 1

        self.set_u64(workspace_word, ref.capacity)
        self.slt64(match, index, workspace_word)
        self.packed64.clear(selected)
        self._int64_low_byte(index_byte, index)

        bf.begin_while(match)
        bf.add_const(match, -1)
        for i in range(ref.capacity):
            self._eq_byte_const(slot_match, index_byte, i)
            bf.begin_while(slot_match)
            bf.add_const(slot_match, -1)
            self.packed64.copy(selected, ref.item(i))
            bf.end_while(slot_match)
        bf.end_while(match)

        self.packed64.to_int64(dst, selected)
        self.packed64.clear(selected)
        bf.clear(index_byte)
        bf.clear(slot_match)
        self._clear_scratch()

    def set_dynamic(
        self,
        ref: IntListRef,
        index: Int64Ref,
        value: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Store through a runtime index using one packed conversion."""
        bf = self.bf
        packed_value = PackedI64Ref(workspace_word.base)
        index_byte = workspace_word.base + I64_BYTES
        slot_match = workspace_word.base + I64_BYTES + 1

        self.set_u64(workspace_word, ref.capacity)
        self.slt64(match, index, workspace_word)
        self.packed64.from_int64(packed_value, value)
        self._int64_low_byte(index_byte, index)

        bf.begin_while(match)
        bf.add_const(match, -1)
        for i in range(ref.capacity):
            self._eq_byte_const(slot_match, index_byte, i)
            bf.begin_while(slot_match)
            bf.add_const(slot_match, -1)
            self.packed64.copy(ref.item(i), packed_value)
            bf.end_while(slot_match)
        bf.end_while(match)

        self.packed64.clear(packed_value)
        bf.clear(index_byte)
        bf.clear(slot_match)
        self._clear_scratch()

    def append(
        self,
        ref: IntListRef,
        value: Int64Ref,
        length_copy: int,
        match: int,
        packed_tmp: PackedI64Ref,
    ) -> None:
        """Append using one int64->packed conversion for all candidate slots."""
        bf = self.bf
        self.packed64.from_int64(packed_tmp, value)
        self.copy_cell(ref.length_cell, length_copy, self.s0)
        for i in range(ref.capacity):
            self._eq_byte_const(match, length_copy, i)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.packed64.copy(ref.item(i), packed_tmp)
            bf.add_const(ref.length_cell, 1)
            bf.end_while(match)
        bf.clear(length_copy)
        self.packed64.clear(packed_tmp)
        self._clear_scratch()

    def read_int_list_line(
        self,
        ref: IntListRef,
        workspace_base: int,
        active: int,
        gate: int,
        has_token: int,
        end_line: int,
    ) -> None:
        """Implement ``list(map(int, input().split()))`` with one parser body.

        The old implementation emitted the full signed-decimal parser once per
        list capacity slot.  Here one parser sits inside a Brainfuck runtime
        loop and appends each token.  Extra tokens after capacity are still
        consumed so the following ``input()`` begins on the next physical line.
        """
        bf = self.bf
        self.clear_list(ref)
        for c in (active, gate, has_token, end_line):
            bf.clear(c)

        token = Int64Ref(workspace_base + WORD_BITS * 2 + 16)
        packed_tmp = PackedI64Ref(token.base + WORD_BITS)
        length_copy = packed_tmp.base + I64_BYTES
        append_match = length_copy + 1

        bf.set_const(active, 1)
        bf.begin_while(active)

        self.read_s64_line_token(token, has_token, end_line, workspace_base)

        self.copy_cell(has_token, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        self.append(ref, token, length_copy, append_match, packed_tmp)
        bf.end_while(gate)

        self.copy_cell(end_line, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.clear(active)
        bf.end_while(gate)

        bf.end_while(active)
        self._clear_scratch()


__all__ = ["IntListRef", "BinaryListIO"]
