"""Fixed-capacity int64 list primitives for the Brainfuck runtime.

Layout::

    [length: one byte][element 0: 64 bits]...[element N-1: 64 bits]

The one-byte length is sufficient because capacities are capped at 255.  It
avoids spending another 64 cells on list metadata while elements retain the
normal signed two's-complement word representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import Int64Ref, WORD_BITS
from bftokens import BinaryTokenIO


@dataclass(frozen=True)
class IntListRef:
    base: int
    capacity: int

    @property
    def length_cell(self) -> int:
        return self.base

    def item(self, index: int) -> Int64Ref:
        if not 0 <= index < self.capacity:
            raise IndexError(index)
        return Int64Ref(self.base + 1 + index * WORD_BITS)

    @property
    def cells(self) -> int:
        return 1 + self.capacity * WORD_BITS


class BinaryListIO(BinaryTokenIO):
    def clear_list(self, ref: IntListRef) -> None:
        self.bf.clear(ref.length_cell)
        for i in range(ref.capacity):
            self._clear_word(ref.item(i))

    def set_list_literal(self, ref: IntListRef, values: list[int]) -> None:
        if len(values) > ref.capacity:
            raise ValueError("list literal exceeds allocated list capacity")
        self.clear_list(ref)
        self.bf.set_const(ref.length_cell, len(values))
        for i, value in enumerate(values):
            self.set_u64(ref.item(i), value)

    def copy_list(self, dst: IntListRef, src: IntListRef) -> None:
        if dst.capacity < src.capacity:
            raise ValueError("destination list capacity is smaller than source")
        self.clear_list(dst)
        self.copy_cell(src.length_cell, dst.length_cell, self.s0)
        for i in range(src.capacity):
            self.copy64(dst.item(i), src.item(i))
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
        self.copy64(dst, ref.item(index))

    def set_const(self, ref: IntListRef, index: int, value: Int64Ref) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.copy64(ref.item(index), value)

    def get_dynamic(
        self,
        dst: Int64Ref,
        ref: IntListRef,
        index: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Load a non-negative runtime index; out-of-range currently yields 0."""
        bf = self.bf
        self._clear_word(dst)
        for i in range(ref.capacity):
            self.set_u64(workspace_word, i)
            self.eq64(match, index, workspace_word)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.copy64(dst, ref.item(i))
            bf.end_while(match)
        self._clear_scratch()

    def set_dynamic(
        self,
        ref: IntListRef,
        index: Int64Ref,
        value: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Store to a non-negative runtime index; out-of-range is a no-op."""
        bf = self.bf
        for i in range(ref.capacity):
            self.set_u64(workspace_word, i)
            self.eq64(match, index, workspace_word)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.copy64(ref.item(i), value)
            bf.end_while(match)
        self._clear_scratch()

    def append(
        self,
        ref: IntListRef,
        value: Int64Ref,
        length_copy: int,
        match: int,
    ) -> None:
        """Append when capacity remains; a full list currently ignores append."""
        bf = self.bf
        # Snapshot the original length.  Comparing against ref.length_cell
        # directly would cascade after the increment and fill every remaining
        # slot during one append call.
        self.copy_cell(ref.length_cell, length_copy, self.s0)
        for i in range(ref.capacity):
            self._eq_byte_const(match, length_copy, i)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.copy64(ref.item(i), value)
            bf.add_const(ref.length_cell, 1)
            bf.end_while(match)
        bf.clear(length_copy)
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
        """Implement ``list(map(int, input().split()))`` into fixed capacity.

        Up to ``capacity`` values are stored.  If a valid input line contains
        additional values, they are parsed and discarded until newline so the
        next input operation remains aligned to the next line.
        """
        bf = self.bf
        self.clear_list(ref)
        for c in (active, gate, has_token, end_line):
            bf.clear(c)
        bf.set_const(active, 1)

        for i in range(ref.capacity):
            self.copy_cell(active, gate, self.s0)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            self.read_s64_line_token(
                ref.item(i), has_token, end_line, workspace_base
            )
            self.copy_cell(has_token, gate, self.s0)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.add_const(ref.length_cell, 1)
            bf.end_while(gate)
            self.copy_cell(end_line, gate, self.s0)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.clear(active)
            bf.end_while(gate)
            bf.end_while(gate)

        # Capacity was exhausted before newline: drain whole remaining tokens.
        discarded = Int64Ref(workspace_base + WORD_BITS * 2 + 16)
        bf.begin_while(active)
        self.read_s64_line_token(discarded, has_token, end_line, workspace_base)
        self.copy_cell(end_line, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.clear(active)
        bf.end_while(gate)
        bf.end_while(active)
        self._clear_scratch()
