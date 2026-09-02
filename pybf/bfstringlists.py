"""Fixed-capacity ``list[str]`` support for the Brainfuck runtime.

Layout::

    [length: one byte][string slot 0]...[string slot N-1]

Each string slot is the normal fixed-capacity NUL-terminated ``StringRef``.
The representation is intentionally static: it is large, but it lets a plain
Brainfuck interpreter execute Python-style ``input().split()`` without help
from Python at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import Int64Ref
from bflists import BinaryListIO
from bfstrings import StringRef


@dataclass(frozen=True)
class StringListRef:
    base: int
    capacity: int
    string_capacity: int

    @property
    def length_cell(self) -> int:
        return self.base

    @property
    def stride(self) -> int:
        return self.string_capacity + 1

    def item(self, index: int) -> StringRef:
        if not 0 <= index < self.capacity:
            raise IndexError(index)
        return StringRef(self.base + 1 + index * self.stride, self.string_capacity)

    @property
    def cells(self) -> int:
        return 1 + self.capacity * self.stride


class BinaryStringListIO(BinaryListIO):
    def clear_string_list(self, ref: StringListRef) -> None:
        self.bf.clear(ref.length_cell)
        for i in range(ref.capacity):
            self.clear_string(ref.item(i))

    def copy_string_list(self, dst: StringListRef, src: StringListRef) -> None:
        if dst.capacity < src.capacity or dst.string_capacity < src.string_capacity:
            raise ValueError("destination string list is smaller than source")
        self.clear_string_list(dst)
        self.copy_cell(src.length_cell, dst.length_cell, self.s0)
        for i in range(src.capacity):
            self.copy_string(dst.item(i), src.item(i))
        self._clear_scratch()

    def get_string_const(self, dst: StringRef, ref: StringListRef, index: int) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.copy_string(dst, ref.item(index))

    def set_string_const(self, ref: StringListRef, index: int, value: StringRef) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.copy_string(ref.item(index), value)

    def get_string_dynamic(
        self,
        dst: StringRef,
        ref: StringListRef,
        index: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Load a runtime index; out-of-range currently yields an empty string."""
        bf = self.bf
        self.clear_string(dst)
        for i in range(ref.capacity):
            self.set_u64(workspace_word, i)
            self.eq64(match, index, workspace_word)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.copy_string(dst, ref.item(i))
            bf.end_while(match)
        self._clear_scratch()

    def set_string_dynamic(
        self,
        ref: StringListRef,
        index: Int64Ref,
        value: StringRef,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Store to a runtime index; out-of-range currently does nothing."""
        bf = self.bf
        for i in range(ref.capacity):
            self.set_u64(workspace_word, i)
            self.eq64(match, index, workspace_word)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.copy_string(ref.item(i), value)
            bf.end_while(match)
        self._clear_scratch()

    def append_string(
        self,
        ref: StringListRef,
        value: StringRef,
        length_copy: int,
        match: int,
    ) -> None:
        """Append when capacity remains; a full fixed list ignores append."""
        bf = self.bf
        self.copy_cell(ref.length_cell, length_copy, self.s0)
        for i in range(ref.capacity):
            self._eq_byte_const(match, length_copy, i)
            bf.begin_while(match)
            bf.add_const(match, -1)
            self.copy_string(ref.item(i), value)
            bf.add_const(ref.length_cell, 1)
            bf.end_while(match)
        bf.clear(length_copy)
        self._clear_scratch()

    def read_string_list_line(
        self,
        ref: StringListRef,
        workspace_base: int,
        active: int,
        gate: int,
        has_token: int,
        end_line: int,
    ) -> None:
        """Implement ``input().split()`` as a fixed-capacity ``list[str]``.

        Tokenization, whitespace handling, truncation, and newline consumption
        are all emitted as Brainfuck.  Excess tokens are drained so the next
        input operation starts at the next logical line.
        """
        bf = self.bf
        self.clear_string_list(ref)
        for c in (active, gate, has_token, end_line):
            bf.clear(c)
        bf.set_const(active, 1)

        for i in range(ref.capacity):
            self.copy_cell(active, gate, self.s0)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            self.read_string_line_token(
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

        # Capacity exhausted before newline: discard the rest of that line.
        self.drain_to_line_end(active, workspace_base)
        self._clear_scratch()
