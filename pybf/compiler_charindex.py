"""Range-safe compact indexing for string-backed character-list views.

``compiler_charconv`` establishes the source-level view semantics.  This layer
removes a correctness-first constant-index shortcut that could touch hidden NUL
suffix slots when an index was below static capacity but beyond the current
logical input length.  Until runtime IndexError propagation exists, every
out-of-range dynamic or constant index is consistently mapped to the reserved
selector 255 and therefore loads empty / stores nothing.

Character loads also use a capacity-one temporary StringRef.  A list element is
known to be exactly one character, so allocating/clearing the full scalar
string capacity only bloated generated Brainfuck.
"""

from __future__ import annotations

import ast

from bfstrings import StringRef
from compiler_charconv import CompileError
from compiler_charconv import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Character-view compiler with safe logical-length indexing."""

    def _new_char_buffer(self) -> StringRef:
        ref = StringRef(self.temps.top, 1)
        self.temps.top += ref.cells
        return ref

    def _load_char_list_subscript(self, node: ast.Subscript):
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]

        constant = self._constant_int(node.slice)
        if constant is not None and constant >= ref.capacity:
            raise self._error(node, "constant character-list index exceeds capacity")

        # Even a nonnegative constant below static capacity must be checked
        # against the runtime logical length.  Example: index 10 into "abc"
        # must not expose the otherwise hidden cleared suffix cell 10.
        index_byte = self._char_list_runtime_index_byte(node.slice, ref)
        result = self._new_char_buffer()
        self._load_string_char_at(result, ref, index_byte)
        return result

    def _store_char_list_subscript(self, node: ast.Subscript, value) -> None:
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]

        constant = self._constant_int(node.slice)
        if constant is not None and constant >= ref.capacity:
            raise self._error(node, "constant character-list index exceeds capacity")

        index_byte = self._char_list_runtime_index_byte(node.slice, ref)
        match = self.temps.cell()
        for i in range(ref.capacity):
            self.backend._eq_byte_const(match, index_byte, i)
            self.bf.begin_while(match)
            self.bf.add_const(match, -1)
            self.backend.copy_cell(value.char(0), ref.char(i), self.backend.s0)
            self.bf.end_while(match)
        self.backend._clear_scratch()


__all__ = ["CompileError", "PythonToBFStream"]
