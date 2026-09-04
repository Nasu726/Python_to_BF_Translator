"""Range-safe compact indexing for string-backed character-list views.

``compiler_charconv`` establishes the source-level view semantics. This layer
removes a correctness-first constant-index shortcut that could touch hidden NUL
suffix slots when an index was below static capacity but beyond the current
logical input length. Until runtime IndexError propagation exists, every
out-of-range dynamic or constant index is consistently mapped to the reserved
selector 255 and therefore loads empty / stores nothing.

The currently supported character-list view permits element replacement but no
append/delete/insert. Its logical length is therefore immutable between
``list(input())`` assignments. We cache that length once in a persistent Quad64
word and reuse it for every later index check instead of rescanning up to 255
string cells on every access.

Character loads also use a capacity-one temporary StringRef. A list element is
known to be exactly one character, so allocating/clearing the full scalar
string capacity only bloated generated Brainfuck.
"""

from __future__ import annotations

import ast

from bfstrings import StringRef
from compiler_charconv import CompileError, _is_list_input
from compiler_charconv import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Character-view compiler with safe logical-length indexing."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
    ) -> None:
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
        )

        # These words are allocated before statement lowering begins. Every
        # compile_stmt mark therefore starts above them, so statement-local temp
        # rewinds cannot reclaim the cached metadata. compiler_layout later
        # places its PeakTempArena above this persistent prefix as well.
        self.char_list_lengths = {
            name: self._new_word(0) for name in sorted(self.char_list_names)
        }
        self._char_list_name_by_base = {
            self.strings[name].base: name for name in self.char_list_names
        }

    def _new_char_buffer(self) -> StringRef:
        ref = StringRef(self.temps.top, 1)
        self.temps.top += ref.cells
        return ref

    def _cached_char_list_length(self, ref):
        name = self._char_list_name_by_base.get(ref.base)
        if name is None:
            return None
        return self.char_list_lengths[name]

    def _char_list_runtime_index_byte(self, node: ast.AST, ref) -> int:
        length = self._cached_char_list_length(ref)
        if length is None:
            return super()._char_list_runtime_index_byte(node, ref)

        raw = self.compile_expr(node)
        index = self._new_word()
        self.backend.copy64(index, raw)
        zero = self._new_word(0)

        # Python negative indexing: normalize once by the cached logical length.
        negative = self.temps.cell()
        self.backend.slt64(negative, index, zero)
        self.bf.begin_while(negative)
        self.bf.add_const(negative, -1)
        summed = self._new_word()
        self.backend.add64(summed, index, length)
        self.backend.copy64(index, summed)
        self.bf.end_while(negative)

        valid = self.temps.cell()
        still_negative = self.temps.cell()
        self.backend.slt64(valid, index, length)
        self.backend.slt64(still_negative, index, zero)
        self.bf.begin_while(still_negative)
        self.bf.add_const(still_negative, -1)
        self.bf.clear(valid)
        self.bf.end_while(still_negative)

        out = self._index_word_to_byte(index)
        invalid = self.temps.cell()
        invalid_tmp = self.temps.cell()
        self._flag_not(invalid, valid, invalid_tmp)
        self.bf.begin_while(invalid)
        self.bf.add_const(invalid, -1)
        self.bf.set_const(out, 255)
        self.bf.end_while(invalid)
        return out

    def _load_char_list_subscript(self, node: ast.Subscript):
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]

        constant = self._constant_int(node.slice)
        if constant is not None and constant >= ref.capacity:
            raise self._error(node, "constant character-list index exceeds capacity")

        # Even a nonnegative constant below static capacity must be checked
        # against the runtime logical length. Example: index 10 into "abc"
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

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.char_list_names
            and _is_list_input(node.value)
        ):
            name = node.targets[0].id
            ref = self.strings[name]
            self.backend.read_line(ref, self.workspace_base)
            self.backend.string_length(
                self.char_list_lengths[name], ref, self.temps.cell()
            )
            return

        return super()._compile_stmt_inner(node)


__all__ = ["CompileError", "PythonToBFStream"]
