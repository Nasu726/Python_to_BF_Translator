"""Quad-scalar allocation layer for the final compact compiler.

The AST/frontend stack remains unchanged.  Only scalar physical storage and hot
numeric primitives change: scalar values use ``Quad64Ref`` so add/sub/compare
can execute one repeated Brainfuck lane body at runtime.  Strings and lists
retain their established fixed representations.
"""

from __future__ import annotations

import ast

from bfcore import BFEmitter, Binary64Core
from bflists import IntListRef
from bfmemory import allocate_live_blocks
from bfpacked64 import PackedI64Ref
from bfquad import WORD_CELLS, Quad64Ref
from bfquadbackend import QuadBinaryStringListIO
from bfstringlists import StringListRef
from bfstrings import StringRef
from compiler import CompileError, _infer_int_list_names, _infer_string_list_names
from compiler_compact import PythonToBFCompact, _compact_char_names
from compiler_strings import _infer_scalar_string_loop_targets
from transpiler import _is_list_map_int_input_split
from transpiler_full import _LoopContext
from transpiler_inputs import infer_split_string_names
from transpiler_v2 import MASK64, _TempArena
from transpiler_v3 import infer_string_names


class PythonToBFQuad(PythonToBFCompact):
    """PythonToBFCompact with source-compact strided scalar words."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
    ) -> None:
        if not 1 <= string_capacity <= 255:
            raise ValueError("string_capacity must be between 1 and 255")
        if not 1 <= list_capacity <= 255:
            raise ValueError("list_capacity must be between 1 and 255")
        self.string_capacity = string_capacity
        self.list_capacity = list_capacity

        base_strings, all_names = infer_string_names(tree)
        string_list_names = _infer_string_list_names(tree)
        loop_chars = _infer_scalar_string_loop_targets(tree)
        string_names = infer_split_string_names(
            tree, base_strings | loop_chars, string_list_names
        )
        compact_chars = _compact_char_names(tree, string_names)
        int_list_names = _infer_int_list_names(tree, string_list_names)

        overlaps = (
            (string_names & int_list_names)
            | (string_names & string_list_names)
            | (int_list_names & string_list_names)
        )
        if overlaps:
            raise CompileError(
                "variables cannot change between scalar/list element types: "
                + ", ".join(sorted(overlaps))
            )

        self.string_names = string_names
        self.list_names = int_list_names
        self.string_list_names = string_list_names
        self.compact_char_names = compact_chars

        int_list_cells = IntListRef(0, list_capacity).cells
        string_list_cells = StringListRef(0, list_capacity, string_capacity).cells

        def scalar_string_cells(name: str) -> int:
            capacity = 1 if name in compact_chars else string_capacity
            return capacity + 1

        sizes = {
            name: (
                string_list_cells
                if name in string_list_names
                else int_list_cells
                if name in int_list_names
                else scalar_string_cells(name)
                if name in string_names
                else WORD_CELLS
            )
            for name in all_names
        }
        blocks, static_top = allocate_live_blocks(tree, sizes)

        self.variables: dict[str, Quad64Ref] = {
            name: Quad64Ref(blocks[name].base)
            for name in all_names
            if name not in string_names
            and name not in int_list_names
            and name not in string_list_names
        }
        self.strings: dict[str, StringRef] = {
            name: StringRef(
                blocks[name].base,
                1 if name in compact_chars else string_capacity,
            )
            for name in string_names
        }
        self.lists: dict[str, IntListRef] = {
            name: IntListRef(blocks[name].base, list_capacity)
            for name in int_list_names
        }
        self.string_lists: dict[str, StringListRef] = {
            name: StringListRef(blocks[name].base, list_capacity, string_capacity)
            for name in string_list_names
        }

        self.bf = BFEmitter()
        self.scratch_base = static_top
        self.workspace_base = self.scratch_base + Binary64Core.SCRATCH_CELLS
        self.backend = QuadBinaryStringListIO(
            self.bf, scratch_base=self.scratch_base
        )
        self.backend.set_quad_workspace(self.workspace_base)
        self.temps = _TempArena(
            self.workspace_base + self.SHARED_WORKSPACE_CELLS
        )
        self._loop_stack: list[_LoopContext] = []

    def _new_word(self, value: int | None = None) -> Quad64Ref:
        word = Quad64Ref(self.temps.top)
        self.temps.top += WORD_CELLS
        if value is not None:
            self.backend.set_u64(word, value & MASK64)
        return word

    def _packed_input_token(self) -> PackedI64Ref:
        return PackedI64Ref(self.workspace_base + 160)

    def _read_single_int_line(self, dst: Quad64Ref) -> None:
        """Read int(input()) through the packed decimal parser, then expand."""
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        line_open = self.temps.cell()
        token = self._packed_input_token()

        self.backend.packed64.clear(token)
        self.bf.set_const(line_open, 1)
        self.backend.read_packed_s64_line_token(
            token,
            has_token,
            end_line,
            self.workspace_base,
        )
        self.backend.copy64(dst, token)
        self._close_line_if_end(line_open, end_line)
        self.backend.drain_to_line_end(line_open, self.workspace_base)
        self.backend.packed64.clear(token)

    def _read_int_unpack_line(self, targets: list[ast.AST], node: ast.AST) -> None:
        """Read map(int, input().split()) without contiguous-word assumptions."""
        if not targets or not all(isinstance(t, ast.Name) for t in targets):
            raise self._error(
                node,
                "map(int, input().split()) unpacking requires simple names",
            )

        line_open = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        token = self._packed_input_token()
        self.bf.set_const(line_open, 1)
        self.backend.packed64.clear(token)

        for target in targets:
            assert isinstance(target, ast.Name)
            if (
                target.id in self.strings
                or target.id in self.lists
                or target.id in self.string_lists
            ):
                raise self._error(target, "integer token requires an integer variable")

            dst = self._var(target)
            self.backend.set_u64(dst, 0)
            gate = self.temps.cell()
            self.backend.copy_cell(line_open, gate, self.backend.s0)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.backend.read_packed_s64_line_token(
                token,
                has_token,
                end_line,
                self.workspace_base,
            )
            self.backend.copy64(dst, token)
            self.backend.packed64.clear(token)
            self._close_line_if_end(line_open, end_line)
            self.bf.end_while(gate)

        self.backend.drain_to_line_end(line_open, self.workspace_base)
        self.backend.packed64.clear(token)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.lists
            and _is_list_map_int_input_split(node.value)
        ):
            active = self.temps.cell()
            gate = self.temps.cell()
            has_token = self.temps.cell()
            end_line = self.temps.cell()
            self.backend.read_int_list_line(
                self.lists[node.targets[0].id],
                self.workspace_base,
                active,
                gate,
                has_token,
                end_line,
            )
            return

        return super()._compile_stmt_inner(node)


__all__ = ["PythonToBFQuad"]
