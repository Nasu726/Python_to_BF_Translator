"""Quad-scalar allocation layer for the final compact compiler.

The AST/frontend stack remains unchanged. Only scalar physical storage and hot
numeric primitives change: scalar values use ``Quad64Ref`` so add/sub/compare
can execute one repeated Brainfuck lane body at runtime. Strings and lists
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
from bfquadinplace import add64_inplace, sub_double64
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


def _times_two_name(node: ast.AST) -> str | None:
    """Return the scalar name in exactly ``2*x`` or ``x*2``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    if (
        isinstance(node.left, ast.Constant)
        and type(node.left.value) is int
        and node.left.value == 2
        and isinstance(node.right, ast.Name)
    ):
        return node.right.id
    if (
        isinstance(node.right, ast.Constant)
        and type(node.right.value) is int
        and node.right.value == 2
        and isinstance(node.left, ast.Name)
    ):
        return node.left.id
    return None


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

        # Some higher frontend layers rewrite expressions only to teach the type
        # inference pass about representation aliases. Those rewrites must not
        # shorten real source-level lifetimes. When such a layer provides the
        # original AST, use it solely for interval/pinning analysis while keeping
        # the rewritten tree above for type and size inference.
        liveness_tree = getattr(self, "_liveness_tree_override", tree)
        blocks, static_top = allocate_live_blocks(liveness_tree, sizes)

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

    def compile_expr(self, node: ast.AST):
        # ``x - 2*y`` is common in partition/prefix-sum code.  Both operands
        # are pure scalar names here, so one fused lane pass is exactly
        # equivalent to materializing ``2*y`` and subtracting it, while avoiding
        # one complete temporary Quad word and its copy/shift traffic.
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Sub)
            and isinstance(node.left, ast.Name)
            and node.left.id in self.variables
        ):
            doubled = _times_two_name(node.right)
            if doubled is not None and doubled in self.variables:
                result = self._new_word()
                if sub_double64(
                    self.backend,
                    result,
                    self.variables[node.left.id],
                    self.variables[doubled],
                ):
                    return result

        return super().compile_expr(node)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        # Common reduction ``sum += values[i]``.  Do not lower the subscript as
        # a general expression: that would allocate a distant compiler temp for
        # the loaded Quad value and another temp for the bounds workspace.  The
        # two reserved Quad words are dead at this statement boundary, so use
        # them as the load destination and bounds workspace, then add directly
        # into the live scalar.
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.Add)
            and isinstance(node.target, ast.Name)
            and node.target.id in self.variables
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in self.lists
        ):
            dst = self._var(node.target)
            ref = self.lists[node.value.value.id]
            rhs = self.backend._qtmp(0)

            shadow = None
            if isinstance(node.value.slice, ast.Name):
                shadow = getattr(self, "_range_index_shadows", {}).get(
                    node.value.slice.id
                )

            if shadow is not None:
                index_byte, valid = shadow
                gate = self.temps.cell()
                self.backend.packed64.clear(ref.result(0))
                self.backend.copy_cell(valid, gate, self.backend.s0)
                self.bf.begin_while(gate)
                self.bf.add_const(gate, -1)
                self.backend._arm_walk_from_byte(ref, index_byte)
                self.backend._run_walk(ref, write=False)
                self.bf.end_while(gate)
                self.backend.copy64(rhs, ref.result(0))
                self.backend.packed64.clear(ref.result(0))
                self.bf.clear(gate)
                self.backend._clear_scratch()
            else:
                index, constant = self._list_index_word(node.value.slice, ref)
                if constant is not None:
                    if constant >= ref.capacity:
                        raise self._error(
                            node.value,
                            "constant list index exceeds configured capacity",
                        )
                    self.backend.get_const(rhs, ref, constant)
                else:
                    workspace_word = self.backend._qtmp(1)
                    match = self.temps.cell()
                    self.backend.get_dynamic(
                        rhs,
                        ref,
                        index,
                        workspace_word,
                        match,
                    )

            if not add64_inplace(self.backend, dst, rhs):
                raise RuntimeError("unexpected alias in fused list augmented assignment")
            return

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
