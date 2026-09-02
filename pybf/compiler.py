"""Current fixed-ABI compiler frontend.

This module is the final lowering layer used by the public ``pybf`` API.  It
keeps all ``input().split()`` forms line-scoped, including the older
``map(int, ...)`` path, and resolves the final fixed-runtime type layout.
"""

from __future__ import annotations

import ast

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bflists import IntListRef
from bfmemory import allocate_live_blocks
from bfstringlists import BinaryStringListIO, StringListRef
from bfstrings import StringRef
from transpiler import _is_map_int_input_split, infer_list_names
from transpiler_full import _LoopContext
from transpiler_inputs import (
    PythonToBFInputs,
    _SimpleAssignments,
    _is_input_split,
    _is_list_wrapper,
    _is_map_str_input_split,
    infer_split_string_names,
)
from transpiler_v2 import CompileError, _TempArena
from transpiler_v3 import infer_string_names


def _string_list_expr(node: ast.AST, names: set[str]) -> bool:
    if _is_input_split(node):
        return True
    if _is_list_wrapper(node, _is_input_split):
        return True
    if _is_list_wrapper(node, _is_map_str_input_split):
        return True
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.List):
        # An empty list has no element-type evidence.  Preserve the pre-existing
        # compiler convention and let the int-list inference own ``[]``.
        return bool(node.elts) and all(
            isinstance(x, ast.Constant) and isinstance(x.value, str)
            for x in node.elts
        )
    if isinstance(node, ast.IfExp):
        return _string_list_expr(node.body, names) and _string_list_expr(
            node.orelse, names
        )
    return False


def _infer_string_list_names(tree: ast.AST) -> set[str]:
    facts = _SimpleAssignments()
    facts.visit(tree)
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for target, value in facts.assignments:
            if _string_list_expr(value, result) and target not in result:
                result.add(target)
                changed = True
    return result


class PythonToBFCompiler(PythonToBFInputs):
    def __init__(
        self,
        tree: ast.AST,
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
        string_names = infer_split_string_names(tree, base_strings, string_list_names)
        int_list_names = infer_list_names(tree) - string_list_names

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

        int_list_cells = IntListRef(0, list_capacity).cells
        string_list_cells = StringListRef(0, list_capacity, string_capacity).cells
        sizes = {
            name: (
                string_list_cells
                if name in string_list_names
                else int_list_cells
                if name in int_list_names
                else string_capacity + 1
                if name in string_names
                else WORD_BITS
            )
            for name in all_names
        }
        blocks, static_top = allocate_live_blocks(tree, sizes)

        self.variables: dict[str, Int64Ref] = {
            name: Int64Ref(blocks[name].base)
            for name in all_names
            if name not in string_names
            and name not in int_list_names
            and name not in string_list_names
        }
        self.strings: dict[str, StringRef] = {
            name: StringRef(blocks[name].base, string_capacity)
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
        self.workspace_base = static_top
        self.scratch_base = self.workspace_base + self.SHARED_WORKSPACE_CELLS
        self.backend = BinaryStringListIO(self.bf, scratch_base=self.scratch_base)
        self.temps = _TempArena(self.scratch_base + Binary64Core.SCRATCH_CELLS)
        self._loop_stack: list[_LoopContext] = []

    def _finish_split_line(self, end_line: int) -> None:
        """Drain any unused bytes from the current logical input line."""
        active = self.temps.cell()
        gate = self.temps.cell()
        self.bf.set_const(active, 1)
        self.backend.copy_cell(end_line, gate, self.backend.s0)
        self.bf.begin_while(gate)
        self.bf.add_const(gate, -1)
        self.bf.clear(active)
        self.bf.end_while(gate)
        self.backend.drain_to_line_end(active, self.workspace_base)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        # The old frontend used read_s64_token(), which skips arbitrary
        # whitespace and could cross a newline.  Python input().split() is
        # explicitly one-line input, so use the line-token primitive here.
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and _is_map_int_input_split(node.value)
        ):
            targets = node.targets[0].elts
            if not targets or not all(isinstance(t, ast.Name) for t in targets):
                raise self._error(node, "map(int, input().split()) unpacking requires simple names")

            has_token = self.temps.cell()
            end_line = self.temps.cell()
            for target in targets:
                assert isinstance(target, ast.Name)
                if (
                    target.id in self.strings
                    or target.id in self.lists
                    or target.id in self.string_lists
                ):
                    raise self._error(target, "integer token requires an integer variable")
                self.backend.read_s64_line_token(
                    self._var(target), has_token, end_line, self.workspace_base
                )

            self._finish_split_line(end_line)
            return

        return super()._compile_stmt_inner(node)


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFCompiler(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
