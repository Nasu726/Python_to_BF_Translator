"""Current fixed-ABI compiler frontend.

This module is the final lowering layer used by the public ``pybf`` API. It
combines strict one-line ``int(input())`` semantics with generic line-scoped
``input().split()`` lowering for integer and string tokens.
"""

from __future__ import annotations

import ast

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bflists import IntListRef
from bfmemory import allocate_live_blocks
from bfstringlists import BinaryStringListIO, StringListRef
from bfstrings import StringRef
from transpiler import (
    _is_list_map_int_input_split,
    _is_map_int_input_split,
    infer_list_names,
)
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


def _is_int_input_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "input"
        and not node.args[0].args
        and not node.args[0].keywords
    )


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
        # An empty list has no element-type evidence. Preserve the established
        # convention and let int-list inference own ``[]``.
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


def _int_list_expr(
    node: ast.AST,
    names: set[str],
    string_list_names: set[str],
) -> bool:
    """Classify int-list expressions including Python list repetition."""
    if _string_list_expr(node, string_list_names):
        return False
    if isinstance(node, ast.List):
        return True
    if isinstance(node, ast.Name):
        return node.id in names
    if _is_list_map_int_input_split(node):
        return True
    if isinstance(node, ast.IfExp):
        return _int_list_expr(node.body, names, string_list_names) and _int_list_expr(
            node.orelse, names, string_list_names
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_list = _int_list_expr(node.left, names, string_list_names)
        right_list = _int_list_expr(node.right, names, string_list_names)
        # Exactly one operand must be the list; the other is the repeat count.
        return left_list != right_list
    return False


def _infer_int_list_names(
    tree: ast.AST,
    string_list_names: set[str],
) -> set[str]:
    """Extend the existing list inference with ``list * int`` propagation."""
    result = infer_list_names(tree) - string_list_names
    facts = _SimpleAssignments()
    facts.visit(tree)
    changed = True
    while changed:
        changed = False
        for target, value in facts.assignments:
            if (
                _int_list_expr(value, result, string_list_names)
                and target not in result
            ):
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

    def _expr_is_string_list(self, node: ast.AST) -> bool:
        # Keep runtime expression classification consistent with the final
        # inference rule: [] has no string evidence and remains an int list.
        return _string_list_expr(node, self.string_list_names)

    def _expr_is_list(self, node: ast.AST) -> bool:
        return _int_list_expr(node, self.list_names, self.string_list_names)

    def _split_int_list_repeat(
        self, node: ast.AST
    ) -> tuple[ast.AST, ast.AST] | None:
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            return None
        left_list = self._expr_is_list(node.left)
        right_list = self._expr_is_list(node.right)
        if left_list and not right_list:
            return node.left, node.right
        if right_list and not left_list:
            return node.right, node.left
        return None

    def _repeat_int_list(self, src: IntListRef, count: Int64Ref) -> IntListRef:
        """Emit Python-style list repetition into the current fixed ABI.

        Negative counts naturally produce an empty list. The current list ABI
        still has a fixed capacity, so repetition stops when that capacity is
        reached; the later heap-list phase will remove this temporary ceiling.
        """
        result = self._new_list()
        self.backend.clear_list(result)

        repeat_index = self._new_word(0)
        output_index = self._new_word(0)
        source_length = self._new_word()
        capacity = self._new_word(self.list_capacity)
        self.backend.list_length(source_length, src, self.temps.cell())

        outer = self.temps.cell()
        full = self.temps.cell()
        self.backend.slt64(outer, repeat_index, count)
        self.backend.eq64(full, output_index, capacity)
        self.bf.begin_while(full)
        self.bf.add_const(full, -1)
        self.bf.clear(outer)
        self.bf.end_while(full)

        self.bf.begin_while(outer)
        self.bf.add_const(outer, -1)

        inner_index = self._new_word(0)
        inner = self.temps.cell()
        inner_full = self.temps.cell()
        loaded = self._new_word()
        workspace_word = self._new_word()
        match = self.temps.cell()
        length_copy = self.temps.cell()
        append_match = self.temps.cell()

        self.backend.slt64(inner, inner_index, source_length)
        self.backend.eq64(inner_full, output_index, capacity)
        self.bf.begin_while(inner_full)
        self.bf.add_const(inner_full, -1)
        self.bf.clear(inner)
        self.bf.end_while(inner_full)

        self.bf.begin_while(inner)
        self.bf.add_const(inner, -1)
        self.backend.get_dynamic(
            loaded, src, inner_index, workspace_word, match
        )
        self.backend.append(result, loaded, length_copy, append_match)
        self.backend._inc64_inplace(inner_index)
        self.backend._inc64_inplace(output_index)

        self.backend.slt64(inner, inner_index, source_length)
        self.backend.eq64(inner_full, output_index, capacity)
        self.bf.begin_while(inner_full)
        self.bf.add_const(inner_full, -1)
        self.bf.clear(inner)
        self.bf.end_while(inner_full)
        self.bf.end_while(inner)

        self.backend._inc64_inplace(repeat_index)
        self.backend.slt64(outer, repeat_index, count)
        self.backend.eq64(full, output_index, capacity)
        self.bf.begin_while(full)
        self.bf.add_const(full, -1)
        self.bf.clear(outer)
        self.bf.end_while(full)
        self.bf.end_while(outer)
        return result

    def _list_ref_from_expr(self, node: ast.AST) -> IntListRef:
        repeat = self._split_int_list_repeat(node)
        if repeat is not None:
            list_node, count_node = repeat
            src = super()._list_ref_from_expr(list_node)
            count = self.compile_expr(count_node)
            # Preserve evaluation of the repeat count even for an empty literal,
            # while avoiding a pointless runtime loop for the common [] * n case.
            if isinstance(list_node, ast.List) and not list_node.elts:
                result = self._new_list()
                self.backend.clear_list(result)
                return result
            return self._repeat_int_list(src, count)
        return super()._list_ref_from_expr(node)

    def _list_index_word(
        self, node: ast.AST, ref: IntListRef
    ) -> tuple[Int64Ref, int | None]:
        """Normalize runtime negative indices as ``len(ref) + index``."""
        raw, constant = super()._list_index_word(node, ref)
        if constant is not None:
            return raw, constant

        index = self._new_word()
        self.backend.copy64(index, raw)
        zero = self._new_word(0)
        negative = self.temps.cell()
        self.backend.slt64(negative, index, zero)
        self.bf.begin_while(negative)
        self.bf.add_const(negative, -1)
        length = self._new_word()
        normalized = self._new_word()
        self.backend.list_length(length, ref, self.temps.cell())
        self.backend.add64(normalized, index, length)
        self.backend.copy64(index, normalized)
        self.bf.end_while(negative)
        return index, None

    def _close_line_if_end(self, line_open: int, end_line: int) -> None:
        """Set ``line_open = 0`` when a token reader consumed newline/EOF."""
        gate = self.temps.cell()
        self.backend.copy_cell(end_line, gate, self.backend.s0)
        self.bf.begin_while(gate)
        self.bf.add_const(gate, -1)
        self.bf.clear(line_open)
        self.bf.end_while(gate)

    def _read_single_int_line(self, dst: Int64Ref) -> None:
        """Lower ``int(input())`` while consuming exactly one logical line."""
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        line_open = self.temps.cell()
        self.bf.set_const(line_open, 1)

        self.backend.read_s64_line_token(
            dst, has_token, end_line, self.workspace_base
        )
        self._close_line_if_end(line_open, end_line)

        # ``int(input())`` owns the entire line. If the first token ended on
        # horizontal whitespace, discard the remainder before later input().
        self.backend.drain_to_line_end(line_open, self.workspace_base)

    def _read_int_unpack_line(self, targets: list[ast.AST], node: ast.AST) -> None:
        """Lower fixed-arity ``map(int, input().split())`` without line bleed."""
        if not targets or not all(isinstance(t, ast.Name) for t in targets):
            raise self._error(node, "map(int, input().split()) unpacking requires simple names")

        line_open = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        self.bf.set_const(line_open, 1)

        for target in targets:
            assert isinstance(target, ast.Name)
            if (
                target.id in self.strings
                or target.id in self.lists
                or target.id in self.string_lists
            ):
                raise self._error(target, "integer token requires an integer variable")

            dst = self._var(target)
            # If the line ended before this target, leave the fixed-runtime
            # fallback value at zero and never consume the next line.
            self.backend._clear_word(dst)
            gate = self.temps.cell()
            self.backend.copy_cell(line_open, gate, self.backend.s0)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.backend.read_s64_line_token(
                dst, has_token, end_line, self.workspace_base
            )
            self._close_line_if_end(line_open, end_line)
            self.bf.end_while(gate)

        # Extra tokens would raise ValueError in CPython. Exceptions are not
        # implemented yet, so discard them while preserving the next line.
        self.backend.drain_to_line_end(line_open, self.workspace_base)

    def _read_string_unpack_line(self, targets: list[ast.AST], node: ast.AST) -> None:
        """Lower ``input().split()`` / ``map(str, ...)`` within one line."""
        if not targets or not all(isinstance(t, ast.Name) for t in targets):
            raise self._error(node, "split unpacking requires simple names")

        line_open = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        self.bf.set_const(line_open, 1)

        for target in targets:
            assert isinstance(target, ast.Name)
            if target.id not in self.strings:
                raise self._error(target, "string token requires a string variable")

            dst = self.strings[target.id]
            self.backend.clear_string(dst)
            gate = self.temps.cell()
            self.backend.copy_cell(line_open, gate, self.backend.s0)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.backend.read_string_line_token(
                dst, has_token, end_line, self.workspace_base
            )
            self._close_line_if_end(line_open, end_line)
            self.bf.end_while(gate)

        self.backend.drain_to_line_end(line_open, self.workspace_base)

    def compile_expr(self, node: ast.AST) -> Int64Ref:
        # Preserve PR #2 semantics after the split/string frontend is layered on
        # top: int(input()) must consume exactly one source-level input line.
        if _is_int_input_call(node):
            result = self._new_word()
            self._read_single_int_line(result)
            return result
        return super().compile_expr(node)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
        ):
            targets = node.targets[0].elts
            if _is_map_int_input_split(node.value):
                self._read_int_unpack_line(targets, node)
                return
            if _is_input_split(node.value) or _is_map_str_input_split(node.value):
                self._read_string_unpack_line(targets, node)
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


__all__ = ["CompileError", "PythonToBFCompiler", "compile_source"]
