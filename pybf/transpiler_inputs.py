"""Line-oriented split input for the fixed-ABI Python -> Brainfuck compiler.

This layer generalizes the previous int-only special cases.  ``input().split()``
is represented as a fixed-capacity ``list[str]`` on the Brainfuck tape, while
``map(int, ...)`` converts tokens into the existing signed int64 representation.
All tokenization happens in generated Brainfuck; Python is not involved after
compilation.
"""

from __future__ import annotations

import ast

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bflists import IntListRef
from bfmemory import allocate_live_blocks
from bfstringlists import BinaryStringListIO, StringListRef
from bfstrings import StringRef
from transpiler import (
    PythonToBF,
    _is_map_int_input_split,
    infer_list_names,
)
from transpiler_full import PythonToBFFull, _LoopContext
from transpiler_v2 import CompileError, _TempArena
from transpiler_v3 import _is_input_call, infer_string_names


def _is_input_split(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "split"
        and not node.args
        and not node.keywords
        and _is_input_call(node.func.value)
    )


def _is_map_str_input_split(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "map"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "str"
        and _is_input_split(node.args[1])
    )


def _is_list_wrapper(node: ast.AST, inner_predicate) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and not node.keywords
        and inner_predicate(node.args[0])
    )


def _is_string_list_source(node: ast.AST, string_list_names: set[str]) -> bool:
    if _is_input_split(node):
        return True
    if _is_list_wrapper(node, _is_input_split):
        return True
    if _is_list_wrapper(node, _is_map_str_input_split):
        return True
    if isinstance(node, ast.Name):
        return node.id in string_list_names
    if isinstance(node, ast.List):
        return all(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in node.elts)
    if isinstance(node, ast.IfExp):
        return _is_string_list_source(node.body, string_list_names) and _is_string_list_source(
            node.orelse, string_list_names
        )
    return False


class _SimpleAssignments(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assignments: list[tuple[str, ast.AST]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments.append((target.id, node.value))
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assignments.append((node.target.id, node.value))
        if node.value is not None:
            self.generic_visit(node.value)


def infer_string_list_names(tree: ast.AST) -> set[str]:
    facts = _SimpleAssignments()
    facts.visit(tree)
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for target, value in facts.assignments:
            if _is_string_list_source(value, result) and target not in result:
                result.add(target)
                changed = True
    return result


def _extended_string_expr(
    node: ast.AST,
    strings: set[str],
    string_lists: set[str],
) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Name):
        return node.id in strings
    if _is_input_call(node):
        return True
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in string_lists
    ):
        return True
    if isinstance(node, ast.IfExp):
        return _extended_string_expr(node.body, strings, string_lists) and _extended_string_expr(
            node.orelse, strings, string_lists
        )
    return False


def infer_split_string_names(
    tree: ast.AST,
    initial: set[str],
    string_lists: set[str],
) -> set[str]:
    """Extend scalar-string inference with split unpacking and string-list use."""
    strings = set(initial)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, (ast.Tuple, ast.List)) and (
                _is_input_split(node.value) or _is_map_str_input_split(node.value)
            ):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        strings.add(elt.id)
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in string_lists
            and isinstance(node.target, ast.Name)
        ):
            strings.add(node.target.id)

    facts = _SimpleAssignments()
    facts.visit(tree)
    changed = True
    while changed:
        changed = False
        for target, value in facts.assignments:
            if _extended_string_expr(value, strings, string_lists) and target not in strings:
                strings.add(target)
                changed = True
    return strings


class PythonToBFInputs(PythonToBFFull):
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
        string_list_names = infer_string_list_names(tree)
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

    # ------------------------------------------------------------------
    # type predicates / temporary values
    # ------------------------------------------------------------------
    def _expr_is_string_list(self, node: ast.AST) -> bool:
        return _is_string_list_source(node, self.string_list_names)

    def _expr_is_string(self, node: ast.AST) -> bool:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.string_lists
        ):
            return True
        return super()._expr_is_string(node)

    def _new_string_list(self) -> StringListRef:
        result = StringListRef(self.temps.top, self.list_capacity, self.string_capacity)
        self.temps.top += result.cells
        return result

    def _string_list_ref_from_expr(self, node: ast.AST) -> StringListRef:
        if isinstance(node, ast.Name) and node.id in self.string_lists:
            result = self._new_string_list()
            self.backend.copy_string_list(result, self.string_lists[node.id])
            return result

        if isinstance(node, ast.List):
            if len(node.elts) > self.list_capacity:
                raise self._error(node, "string list literal exceeds fixed list capacity")
            result = self._new_string_list()
            self.backend.clear_string_list(result)
            self.bf.set_const(result.length_cell, len(node.elts))
            for i, elt in enumerate(node.elts):
                if not self._expr_is_string(elt):
                    raise self._error(elt, "string lists currently contain strings only")
                value = self._eval_string(elt)
                self.backend.copy_string(result.item(i), value)
            return result

        if _is_input_split(node) or _is_list_wrapper(node, _is_input_split) or _is_list_wrapper(
            node, _is_map_str_input_split
        ):
            result = self._new_string_list()
            active, gate = self.temps.cell(), self.temps.cell()
            has_token, end_line = self.temps.cell(), self.temps.cell()
            self.backend.read_string_list_line(
                result,
                self.workspace_base,
                active,
                gate,
                has_token,
                end_line,
            )
            return result

        raise self._error(node, "unsupported string-list expression")

    # ------------------------------------------------------------------
    # string-list indexing
    # ------------------------------------------------------------------
    def _load_string_subscript(self, node: ast.Subscript) -> StringRef:
        assert isinstance(node.value, ast.Name)
        ref = self.string_lists[node.value.id]
        index, constant = self._list_index_word(node.slice, ref)  # length-cell compatible
        result = self._new_string()
        if constant is not None:
            if constant >= ref.capacity:
                raise self._error(node, "constant string-list index exceeds fixed capacity")
            self.backend.get_string_const(result, ref, constant)
        else:
            workspace_word = self._new_word()
            self.backend.get_string_dynamic(
                result, ref, index, workspace_word, self.temps.cell()
            )
        return result

    def _store_string_subscript(self, node: ast.Subscript, value: StringRef) -> None:
        assert isinstance(node.value, ast.Name)
        ref = self.string_lists[node.value.id]
        index, constant = self._list_index_word(node.slice, ref)
        if constant is not None:
            if constant >= ref.capacity:
                raise self._error(node, "constant string-list index exceeds fixed capacity")
            self.backend.set_string_const(ref, constant, value)
        else:
            workspace_word = self._new_word()
            self.backend.set_string_dynamic(
                ref, index, value, workspace_word, self.temps.cell()
            )

    def _eval_string(self, node: ast.AST) -> StringRef:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.string_lists
        ):
            return self._load_string_subscript(node)
        return super()._eval_string(node)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if isinstance(node, ast.Name) and node.id in self.string_lists:
            result = self._new_word()
            self.backend.list_length(result, self.string_lists[node.id], self.temps.cell())
            return result

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len" and len(node.args) == 1 and not node.keywords:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id in self.string_lists:
                    result = self._new_word()
                    self.backend.list_length(result, self.string_lists[arg.id], self.temps.cell())
                    return result

        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.string_lists
        ):
            value = self._load_string_subscript(node)
            result = self._new_word()
            self.backend.string_length(result, value, self.temps.cell())
            return result

        if isinstance(node, ast.BinOp) and (
            self._expr_is_string_list(node.left) or self._expr_is_string_list(node.right)
        ):
            raise self._error(node, "string-list arithmetic/concatenation is not lowered yet")

        return super().compile_expr(node)

    # ------------------------------------------------------------------
    # statements / input
    # ------------------------------------------------------------------
    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        # a, b = input().split() / map(str, input().split())
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and (_is_input_split(node.value) or _is_map_str_input_split(node.value))
        ):
            targets = node.targets[0].elts
            if not targets or not all(isinstance(t, ast.Name) for t in targets):
                raise self._error(node, "split unpacking requires simple names")
            has_token, end_line = self.temps.cell(), self.temps.cell()
            for target in targets:
                assert isinstance(target, ast.Name)
                if target.id not in self.strings:
                    raise self._error(target, "string token requires a string variable")
                self.backend.read_string_line_token(
                    self.strings[target.id], has_token, end_line, self.workspace_base
                )

            # Valid Python unpacking has exactly this many tokens.  If extra
            # input is present, consume it anyway so later input() calls remain
            # line-aligned instead of inheriting half of this line.
            active = self.temps.cell()
            gate = self.temps.cell()
            self.bf.set_const(active, 1)
            self.backend.copy_cell(end_line, gate, self.backend.s0)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.bf.clear(active)
            self.bf.end_while(gate)
            self.backend.drain_to_line_end(active, self.workspace_base)
            return

        if isinstance(node, ast.Assign):
            # S[i] = "value"
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in self.string_lists
            ):
                if not self._expr_is_string(node.value):
                    raise self._error(node.value, "string-list assignment requires a string")
                self._store_string_subscript(node.targets[0], self._eval_string(node.value))
                return

            # S = input().split(), S = list(input().split()), string-list copy.
            if all(isinstance(t, ast.Name) for t in node.targets) and self._expr_is_string_list(
                node.value
            ):
                value = self._string_list_ref_from_expr(node.value)
                for target in node.targets:
                    assert isinstance(target, ast.Name)
                    if target.id not in self.string_lists:
                        raise self._error(target, "cannot assign string list to another type")
                    self.backend.copy_string_list(self.string_lists[target.id], value)
                return

        # S.append("value")
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in self.string_lists
                and len(call.args) == 1
                and not call.keywords
            ):
                if not self._expr_is_string(call.args[0]):
                    raise self._error(call.args[0], "string-list append requires a string")
                value = self._eval_string(call.args[0])
                self.backend.append_string(
                    self.string_lists[call.func.value.id],
                    value,
                    self.temps.cell(),
                    self.temps.cell(),
                )
                return

        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in self.string_lists
        ):
            self._compile_for_string_list_control(node)
            return

        return super()._compile_stmt_inner(node)

    # ------------------------------------------------------------------
    # string-list loops / output
    # ------------------------------------------------------------------
    def _compile_for_string_list_control(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name) or node.target.id not in self.strings:
            raise self._error(node, "string-list iteration target must be a string variable")
        assert isinstance(node.iter, ast.Name)
        ref = self.string_lists[node.iter.id]
        target = self.strings[node.target.id]

        index = self._new_word(0)
        length = self._new_word()
        self.backend.list_length(length, ref, self.temps.cell())
        control = self.temps.cell()
        body_active = self.temps.cell()
        broke = self.temps.cell()
        self.bf.clear(broke)
        self.backend.slt64(control, index, length)

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        workspace_word = self._new_word()
        self.backend.get_string_dynamic(
            target, ref, index, workspace_word, self.temps.cell()
        )
        self.bf.set_const(body_active, 1)

        ctx = _LoopContext(body_active, broke)
        self._loop_stack.append(ctx)
        try:
            for stmt in node.body:
                self.compile_stmt(stmt)
        finally:
            self._loop_stack.pop()

        proceed = self.temps.cell()
        tmp_flag = self.temps.cell()
        self._flag_not(proceed, broke, tmp_flag)
        self.bf.begin_while(proceed)
        self.bf.add_const(proceed, -1)
        self.backend._inc64_inplace(index)
        self.backend.slt64(control, index, length)
        self.bf.end_while(proceed)
        self.bf.end_while(control)

        self._compile_guarded_else(node.orelse, broke)

    def _compile_print(self, call: ast.Call) -> None:
        sep = " "
        end = "\n"
        for kw in call.keywords:
            if (
                kw.arg not in ("sep", "end")
                or not isinstance(kw.value, ast.Constant)
                or not isinstance(kw.value.value, str)
            ):
                raise self._error(call, "print only supports constant-string sep= and end=")
            if kw.arg == "sep":
                sep = kw.value.value
            else:
                end = kw.value.value

        for arg_index, arg in enumerate(call.args):
            if arg_index:
                self._emit_string(sep)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self._emit_string(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in self.string_lists:
                self._print_string_list(self.string_lists[arg.id])
            elif self._expr_is_string(arg):
                value = self._eval_string(arg)
                self.backend.print_string(value, self.temps.cell())
            elif isinstance(arg, ast.Name) and arg.id in self.lists:
                self._print_list(self.lists[arg.id])
            else:
                value = self.compile_expr(arg)
                self.backend.print_s64(value, self.workspace_base)
        self._emit_string(end)

    def _print_string_list(self, ref: StringListRef) -> None:
        self._emit_string("[")
        index = self._new_word(0)
        length = self._new_word()
        self.backend.list_length(length, ref, self.temps.cell())
        control = self.temps.cell()
        self.backend.slt64(control, index, length)

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        nonzero_index = self.temps.cell()
        self.backend._is_nonzero64(nonzero_index, index)
        self.bf.begin_while(nonzero_index)
        self.bf.add_const(nonzero_index, -1)
        self._emit_string(", ")
        self.bf.end_while(nonzero_index)

        loaded = self._new_string()
        workspace_word = self._new_word()
        self.backend.get_string_dynamic(
            loaded, ref, index, workspace_word, self.temps.cell()
        )
        self._emit_string("'")
        self.backend.print_string(loaded, self.temps.cell())
        self._emit_string("'")
        self.backend._inc64_inplace(index)
        self.backend.slt64(control, index, length)
        self.bf.end_while(control)
        self._emit_string("]")


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFInputs(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
