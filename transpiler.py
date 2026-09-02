"""Current recommended Python -> Brainfuck compiler entrypoint.

This frontend targets common competitive-programming Python while preserving a
small, explicit runtime model: signed int64 scalars, fixed-capacity byte
strings, and fixed-capacity int64 lists.  Versioned v2/v3 frontends remain as
regression/reference layers.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bflists import BinaryListIO, IntListRef
from bfmemory import allocate_live_blocks
from transpiler_v2 import CompileError, _TempArena, clean_bf
from transpiler_v3 import PythonToBFV3, StringRef, infer_string_names


DEFAULT_LIST_CAPACITY = 32


def _is_map_int_input_split(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "map"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "int"
    ):
        return False
    split = node.args[1]
    return (
        isinstance(split, ast.Call)
        and isinstance(split.func, ast.Attribute)
        and split.func.attr == "split"
        and not split.args
        and not split.keywords
        and isinstance(split.func.value, ast.Call)
        and isinstance(split.func.value.func, ast.Name)
        and split.func.value.func.id == "input"
        and not split.func.value.args
        and not split.func.value.keywords
    )


def _is_list_map_int_input_split(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and not node.keywords
        and _is_map_int_input_split(node.args[0])
    )


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in node.elts:
            out.extend(_target_names(elt))
        return out
    return []


class _ListFacts(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assignments: list[tuple[str, ast.AST]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name in _target_names(target):
                self.assignments.append((name, node.value))
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            for name in _target_names(node.target):
                self.assignments.append((name, node.value))
            self.generic_visit(node.value)


def _list_expr(node: ast.AST, list_names: set[str]) -> bool:
    if isinstance(node, ast.List):
        return True
    if isinstance(node, ast.Name):
        return node.id in list_names
    if _is_list_map_int_input_split(node):
        return True
    if isinstance(node, ast.IfExp):
        return _list_expr(node.body, list_names) and _list_expr(node.orelse, list_names)
    return False


def infer_list_names(tree: ast.AST) -> set[str]:
    facts = _ListFacts()
    facts.visit(tree)
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for target, value in facts.assignments:
            if _list_expr(value, result) and target not in result:
                result.add(target)
                changed = True
    return result


class PythonToBF(PythonToBFV3):
    def __init__(
        self,
        tree: ast.AST,
        *,
        string_capacity: int = 128,
        list_capacity: int = DEFAULT_LIST_CAPACITY,
    ) -> None:
        if not 1 <= string_capacity <= 255:
            raise ValueError("string_capacity must be between 1 and 255")
        if not 1 <= list_capacity <= 255:
            raise ValueError("list_capacity must be between 1 and 255")
        self.string_capacity = string_capacity
        self.list_capacity = list_capacity

        string_names, all_names = infer_string_names(tree)
        list_names = infer_list_names(tree)
        overlap = string_names & list_names
        if overlap:
            raise CompileError(
                "variables cannot change between string and list types: "
                + ", ".join(sorted(overlap))
            )
        self.string_names = string_names
        self.list_names = list_names

        list_cells = IntListRef(0, list_capacity).cells
        sizes = {
            name: (
                list_cells
                if name in list_names
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
            if name not in string_names and name not in list_names
        }
        self.strings: dict[str, StringRef] = {
            name: StringRef(blocks[name].base, string_capacity)
            for name in string_names
        }
        self.lists: dict[str, IntListRef] = {
            name: IntListRef(blocks[name].base, list_capacity)
            for name in list_names
        }

        self.bf = BFEmitter()
        self.workspace_base = static_top
        self.scratch_base = self.workspace_base + self.SHARED_WORKSPACE_CELLS
        self.backend = BinaryListIO(self.bf, scratch_base=self.scratch_base)
        self.temps = _TempArena(self.scratch_base + Binary64Core.SCRATCH_CELLS)

    # ------------------------------------------------------------------
    # list helpers
    # ------------------------------------------------------------------
    def _new_list(self) -> IntListRef:
        result = IntListRef(self.temps.top, self.list_capacity)
        self.temps.top += result.cells
        return result

    def _expr_is_list(self, node: ast.AST) -> bool:
        return _list_expr(node, self.list_names)

    def _list_ref_from_expr(self, node: ast.AST) -> IntListRef:
        if isinstance(node, ast.Name) and node.id in self.lists:
            result = self._new_list()
            self.backend.copy_list(result, self.lists[node.id])
            return result
        if isinstance(node, ast.List):
            if len(node.elts) > self.list_capacity:
                raise self._error(node, "list literal exceeds configured list capacity")
            result = self._new_list()
            self.backend.clear_list(result)
            self.bf.set_const(result.length_cell, len(node.elts))
            for i, elt in enumerate(node.elts):
                if self._expr_is_string(elt) or self._expr_is_list(elt):
                    raise self._error(elt, "current lists contain int64 values only")
                value = self.compile_expr(elt)
                self.backend.copy64(result.item(i), value)
            return result
        if _is_list_map_int_input_split(node):
            result = self._new_list()
            active, gate = self.temps.cell(), self.temps.cell()
            has_token, end_line = self.temps.cell(), self.temps.cell()
            self.backend.read_int_list_line(
                result,
                self.workspace_base,
                active,
                gate,
                has_token,
                end_line,
            )
            return result
        raise self._error(node, "unsupported list expression")

    def _list_index_word(self, node: ast.AST, ref: IntListRef) -> tuple[Int64Ref, int | None]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            raw = node.value
            if raw >= 0:
                return self._new_word(raw), raw
            # Python negative indexing is relative to the runtime length.
            length = self._new_word()
            self.backend.list_length(length, ref, self.temps.cell())
            offset = self._new_word(raw)
            result = self._new_word()
            self.backend.add64(result, length, offset)
            return result, None
        return self.compile_expr(node), None

    def _load_subscript(self, node: ast.Subscript) -> Int64Ref:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.lists:
            raise self._error(node, "only int-list subscripting is currently supported")
        ref = self.lists[node.value.id]
        index, constant = self._list_index_word(node.slice, ref)
        result = self._new_word()
        if constant is not None:
            if constant >= ref.capacity:
                raise self._error(node, "constant list index exceeds configured capacity")
            self.backend.get_const(result, ref, constant)
        else:
            workspace_word = self._new_word()
            self.backend.get_dynamic(result, ref, index, workspace_word, self.temps.cell())
        return result

    def _store_subscript(self, node: ast.Subscript, value: Int64Ref) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.lists:
            raise self._error(node, "only int-list subscript assignment is supported")
        ref = self.lists[node.value.id]
        index, constant = self._list_index_word(node.slice, ref)
        if constant is not None:
            if constant >= ref.capacity:
                raise self._error(node, "constant list index exceeds configured capacity")
            self.backend.set_const(ref, constant, value)
        else:
            workspace_word = self._new_word()
            self.backend.set_dynamic(ref, index, value, workspace_word, self.temps.cell())

    # ------------------------------------------------------------------
    # expression extensions
    # ------------------------------------------------------------------
    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if isinstance(node, ast.Name) and node.id in self.lists:
            # As a condition, list truth is exactly length != 0.  Returning the
            # length word also makes bool(A) work through the inherited path.
            result = self._new_word()
            self.backend.list_length(result, self.lists[node.id], self.temps.cell())
            return result

        if isinstance(node, ast.Subscript):
            return self._load_subscript(node)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len" and len(node.args) == 1 and not node.keywords:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id in self.lists:
                    result = self._new_word()
                    self.backend.list_length(result, self.lists[arg.id], self.temps.cell())
                    return result

        if isinstance(node, ast.BinOp) and (
            self._expr_is_list(node.left) or self._expr_is_list(node.right)
        ):
            raise self._error(node, "list arithmetic/concatenation is not lowered yet")

        return super().compile_expr(node)

    # ------------------------------------------------------------------
    # statement extensions
    # ------------------------------------------------------------------
    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        # Common fixed-arity contest input: a, b = map(int, input().split())
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and _is_map_int_input_split(node.value)
        ):
            targets = node.targets[0].elts
            if not targets or not all(isinstance(t, ast.Name) for t in targets):
                raise self._error(node, "map(int, input().split()) unpacking requires simple names")
            for target in targets:
                assert isinstance(target, ast.Name)
                if target.id in self.strings or target.id in self.lists:
                    raise self._error(target, "integer token requires an integer variable")
                self.backend.read_s64_token(self._var(target), self.workspace_base)
            return

        if isinstance(node, ast.Assign):
            # A[i] = value
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                value = self.compile_expr(node.value)
                self._store_subscript(node.targets[0], value)
                return

            # A = [...], A = B, A = list(map(int, input().split()))
            if all(isinstance(t, ast.Name) for t in node.targets) and self._expr_is_list(node.value):
                value = self._list_ref_from_expr(node.value)
                for target in node.targets:
                    assert isinstance(target, ast.Name)
                    if target.id not in self.lists:
                        raise self._error(target, "cannot assign list to non-list variable")
                    self.backend.copy_list(self.lists[target.id], value)
                return

        # A.append(value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in self.lists
                and len(call.args) == 1
                and not call.keywords
            ):
                value = self.compile_expr(call.args[0])
                self.backend.append(
                    self.lists[call.func.value.id],
                    value,
                    self.temps.cell(),
                    self.temps.cell(),
                )
                return

        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id in self.lists:
            self._compile_for_list(node)
            return

        return super()._compile_stmt_inner(node)

    def _compile_for_list(self, node: ast.For) -> None:
        if node.orelse:
            raise self._error(node, "for ... else is not lowered yet")
        if not isinstance(node.target, ast.Name) or node.target.id in self.strings or node.target.id in self.lists:
            raise self._error(node, "list iteration target must be an integer variable")
        assert isinstance(node.iter, ast.Name)
        ref = self.lists[node.iter.id]
        target = self._var(node.target)

        index = self._new_word(0)
        length = self._new_word()
        self.backend.list_length(length, ref, self.temps.cell())
        control = self.temps.cell()
        self.backend.slt64(control, index, length)

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        loaded = self._new_word()
        workspace_word = self._new_word()
        self.backend.get_dynamic(loaded, ref, index, workspace_word, self.temps.cell())
        self.backend.copy64(target, loaded)
        for stmt in node.body:
            self.compile_stmt(stmt)
        self.backend._inc64_inplace(index)
        self.backend.slt64(control, index, length)
        self.bf.end_while(control)

    def _compile_print(self, call: ast.Call) -> None:
        # Reimplement the small dispatcher so list values render in familiar
        # Python form while scalar/string formatting remains inherited policy.
        sep = " "
        end = "\n"
        for kw in call.keywords:
            if kw.arg not in ("sep", "end") or not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
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
            elif isinstance(arg, ast.Name) and arg.id in self.strings:
                self.backend.print_string(self.strings[arg.id], self.temps.cell())
            elif isinstance(arg, ast.Name) and arg.id in self.lists:
                self._print_list(self.lists[arg.id])
            else:
                value = self.compile_expr(arg)
                self.backend.print_s64(value, self.workspace_base)
        self._emit_string(end)

    def _print_list(self, ref: IntListRef) -> None:
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

        loaded = self._new_word()
        workspace_word = self._new_word()
        self.backend.get_dynamic(loaded, ref, index, workspace_word, self.temps.cell())
        self.backend.print_s64(loaded, self.workspace_base)
        self.backend._inc64_inplace(index)
        self.backend.slt64(control, index, length)
        self.bf.end_while(control)
        self._emit_string("]")


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 128,
    list_capacity: int = DEFAULT_LIST_CAPACITY,
) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBF(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the supported Python subset to Brainfuck")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--string-capacity", type=int, default=128)
    parser.add_argument("--list-capacity", type=int, default=DEFAULT_LIST_CAPACITY)
    args = parser.parse_args()
    source = args.input.read_text(encoding="utf-8")
    try:
        code = compile_source(
            source,
            str(args.input),
            string_capacity=args.string_capacity,
            list_capacity=args.list_capacity,
        )
    except (CompileError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output or args.input.with_suffix(".bf")
    output.write_text(clean_bf(code), encoding="utf-8")
    print(f"wrote {output} ({len(code):,} BF commands)")


if __name__ == "__main__":
    main()
