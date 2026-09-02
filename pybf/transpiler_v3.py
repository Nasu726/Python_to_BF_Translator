"""Typed Python -> Brainfuck frontend built on the v2 arithmetic backend.

v3 keeps the fixed-width contest-oriented model, but removes the assumption
that every variable is an int64.  It adds fixed-capacity byte strings and a
conservative static tape allocator while preserving v2's tested arithmetic.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bfmemory import allocate_live_blocks
from bfstrings import BinaryStringIO, StringRef
from transpiler_v2 import CompileError, PythonToBFV2, _TempArena, clean_bf


MASK64 = (1 << WORD_BITS) - 1


def _is_input_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
        and not node.args
        and not node.keywords
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


class _TypeFacts(ast.NodeVisitor):
    """Small fixed-point type inference for int/bool versus byte-string."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.assignments: list[tuple[str, ast.AST]] = []
        self.for_targets: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name in _target_names(target):
                self.names.add(name)
                self.assignments.append((name, node.value))
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in _target_names(node.target):
            self.names.add(name)
            if node.value is not None:
                self.assignments.append((name, node.value))
        if node.value is not None:
            self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _target_names(node.target):
            self.names.add(name)
        self.generic_visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        for name in _target_names(node.target):
            self.names.add(name)
            self.for_targets.add(name)
        self.generic_visit(node.iter)
        for stmt in node.body + node.orelse:
            self.visit(stmt)


def _string_expr(node: ast.AST, string_names: set[str]) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.Name):
        return node.id in string_names
    if _is_input_call(node):
        return True
    if isinstance(node, ast.IfExp):
        return _string_expr(node.body, string_names) and _string_expr(node.orelse, string_names)
    return False


def infer_string_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    facts = _TypeFacts()
    facts.visit(tree)
    strings: set[str] = set()

    changed = True
    while changed:
        changed = False
        for target, value in facts.assignments:
            if _string_expr(value, strings) and target not in strings:
                strings.add(target)
                changed = True

    # Loop targets are integer counters in the currently supported range form.
    strings.difference_update(facts.for_targets)
    return strings, facts.names


class PythonToBFV3(PythonToBFV2):
    def __init__(self, tree: ast.AST, *, string_capacity: int = 128) -> None:
        if not 1 <= string_capacity <= 255:
            raise ValueError("string_capacity must be between 1 and 255")
        self.string_capacity = string_capacity
        string_names, all_names = infer_string_names(tree)
        self.string_names = string_names

        sizes = {
            name: (string_capacity + 1 if name in string_names else WORD_BITS)
            for name in all_names
        }
        blocks, static_top = allocate_live_blocks(tree, sizes)

        self.variables: dict[str, Int64Ref] = {
            name: Int64Ref(blocks[name].base)
            for name in all_names
            if name not in string_names
        }
        self.strings: dict[str, StringRef] = {
            name: StringRef(blocks[name].base, string_capacity)
            for name in string_names
        }

        self.bf = BFEmitter()
        self.workspace_base = static_top
        self.scratch_base = self.workspace_base + self.SHARED_WORKSPACE_CELLS
        self.backend = BinaryStringIO(self.bf, scratch_base=self.scratch_base)
        self.temps = _TempArena(self.scratch_base + Binary64Core.SCRATCH_CELLS)

    # ------------------------------------------------------------------
    # typed helpers
    # ------------------------------------------------------------------
    def _string_ref(self, node: ast.Name) -> StringRef:
        try:
            return self.strings[node.id]
        except KeyError as exc:
            raise self._error(node, f"{node.id!r} is not a string variable") from exc

    def _new_string(self) -> StringRef:
        result = StringRef(self.temps.top, self.string_capacity)
        self.temps.top += result.cells
        return result

    def _copy_string_new(self, src: StringRef) -> StringRef:
        dst = self._new_string()
        self.backend.copy_string(dst, src)
        return dst

    def _eval_string(self, node: ast.AST) -> StringRef:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            dst = self._new_string()
            try:
                self.backend.set_string_literal(dst, node.value)
            except ValueError as exc:
                raise self._error(node, str(exc)) from exc
            return dst
        if isinstance(node, ast.Name) and node.id in self.strings:
            return self._copy_string_new(self.strings[node.id])
        if _is_input_call(node):
            dst = self._new_string()
            self.backend.read_line(dst, self.workspace_base)
            return dst
        raise self._error(node, "unsupported string expression (concatenation is not lowered yet)")

    def _expr_is_string(self, node: ast.AST) -> bool:
        return _string_expr(node, self.string_names)

    def _emit_compare(self, op: ast.cmpop, out: int, left: Int64Ref, right: Int64Ref) -> None:
        if isinstance(op, ast.Eq):
            self.backend.eq64(out, left, right)
        elif isinstance(op, ast.NotEq):
            self.backend.eq64(out, left, right)
            self.backend._toggle_bit(out, self.backend.s0)
            self.backend._clear_scratch()
        elif isinstance(op, ast.Lt):
            self.backend.slt64(out, left, right)
        elif isinstance(op, ast.LtE):
            self.backend.sle64(out, left, right)
        elif isinstance(op, ast.Gt):
            self.backend.sgt64(out, left, right)
        elif isinstance(op, ast.GtE):
            self.backend.sge64(out, left, right)
        else:
            raise CompileError(f"unsupported comparison {type(op).__name__}")

    # ------------------------------------------------------------------
    # expression extensions
    # ------------------------------------------------------------------
    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if isinstance(node, ast.Name) and node.id in self.strings:
            # In scalar/condition context a string contributes its truth value
            # through its length.  Operations that would incorrectly treat a
            # string as an integer are rejected below before recursion.
            result = self._new_word()
            control = self.temps.cell()
            self.backend.string_length(result, self.strings[node.id], control)
            return result

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "len" and len(node.args) == 1 and not node.keywords:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id in self.strings:
                    result = self._new_word()
                    self.backend.string_length(result, self.strings[arg.id], self.temps.cell())
                    return result
            if name == "abs" and len(node.args) == 1 and not node.keywords:
                result = self._copy_new(self.compile_expr(node.args[0]))
                sign = self.temps.cell()
                self.backend.copy_cell(result.bit(WORD_BITS - 1), sign, self.backend.s0)
                self.bf.begin_while(sign)
                self.bf.add_const(sign, -1)
                self.backend._neg64_inplace(result)
                self.bf.end_while(sign)
                return result
            if name == "bool" and len(node.args) == 1 and not node.keywords:
                value = self.compile_expr(node.args[0])
                result = self._new_word(0)
                self.backend._is_nonzero64(result.bit(0), value)
                return result
            if name in ("min", "max") and len(node.args) == 2 and not node.keywords:
                a = self.compile_expr(node.args[0])
                b_ref = self.compile_expr(node.args[1])
                result = self._copy_new(a)
                choose_b = self.temps.cell()
                if name == "min":
                    self.backend.sgt64(choose_b, a, b_ref)
                else:
                    self.backend.slt64(choose_b, a, b_ref)
                self.bf.begin_while(choose_b)
                self.bf.add_const(choose_b, -1)
                self.backend.copy64(result, b_ref)
                self.bf.end_while(choose_b)
                return result

        if isinstance(node, ast.BinOp):
            if self._expr_is_string(node.left) or self._expr_is_string(node.right):
                raise self._error(node, "string arithmetic/concatenation is not lowered yet")
            if isinstance(node.op, ast.RShift):
                if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                    raise self._error(node, "v3 currently requires a constant shift amount")
                amount = node.right.value
                if amount < 0:
                    raise self._error(node, "negative shift count")
                left = self.compile_expr(node.left)
                result = self._new_word()
                self.backend.shr_const(result, left, amount)
                sign = self.temps.cell()
                self.backend.copy_cell(left.bit(WORD_BITS - 1), sign, self.backend.s0)
                self.bf.begin_while(sign)
                self.bf.add_const(sign, -1)
                first = max(0, WORD_BITS - amount)
                for i in range(first, WORD_BITS):
                    self.bf.set_const(result.bit(i), 1)
                self.bf.end_while(sign)
                return result

        if isinstance(node, ast.UnaryOp) and self._expr_is_string(node.operand):
            if not isinstance(node.op, ast.Not):
                raise self._error(node, "only 'not' is valid for strings in scalar context")

        if isinstance(node, ast.BoolOp) and any(self._expr_is_string(v) for v in node.values):
            raise self._error(node, "string-valued and/or is not lowered yet")

        if isinstance(node, ast.Compare):
            any_string = self._expr_is_string(node.left) or any(
                self._expr_is_string(x) for x in node.comparators
            )
            if any_string:
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise self._error(node, "chained string comparisons are not lowered yet")
                if not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                    raise self._error(node, "only == and != are supported for strings")
                a = self._eval_string(node.left)
                b_ref = self._eval_string(node.comparators[0])
                result = self._new_word(0)
                self.backend.eq_string(result.bit(0), a, b_ref)
                if isinstance(node.ops[0], ast.NotEq):
                    self.backend._toggle_bit(result.bit(0), self.backend.s0)
                    self.backend._clear_scratch()
                return result

            if len(node.ops) > 1:
                # Python evaluates chained comparisons left-to-right and stops
                # before later comparator expressions after the first failure.
                result = self._new_word(1)
                active = self.temps.cell()
                self.bf.set_const(active, 1)
                left = self.compile_expr(node.left)
                for op, rhs_node in zip(node.ops, node.comparators):
                    gate = self.temps.cell()
                    self.backend.copy_cell(active, gate, self.backend.s0)
                    self.bf.begin_while(gate)
                    self.bf.add_const(gate, -1)
                    right = self.compile_expr(rhs_node)
                    pair = self.temps.cell()
                    self._emit_compare(op, pair, left, right)
                    self.backend.copy_cell(pair, active, self.backend.s0)
                    fail = self.temps.cell()
                    self.backend.copy_cell(pair, fail, self.backend.s0)
                    self.backend._toggle_bit(fail, self.backend.s0)
                    self.bf.begin_while(fail)
                    self.bf.add_const(fail, -1)
                    self._clear_word(result)
                    self.bf.end_while(fail)
                    left = right
                    self.bf.end_while(gate)
                return result

        return super().compile_expr(node)

    # ------------------------------------------------------------------
    # statement / print extensions
    # ------------------------------------------------------------------
    def _assign_string_to(self, target: ast.Name, value: StringRef) -> None:
        if target.id not in self.strings:
            raise self._error(target, f"cannot assign string to integer variable {target.id!r}")
        self.backend.copy_string(self.strings[target.id], value)

    def _assign_scalar_to(self, target: ast.Name, value: Int64Ref) -> None:
        if target.id in self.strings:
            raise self._error(target, f"cannot assign integer to string variable {target.id!r}")
        dst = self._var(target)
        if dst.base != value.base:
            self.backend.copy64(dst, value)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            # Tuple/list unpacking evaluates all RHS expressions before stores.
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
            ):
                targets = node.targets[0].elts
                values = node.value.elts
                if len(targets) != len(values) or not all(isinstance(t, ast.Name) for t in targets):
                    raise self._error(node, "unpacking requires equally-sized simple-name tuples/lists")
                evaluated: list[tuple[str, object]] = []
                for value_node in values:
                    if self._expr_is_string(value_node):
                        evaluated.append(("str", self._eval_string(value_node)))
                    else:
                        evaluated.append(("int", self._copy_new(self.compile_expr(value_node))))
                for target, (kind, value) in zip(targets, evaluated):
                    assert isinstance(target, ast.Name)
                    if kind == "str":
                        self._assign_string_to(target, value)  # type: ignore[arg-type]
                    else:
                        self._assign_scalar_to(target, value)  # type: ignore[arg-type]
                return

            if all(isinstance(t, ast.Name) for t in node.targets):
                if self._expr_is_string(node.value):
                    value = self._eval_string(node.value)
                    for target in node.targets:
                        assert isinstance(target, ast.Name)
                        self._assign_string_to(target, value)
                else:
                    value = self.compile_expr(node.value)
                    for target in node.targets:
                        assert isinstance(target, ast.Name)
                        self._assign_scalar_to(target, value)
                return

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            fake = ast.Assign(targets=[node.target], value=node.value)
            ast.copy_location(fake, node)
            self._compile_stmt_inner(fake)
            return

        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id in self.strings:
                raise self._error(node, "string augmented assignment is not lowered yet")
            dst = self._var(node.target)
            rhs = self.compile_expr(node.value)
            tmp = self._new_word()
            op = node.op
            if isinstance(op, ast.Add):
                self.backend.add64(tmp, dst, rhs)
            elif isinstance(op, ast.Sub):
                self.backend.sub64(tmp, dst, rhs)
            elif isinstance(op, ast.Mult):
                self.backend.mul64(tmp, dst, rhs, self.workspace_base)
            elif isinstance(op, (ast.FloorDiv, ast.Mod)):
                q, r = self._new_word(), self._new_word()
                self.backend.sdivmod64(q, r, dst, rhs, self.workspace_base)
                tmp = q if isinstance(op, ast.FloorDiv) else r
            elif isinstance(op, ast.BitAnd):
                self.backend.and64(tmp, dst, rhs)
            elif isinstance(op, ast.BitOr):
                self.backend.or64(tmp, dst, rhs)
            elif isinstance(op, ast.BitXor):
                self.backend.xor64(tmp, dst, rhs)
            else:
                return super()._compile_stmt_inner(node)
            self.backend.copy64(dst, tmp)
            return

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
                self._compile_print(node.value)
                return
            if _is_input_call(node.value):
                discarded = self._new_string()
                self.backend.read_line(discarded, self.workspace_base)
                return

        return super()._compile_stmt_inner(node)

    def _compile_print(self, call: ast.Call) -> None:
        sep = " "
        end = "\n"
        for kw in call.keywords:
            if kw.arg not in ("sep", "end") or not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                raise self._error(call, "print only supports constant-string sep= and end=")
            if kw.arg == "sep":
                sep = kw.value.value
            else:
                end = kw.value.value

        for index, arg in enumerate(call.args):
            if index:
                self._emit_string(sep)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self._emit_string(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in self.strings:
                self.backend.print_string(self.strings[arg.id], self.temps.cell())
            else:
                value = self.compile_expr(arg)
                self.backend.print_s64(value, self.workspace_base)
        self._emit_string(end)


def compile_source(source: str, filename: str = "<string>", *, string_capacity: int = 128) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFV3(tree, string_capacity=string_capacity).compile_module(tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the richer v3 Python subset to Brainfuck")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--string-capacity", type=int, default=128)
    args = parser.parse_args()
    source = args.input.read_text(encoding="utf-8")
    code = compile_source(source, str(args.input), string_capacity=args.string_capacity)
    output = args.output or args.input.with_suffix(".bf")
    output.write_text(clean_bf(code), encoding="utf-8")
    print(f"wrote {output} ({len(code)} BF commands)")


if __name__ == "__main__":
    main()
