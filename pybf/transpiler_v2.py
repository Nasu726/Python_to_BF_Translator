"""Structured Python -> Brainfuck transpiler (v2).

This frontend intentionally targets a contest-oriented, fixed-width subset of
Python rather than pretending to implement CPython.  Every scalar variable is
a signed two's-complement int64.  The backend is separated from AST lowering,
so arithmetic correctness and Python control-flow lowering can be tested
independently.

Currently lowered:
* int/bool constants and scalar variables
* assignment / augmented assignment
* +, -, *, //, %, constant non-negative **
* bitwise &, |, ^, ~ and constant shifts
* ==, !=, <, <=, >, >= (signed comparisons)
* Python-value short-circuit ``and`` / ``or`` and boolean ``not``
* int(input()) for newline-terminated decimal input
* print(int expressions, "literal strings", ...)
* if/else, while, for ... in range(...) with constant nonzero step

Integer arithmetic wraps modulo 2**64.  Comparisons, //, %, decimal input and
decimal output interpret those bits as signed two's-complement values.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bfio import Binary64IO


MASK64 = (1 << WORD_BITS) - 1


class CompileError(Exception):
    pass


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


@dataclass
class _TempArena:
    top: int

    def mark(self) -> int:
        return self.top

    def rewind(self, mark: int) -> None:
        self.top = mark

    def cell(self) -> int:
        result = self.top
        self.top += 1
        return result

    def word(self) -> Int64Ref:
        result = Int64Ref(self.top)
        self.top += WORD_BITS
        return result


class PythonToBFV2:
    # Signed divmod is currently the largest shared operation workspace.
    SHARED_WORKSPACE_CELLS = Binary64IO.SDIVMOD_WORKSPACE_CELLS

    def __init__(self, tree: ast.AST) -> None:
        names = _AssignedNames()
        names.visit(tree)

        self.variables: dict[str, Int64Ref] = {}
        static_top = 0
        for name in sorted(names.names):
            self.variables[name] = Int64Ref(static_top)
            static_top += WORD_BITS

        self.bf = BFEmitter()
        self.workspace_base = static_top
        self.scratch_base = self.workspace_base + self.SHARED_WORKSPACE_CELLS
        self.backend = Binary64IO(self.bf, scratch_base=self.scratch_base)
        self.temps = _TempArena(self.scratch_base + Binary64Core.SCRATCH_CELLS)

    # ------------------------------------------------------------------
    # generic helpers
    # ------------------------------------------------------------------
    def _error(self, node: ast.AST, message: str) -> CompileError:
        line = getattr(node, 'lineno', '?')
        return CompileError(f'line {line}: {message}')

    def _var(self, node: ast.Name) -> Int64Ref:
        try:
            return self.variables[node.id]
        except KeyError as exc:
            raise self._error(node, f'unknown scalar variable {node.id!r}') from exc

    def _new_word(self, value: int | None = None) -> Int64Ref:
        word = self.temps.word()
        if value is not None:
            self.backend.set_u64(word, value & MASK64)
        return word

    def _copy_new(self, src: Int64Ref) -> Int64Ref:
        dst = self._new_word()
        self.backend.copy64(dst, src)
        return dst

    def _clear_word(self, word: Int64Ref) -> None:
        for i in range(WORD_BITS):
            self.bf.clear(word.bit(i))

    def _bool_word_from_cell(self, cell: int) -> Int64Ref:
        result = self._new_word(0)
        self.backend.copy_cell(cell, result.bit(0), self.backend.s0)
        return result

    def _truth_cell(self, value: Int64Ref) -> int:
        result = self.temps.cell()
        self.backend._is_nonzero64(result, value)
        return result

    def _emit_string(self, text: str) -> None:
        # The shared workspace is dead between backend calls; its first byte is
        # a convenient character output cell.
        cell = self.workspace_base
        for ch in text:
            self.backend.print_char(ord(ch), cell)

    # ------------------------------------------------------------------
    # expression lowering
    # ------------------------------------------------------------------
    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return self._new_word(int(node.value))
            if isinstance(node.value, int):
                return self._new_word(node.value)
            raise self._error(node, f'unsupported scalar constant {node.value!r}')

        if isinstance(node, ast.Name):
            return self._var(node)

        if isinstance(node, ast.UnaryOp):
            value = self.compile_expr(node.operand)
            if isinstance(node.op, ast.UAdd):
                return self._copy_new(value)
            if isinstance(node.op, ast.USub):
                result = self._copy_new(value)
                self.backend._neg64_inplace(result)
                return result
            if isinstance(node.op, ast.Invert):
                result = self._new_word()
                self.backend.not64(result, value)
                return result
            if isinstance(node.op, ast.Not):
                result = self._new_word(0)
                self.backend._is_nonzero64(result.bit(0), value)
                self.backend._toggle_bit(result.bit(0), self.backend.s0)
                self.backend._clear_scratch()
                return result
            raise self._error(node, f'unsupported unary operator {type(node.op).__name__}')

        if isinstance(node, ast.BoolOp):
            if not node.values:
                raise self._error(node, 'empty boolean expression')
            # Python and/or return one of their operands, not a coerced bool.
            result = self._copy_new(self.compile_expr(node.values[0]))
            is_and = isinstance(node.op, ast.And)
            is_or = isinstance(node.op, ast.Or)
            if not (is_and or is_or):
                raise self._error(node, f'unsupported BoolOp {type(node.op).__name__}')

            for rhs_node in node.values[1:]:
                control = self._truth_cell(result)
                if is_or:
                    self.backend._toggle_bit(control, self.backend.s0)
                    self.backend._clear_scratch()
                self.bf.begin_while(control)
                self.bf.add_const(control, -1)
                rhs = self.compile_expr(rhs_node)
                self.backend.copy64(result, rhs)
                self.bf.end_while(control)
            return result

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return self._compile_constant_pow(node)

            left = self.compile_expr(node.left)
            right = self.compile_expr(node.right)

            if isinstance(node.op, ast.Add):
                result = self._new_word()
                self.backend.add64(result, left, right)
                return result
            if isinstance(node.op, ast.Sub):
                result = self._new_word()
                self.backend.sub64(result, left, right)
                return result
            if isinstance(node.op, ast.Mult):
                result = self._new_word()
                self.backend.mul64(result, left, right, self.workspace_base)
                return result
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                quotient = self._new_word()
                remainder = self._new_word()
                self.backend.sdivmod64(
                    quotient, remainder, left, right, self.workspace_base
                )
                return quotient if isinstance(node.op, ast.FloorDiv) else remainder
            if isinstance(node.op, ast.BitAnd):
                result = self._new_word()
                self.backend.and64(result, left, right)
                return result
            if isinstance(node.op, ast.BitOr):
                result = self._new_word()
                self.backend.or64(result, left, right)
                return result
            if isinstance(node.op, ast.BitXor):
                result = self._new_word()
                self.backend.xor64(result, left, right)
                return result
            if isinstance(node.op, (ast.LShift, ast.RShift)):
                if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                    raise self._error(node, 'v2 currently requires a constant shift amount')
                amount = node.right.value
                if amount < 0:
                    raise self._error(node, 'negative shift count')
                result = self._new_word()
                if isinstance(node.op, ast.LShift):
                    self.backend.shl_const(result, left, amount)
                else:
                    # Python signed >> is arithmetic.  The current primitive is
                    # logical, so only nonnegative/unsigned behavior is exposed
                    # until arithmetic-shift lowering is added.
                    self.backend.shr_const(result, left, amount)
                return result
            raise self._error(node, f'unsupported binary operator {type(node.op).__name__}')

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise self._error(node, 'chained comparisons are not lowered yet')
            left = self.compile_expr(node.left)
            right = self.compile_expr(node.comparators[0])
            result = self._new_word(0)
            out = result.bit(0)
            op = node.ops[0]
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
                raise self._error(node, f'unsupported comparison {type(op).__name__}')
            return result

        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == 'int'
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == 'input'
                and not node.args[0].args
                and not node.args[0].keywords
            ):
                result = self._new_word()
                self.backend.read_s64(result, self.workspace_base)
                return result
            raise self._error(node, 'unsupported call in scalar expression')

        if isinstance(node, ast.IfExp):
            result = self._new_word()
            cond = self.compile_expr(node.test)
            control = self._truth_cell(cond)
            else_control = self.temps.cell()
            self.bf.set_const(else_control, 1)

            self.bf.begin_while(control)
            self.bf.add_const(control, -1)
            self.bf.clear(else_control)
            yes = self.compile_expr(node.body)
            self.backend.copy64(result, yes)
            self.bf.end_while(control)

            self.bf.begin_while(else_control)
            self.bf.add_const(else_control, -1)
            no = self.compile_expr(node.orelse)
            self.backend.copy64(result, no)
            self.bf.end_while(else_control)
            return result

        raise self._error(node, f'unsupported expression {type(node).__name__}')

    def _compile_constant_pow(self, node: ast.BinOp) -> Int64Ref:
        if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
            raise self._error(node, 'v2 currently requires a constant integer exponent')
        exponent = node.right.value
        if exponent < 0:
            raise self._error(node, 'negative exponent would require non-integer TrueDiv')

        base = self._copy_new(self.compile_expr(node.left))
        result = self._new_word(1)
        e = exponent
        while e:
            if e & 1:
                product = self._new_word()
                self.backend.mul64(product, result, base, self.workspace_base)
                result = product
            e >>= 1
            if e:
                squared = self._new_word()
                self.backend.mul64(squared, base, base, self.workspace_base)
                base = squared
        return result

    # ------------------------------------------------------------------
    # statements / control flow
    # ------------------------------------------------------------------
    def compile_stmt(self, node: ast.stmt) -> None:
        mark = self.temps.mark()
        try:
            self._compile_stmt_inner(node)
        finally:
            self.temps.rewind(mark)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise self._error(node, 'only simple scalar assignment is supported')
            dst = self._var(node.targets[0])
            value = self.compile_expr(node.value)
            if dst.base != value.base:
                self.backend.copy64(dst, value)
            return

        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise self._error(node, 'only simple scalar augmented assignment is supported')
            dst = self._var(node.target)
            rhs = self.compile_expr(node.value)
            tmp = self._new_word()
            if isinstance(node.op, ast.Add):
                self.backend.add64(tmp, dst, rhs)
            elif isinstance(node.op, ast.Sub):
                self.backend.sub64(tmp, dst, rhs)
            elif isinstance(node.op, ast.Mult):
                self.backend.mul64(tmp, dst, rhs, self.workspace_base)
            else:
                raise self._error(node, f'unsupported augmented operator {type(node.op).__name__}')
            self.backend.copy64(dst, tmp)
            return

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == 'print':
                self._compile_print(call)
                return
            # Evaluate other supported calls for their side effects.
            self.compile_expr(call)
            return

        if isinstance(node, ast.If):
            cond = self.compile_expr(node.test)
            control = self._truth_cell(cond)
            else_control = self.temps.cell() if node.orelse else None
            if else_control is not None:
                self.bf.set_const(else_control, 1)

            self.bf.begin_while(control)
            self.bf.add_const(control, -1)
            if else_control is not None:
                self.bf.clear(else_control)
            for stmt in node.body:
                self.compile_stmt(stmt)
            self.bf.end_while(control)

            if else_control is not None:
                self.bf.begin_while(else_control)
                self.bf.add_const(else_control, -1)
                for stmt in node.orelse:
                    self.compile_stmt(stmt)
                self.bf.end_while(else_control)
            return

        if isinstance(node, ast.While):
            if node.orelse:
                raise self._error(node, 'while ... else is not lowered yet')
            cond0 = self.compile_expr(node.test)
            control = self._truth_cell(cond0)
            self.bf.begin_while(control)
            self.bf.add_const(control, -1)
            for stmt in node.body:
                self.compile_stmt(stmt)
            cond_next = self.compile_expr(node.test)
            self.backend._is_nonzero64(control, cond_next)
            self.bf.end_while(control)
            return

        if isinstance(node, ast.For):
            self._compile_for_range(node)
            return

        if isinstance(node, ast.Pass):
            return

        raise self._error(node, f'unsupported statement {type(node).__name__}')

    def _compile_print(self, call: ast.Call) -> None:
        if call.keywords:
            raise self._error(call, 'print keyword arguments are not lowered yet')
        for index, arg in enumerate(call.args):
            if index:
                self.backend.print_space(self.workspace_base)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self._emit_string(arg.value)
            else:
                value = self.compile_expr(arg)
                self.backend.print_s64(value, self.workspace_base)
        self.backend.print_newline(self.workspace_base)

    def _range_parts(self, node: ast.For):
        if not isinstance(node.target, ast.Name):
            raise self._error(node, 'range target must be a simple name')
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == 'range'
            and not node.iter.keywords
        ):
            raise self._error(node, 'only for ... in range(...) is supported')
        args = node.iter.args
        if len(args) == 1:
            start_node, stop_node, step = ast.Constant(value=0), args[0], 1
        elif len(args) == 2:
            start_node, stop_node, step = args[0], args[1], 1
        elif len(args) == 3:
            start_node, stop_node = args[0], args[1]
            if not isinstance(args[2], ast.Constant) or not isinstance(args[2].value, int):
                raise self._error(node, 'range step must currently be a constant integer')
            step = args[2].value
        else:
            raise self._error(node, 'range expects one to three arguments')
        if step == 0:
            raise self._error(node, 'range() arg 3 must not be zero')
        return node.target, start_node, stop_node, step

    def _compile_for_range(self, node: ast.For) -> None:
        if node.orelse:
            raise self._error(node, 'for ... else is not lowered yet')
        target_node, start_node, stop_node, step = self._range_parts(node)
        target = self._var(target_node)

        # range arguments are evaluated once.  Hidden current also means body
        # assignments to the visible loop variable do not disturb iteration,
        # matching Python's behavior.
        start = self.compile_expr(start_node)
        stop_value = self.compile_expr(stop_node)
        current = self._copy_new(start)
        stop = self._copy_new(stop_value)
        control = self.temps.cell()

        if step > 0:
            self.backend.slt64(control, current, stop)
        else:
            self.backend.sgt64(control, current, stop)

        step_word = None
        add_tmp = None
        if step not in (1,):
            step_word = self._new_word(step)
            add_tmp = self._new_word()

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        self.backend.copy64(target, current)

        for stmt in node.body:
            self.compile_stmt(stmt)

        if step == 1:
            self.backend._inc64_inplace(current)
        else:
            assert step_word is not None and add_tmp is not None
            self.backend.add64(add_tmp, current, step_word)
            self.backend.copy64(current, add_tmp)

        if step > 0:
            self.backend.slt64(control, current, stop)
        else:
            self.backend.sgt64(control, current, stop)
        self.bf.end_while(control)

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError('expected ast.Module')
        for stmt in tree.body:
            self.compile_stmt(stmt)
        return clean_bf(self.bf.code())


def clean_bf(code: str) -> str:
    """Conservative adjacent cancellation; never rewrites loop structure."""
    stack: list[str] = []
    opposite = {'>': '<', '<': '>', '+': '-', '-': '+'}
    for ch in code:
        if ch not in '><+-[],.':
            continue
        if ch in opposite and stack and stack[-1] == opposite[ch]:
            stack.pop()
        else:
            stack.append(ch)
    return ''.join(stack)


def compile_source(source: str, filename: str = '<string>') -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFV2(tree).compile_module(tree)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compile the supported Python subset to Brainfuck')
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()

    source = args.input.read_text(encoding='utf-8')
    code = compile_source(source, str(args.input))
    output = args.output or args.input.with_suffix('.bf')
    output.write_text(code, encoding='utf-8')
    print(f'wrote {output} ({len(code)} BF commands)')


if __name__ == '__main__':
    main()
