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


class _TempArena:
    """Compile-time temporary allocator with high-water tracking.

    Older compiler layers mutate ``temps.top`` directly, so ``top`` remains a
    read/write property rather than forcing every frontend to adopt a new API.
    Recording ``peak`` is useful beyond diagnostics: a future two-pass lowering
    can measure the maximum temporary footprint first and place the runtime
    variable-sized heap immediately after that proven bound on the second pass.
    """

    def __init__(self, top: int) -> None:
        self._top = top
        self.base = top
        self.peak = top

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int) -> None:
        self._top = value
        if value > self.peak:
            self.peak = value

    @property
    def cells_used_peak(self) -> int:
        return self.peak - self.base

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

    def _flag_not(self, dst: int, src: int, tmp: int) -> None:
        self.bf.set_const(dst, 1)
        self.backend.copy_cell(src, tmp, self.backend.s0)
        self.bf.begin_while(tmp)
        self.bf.clear(tmp)
        self.bf.clear(dst)
        self.bf.end_while(tmp)

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return self._new_word(1 if node.value else 0)
            if type(node.value) is int:
                return self._new_word(node.value)
            raise self._error(node, 'only int/bool constants are supported here')

        if isinstance(node, ast.Name):
            return self._var(node)

        if isinstance(node, ast.UnaryOp):
            value = self.compile_expr(node.operand)
            result = self._new_word()
            if isinstance(node.op, ast.USub):
                zero = self._new_word(0)
                self.backend.sub64(result, zero, value)
                return result
            if isinstance(node.op, ast.UAdd):
                self.backend.copy64(result, value)
                return result
            if isinstance(node.op, ast.Invert):
                self.backend.not64(result, value)
                return result
            if isinstance(node.op, ast.Not):
                nonzero = self.temps.cell()
                self.backend._is_nonzero64(nonzero, value)
                self._flag_not(nonzero, nonzero, self.temps.cell())
                return self._bool_word_from_cell(nonzero)
            raise self._error(node, 'unsupported unary operator')

        if isinstance(node, ast.BinOp):
            left = self.compile_expr(node.left)
            right = self.compile_expr(node.right)
            result = self._new_word()
            if isinstance(node.op, ast.Add):
                self.backend.add64(result, left, right)
            elif isinstance(node.op, ast.Sub):
                self.backend.sub64(result, left, right)
            elif isinstance(node.op, ast.Mult):
                self.backend.mul64(result, left, right, self.workspace_base)
            elif isinstance(node.op, ast.FloorDiv):
                rem = self._new_word()
                self.backend.sdivmod64(result, rem, left, right, self.workspace_base)
            elif isinstance(node.op, ast.Mod):
                quot = self._new_word()
                self.backend.sdivmod64(quot, result, left, right, self.workspace_base)
            elif isinstance(node.op, ast.BitAnd):
                self.backend.and64(result, left, right)
            elif isinstance(node.op, ast.BitOr):
                self.backend.or64(result, left, right)
            elif isinstance(node.op, ast.BitXor):
                self.backend.xor64(result, left, right)
            elif isinstance(node.op, ast.LShift):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or node.right.value < 0:
                    raise self._error(node, 'shift count must be a non-negative integer constant')
                self.backend.shl_const(result, left, node.right.value)
            elif isinstance(node.op, ast.RShift):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or node.right.value < 0:
                    raise self._error(node, 'shift count must be a non-negative integer constant')
                self.backend.shr_const(result, left, node.right.value)
            elif isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or node.right.value < 0:
                    raise self._error(node, 'exponent must be a non-negative integer constant')
                self.backend.pow_const(result, left, node.right.value, self.workspace_base)
            else:
                raise self._error(node, 'unsupported binary operator')
            return result

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise self._error(node, 'only one comparison at a time is supported')
            left = self.compile_expr(node.left)
            right = self.compile_expr(node.comparators[0])
            flag = self.temps.cell()
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                self.backend.eq64(flag, left, right)
            elif isinstance(op, ast.NotEq):
                self.backend.eq64(flag, left, right)
                self.backend._toggle_bit(flag, self.backend.s0)
                self.backend._clear_scratch()
            elif isinstance(op, ast.Lt):
                self.backend.slt64(flag, left, right)
            elif isinstance(op, ast.LtE):
                self.backend.sle64(flag, left, right)
            elif isinstance(op, ast.Gt):
                self.backend.sgt64(flag, left, right)
            elif isinstance(op, ast.GtE):
                self.backend.sge64(flag, left, right)
            else:
                raise self._error(node, 'unsupported comparison')
            return self._bool_word_from_cell(flag)

        if isinstance(node, ast.BoolOp):
            if len(node.values) < 2:
                raise self._error(node, 'boolean expression needs at least two operands')
            result = self._copy_new(self.compile_expr(node.values[0]))
            for rhs_node in node.values[1:]:
                nonzero = self.temps.cell()
                self.backend._is_nonzero64(nonzero, result)
                gate = self.temps.cell()
                if isinstance(node.op, ast.And):
                    self.backend.copy_cell(nonzero, gate, self.backend.s0)
                elif isinstance(node.op, ast.Or):
                    self._flag_not(gate, nonzero, self.temps.cell())
                else:
                    raise self._error(node, 'unsupported boolean operator')
                self.bf.begin_while(gate)
                self.bf.add_const(gate, -1)
                rhs = self.compile_expr(rhs_node)
                self.backend.copy64(result, rhs)
                self.bf.end_while(gate)
            return result

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'int' and len(node.args) == 1:
                arg = node.args[0]
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'input' and not arg.args and not arg.keywords:
                    result = self._new_word()
                    self.backend.read_s64(result, self.workspace_base)
                    return result
            raise self._error(node, 'unsupported call expression')

        raise self._error(node, f'unsupported expression: {type(node).__name__}')

    # ------------------------------------------------------------------
    # statements / control flow
    # ------------------------------------------------------------------
    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise self._error(node, 'assignment requires one simple name target')
            value = self.compile_expr(node.value)
            self.backend.copy64(self._var(node.targets[0]), value)
            return

        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise self._error(node, 'augmented assignment requires a simple name target')
            target = self._var(node.target)
            rhs = self.compile_expr(node.value)
            result = self._new_word()
            if isinstance(node.op, ast.Add):
                self.backend.add64(result, target, rhs)
            elif isinstance(node.op, ast.Sub):
                self.backend.sub64(result, target, rhs)
            elif isinstance(node.op, ast.Mult):
                self.backend.mul64(result, target, rhs, self.workspace_base)
            else:
                raise self._error(node, 'unsupported augmented assignment operator')
            self.backend.copy64(target, result)
            return

        if isinstance(node, ast.Expr):
            # print(...)
            call = node.value
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'print':
                raise self._error(node, 'only print(...) expression statements are supported')
            for i, arg in enumerate(call.args):
                if i:
                    self.backend.write_literal(' ')
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.backend.write_literal(arg.value)
                else:
                    self.backend.print_s64(self.compile_expr(arg), self.workspace_base)
            self.backend.write_literal('\n')
            return

        if isinstance(node, ast.If):
            cond_word = self.compile_expr(node.test)
            cond = self.temps.cell()
            self.backend._is_nonzero64(cond, cond_word)
            else_flag = self.temps.cell()
            self.bf.set_const(else_flag, 1)
            self.bf.begin_while(cond)
            self.bf.add_const(cond, -1)
            self.bf.clear(else_flag)
            for stmt in node.body:
                self.compile_stmt(stmt)
            self.bf.end_while(cond)
            self.bf.begin_while(else_flag)
            self.bf.add_const(else_flag, -1)
            for stmt in node.orelse:
                self.compile_stmt(stmt)
            self.bf.end_while(else_flag)
            return

        if isinstance(node, ast.While):
            if node.orelse:
                raise self._error(node, 'while ... else is not supported yet')
            control = self.temps.cell()
            cond_word = self.compile_expr(node.test)
            self.backend._is_nonzero64(control, cond_word)
            self.bf.begin_while(control)
            self.bf.add_const(control, -1)
            for stmt in node.body:
                self.compile_stmt(stmt)
            cond_word = self.compile_expr(node.test)
            self.backend._is_nonzero64(control, cond_word)
            self.bf.end_while(control)
            return

        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise self._error(node, 'for target must be a simple name')
            if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != 'range':
                raise self._error(node, 'only for ... in range(...) is supported')
            args = node.iter.args
            if len(args) == 1:
                start_node, stop_node, step = ast.Constant(0), args[0], 1
            elif len(args) == 2:
                start_node, stop_node, step = args[0], args[1], 1
            elif len(args) == 3 and isinstance(args[2], ast.Constant) and type(args[2].value) is int and args[2].value != 0:
                start_node, stop_node, step = args[0], args[1], args[2].value
            else:
                raise self._error(node, 'range() requires 1/2 args or constant nonzero step')

            target = self._var(node.target)
            current = self._copy_new(self.compile_expr(start_node))
            stop = self._copy_new(self.compile_expr(stop_node))
            control = self.temps.cell()
            if step > 0:
                self.backend.slt64(control, current, stop)
            else:
                self.backend.sgt64(control, current, stop)
            step_word = self._new_word(step)
            add_tmp = self._new_word()
            self.bf.begin_while(control)
            self.bf.add_const(control, -1)
            self.backend.copy64(target, current)
            for stmt in node.body:
                self.compile_stmt(stmt)
            self.backend.add64(add_tmp, current, step_word)
            self.backend.copy64(current, add_tmp)
            if step > 0:
                self.backend.slt64(control, current, stop)
            else:
                self.backend.sgt64(control, current, stop)
            self.bf.end_while(control)
            return

        raise self._error(node, f'unsupported statement: {type(node).__name__}')

    def compile_stmt(self, node: ast.stmt) -> None:
        mark = self.temps.mark()
        try:
            self._compile_stmt_inner(node)
        finally:
            self.temps.rewind(mark)

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError('expected ast.Module')
        for stmt in tree.body:
            self.compile_stmt(stmt)
        return self.bf.code()


def compile_source(source: str, filename: str = '<string>') -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFV2(tree).compile_module(tree)


def _cli() -> int:
    parser = argparse.ArgumentParser(description='Compile a small Python subset to Brainfuck')
    parser.add_argument('input', type=Path, help='Python source file')
    parser.add_argument('-o', '--output', type=Path, help='Output .bf file')
    args = parser.parse_args()

    source = args.input.read_text(encoding='utf-8')
    code = compile_source(source, filename=str(args.input))
    if args.output:
        args.output.write_text(code, encoding='ascii')
    else:
        print(code)
    return 0


if __name__ == '__main__':
    raise SystemExit(_cli())
