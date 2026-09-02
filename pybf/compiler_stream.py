"""Streaming and source-size lowering for the final compiler path."""

from __future__ import annotations

import ast

from bfopt import optimize_bf
from compiler_compact import CompileError, _body_rebinds
from compiler_quad import PythonToBFQuad
from transpiler_full import _LoopContext
from transpiler_v2 import clean_bf


def _is_input_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
        and not node.args
        and not node.keywords
    )


def _simple_input_assignment(node: ast.stmt) -> str | None:
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _is_input_call(node.value)
    ):
        return None
    return node.targets[0].id


def _mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(node))


def _contains_input_call(nodes: list[ast.stmt]) -> bool:
    for stmt in nodes:
        for item in ast.walk(stmt):
            if _is_input_call(item):
                return True
    return False


def _contains_loop_transfer(nodes: list[ast.stmt]) -> bool:
    # Conservative by design: even a break/continue owned by a nested loop
    # disables this source-size optimization. Correctness matters more than
    # squeezing the rare nested-control case.
    return any(
        isinstance(item, (ast.Break, ast.Continue))
        for stmt in nodes
        for item in ast.walk(stmt)
    )


def _positive_power_of_two_constant(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, int):
        return None
    value = node.value
    if value <= 0 or value & (value - 1):
        return None
    return value.bit_length() - 1


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _self_min_abs_assignment(node: ast.stmt) -> tuple[str, ast.AST] | None:
    """Recognize ``x = min(x, abs(fresh_binary_expression))``."""
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "min"
        and len(node.value.args) == 2
        and not node.value.keywords
    ):
        return None

    target = node.targets[0].id
    first, second = node.value.args
    if not isinstance(first, ast.Name) or first.id != target:
        return None
    if not (
        isinstance(second, ast.Call)
        and isinstance(second.func, ast.Name)
        and second.func.id == "abs"
        and len(second.args) == 1
        and not second.keywords
        and isinstance(second.args[0], ast.BinOp)
    ):
        return None
    return target, second.args[0]


class PythonToBFStream(PythonToBFQuad):
    """Quad-scalar compact compiler plus proven-safe contest lowering."""

    def compile_expr(self, node: ast.AST):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            left_shift = _positive_power_of_two_constant(node.left)
            right_shift = _positive_power_of_two_constant(node.right)

            if left_shift is not None:
                value = self.compile_expr(node.right)
                result = self._new_word()
                self.backend.shl_const(result, value, left_shift)
                return result
            if right_shift is not None:
                value = self.compile_expr(node.left)
                result = self._new_word()
                self.backend.shl_const(result, value, right_shift)
                return result

        return super().compile_expr(node)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        fused = _self_min_abs_assignment(node)
        if fused is not None:
            target_name, magnitude_expr = fused
            if target_name not in self.variables:
                return super()._compile_stmt_inner(node)

            candidate = self.compile_expr(magnitude_expr)
            sign = self.temps.cell()
            self.backend.copy_cell(candidate.bit(63), sign, self.backend.s0)
            self.bf.begin_while(sign)
            self.bf.add_const(sign, -1)
            self.backend._neg64_inplace(candidate)
            self.bf.end_while(sign)

            dst = self.variables[target_name]
            choose = self.temps.cell()
            self.backend.slt64(choose, candidate, dst)
            self.bf.begin_while(choose)
            self.bf.add_const(choose, -1)
            self.backend.copy64(dst, candidate)
            self.bf.end_while(choose)
            return

        return super()._compile_stmt_inner(node)

    def _list_index_word(self, node: ast.AST, ref):
        proven = getattr(self, "_proven_nonnegative_indices", set())
        if isinstance(node, ast.Name) and node.id in proven:
            return self.compile_expr(node), None
        return super()._list_index_word(node, ref)

    def _compile_for_range_simple(self, node: ast.For) -> None:
        """Range loop without break/continue/else bookkeeping."""
        target_node, start_node, stop_node, step = self._range_parts(node)
        target = self._var(target_node)
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
        if step != 1:
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

    def _compile_for_range_control(self, node: ast.For) -> None:
        target, start, _stop, step = self._range_parts(node)
        name = target.id if isinstance(target, ast.Name) else None
        start_value = _literal_int(start)
        safe_nonnegative = (
            name is not None
            and step > 0
            and start_value is not None
            and start_value >= 0
            and not _body_rebinds(name, node.body)
        )
        lean_control = not node.orelse and not _contains_loop_transfer(node.body)

        proven = getattr(self, "_proven_nonnegative_indices", None)
        if proven is None:
            proven = set()
            self._proven_nonnegative_indices = proven
        already = name in proven if name is not None else False
        if safe_nonnegative and name is not None:
            proven.add(name)
        try:
            if lean_control:
                self._compile_for_range_simple(node)
            else:
                super()._compile_for_range_control(node)
        finally:
            if safe_nonnegative and name is not None and not already:
                proven.remove(name)

    def compile_stmt(self, node: ast.stmt) -> None:
        """Delegate normally while attributing nested statement source size."""
        start = len(self.bf.parts)
        super().compile_stmt(node)
        if hasattr(self, "detail_sizes"):
            emitted = sum(len(part) for part in self.bf.parts[start:])
            self.detail_sizes.append(
                (getattr(node, "lineno", 0), type(node).__name__, emitted)
            )

    def _can_fuse_input_string_for(
        self,
        body: list[ast.stmt],
        index: int,
    ) -> bool:
        if index + 1 >= len(body):
            return False
        source = _simple_input_assignment(body[index])
        loop = body[index + 1]
        if source is None or not isinstance(loop, ast.For):
            return False
        if not (
            isinstance(loop.iter, ast.Name)
            and loop.iter.id == source
            and isinstance(loop.target, ast.Name)
            and loop.target.id in self.compact_char_names
            and loop.target.id != source
        ):
            return False
        if _contains_input_call(loop.body) or _contains_input_call(loop.orelse):
            return False
        for stmt in loop.body + loop.orelse:
            if _mentions_name(stmt, source):
                return False
        for stmt in body[index + 2 :]:
            if _mentions_name(stmt, source):
                return False
        return True

    def _compile_streaming_input_string_for(self, node: ast.For) -> None:
        assert isinstance(node.target, ast.Name)
        target = self.strings[node.target.id]

        input_cell = self.temps.cell()
        is_end = self.temps.cell()
        end_tmp = self.temps.cell()
        active = self.temps.cell()
        data = self.temps.cell()
        data_tmp = self.temps.cell()
        body_active = self.temps.cell()
        broke = self.temps.cell()
        proceed = self.temps.cell()
        proceed_tmp = self.temps.cell()

        for cell in (
            input_cell,
            is_end,
            end_tmp,
            active,
            data,
            data_tmp,
            body_active,
            broke,
            proceed,
            proceed_tmp,
        ):
            self.bf.clear(cell)
        self.bf.set_const(active, 1)

        self.bf.begin_while(active)
        self.bf.add_const(active, -1)
        self.bf.move(input_cell)
        self.bf.emit(",")
        self.backend._set_line_end(is_end, input_cell, end_tmp)
        self._flag_not(data, is_end, data_tmp)

        self.bf.begin_while(data)
        self.bf.add_const(data, -1)
        self.backend.clear_string(target)
        self.backend.copy_cell(input_cell, target.char(0), self.backend.s0)
        self.bf.set_const(body_active, 1)

        ctx = _LoopContext(body_active, broke)
        self._loop_stack.append(ctx)
        try:
            for stmt in node.body:
                self.compile_stmt(stmt)
        finally:
            self._loop_stack.pop()

        self._flag_not(proceed, broke, proceed_tmp)
        self.bf.begin_while(proceed)
        self.bf.add_const(proceed, -1)
        self.bf.set_const(active, 1)
        self.bf.end_while(proceed)
        self.bf.end_while(data)
        self.bf.end_while(active)

        drain_gate = self.temps.cell()
        line_open = self.temps.cell()
        self.backend.copy_cell(broke, drain_gate, self.backend.s0)
        self.bf.begin_while(drain_gate)
        self.bf.add_const(drain_gate, -1)
        self.bf.set_const(line_open, 1)
        self.backend.drain_to_line_end(line_open, self.workspace_base)
        self.bf.end_while(drain_gate)

        self._compile_guarded_else(node.orelse, broke)

    def _record_size(self, node: ast.AST, part_start: int) -> None:
        emitted = sum(len(part) for part in self.bf.parts[part_start:])
        self.statement_sizes.append(
            (getattr(node, "lineno", 0), type(node).__name__, emitted)
        )

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError("expected ast.Module")

        self.statement_sizes: list[tuple[int, str, int]] = []
        self.detail_sizes: list[tuple[int, str, int]] = []

        i = 0
        while i < len(tree.body):
            part_start = len(self.bf.parts)
            if self._can_fuse_input_string_for(tree.body, i):
                first = tree.body[i]
                loop = tree.body[i + 1]
                assert isinstance(loop, ast.For)
                mark = self.temps.mark()
                try:
                    self._compile_streaming_input_string_for(loop)
                finally:
                    self.temps.rewind(mark)
                self._record_size(first, part_start)
                i += 2
                continue
            stmt = tree.body[i]
            self.compile_stmt(stmt)
            self._record_size(stmt, part_start)
            i += 1
        return clean_bf(self.bf.code())


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    tree = ast.parse(source, filename=filename)
    code = PythonToBFStream(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
    return optimize_bf(code)


__all__ = ["CompileError", "PythonToBFStream", "compile_source"]
