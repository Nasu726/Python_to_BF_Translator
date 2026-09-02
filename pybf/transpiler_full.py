"""Control-flow-complete layer over the current contest frontend.

Brainfuck cannot jump out of a loop body, so Python ``break``/``continue`` are
lowered with explicit runtime flags.  Every statement in a loop body is gated
by ``body_active``.  ``continue`` clears that flag; ``break`` also sets a
persistent break flag so the loop condition/update is not re-enabled.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bfcore import Int64Ref
from transpiler import PythonToBF
from transpiler_v2 import CompileError


@dataclass(frozen=True)
class _LoopContext:
    body_active: int
    broke: int


class PythonToBFFull(PythonToBF):
    def __init__(self, tree: ast.AST, **kwargs) -> None:
        super().__init__(tree, **kwargs)
        self._loop_stack: list[_LoopContext] = []

    def _flag_not(self, result: int, flag: int, tmp: int) -> None:
        bf = self.bf
        bf.set_const(result, 1)
        self.backend.copy_cell(flag, tmp, self.backend.s0)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        bf.clear(result)
        bf.end_while(tmp)

    def compile_stmt(self, node: ast.stmt) -> None:
        """Statement region allocation plus runtime continue/break gating."""
        mark = self.temps.mark()
        try:
            if not self._loop_stack:
                self._compile_stmt_inner(node)
                return

            ctx = self._loop_stack[-1]
            gate = self.temps.cell()
            self.backend.copy_cell(ctx.body_active, gate, self.backend.s0)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self._compile_stmt_inner(node)
            self.bf.end_while(gate)
        finally:
            self.temps.rewind(mark)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Break):
            if not self._loop_stack:
                raise self._error(node, "break outside loop")
            ctx = self._loop_stack[-1]
            self.bf.set_const(ctx.broke, 1)
            self.bf.clear(ctx.body_active)
            return

        if isinstance(node, ast.Continue):
            if not self._loop_stack:
                raise self._error(node, "continue outside loop")
            self.bf.clear(self._loop_stack[-1].body_active)
            return

        if isinstance(node, ast.While):
            self._compile_while_control(node)
            return

        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Name) and node.iter.id in self.lists:
                self._compile_for_list_control(node)
                return
            # Parent helper validates the range shape and raises for anything
            # else; do not let the v2 implementation bypass our loop flags.
            self._compile_for_range_control(node)
            return

        return super()._compile_stmt_inner(node)

    def _compile_guarded_else(self, body: list[ast.stmt], broke: int) -> None:
        if not body:
            return
        run_else = self.temps.cell()
        tmp = self.temps.cell()
        self._flag_not(run_else, broke, tmp)
        self.bf.begin_while(run_else)
        self.bf.add_const(run_else, -1)
        for stmt in body:
            # The loop context has already been popped, so this else suite is
            # not suppressed by the completed loop's body_active flag.
            self.compile_stmt(stmt)
        self.bf.end_while(run_else)

    def _compile_while_control(self, node: ast.While) -> None:
        cond0 = self.compile_expr(node.test)
        control = self._truth_cell(cond0)
        body_active = self.temps.cell()
        broke = self.temps.cell()
        self.bf.clear(broke)

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        self.bf.set_const(body_active, 1)
        ctx = _LoopContext(body_active, broke)
        self._loop_stack.append(ctx)
        try:
            for stmt in node.body:
                self.compile_stmt(stmt)
        finally:
            self._loop_stack.pop()

        proceed = self.temps.cell()
        tmp = self.temps.cell()
        self._flag_not(proceed, broke, tmp)
        self.bf.begin_while(proceed)
        self.bf.add_const(proceed, -1)
        cond_next = self.compile_expr(node.test)
        self.backend._is_nonzero64(control, cond_next)
        self.bf.end_while(proceed)
        self.bf.end_while(control)

        self._compile_guarded_else(node.orelse, broke)

    def _compile_for_range_control(self, node: ast.For) -> None:
        target_node, start_node, stop_node, step = self._range_parts(node)
        target = self._var(target_node)
        start = self.compile_expr(start_node)
        stop_value = self.compile_expr(stop_node)
        current = self._copy_new(start)
        stop = self._copy_new(stop_value)
        control = self.temps.cell()
        body_active = self.temps.cell()
        broke = self.temps.cell()
        self.bf.clear(broke)

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
        self.bf.end_while(proceed)
        self.bf.end_while(control)

        self._compile_guarded_else(node.orelse, broke)

    def _compile_for_list_control(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name) or node.target.id in self.strings or node.target.id in self.lists:
            raise self._error(node, "list iteration target must be an integer variable")
        assert isinstance(node.iter, ast.Name)
        ref = self.lists[node.iter.id]
        target = self._var(node.target)

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
        loaded = self._new_word()
        workspace_word = self._new_word()
        self.backend.get_dynamic(loaded, ref, index, workspace_word, self.temps.cell())
        self.backend.copy64(target, loaded)
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


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 128,
    list_capacity: int = 32,
) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFFull(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
