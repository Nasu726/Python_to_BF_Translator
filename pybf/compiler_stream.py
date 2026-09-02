"""Streaming fusion for dead materialized input strings.

A very common contest pattern is::

    s = input()
    for c in s:
        ...

If ``s`` is never observed as a value, materializing a 255-byte buffer and then
emitting a dynamic-index traversal is pure overhead.  This layer fuses the two
statements into one line-scoped input loop while preserving Python-visible I/O
ordering.  The optimization is deliberately conservative and is disabled when
its proof conditions are not met.
"""

from __future__ import annotations

import ast

from bfopt import optimize_bf
from compiler_compact import CompileError, PythonToBFCompact
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


class PythonToBFStream(PythonToBFCompact):
    """Compact compiler plus proven-safe input/string-loop fusion."""

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

        # Original input() consumes the complete source line before the first
        # loop-body statement runs.  A fused stream must therefore reject any
        # body that performs another input operation before this line is fully
        # consumed.
        if _contains_input_call(loop.body) or _contains_input_call(loop.orelse):
            return False

        # The source value is intentionally never materialized.  Reject any
        # observation/rebinding of it in the loop or later module statements.
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

        # Read one byte at a time.  active is re-enabled only after a normal
        # data byte whose loop body did not break.
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

        # In unfused Python, input() had already consumed the whole line before
        # a possible break.  Drain the unread suffix so subsequent input() sees
        # exactly the same next line.
        drain_gate = self.temps.cell()
        line_open = self.temps.cell()
        self.backend.copy_cell(broke, drain_gate, self.backend.s0)
        self.bf.begin_while(drain_gate)
        self.bf.add_const(drain_gate, -1)
        self.bf.set_const(line_open, 1)
        self.backend.drain_to_line_end(line_open, self.workspace_base)
        self.bf.end_while(drain_gate)

        self._compile_guarded_else(node.orelse, broke)

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError("expected ast.Module")

        i = 0
        while i < len(tree.body):
            if self._can_fuse_input_string_for(tree.body, i):
                loop = tree.body[i + 1]
                assert isinstance(loop, ast.For)
                mark = self.temps.mark()
                try:
                    self._compile_streaming_input_string_for(loop)
                finally:
                    self.temps.rewind(mark)
                i += 2
                continue
            self.compile_stmt(tree.body[i])
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
