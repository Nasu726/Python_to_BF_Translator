"""Streaming fusion for single-use ``list(map(int, input().split()))`` values.

This layer eliminates a temporary Python list when it is produced from one
input line and immediately consumed by exactly one ``for x in list`` loop with
no later uses.  The optimization is structural and semantics-preserving for the
accepted shape: tokens are parsed one-at-a-time into the normal scalar target,
so emitted Brainfuck source and runtime storage are independent of line length.

This is intentionally a producer/consumer optimization, not a replacement for
full mutable list semantics.  Programs that retain, index, alias, sort or reuse
the list fall back unchanged to the established generic compiler.
"""

from __future__ import annotations

import ast

from compiler_stream_generic import CompileError
from compiler_stream_generic import PythonToBFStream as _BasePythonToBFStream
from compiler_stream_generic import _contains_input_call, _mentions_name
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


def _is_input_split(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "split"
        and not node.args
        and not node.keywords
        and _is_input_call(node.func.value)
    )


def _is_map_int_input_split(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "map"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "int"
        and _is_input_split(node.args[1])
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


def _simple_int_list_input_assignment(node: ast.stmt) -> str | None:
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _is_list_map_int_input_split(node.value)
    ):
        return None
    return node.targets[0].id


class PythonToBFStream(_BasePythonToBFStream):
    """Established generic compiler plus dead-list producer/consumer fusion."""

    def _can_fuse_input_int_list_for(
        self,
        body: list[ast.stmt],
        index: int,
    ) -> bool:
        if index + 1 >= len(body):
            return False

        source = _simple_int_list_input_assignment(body[index])
        loop = body[index + 1]
        if source is None or not isinstance(loop, ast.For):
            return False
        if not (
            isinstance(loop.iter, ast.Name)
            and loop.iter.id == source
            and isinstance(loop.target, ast.Name)
            and loop.target.id in self.variables
            and loop.target.id != source
        ):
            return False

        # Reading input inside the consumer would change byte-stream ordering
        # relative to first materializing the source line.
        if _contains_input_call(loop.body) or _contains_input_call(loop.orelse):
            return False

        # The list object itself must be dead except as this loop's iterator.
        # That excludes len/index/alias/mutation/reuse and keeps this transform a
        # true allocation elimination rather than a change to list semantics.
        for stmt in loop.body + loop.orelse:
            if _mentions_name(stmt, source):
                return False
        for stmt in body[index + 2 :]:
            if _mentions_name(stmt, source):
                return False
        return True

    def _compile_streaming_input_int_list_for(self, node: ast.For) -> None:
        assert isinstance(node.target, ast.Name)
        target = self.variables[node.target.id]

        active = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        token_gate = self.temps.cell()
        body_active = self.temps.cell()
        broke = self.temps.cell()
        proceed = self.temps.cell()
        proceed_tmp = self.temps.cell()
        line_open = self.temps.cell()
        line_tmp = self.temps.cell()
        rearm_gate = self.temps.cell()
        drain_gate = self.temps.cell()
        drain_active = self.temps.cell()

        for cell in (
            active,
            has_token,
            end_line,
            token_gate,
            body_active,
            broke,
            proceed,
            proceed_tmp,
            line_open,
            line_tmp,
            rearm_gate,
            drain_gate,
            drain_active,
        ):
            self.bf.clear(cell)

        self.bf.set_const(active, 1)
        self.bf.begin_while(active)
        self.bf.add_const(active, -1)

        self.backend.read_s64_line_token(
            target,
            has_token,
            end_line,
            self.workspace_base,
        )

        self.backend.copy_cell(has_token, token_gate, self.backend.s0)
        self.bf.begin_while(token_gate)
        self.bf.add_const(token_gate, -1)
        self.bf.set_const(body_active, 1)

        ctx = _LoopContext(body_active, broke)
        self._loop_stack.append(ctx)
        try:
            for stmt in node.body:
                self.compile_stmt(stmt)
        finally:
            self._loop_stack.pop()

        # Rearm only when the loop did not break and the current token did not
        # already consume LF/EOF.  A continue leaves broke == 0, matching Python.
        self._flag_not(proceed, broke, proceed_tmp)
        self._flag_not(line_open, end_line, line_tmp)
        self.backend.copy_cell(proceed, rearm_gate, self.backend.s0)
        self.bf.begin_while(rearm_gate)
        self.bf.add_const(rearm_gate, -1)
        self.backend.copy_cell(line_open, active, self.backend.s0)
        self.bf.end_while(rearm_gate)

        self.bf.end_while(token_gate)
        self.bf.end_while(active)

        # A break in the middle of the line must consume the rest of that line
        # so the following input() starts correctly.  If the breaking token
        # itself consumed LF, line_open == 0 and we must not touch the next line.
        self.backend.copy_cell(broke, drain_gate, self.backend.s0)
        self.bf.begin_while(drain_gate)
        self.bf.add_const(drain_gate, -1)
        self.backend.copy_cell(line_open, drain_active, self.backend.s0)
        self.bf.begin_while(drain_active)
        self.bf.add_const(drain_active, -1)
        self.bf.set_const(drain_active, 1)
        self.backend.drain_to_line_end(drain_active, self.workspace_base)
        self.bf.clear(drain_active)
        self.bf.end_while(drain_active)
        self.bf.end_while(drain_gate)

        self._compile_guarded_else(node.orelse, broke)

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError("expected ast.Module")

        self.statement_sizes = []
        self.detail_sizes = []

        i = 0
        while i < len(tree.body):
            part_start = len(self.bf.parts)
            if self._can_fuse_input_int_list_for(tree.body, i):
                first = tree.body[i]
                loop = tree.body[i + 1]
                assert isinstance(loop, ast.For)
                mark = self.temps.mark()
                try:
                    self._compile_streaming_input_int_list_for(loop)
                finally:
                    self.temps.rewind(mark)
                self._record_size(first, part_start)
                i += 2
                continue

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


__all__ = ["CompileError", "PythonToBFStream"]
