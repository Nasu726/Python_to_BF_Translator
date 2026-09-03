"""Streaming fusion for single-use ``list(map(int, input().split()))`` values.

This layer eliminates a temporary Python list when it is produced from one
input line and consumed by exactly one later ``for x in list`` loop with no
other uses.  Pure scalar assignments may occur between producer and consumer.
Tokens are parsed one-at-a-time into the normal scalar target, so emitted
Brainfuck source and runtime storage are independent of line length.

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


def _pure_scalar_expr(node: ast.AST) -> bool:
    """Conservative expression class safe to move across an input read."""
    forbidden = (
        ast.Call,
        ast.Subscript,
        ast.Attribute,
        ast.List,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Lambda,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.NamedExpr,
    )
    return not any(isinstance(item, forbidden) for item in ast.walk(node))


class PythonToBFStream(_BasePythonToBFStream):
    """Established generic compiler plus dead-list producer/consumer fusion."""

    def _reorder_safe_setup(self, stmt: ast.stmt) -> bool:
        """Whether stmt may execute before the producer's input consumption."""
        if isinstance(stmt, ast.Assign):
            return (
                bool(stmt.targets)
                and all(
                    isinstance(target, ast.Name) and target.id in self.variables
                    for target in stmt.targets
                )
                and _pure_scalar_expr(stmt.value)
            )
        if isinstance(stmt, ast.AnnAssign):
            return (
                isinstance(stmt.target, ast.Name)
                and stmt.target.id in self.variables
                and stmt.value is not None
                and _pure_scalar_expr(stmt.value)
            )
        return False

    def _find_input_int_list_consumer(
        self,
        body: list[ast.stmt],
        index: int,
    ) -> int | None:
        source = _simple_int_list_input_assignment(body[index])
        if source is None:
            return None

        for loop_index in range(index + 1, len(body)):
            stmt = body[loop_index]
            if (
                isinstance(stmt, ast.For)
                and isinstance(stmt.iter, ast.Name)
                and stmt.iter.id == source
            ):
                if not (
                    isinstance(stmt.target, ast.Name)
                    and stmt.target.id in self.variables
                    and stmt.target.id != source
                ):
                    return None
                if _contains_input_call(stmt.body) or _contains_input_call(stmt.orelse):
                    return None
                for nested in stmt.body + stmt.orelse:
                    if _mentions_name(nested, source):
                        return None
                for later in body[loop_index + 1 :]:
                    if _mentions_name(later, source):
                        return None
                return loop_index

            # Any use of the list requires real list state.  Even an A-free
            # statement may be crossed only when it is a pure scalar assignment;
            # this prevents moving print/calls/mutations before the input read.
            if _mentions_name(stmt, source) or not self._reorder_safe_setup(stmt):
                return None

        return None

    def _compile_streaming_input_int_list_for(self, node: ast.For) -> None:
        assert isinstance(node.target, ast.Name)
        target = self.variables[node.target.id]
        token = self._packed_input_token()

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
        self.backend.packed64.clear(token)

        self.bf.set_const(active, 1)
        self.bf.begin_while(active)
        self.bf.add_const(active, -1)

        # The final generic compiler stores scalars as Quad64Ref rather than the
        # old contiguous Boolean Int64Ref. Parse through the same packed-token
        # ABI used by PythonToBFQuad, then expand/copy into the loop target.
        self.backend.read_packed_s64_line_token(
            token,
            has_token,
            end_line,
            self.workspace_base,
        )
        self.backend.copy64(target, token)
        self.backend.packed64.clear(token)

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
        # already consume LF/EOF. A continue leaves broke == 0, matching Python.
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
        # so the following input() starts correctly. If the breaking token
        # itself consumed LF, line_open == 0 and drain_to_line_end is a no-op.
        self.backend.copy_cell(broke, drain_gate, self.backend.s0)
        self.bf.begin_while(drain_gate)
        self.bf.add_const(drain_gate, -1)
        self.backend.copy_cell(line_open, drain_active, self.backend.s0)
        self.backend.drain_to_line_end(drain_active, self.workspace_base)
        self.bf.end_while(drain_gate)

        self.backend.packed64.clear(token)
        self._compile_guarded_else(node.orelse, broke)

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError("expected ast.Module")

        self.statement_sizes = []
        self.detail_sizes = []

        i = 0
        while i < len(tree.body):
            part_start = len(self.bf.parts)
            consumer_index = self._find_input_int_list_consumer(tree.body, i)
            if consumer_index is not None:
                first = tree.body[i]
                for setup_stmt in tree.body[i + 1 : consumer_index]:
                    self.compile_stmt(setup_stmt)
                loop = tree.body[consumer_index]
                assert isinstance(loop, ast.For)
                mark = self.temps.mark()
                try:
                    self._compile_streaming_input_int_list_for(loop)
                finally:
                    self.temps.rewind(mark)
                self._record_size(first, part_start)
                i = consumer_index + 1
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
