"""Packed lowering for nonescaping loop-local integer inputs.

A common contest loop reads a fixed tuple of integers and uses those values only
inside that iteration::

    for ...:
        t, a, b = map(int, input().split())
        ...

The generic compiler historically parsed each token into eight packed bytes and
immediately expanded it into a 99-cell Quad word.  Dynamic character indexing
then packed the same value again.  This layer keeps a statically proven set of
nonescaping loop-local inputs in their eight-byte form and uses ``PackedI64Ops``
for simple arithmetic/comparisons. Unsupported expressions still expand a
preserved packed copy to Quad, so optimization never changes accepted semantics.

This is structural lowering, not task matching.  It applies only when all uses
of the selected names are inside one range-loop and their stores are forms this
layer can keep synchronized.
"""

from __future__ import annotations

import ast
from collections import Counter

from bfpacked64 import PackedI64Ref
from bfpackedops import PackedI64Ops
from compiler_quadlocal import CompileError
from compiler_quadlocal import PythonToBFStream as _BasePythonToBFStream
from transpiler import _is_map_int_input_split


PACKED_LOCAL_LIMIT = 4
PACKED_CACHE_LIMIT = 4
PACKED_BYTES = 8
PACKED_EXTRA_CELLS = (
    (PACKED_LOCAL_LIMIT + PACKED_CACHE_LIMIT) * PACKED_BYTES
    + PackedI64Ops.SCRATCH_CELLS
)


def _fixed_int_unpack_names(stmt: ast.stmt) -> list[str] | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], (ast.Tuple, ast.List))
        and _is_map_int_input_split(stmt.value)
    ):
        return None
    names: list[str] = []
    for target in stmt.targets[0].elts:
        if not isinstance(target, ast.Name):
            return None
        names.append(target.id)
    return names or None


class PythonToBFStream(_BasePythonToBFStream):
    """Final compiler plus conservative packed loop-local scalar shadows."""

    SHARED_WORKSPACE_CELLS = (
        _BasePythonToBFStream.SHARED_WORKSPACE_CELLS + PACKED_EXTRA_CELLS
    )

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
        runtime_charlist_base: int | None = None,
    ) -> None:
        self._packed_original_tree = tree
        self._packed_parent = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self._active_packed_locals: dict[str, PackedI64Ref] = {}
        self._active_packed_readonly: dict[str, PackedI64Ref] = {}
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
            runtime_charlist_base=runtime_charlist_base,
        )
        self._packed_extra_base = (
            self.workspace_base + _BasePythonToBFStream.SHARED_WORKSPACE_CELLS
        )
        scratch = self._packed_extra_base + (
            PACKED_LOCAL_LIMIT + PACKED_CACHE_LIMIT
        ) * PACKED_BYTES
        self.packed_ops = PackedI64Ops(self.bf, scratch)

    # ------------------------------------------------------------------
    # fixed packed storage inside the shared workspace
    # ------------------------------------------------------------------
    def _packed_local_slot(self, index: int) -> PackedI64Ref:
        if not 0 <= index < PACKED_LOCAL_LIMIT:
            raise IndexError(index)
        return PackedI64Ref(self._packed_extra_base + index * PACKED_BYTES)

    def _packed_cache_slot(self, index: int) -> PackedI64Ref:
        if not 0 <= index < PACKED_CACHE_LIMIT:
            raise IndexError(index)
        return PackedI64Ref(
            self._packed_extra_base
            + (PACKED_LOCAL_LIMIT + index) * PACKED_BYTES
        )

    def _new_packed_temp(self) -> PackedI64Ref:
        result = PackedI64Ref(self.temps.top)
        self.temps.top += PACKED_BYTES
        return result

    def _packed_name(self, name: str) -> PackedI64Ref | None:
        return self._active_packed_locals.get(name) or self._active_packed_readonly.get(name)

    def _literal_packed(self, value: int) -> PackedI64Ref:
        result = self._new_packed_temp()
        self.packed_ops.set_u64(result, value)
        return result

    def _packed_ref_for_expr(self, node: ast.AST) -> PackedI64Ref | None:
        if isinstance(node, ast.Name):
            return self._packed_name(node.id)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return self._literal_packed(node.value)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) is int
            and isinstance(node.op, (ast.UAdd, ast.USub))
        ):
            value = node.operand.value
            if isinstance(node.op, ast.USub):
                value = -value
            return self._literal_packed(value)
        return None

    def _quad_from_packed(self, src: PackedI64Ref):
        """Expand a preserved packed value through a disposable packed copy."""
        copy = self._new_packed_temp()
        self.packed_ops.copy(copy, src)
        result = self._new_word()
        self.backend.copy64(result, copy)
        return result

    def _assign_packed_from_quad(self, dst: PackedI64Ref, value) -> None:
        packed = self._pack_word(value)
        self.packed_ops.copy(dst, packed)

    # ------------------------------------------------------------------
    # conservative range-loop selection
    # ------------------------------------------------------------------
    def _store_supported(self, node: ast.Name, first_stmt: ast.stmt) -> bool:
        parent = self._packed_parent.get(node)
        if parent is first_stmt:
            return True
        if isinstance(parent, (ast.Tuple, ast.List)) and self._packed_parent.get(parent) is first_stmt:
            return True
        if isinstance(parent, ast.AugAssign) and parent.target is node:
            return True
        if (
            isinstance(parent, ast.Assign)
            and len(parent.targets) == 1
            and parent.targets[0] is node
        ):
            return True
        return False

    def _packed_loop_candidate(self, node: ast.For) -> tuple[list[str], list[str]] | None:
        if not node.body:
            return None
        targets = _fixed_int_unpack_names(node.body[0])
        if targets is None or len(targets) > PACKED_LOCAL_LIMIT:
            return None
        if len(set(targets)) != len(targets):
            return None

        loop_nodes = set(ast.walk(node))
        first_stmt = node.body[0]
        loop_target_name = node.target.id if isinstance(node.target, ast.Name) else None
        if loop_target_name in targets:
            return None

        # Optimized values are authoritative only in packed storage. They must
        # not escape the loop, and every in-loop store must be synchronizable.
        for name in targets:
            for occurrence in ast.walk(self._packed_original_tree):
                if not isinstance(occurrence, ast.Name) or occurrence.id != name:
                    continue
                if occurrence not in loop_nodes:
                    return None
                if isinstance(occurrence.ctx, ast.Store) and not self._store_supported(
                    occurrence, first_stmt
                ):
                    return None

        # Cache a few frequently loaded, read-only external integer scalars.
        stores = Counter(
            occurrence.id
            for occurrence in ast.walk(node)
            if isinstance(occurrence, ast.Name)
            and isinstance(occurrence.ctx, ast.Store)
        )
        loads = Counter(
            occurrence.id
            for occurrence in ast.walk(node)
            if isinstance(occurrence, ast.Name)
            and isinstance(occurrence.ctx, ast.Load)
            and occurrence.id in self.variables
            and occurrence.id not in targets
            and occurrence.id != loop_target_name
        )
        readonly = [
            name
            for name, _count in loads.most_common()
            if stores[name] == 0
        ][:PACKED_CACHE_LIMIT]
        return targets, readonly

    def _prepare_packed_cache(self, names: list[str]) -> dict[str, PackedI64Ref]:
        mapping: dict[str, PackedI64Ref] = {}
        for index, name in enumerate(names):
            dst = self._packed_cache_slot(index)
            mark = self.temps.mark()
            try:
                packed = self._pack_word(self.variables[name])
                self.packed_ops.copy(dst, packed)
            finally:
                self.temps.rewind(mark)
            mapping[name] = dst
        return mapping

    def _compile_for_range_control(self, node: ast.For) -> None:
        # Nested packed scopes would need disjoint mobile storage. Keep the first
        # slice simple and fall back for nested candidates.
        if self._active_packed_locals:
            return super()._compile_for_range_control(node)

        candidate = self._packed_loop_candidate(node)
        if candidate is None:
            return super()._compile_for_range_control(node)
        target_names, readonly_names = candidate

        locals_map = {
            name: self._packed_local_slot(index)
            for index, name in enumerate(target_names)
        }
        readonly_map = self._prepare_packed_cache(readonly_names)
        previous_locals = self._active_packed_locals
        previous_readonly = self._active_packed_readonly
        self._active_packed_locals = locals_map
        self._active_packed_readonly = readonly_map
        try:
            super()._compile_for_range_control(node)
        finally:
            self._active_packed_locals = previous_locals
            self._active_packed_readonly = previous_readonly

    # ------------------------------------------------------------------
    # fixed-arity input directly into active packed shadows
    # ------------------------------------------------------------------
    def _read_int_unpack_line(self, targets: list[ast.AST], node: ast.AST) -> None:
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if (
            len(names) != len(targets)
            or not names
            or any(name not in self._active_packed_locals for name in names)
        ):
            return super()._read_int_unpack_line(targets, node)

        destinations = [self._active_packed_locals[name] for name in names]
        line_open = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        active = self.temps.cell()
        selector = self.temps.cell()
        route = self.temps.cell()
        token_gate = self.temps.cell()
        done = self.temps.cell()
        token = self._packed_input_token()

        for destination in destinations:
            self.packed_ops.clear(destination)
        for cell in (
            line_open,
            has_token,
            end_line,
            active,
            selector,
            route,
            token_gate,
            done,
        ):
            self.bf.clear(cell)
        self.backend.packed64.clear(token)
        self.bf.set_const(line_open, 1)
        self.bf.set_const(active, 1)

        self.bf.begin_while(active)
        self.bf.add_const(active, -1)
        self.backend.read_packed_s64_line_token(
            token,
            has_token,
            end_line,
            self.workspace_base,
        )
        for target_index, destination in enumerate(destinations):
            self.backend._eq_byte_const(route, selector, target_index)
            self.bf.begin_while(route)
            self.bf.add_const(route, -1)
            self.backend.copy_cell(has_token, token_gate, self.backend.s0)
            self.bf.begin_while(token_gate)
            self.bf.add_const(token_gate, -1)
            self.packed_ops.copy(destination, token)
            self.bf.end_while(token_gate)
            self.bf.end_while(route)

        self.bf.add_const(selector, 1)
        self._close_line_if_end(line_open, end_line)
        self.backend.copy_cell(line_open, active, self.backend.s0)
        self.backend._eq_byte_const(done, selector, len(destinations))
        self.bf.begin_while(done)
        self.bf.add_const(done, -1)
        self.bf.clear(active)
        self.bf.end_while(done)
        self.bf.end_while(active)

        self.backend.drain_to_line_end(line_open, self.workspace_base)
        self.backend.packed64.clear(token)

    # ------------------------------------------------------------------
    # expressions / indexes
    # ------------------------------------------------------------------
    def _pack_index(self, node: ast.AST) -> PackedI64Ref:
        packed = self._packed_ref_for_expr(node)
        if packed is not None:
            return packed
        return super()._pack_index(node)

    def compile_expr(self, node: ast.AST):
        if isinstance(node, ast.Name):
            packed = self._packed_name(node.id)
            if packed is not None:
                return self._quad_from_packed(packed)

        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
        ):
            left = self._packed_ref_for_expr(node.left)
            right = self._packed_ref_for_expr(node.comparators[0])
            if left is not None and right is not None:
                result = self._new_word(0)
                out = result.bit(0)
                op = node.ops[0]
                if isinstance(op, ast.Eq):
                    self.packed_ops.equal(out, left, right)
                elif isinstance(op, ast.NotEq):
                    self.packed_ops.equal(out, left, right)
                    self.backend._toggle_bit(out, self.backend.s0)
                    self.backend._clear_scratch()
                elif isinstance(op, ast.Lt):
                    self.packed_ops.signed_lt(out, left, right)
                elif isinstance(op, ast.Gt):
                    self.packed_ops.signed_lt(out, right, left)
                elif isinstance(op, ast.LtE):
                    self.packed_ops.signed_lt(out, right, left)
                    self.backend._toggle_bit(out, self.backend.s0)
                    self.backend._clear_scratch()
                elif isinstance(op, ast.GtE):
                    self.packed_ops.signed_lt(out, left, right)
                    self.backend._toggle_bit(out, self.backend.s0)
                    self.backend._clear_scratch()
                else:
                    return super().compile_expr(node)
                return result

        return super().compile_expr(node)

    # ------------------------------------------------------------------
    # stores / augmented arithmetic
    # ------------------------------------------------------------------
    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self._active_packed_locals
        ):
            dst = self._active_packed_locals[node.targets[0].id]
            packed = self._packed_ref_for_expr(node.value)
            if packed is not None:
                self.packed_ops.copy(dst, packed)
            else:
                self._assign_packed_from_quad(dst, self.compile_expr(node.value))
            return

        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in self._active_packed_locals
        ):
            dst = self._active_packed_locals[node.target.id]
            rhs = self._packed_ref_for_expr(node.value)
            if rhs is not None and isinstance(node.op, ast.Add):
                self.packed_ops.add_inplace(dst, rhs)
                return
            if rhs is not None and isinstance(node.op, ast.Sub):
                self.packed_ops.sub_inplace(dst, rhs)
                return

            # Correctness fallback for uncommon augmented operators: synchronize
            # the shadow into the normal Quad variable, use established lowering,
            # then repack the updated value.
            quad_dst = self.variables[node.target.id]
            current = self._quad_from_packed(dst)
            self.backend.copy64(quad_dst, current)
            super()._compile_stmt_inner(node)
            self._assign_packed_from_quad(dst, quad_dst)
            return

        return super()._compile_stmt_inner(node)


__all__ = ["CompileError", "PythonToBFStream"]
