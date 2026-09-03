"""Generic heap-backed runtime ``list[int]`` lowering.

This is the first bridge from the reusable dynamic-object runtime into ordinary
Python AST lowering.  It is deliberately a *generic feature route*, not a
whole-program algorithm specialization: any program expressible by the scalar
v2 subset plus the list operations below can use it.

Supported dynamic-list surface:

* ``list(map(int, input().split()))`` with runtime line length;
* integer list literals, including ``[]``;
* Python reference assignment (``b = a`` aliases the same list object);
* ``len(a)`` and list truth values;
* non-negative or Python-style negative ``a[i]`` reads;
* ``a.append(expr)``;
* ``for x in a`` and repeated passes over the same list.

The current heap representation is correctness-first (one heap block per
integer and ordinal lookup).  It removes the old fixed-capacity semantic limit
and keeps emitted source independent of runtime list length, but indexed/list
iteration is not yet the final O(n) contiguous/chunked backend.  The route is
therefore isolated so its ABI can be replaced without changing Python syntax.

Tape layout uses two compiler passes.  A planning pass records the temporary
high-water mark with ``PeakTempArena``.  The real pass places the heap sentinel
strictly to the right of that peak, preventing runtime heap data from aliasing a
later compile-time temporary.
"""

from __future__ import annotations

import ast

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bfdynlist import DynamicIntListRuntime
from bfheap import HeapBlockArena
from bfobjects import HANDLE_BYTES, ObjectHandleCore, ObjectHandleRef
from bfopt import optimize_bf
from bfpacked import PackedU32Core, PackedU32Ref, U32_BYTES
from bfpacked64 import I64_BYTES, PackedI64Core, PackedI64Ref
from bftemparena import PeakTempArena
from bftokens import BinaryTokenIO
from transpiler import _is_list_map_int_input_split, infer_list_names
from transpiler_v2 import CompileError, PythonToBFV2, _AssignedNames, clean_bf


MASK64 = (1 << WORD_BITS) - 1
HEAP_GUARD_CELLS = 2
DYNAMIC_CORE_SCRATCH_CELLS = 4


def _signed_literal(node: ast.AST) -> int | None:
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


class DynamicIntListCompiler(PythonToBFV2):
    """Scalar v2 compiler with heap-backed Python ``list[int]`` references."""

    def __init__(self, tree: ast.AST, *, heap_left_sentinel: int | None = None) -> None:
        names = _AssignedNames()
        names.visit(tree)
        self.list_names = infer_list_names(tree)
        self.all_names = set(names.names)

        # The route intentionally owns int lists only.  String variables are
        # left to the established richer generic compiler.
        self.variables: dict[str, Int64Ref] = {}
        self.list_handles: dict[str, ObjectHandleRef] = {}
        static_top = 0
        for name in sorted(self.all_names):
            if name in self.list_names:
                self.list_handles[name] = ObjectHandleRef(static_top)
                static_top += HANDLE_BYTES
            else:
                self.variables[name] = Int64Ref(static_top)
                static_top += WORD_BITS

        self.next_handle = ObjectHandleRef(static_top)
        static_top += HANDLE_BYTES

        self.bf = BFEmitter()
        self.workspace_base = static_top
        self.scratch_base = self.workspace_base + self.SHARED_WORKSPACE_CELLS
        self.backend = BinaryTokenIO(self.bf, scratch_base=self.scratch_base)

        dynamic_scratch = self.scratch_base + Binary64Core.SCRATCH_CELLS
        self.packed = PackedU32Core(self.bf, dynamic_scratch)
        self.handles = ObjectHandleCore(self.bf, dynamic_scratch)
        self.packed64 = PackedI64Core(self.bf, dynamic_scratch)
        self.list_workspace_base = dynamic_scratch + DYNAMIC_CORE_SCRATCH_CELLS
        temp_base = self.list_workspace_base + DynamicIntListRuntime.WORKSPACE_CELLS
        self.temps = PeakTempArena(temp_base)

        # During the planning pass heap placement may overlap future temps; the
        # emitted code is discarded.  The second pass uses peak+guard.
        if heap_left_sentinel is None:
            heap_left_sentinel = temp_base + HEAP_GUARD_CELLS
        self.heap_left_sentinel = heap_left_sentinel
        self.heap = HeapBlockArena(
            self.bf,
            left_sentinel=heap_left_sentinel,
            next_handle=self.next_handle,
            scratch_base=dynamic_scratch,
        )
        self.dynamic_lists = DynamicIntListRuntime(
            self.heap,
            packed=self.packed,
            handles=self.handles,
            packed64=self.packed64,
            workspace_base=self.list_workspace_base,
        )

    # ------------------------------------------------------------------
    # temporary packed/runtime references
    # ------------------------------------------------------------------
    def _new_handle(self) -> ObjectHandleRef:
        ref = ObjectHandleRef(self.temps.top)
        self.temps.top += HANDLE_BYTES
        return ref

    def _new_u32(self) -> PackedU32Ref:
        ref = PackedU32Ref(self.temps.top)
        self.temps.top += U32_BYTES
        return ref

    def _new_packed_i64(self) -> PackedI64Ref:
        ref = PackedI64Ref(self.temps.top)
        self.temps.top += I64_BYTES
        return ref

    def _u32_to_int64(self, src: PackedU32Ref) -> Int64Ref:
        wide = self._new_packed_i64()
        self.packed64.clear(wide)
        for i in range(U32_BYTES):
            self.packed._copy_cell(src.byte(i), wide.byte(i), self.packed.s0)
        result = self._new_word()
        self.packed64.to_int64(result, wide)
        return result

    def _int64_to_u32(self, src: Int64Ref) -> PackedU32Ref:
        wide = self._new_packed_i64()
        self.packed64.from_int64(wide, src)
        result = self._new_u32()
        self.packed.clear(result)
        for i in range(U32_BYTES):
            self.packed._copy_cell(wide.byte(i), result.byte(i), self.packed.s0)
        return result

    # ------------------------------------------------------------------
    # list expression helpers
    # ------------------------------------------------------------------
    def _expr_is_list(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.list_names
        if isinstance(node, ast.List):
            return True
        if _is_list_map_int_input_split(node):
            return True
        return False

    def _read_runtime_int_list(self) -> ObjectHandleRef:
        ref = self._new_handle()
        self.dynamic_lists.create_empty(ref)

        active = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        token_gate = self.temps.cell()
        end_gate = self.temps.cell()
        value = self._new_word()

        self.bf.set_const(active, 1)
        self.bf.begin_while(active)
        self.bf.add_const(active, -1)
        self.backend.read_s64_line_token(
            value,
            has_token,
            end_line,
            self.workspace_base,
        )

        self.backend.copy_cell(has_token, token_gate, self.backend.s0)
        self.bf.begin_while(token_gate)
        self.bf.add_const(token_gate, -1)
        self.dynamic_lists.append_int64(ref, value)
        self.bf.end_while(token_gate)

        # Continue only while the logical line remains open.  A token ending in
        # horizontal space intentionally performs one more token read, which
        # consumes any trailing spaces and stops at LF/EOF without line bleed.
        self.bf.set_const(active, 1)
        self.backend.copy_cell(end_line, end_gate, self.backend.s0)
        self.bf.begin_while(end_gate)
        self.bf.add_const(end_gate, -1)
        self.bf.clear(active)
        self.bf.end_while(end_gate)
        self.bf.end_while(active)
        return ref

    def _list_handle_from_expr(self, node: ast.AST) -> ObjectHandleRef:
        if isinstance(node, ast.Name) and node.id in self.list_handles:
            return self.list_handles[node.id]
        if _is_list_map_int_input_split(node):
            return self._read_runtime_int_list()
        if isinstance(node, ast.List):
            ref = self._new_handle()
            self.dynamic_lists.create_empty(ref)
            for elt in node.elts:
                if self._expr_is_list(elt):
                    raise self._error(elt, "dynamic int lists cannot contain list objects yet")
                value = self.compile_expr(elt)
                self.dynamic_lists.append_int64(ref, value)
            return ref
        raise self._error(node, "unsupported dynamic int-list expression")

    def _list_length(self, ref: ObjectHandleRef) -> Int64Ref:
        packed_length = self._new_u32()
        self.dynamic_lists.read_length(packed_length, ref)
        return self._u32_to_int64(packed_length)

    def _normalized_index(self, node: ast.AST, ref: ObjectHandleRef) -> PackedU32Ref:
        literal = _signed_literal(node)
        if literal is not None and literal >= 0:
            out = self._new_u32()
            self.packed.set_u32(out, literal)
            return out

        raw = self.compile_expr(node)
        normalized = self._copy_new(raw)
        sign = self.temps.cell()
        self.backend.copy_cell(normalized.bit(WORD_BITS - 1), sign, self.backend.s0)
        self.bf.begin_while(sign)
        self.bf.add_const(sign, -1)
        length = self._list_length(ref)
        added = self._new_word()
        self.backend.add64(added, normalized, length)
        self.backend.copy64(normalized, added)
        self.bf.end_while(sign)
        return self._int64_to_u32(normalized)

    def _load_subscript(self, node: ast.Subscript) -> Int64Ref:
        if not isinstance(node.value, ast.Name) or node.value.id not in self.list_handles:
            raise self._error(node, "only dynamic int-list subscripting is supported")
        ref = self.list_handles[node.value.id]
        index = self._normalized_index(node.slice, ref)
        result = self._new_word()
        self.dynamic_lists.get_int64(result, ref, index)
        return result

    # ------------------------------------------------------------------
    # scalar expression extensions
    # ------------------------------------------------------------------
    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if isinstance(node, ast.Name) and node.id in self.list_handles:
            return self._list_length(self.list_handles[node.id])

        if isinstance(node, ast.Subscript):
            return self._load_subscript(node)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len" and len(node.args) == 1 and not node.keywords:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id in self.list_handles:
                    return self._list_length(self.list_handles[arg.id])

        if isinstance(node, ast.BinOp) and (
            self._expr_is_list(node.left) or self._expr_is_list(node.right)
        ):
            raise self._error(node, "dynamic list arithmetic/concatenation is not lowered yet")

        return super().compile_expr(node)

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------
    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            if all(isinstance(t, ast.Name) for t in node.targets) and self._expr_is_list(node.value):
                value = self._list_handle_from_expr(node.value)
                for target in node.targets:
                    assert isinstance(target, ast.Name)
                    if target.id not in self.list_handles:
                        raise self._error(target, "cannot assign list to scalar variable")
                    self.dynamic_lists.alias(self.list_handles[target.id], value)
                return
            if any(isinstance(t, ast.Name) and t.id in self.list_handles for t in node.targets):
                raise self._error(node, "cannot assign scalar value to dynamic list variable")
            if any(isinstance(t, ast.Subscript) for t in node.targets):
                raise self._error(node, "dynamic list item assignment is not lowered yet")

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in self.list_handles
                and len(call.args) == 1
                and not call.keywords
            ):
                value = self.compile_expr(call.args[0])
                self.dynamic_lists.append_int64(
                    self.list_handles[call.func.value.id],
                    value,
                )
                return

        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id in self.list_handles:
            self._compile_for_dynamic_list(node)
            return

        return super()._compile_stmt_inner(node)

    def _compile_for_dynamic_list(self, node: ast.For) -> None:
        if node.orelse:
            raise self._error(node, "for ... else on dynamic lists is not lowered yet")
        if not isinstance(node.target, ast.Name) or node.target.id in self.list_handles:
            raise self._error(node, "dynamic list iteration target must be a scalar name")
        assert isinstance(node.iter, ast.Name)

        ref = self.list_handles[node.iter.id]
        target = self._var(node.target)
        index_scalar = self._new_word(0)
        index_packed = self._new_u32()
        self.packed.clear(index_packed)
        length = self._list_length(ref)
        control = self.temps.cell()
        self.backend.slt64(control, index_scalar, length)

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        loaded = self._new_word()
        self.dynamic_lists.get_int64(loaded, ref, index_packed)
        self.backend.copy64(target, loaded)
        for stmt in node.body:
            self.compile_stmt(stmt)
        self.backend._inc64_inplace(index_scalar)
        self.packed.increment(index_packed)
        self.backend.slt64(control, index_scalar, length)
        self.bf.end_while(control)

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError("expected ast.Module")
        self.heap.initialize()
        for stmt in tree.body:
            self.compile_stmt(stmt)
        return clean_bf(self.bf.code())


class _DynamicRouteFacts(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_runtime_input_list = False
        self.has_obvious_string_value = False

    def visit_Call(self, node: ast.Call) -> None:
        if _is_list_map_int_input_split(node):
            self.has_runtime_input_list = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Literal strings passed directly to print are supported by v2, so only
        # reject strings that participate in assignments/expressions elsewhere
        # during the actual compilation pass. This flag is kept diagnostic-only.
        if isinstance(node.value, str):
            self.has_obvious_string_value = True


def lower_dynamic_int_list_program_if_supported(tree: ast.AST) -> str | None:
    """Try the generic dynamic-int-list route; return None for normal fallback."""
    if not isinstance(tree, ast.Module):
        return None
    facts = _DynamicRouteFacts()
    facts.visit(tree)
    if not facts.has_runtime_input_list:
        return None

    # Pass 1: compile with provisional heap placement solely to observe the
    # maximum temporary address. Unsupported syntax simply falls back to the
    # established fixed-ABI compiler.
    try:
        planning = DynamicIntListCompiler(tree)
        planning.compile_module(tree)
    except CompileError:
        return None

    heap_left_sentinel = planning.temps.peak + HEAP_GUARD_CELLS

    try:
        compiler = DynamicIntListCompiler(
            tree,
            heap_left_sentinel=heap_left_sentinel,
        )
        code = compiler.compile_module(tree)
    except CompileError:
        return None

    if compiler.temps.peak >= heap_left_sentinel:
        raise AssertionError(
            "dynamic-list layout planning underestimated the temporary peak"
        )
    return optimize_bf(code)


__all__ = [
    "DynamicIntListCompiler",
    "lower_dynamic_int_list_program_if_supported",
]
