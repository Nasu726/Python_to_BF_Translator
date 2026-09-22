"""One statically owned runtime integer list with proven alias views.

This route supports one top-level input-list or singleton-repeat construction
and unconditional alias bindings, with len/sum/clear/index uses. It is not the
general heap/object model: rebinding, escaping and multiple dynamic owners stay
on the existing route. Selection is based on uses/definitions, never a problem
name.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from bfcore import Int64Ref
from bfstreamseq import _extract_packed_sign
from bfpacked import PackedU32Ref
from bfpacked64 import PackedI64Ref
from bfpackedops import PackedI64Ops
from bfpackedseq import (ACCESS_WORKSPACE_CELLS, BACK,
                         LOAD_ACCESS_WORKSPACE_CELLS,
                         REDUCTION_WORKSPACE_CELLS, RuntimePackedIntSequence)
from compiler_dynamic_charlist import select_dynamic_char_list
from compiler_stream import CompileError, PythonToBFStream as _Base
from transpiler import _is_list_map_int_input_split


@dataclass(frozen=True)
class DynamicIntListSelection:
    owner: str
    bindings: dict[str, ast.Assign]
    needs_load: bool
    needs_store: bool


def dynamic_int_workspace_cells(selection: DynamicIntListSelection) -> int:
    width = REDUCTION_WORKSPACE_CELLS
    if selection.needs_load:
        width = max(width, LOAD_ACCESS_WORKSPACE_CELLS)
    if selection.needs_store:
        width = max(width, ACCESS_WORKSPACE_CELLS)
    return width


def _repeat_parts(node):
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    if isinstance(node.left, ast.List) and len(node.left.elts) == 1:
        return node.left.elts[0], node.right, False
    if isinstance(node.right, ast.List) and len(node.right.elts) == 1:
        return node.right.elts[0], node.left, True
    return None


def select_dynamic_int_list(tree: ast.Module) -> DynamicIntListSelection | None:
    if not isinstance(tree, ast.Module) or select_dynamic_char_list(tree) is not None:
        return None
    assignments = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)]
    candidates = [node for node in assignments
                  if _is_list_map_int_input_split(node.value)]
    # Preserve the established input-owner route when unrelated fixed repeats
    # coexist. Selecting new repeats must not silently restore the input's old
    # capacity bound. Multiple input owners retain the previous rejection.
    if not candidates:
        candidates = [node for node in assignments if _repeat_parts(node.value) is not None]
    if len(candidates) != 1 or candidates[0] not in tree.body:
        return None
    producer = candidates[0]
    owner = producer.targets[0].id
    bindings = {owner: producer}
    positions = {node: i for i, node in enumerate(tree.body)}
    for statement in tree.body[positions[producer] + 1:]:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Name)
                and statement.value.id in bindings):
            name = statement.targets[0].id
            if name in bindings:
                return None
            bindings[name] = statement
    binding_nodes = set(bindings.values())
    needs_load = False
    needs_store = False
    top_positions = {node: i for i, statement in enumerate(tree.body) for node in ast.walk(statement)}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    builtin_names = {"list", "map", "int", "input", "sum", "len"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return None
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in builtin_names:
            return None
        if not isinstance(node, ast.Name) or node.id not in bindings:
            continue
        binding = bindings[node.id]
        if node is binding.targets[0]:
            continue
        if not isinstance(node.ctx, ast.Load):
            return None
        if top_positions[node] <= positions[binding]:
            return None
        parent = parents[node]
        if parent in binding_nodes and parent.value is node:
            continue
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                and parent.func.id in ("len", "sum") and parent.args == [node]
                and not parent.keywords):
            continue
        if isinstance(parent, ast.Subscript) and parent.value is node:
            if isinstance(parent.slice, ast.Slice):
                return None
            if isinstance(parent.ctx, ast.Load):
                needs_load = True
                continue
            grandparent = parents.get(parent)
            if (isinstance(parent.ctx, ast.Store)
                    and isinstance(grandparent, ast.Assign)
                    and grandparent.targets == [parent]):
                needs_store = True
                continue
        if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "clear":
            call = parents.get(parent)
            if (isinstance(call, ast.Call) and call.func is parent
                    and not call.args and not call.keywords
                    and isinstance(parents.get(call), ast.Expr)):
                continue
        return None
    return DynamicIntListSelection(owner, bindings, needs_load, needs_store)


class PythonToBFStream(_Base):
    """Cache metadata for a statically proven single mutable int-list owner."""

    def __init__(self, tree, *, string_capacity=255, list_capacity=64,
                 runtime_charlist_base=None, runtime_intlist_base=None):
        self.dynamic_int_selection = select_dynamic_int_list(tree)
        inference_tree = tree
        if self.dynamic_int_selection is not None:
            # These names denote one runtime object, not fixed-capacity value
            # slots. Keep unused scalar-sized placeholders in legacy name/layout
            # analysis; shared identity is proven statically by this layer.
            inference_tree = copy.deepcopy(tree)
            selected = set(self.dynamic_int_selection.bindings.values())
            for index, statement in enumerate(tree.body):
                if statement in selected:
                    repeat = _repeat_parts(statement.value)
                    if repeat is None:
                        inference_tree.body[index].value = ast.Constant(value=0)
                    else:
                        # Preserve operand reads in lifetime analysis; replacing
                        # the entire repeat with 0 could reuse x's slot for n.
                        value, count, count_first = copy.deepcopy(repeat)
                        left, right = (count, value) if count_first else (value, count)
                        inference_tree.body[index].value = ast.BinOp(
                            left=left, op=ast.Add(), right=right)
        super().__init__(inference_tree, string_capacity=string_capacity,
                         list_capacity=list_capacity,
                         runtime_charlist_base=runtime_charlist_base)
        self._int_binding_nodes = (set(self.dynamic_int_selection.bindings.values())
                                   if self.dynamic_int_selection is not None else set())
        self._int_provisional_layout = runtime_intlist_base is None
        self.runtime_intlist_base = None
        self.dynamic_int_sequence = None
        if self.dynamic_int_selection is not None:
            self._int_total = PackedI64Ref(self.temps.top)
            self._int_length = PackedI64Ref(self.temps.top + 8)
            self.temps.top += 16  # Persistent header; statement rewind cannot reuse it.
            # Provisional base is used only by the layout discovery pass. The
            # public entrypoint reruns lowering after measuring all temporaries.
            base = runtime_intlist_base
            if base is None:
                base = self.temps.top + dynamic_int_workspace_cells(
                    self.dynamic_int_selection
                ) + 10
            self.runtime_intlist_base = base
            self.dynamic_int_sequence = RuntimePackedIntSequence(base)

    def _dynamic_int_name(self, node):
        return (self.dynamic_int_selection is not None and isinstance(node, ast.Name)
                and node.id in self.dynamic_int_selection.bindings)

    def _cached_int_value(self, ref):
        snapshot = self._new_packed_i64()
        for i in range(8):
            self.bf.clear(snapshot.byte(i))
        for i in range(ref.cells):
            self.backend.copy_cell(ref.byte(i), snapshot.byte(i), self.backend.s0)
        result = self._new_word()
        self.backend.copy64(result, snapshot)  # Conversion consumes only the snapshot.
        return result

    def compile_expr(self, node):
        if (isinstance(node, ast.Subscript)
                and self._dynamic_int_name(node.value)):
            return self._load_dynamic_int(node.slice)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("sum", "len") and len(node.args) == 1
                and not node.keywords and self._dynamic_int_name(node.args[0])):
            ref = self._int_total if node.func.id == "sum" else self._int_length
            return self._cached_int_value(ref)
        return super().compile_expr(node)

    def _dynamic_int_packed_ops(self) -> PackedI64Ops:
        scratch = self.temps.top
        self.temps.top += PackedI64Ops.SCRATCH_CELLS
        return PackedI64Ops(self.bf, scratch)

    def _ensure_provisional_access_base(self) -> None:
        if self._int_provisional_layout:
            self.runtime_intlist_base = (
                self.temps.top
                + dynamic_int_workspace_cells(self.dynamic_int_selection)
                + 10
            )
            self.dynamic_int_sequence = RuntimePackedIntSequence(
                self.runtime_intlist_base
            )

    def _normalize_dynamic_int_index(self, node: ast.AST) -> PackedI64Ref:
        """Evaluate one Python index and normalize a negative value once."""
        normalized = self._packed_int_expr(
            node, "dynamic integer-list index must be an integer"
        )
        scratch = [self.temps.cell() for _ in range(4)]
        negative = scratch[0]
        _extract_packed_sign(self.bf, normalized.byte(7), *scratch)
        ops = self._dynamic_int_packed_ops()
        self.bf.begin_while(negative)
        self.bf.add_const(negative, -1)
        ops.add_inplace(normalized, self._int_length)
        self.bf.end_while(negative)
        return normalized

    def _load_dynamic_int(self, index_node: ast.AST):
        index = self._normalize_dynamic_int_index(index_node)
        packed = self._new_packed_i64()
        self._ensure_provisional_access_base()
        self.dynamic_int_sequence.load_value(self.bf, index, packed)
        result = self._new_word()
        self.backend.copy64(result, packed)
        return result

    def _store_dynamic_int(self, target: ast.Subscript, value_node: ast.AST) -> None:
        # Python evaluates the RHS before the subscription target/index.
        value = self._packed_int_expr(
            value_node, "dynamic integer-list item must be an integer"
        )
        index = self._normalize_dynamic_int_index(target.slice)
        previous = self._new_packed_i64()
        found = self.temps.cell()
        ops = self._dynamic_int_packed_ops()
        self._ensure_provisional_access_base()
        self.dynamic_int_sequence.exchange_value(
            self.bf, index, value, previous, found=found,
        )
        self.bf.begin_while(found)
        self.bf.add_const(found, -1)
        ops.sub_inplace(self._int_total, previous)
        ops.add_inplace(self._int_total, value)
        self.bf.end_while(found)

    def _packed_int_expr(self, node, error_message):
        if (isinstance(node, ast.Constant)
                or (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
                    and isinstance(node.operand, ast.Constant))):
            literal = node.value if isinstance(node, ast.Constant) else node.operand.value
            if isinstance(literal, (int, bool)):
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                    literal = -literal
                packed = self._new_packed_i64()
                for i in range(8):
                    self.bf.set_const(packed.byte(i), (literal >> (8 * i)) & 255)
                return packed
        value = self.compile_expr(node)
        if not isinstance(value, Int64Ref):
            raise CompileError(error_message)
        return self._pack_word(value)

    def _repeat_operand(self, node):
        return self._packed_int_expr(
            node, "dynamic integer-list repetition requires integer operands"
        )

    def _construct_repeat(self, parts):
        value_node, count_node, count_first = parts
        # Snapshot operands immediately: later evaluation may perform input or
        # reuse temporary scalar storage. Both operands evaluate even for n<=0.
        if count_first:
            count = self._repeat_operand(count_node)
            value = self._repeat_operand(value_node)
        else:
            value = self._repeat_operand(value_node)
            count = self._repeat_operand(count_node)
        # Normalize every negative int64 directly in the packed snapshot;
        # positive counts retain all 64 bits, including their upper u32.
        scratch = [self.temps.cell() for _ in range(4)]
        negative = scratch[0]
        _extract_packed_sign(self.bf, count.byte(7), *scratch)
        self.bf.begin_while(negative)
        self.bf.clear(negative)
        for i in range(8):
            self.bf.clear(count.byte(i))
        self.bf.end_while(negative)
        discarded_length = self._new_packed_u32()
        if self._int_provisional_layout:
            self.runtime_intlist_base = self.temps.top + dynamic_int_workspace_cells(
                self.dynamic_int_selection
            ) + 10
            self.dynamic_int_sequence = RuntimePackedIntSequence(self.runtime_intlist_base)
        self.dynamic_int_sequence.repeat_value(self.bf, count, value)
        # Reduction uses a u32 traversal length, but repetition already knows
        # its exact signed-positive count. Retain the original 64-bit length.
        self.dynamic_int_sequence.sum_and_length(self.bf, self._int_total, discarded_length)
        for i in range(8):
            self.backend.copy_cell(count.byte(i), self._int_length.byte(i), self.backend.s0)

    def _compile_stmt_inner(self, node):
        selection = self.dynamic_int_selection
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and self._dynamic_int_name(node.targets[0].value)):
            self._store_dynamic_int(node.targets[0], node.value)
            return
        if selection is not None and node in self._int_binding_nodes:
            if node is selection.bindings[selection.owner]:
                repeat = _repeat_parts(node.value)
                if repeat is not None:
                    self._construct_repeat(repeat)
                else:
                    self.dynamic_int_sequence.read_lf_terminated_s64s(self.bf)
                    for i in range(8):
                        self.bf.clear(self._int_length.byte(i))
                    self.dynamic_int_sequence.sum_and_length(
                        self.bf, self._int_total, PackedU32Ref(self._int_length.base))
            # All aliases denote this same statically proven object. No value
            # copy, additional sequence or heap lookup is emitted.
            return
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "clear"
                and self._dynamic_int_name(node.value.func.value)):
            for i in range(8):
                self.bf.clear(self._int_total.byte(i))
            for i in range(8):
                self.bf.clear(self._int_length.byte(i))
            self.bf.clear(self.dynamic_int_sequence.base)
            self.bf.clear(self.dynamic_int_sequence.base + BACK)
            # Logical clear: old payload is unreachable, not allocator reuse.
            return
        return super()._compile_stmt_inner(node)


__all__ = [
    "CompileError",
    "PythonToBFStream",
    "dynamic_int_workspace_cells",
    "select_dynamic_int_list",
]
