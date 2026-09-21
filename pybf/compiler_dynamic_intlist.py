"""One statically owned runtime integer list with proven alias views.

This route supports a single top-level input-list construction and unconditional
alias bindings, with len/sum/clear uses only. It is not the general heap/object
model: rebinding, escaping, indexing and multiple dynamic owners stay on the
existing route. Selection is based on uses/definitions, never a problem name.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from bfpacked import PackedU32Ref
from bfpacked64 import PackedI64Ref
from bfpackedseq import BACK, REDUCTION_WORKSPACE_CELLS, RuntimePackedIntSequence
from compiler_dynamic_charlist import select_dynamic_char_list
from compiler_stream import CompileError, PythonToBFStream as _Base
from transpiler import _is_list_map_int_input_split


@dataclass(frozen=True)
class DynamicIntListSelection:
    owner: str
    bindings: dict[str, ast.Assign]


def select_dynamic_int_list(tree: ast.Module) -> DynamicIntListSelection | None:
    if not isinstance(tree, ast.Module) or select_dynamic_char_list(tree) is not None:
        return None
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name)
                  and _is_list_map_int_input_split(node.value)]
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
        if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "clear":
            call = parents.get(parent)
            if (isinstance(call, ast.Call) and call.func is parent
                    and not call.args and not call.keywords
                    and isinstance(parents.get(call), ast.Expr)):
                continue
        return None
    return DynamicIntListSelection(owner, bindings)


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
                    inference_tree.body[index].value = ast.Constant(value=0)
        super().__init__(inference_tree, string_capacity=string_capacity,
                         list_capacity=list_capacity,
                         runtime_charlist_base=runtime_charlist_base)
        self._int_binding_nodes = (set(self.dynamic_int_selection.bindings.values())
                                   if self.dynamic_int_selection is not None else set())
        self.runtime_intlist_base = None
        self.dynamic_int_sequence = None
        if self.dynamic_int_selection is not None:
            self._int_total = PackedI64Ref(self.temps.top)
            self._int_length = PackedU32Ref(self.temps.top + 8)
            self.temps.top += 12  # Persistent header; statement rewind cannot reuse it.
            # Provisional base is used only by the layout discovery pass. The
            # public entrypoint reruns lowering after measuring all temporaries.
            base = runtime_intlist_base
            if base is None:
                base = self.temps.top + REDUCTION_WORKSPACE_CELLS + 10
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
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("sum", "len") and len(node.args) == 1
                and not node.keywords and self._dynamic_int_name(node.args[0])):
            ref = self._int_total if node.func.id == "sum" else self._int_length
            return self._cached_int_value(ref)
        return super().compile_expr(node)

    def _compile_stmt_inner(self, node):
        selection = self.dynamic_int_selection
        if selection is not None and node in self._int_binding_nodes:
            if node is selection.bindings[selection.owner]:
                self.dynamic_int_sequence.read_lf_terminated_s64s(self.bf)
                self.dynamic_int_sequence.sum_and_length(self.bf, self._int_total, self._int_length)
            # All aliases denote this same statically proven object. No value
            # copy, additional sequence or heap lookup is emitted.
            return
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "clear"
                and self._dynamic_int_name(node.value.func.value)):
            for i in range(8):
                self.bf.clear(self._int_total.byte(i))
            for i in range(4):
                self.bf.clear(self._int_length.byte(i))
            self.bf.clear(self.dynamic_int_sequence.base)
            self.bf.clear(self.dynamic_int_sequence.base + BACK)
            # Logical clear: old payload is unreachable, not allocator reuse.
            return
        return super()._compile_stmt_inner(node)


__all__ = ["CompileError", "PythonToBFStream", "select_dynamic_int_list"]
