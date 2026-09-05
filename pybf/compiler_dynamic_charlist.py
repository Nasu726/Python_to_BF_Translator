"""Restricted runtime-sized character-list lowering.

This layer connects the ordinary ``list(input())`` frontend to
``RuntimeByteSequence`` without pretending that the compiler already has a
general dynamic-object model. Selection is deliberately conservative: one
name, one construction, and only operations whose representation is already
defined by the byte-sequence primitive.

Unsupported shapes continue through the established fixed ``StringRef`` path.
That preserves compatibility while the scalable route grows one independently
testable semantic slice at a time.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bfpacked64 import PackedI64Ref
from bfstreamseq import RuntimeByteSequence
from compiler_charconv import _empty_join_arg, _is_list_input
from compiler_charindex import _must_char_value_names
from compiler_formatcompact import CompileError
from compiler_formatcompact import PythonToBFStream as _BasePythonToBFStream
from transpiler_full import _LoopContext


@dataclass(frozen=True)
class DynamicCharListSelection:
    """One statically owned runtime-sized character-list name."""

    name: str


def _dynamic_subscript_char_names(tree: ast.AST, char_list_name: str) -> set[str]:
    """Names whose every assignment is a dynamic-list character result.

    This is narrower than the general one-character proof. In particular, a
    name assigned from a literal is left at normal string capacity because the
    established literal materializer creates a normal-capacity temporary.
    """
    writes: dict[str, list[ast.AST]] = {}
    stores: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            writes.setdefault(node.targets[0].id, []).append(node.value)

    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, values in writes.items():
            if name in result or len(values) != stores.get(name, 0):
                continue
            if values and all(
                (
                    isinstance(value, ast.Subscript)
                    and isinstance(value.value, ast.Name)
                    and value.value.id == char_list_name
                )
                or (isinstance(value, ast.Name) and value.id in result)
                for value in values
            ):
                result.add(name)
                changed = True
    return result


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _is_direct_print_join(
    node: ast.Name,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    join = parents.get(node)
    if not isinstance(join, ast.Call) or _empty_join_arg(join) is not node:
        return False
    call = parents.get(join)
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "print"
        and join in call.args
    )


def select_dynamic_char_list(
    tree: ast.Module,
) -> DynamicCharListSelection | None:
    """Select the first safe runtime-sized character-list frontend slice.

    The byte sequence is one-shot storage and does not yet carry general object
    identity. Require exactly one direct ``name = list(input())`` construction
    and allow name occurrences only as:

    * ``name[index]`` load or a simple-assignment target;
    * ``len(name)``;
    * ``for character in name`` with a distinct simple target;
    * a direct ``print(''.join(name))`` argument.

    A rejected selection is not a compilation error by itself. Higher layers
    retain the fixed-capacity compatibility implementation and its existing
    diagnostics.
    """
    if not isinstance(tree, ast.Module):
        return None

    constructions: list[tuple[str, ast.Name]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_list_input(node.value)
        ):
            continue
        constructions.append((node.targets[0].id, node.targets[0]))

    names = {name for name, _target in constructions}
    if len(constructions) != 1 or len(names) != 1:
        return None
    name, construction_target = constructions[0]
    parents = _parents(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id != name:
            continue

        if node is construction_target:
            continue
        if isinstance(node.ctx, ast.Store):
            # Any other binding is a rebind or unsupported target shape.
            return None

        parent = parents.get(node)
        if isinstance(parent, ast.Subscript) and parent.value is node:
            grandparent = parents.get(parent)
            if isinstance(parent.ctx, ast.Load):
                continue
            if (
                isinstance(parent.ctx, ast.Store)
                and isinstance(grandparent, ast.Assign)
                and len(grandparent.targets) == 1
                and grandparent.targets[0] is parent
            ):
                continue
            return None

        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "len"
            and len(parent.args) == 1
            and not parent.keywords
            and parent.args[0] is node
        ):
            continue

        if (
            isinstance(parent, ast.For)
            and parent.iter is node
            and isinstance(parent.target, ast.Name)
            and parent.target.id != name
        ):
            continue

        if _is_direct_print_join(node, parents):
            continue

        return None

    return DynamicCharListSelection(name)


class PythonToBFStream(_BasePythonToBFStream):
    """Final generic compiler with one optional runtime byte sequence."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
        runtime_charlist_base: int | None = None,
    ) -> None:
        selection = (
            select_dynamic_char_list(tree)
            if runtime_charlist_base is not None
            else None
        )
        self.dynamic_char_list_selection = selection
        self._runtime_sized_char_list_names = (
            {selection.name} if selection is not None else set()
        )
        self._forced_compact_char_names = set()
        if selection is not None:
            proven_chars = _must_char_value_names(tree, {selection.name})
            dynamic_results = _dynamic_subscript_char_names(tree, selection.name)
            self._forced_compact_char_names = proven_chars & dynamic_results
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
        )
        self.runtime_charlist_base = (
            runtime_charlist_base if selection is not None else None
        )
        self.dynamic_char_sequence = (
            RuntimeByteSequence(runtime_charlist_base)
            if selection is not None and runtime_charlist_base is not None
            else None
        )

    @property
    def uses_runtime_char_list(self) -> bool:
        return self.dynamic_char_sequence is not None

    def _is_dynamic_char_list_name(self, name: str) -> bool:
        selection = self.dynamic_char_list_selection
        return selection is not None and selection.name == name

    def _new_packed_i64(self) -> PackedI64Ref:
        result = PackedI64Ref(self.temps.top)
        self.temps.top += result.cells
        return result

    def _pack_word(self, value) -> PackedI64Ref:
        # A direct generic ``PackedI64Core.from_int64`` would preserve and copy
        # 64 distant Boolean cells one by one. First snapshot through Quad's
        # single emitted lane-walker. The snapshot is disposable, so each group
        # of eight bits can accumulate in a nearby now-zero marker and move to
        # the adjacent packed result once.
        nearby = self._new_word()
        self.backend.copy64(nearby, value)
        packed = self._new_packed_i64()

        for byte_index in range(8):
            accumulator = nearby.marker(byte_index * 4)
            self.bf.clear(accumulator)
            for within in range(8):
                bit = nearby.bit(byte_index * 8 + within)
                self.bf.begin_while(bit)
                self.bf.add_const(bit, -1)
                self.bf.add_const(accumulator, 1 << within)
                self.bf.end_while(bit)

            dst = packed.byte(byte_index)
            self.bf.clear(dst)
            self.bf.begin_while(accumulator)
            self.bf.add_const(accumulator, -1)
            self.bf.add_const(dst, 1)
            self.bf.end_while(accumulator)
        return packed

    def _pack_index(self, node: ast.AST) -> PackedI64Ref:
        return self._pack_word(self.compile_expr(node))

    def _dynamic_length(self):
        sequence = self.dynamic_char_sequence
        assert sequence is not None

        packed = self._new_packed_i64()
        self.backend.packed64.clear(packed)
        for byte_index in range(4):
            self.backend.copy_cell(
                sequence.length_ref.byte(byte_index),
                packed.byte(byte_index),
                self.backend.s0,
            )

        result = self._new_word()
        # The Quad backend consumes this temporary packed value while expanding
        # it. The sequence's persistent packed-u32 length was copied above and
        # therefore remains intact.
        self.backend.copy64(result, packed)
        return result

    def compile_expr(self, node: ast.AST):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and self._is_dynamic_char_list_name(node.args[0].id)
        ):
            return self._dynamic_length()
        return super().compile_expr(node)

    def _load_char_list_subscript(self, node: ast.Subscript):
        assert isinstance(node.value, ast.Name)
        if not self._is_dynamic_char_list_name(node.value.id):
            return super()._load_char_list_subscript(node)

        sequence = self.dynamic_char_sequence
        assert sequence is not None
        index = self._pack_index(node.slice)
        result = self._new_char_buffer()
        self.backend.clear_string(result)
        sequence.load_byte_signed(self.bf, result.char(0), index)
        self.bf.clear(result.terminator)
        return result

    def _store_char_list_subscript(self, node: ast.Subscript, value) -> None:
        assert isinstance(node.value, ast.Name)
        if not self._is_dynamic_char_list_name(node.value.id):
            return super()._store_char_list_subscript(node, value)

        sequence = self.dynamic_char_sequence
        assert sequence is not None
        index = self._pack_index(node.slice)
        sequence.store_byte_signed(self.bf, index, value.char(0))

    def _load_dynamic_char(self, target, index) -> None:
        sequence = self.dynamic_char_sequence
        assert sequence is not None
        self.backend.clear_string(target)
        packed = self._pack_word(index)
        sequence.load_byte_signed(self.bf, target.char(0), packed)
        self.bf.clear(target.terminator)

    def _compile_for_string_control(self, node: ast.For) -> None:
        if not (
            isinstance(node.iter, ast.Name)
            and self._is_dynamic_char_list_name(node.iter.id)
        ):
            return super()._compile_for_string_control(node)
        if not isinstance(node.target, ast.Name) or node.target.id not in self.strings:
            raise self._error(
                node,
                "runtime character-list iteration target must be a simple string variable",
            )

        target = self.strings[node.target.id]
        index = self._new_word(0)
        length = self._dynamic_length()
        control = self.temps.cell()
        body_active = self.temps.cell()
        broke = self.temps.cell()
        self.bf.clear(broke)
        self.backend.slt64(control, index, length)

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        self._load_dynamic_char(target, index)
        self.bf.set_const(body_active, 1)

        self._loop_stack.append(_LoopContext(body_active, broke))
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

    def _print_string_ref_compact(self, ref) -> None:
        selection = self.dynamic_char_list_selection
        sequence = self.dynamic_char_sequence
        if (
            selection is not None
            and sequence is not None
            and ref is self.strings[selection.name]
        ):
            sequence.write_all_bytes(self.bf)
            return
        return super()._print_string_ref_compact(ref)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and self._is_dynamic_char_list_name(node.targets[0].id)
            and _is_list_input(node.value)
        ):
            sequence = self.dynamic_char_sequence
            assert sequence is not None
            sequence.read_lf_terminated_bytes(self.bf)
            return
        return super()._compile_stmt_inner(node)


__all__ = [
    "CompileError",
    "DynamicCharListSelection",
    "PythonToBFStream",
    "select_dynamic_char_list",
]
