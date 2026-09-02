"""Size-oriented final compiler layer.

This module preserves the public Python semantics while avoiding several large
static expansions that are unnecessary for common contest programs:

* character-only loop variables use a one-byte string payload instead of the
  full 255-byte string ABI when that compact representation is provably safe;
* scratch cells are placed immediately after static variables, with the larger
  shared arithmetic workspace after them, reducing emitted pointer travel;
* ``for c in s`` snapshots ``s`` only when the loop can actually rebind the
  iterable name (or when target and iterable are the same name);
* one-character equality against a literal is lowered as a byte comparison;
* final standard-Brainfuck output passes through ``optimize_bf``.
"""

from __future__ import annotations

import ast

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bflists import IntListRef
from bfmemory import allocate_live_blocks
from bfopt import optimize_bf
from bfstringlists import BinaryStringListIO, StringListRef
from bfstrings import StringRef
from compiler import (
    CompileError,
    _infer_int_list_names,
    _infer_string_list_names,
)
from compiler_strings import (
    PythonToBFStringIteration,
    _infer_scalar_string_loop_targets,
)
from transpiler_full import _LoopContext
from transpiler_inputs import infer_split_string_names
from transpiler_v2 import _TempArena
from transpiler_v3 import infer_string_names


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in node.elts:
            out.update(_target_names(elt))
        return out
    return set()


class _ExplicitWrites(ast.NodeVisitor):
    """Collect writes other than the target assignment performed by ``for``."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.names.update(_target_names(target))
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.names.update(_target_names(node.target))
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.names.update(_target_names(node.target))
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        # Deliberately do not count node.target here.  The caller separately
        # checks whether a name is ever used as a non-string loop target.
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)


class _BodyWrites(ast.NodeVisitor):
    """Check whether one specific variable name can be rebound in a loop body."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.found = False

    def _check(self, target: ast.AST) -> None:
        if self.name in _target_names(target):
            self.found = True

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check(target)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check(node.target)
        if node.value is not None:
            self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check(node.target)
        self.generic_visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._check(node.target)
        self.generic_visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._check(node.target)
        self.generic_visit(node)


def _body_rebinds(name: str, body: list[ast.stmt]) -> bool:
    visitor = _BodyWrites(name)
    for stmt in body:
        visitor.visit(stmt)
        if visitor.found:
            return True
    return False


def _compact_char_names(
    tree: ast.Module,
    all_string_names: set[str],
) -> set[str]:
    """Find loop targets that can safely use capacity-one string storage."""
    candidates = _infer_scalar_string_loop_targets(tree)

    writes = _ExplicitWrites()
    writes.visit(tree)
    candidates.difference_update(writes.names)

    # A variable reused as a range/list/etc. loop target cannot have a permanent
    # one-byte string representation.
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name not in candidates:
            continue
        is_scalar_string_iter = (
            isinstance(node.iter, ast.Name) and node.iter.id in all_string_names
        )
        if not is_scalar_string_iter:
            candidates.discard(name)

    return candidates


class PythonToBFCompact(PythonToBFStringIteration):
    """Public compiler backend optimized for standard-BF source size."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
    ) -> None:
        if not 1 <= string_capacity <= 255:
            raise ValueError("string_capacity must be between 1 and 255")
        if not 1 <= list_capacity <= 255:
            raise ValueError("list_capacity must be between 1 and 255")
        self.string_capacity = string_capacity
        self.list_capacity = list_capacity

        base_strings, all_names = infer_string_names(tree)
        string_list_names = _infer_string_list_names(tree)

        # Teach the fixed-point inference about characters yielded by scalar
        # string loops, then propagate assignments from those character values.
        loop_chars = _infer_scalar_string_loop_targets(tree)
        string_names = infer_split_string_names(
            tree, base_strings | loop_chars, string_list_names
        )
        compact_chars = _compact_char_names(tree, string_names)
        int_list_names = _infer_int_list_names(tree, string_list_names)

        overlaps = (
            (string_names & int_list_names)
            | (string_names & string_list_names)
            | (int_list_names & string_list_names)
        )
        if overlaps:
            raise CompileError(
                "variables cannot change between scalar/list element types: "
                + ", ".join(sorted(overlaps))
            )

        self.string_names = string_names
        self.list_names = int_list_names
        self.string_list_names = string_list_names
        self.compact_char_names = compact_chars

        int_list_cells = IntListRef(0, list_capacity).cells
        string_list_cells = StringListRef(0, list_capacity, string_capacity).cells

        def scalar_string_cells(name: str) -> int:
            capacity = 1 if name in compact_chars else string_capacity
            return capacity + 1

        sizes = {
            name: (
                string_list_cells
                if name in string_list_names
                else int_list_cells
                if name in int_list_names
                else scalar_string_cells(name)
                if name in string_names
                else WORD_BITS
            )
            for name in all_names
        }
        blocks, static_top = allocate_live_blocks(tree, sizes)

        self.variables: dict[str, Int64Ref] = {
            name: Int64Ref(blocks[name].base)
            for name in all_names
            if name not in string_names
            and name not in int_list_names
            and name not in string_list_names
        }
        self.strings: dict[str, StringRef] = {
            name: StringRef(
                blocks[name].base,
                1 if name in compact_chars else string_capacity,
            )
            for name in string_names
        }
        self.lists: dict[str, IntListRef] = {
            name: IntListRef(blocks[name].base, list_capacity)
            for name in int_list_names
        }
        self.string_lists: dict[str, StringListRef] = {
            name: StringListRef(blocks[name].base, list_capacity, string_capacity)
            for name in string_list_names
        }

        self.bf = BFEmitter()

        # The five hot scratch cells are touched vastly more often than the
        # large signed-divmod workspace.  Keeping them adjacent to user storage
        # eliminates millions of emitted '<'/'>' commands in string-heavy code.
        self.scratch_base = static_top
        self.workspace_base = self.scratch_base + Binary64Core.SCRATCH_CELLS
        self.backend = BinaryStringListIO(self.bf, scratch_base=self.scratch_base)
        self.temps = _TempArena(
            self.workspace_base + self.SHARED_WORKSPACE_CELLS
        )
        self._loop_stack: list[_LoopContext] = []

    # ------------------------------------------------------------------
    # compact character operations
    # ------------------------------------------------------------------
    def _char_ref(self, node: ast.AST) -> StringRef | None:
        if isinstance(node, ast.Name) and node.id in self.compact_char_names:
            return self.strings[node.id]
        return None

    @staticmethod
    def _one_byte_literal(node: ast.AST) -> int | None:
        if not (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) == 1
        ):
            return None
        value = ord(node.value)
        if value == 0 or value > 255:
            return None
        return value

    def compile_expr(self, node: ast.AST) -> Int64Ref:
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
        ):
            left_char = self._char_ref(node.left)
            right_char = self._char_ref(node.comparators[0])
            left_lit = self._one_byte_literal(node.left)
            right_lit = self._one_byte_literal(node.comparators[0])

            char: StringRef | None = None
            literal: int | None = None
            if left_char is not None and right_lit is not None:
                char, literal = left_char, right_lit
            elif right_char is not None and left_lit is not None:
                char, literal = right_char, left_lit

            if char is not None and literal is not None:
                result = self._new_word(0)
                self.backend._eq_byte_const(result.bit(0), char.char(0), literal)
                if isinstance(node.ops[0], ast.NotEq):
                    self.backend._toggle_bit(result.bit(0), self.backend.s0)
                    self.backend._clear_scratch()
                return result

            if left_char is not None and right_char is not None:
                result = self._new_word(0)
                # Compare the two payload bytes directly rather than promoting
                # both values into 255-byte temporary strings.
                self.backend.copy_cell(
                    left_char.char(0), self.backend.s0, self.backend.s2
                )
                self.backend.copy_cell(
                    right_char.char(0), self.backend.s1, self.backend.s2
                )
                self.bf.begin_while(self.backend.s1)
                self.bf.add_const(self.backend.s1, -1)
                self.bf.add_const(self.backend.s0, -1)
                self.bf.end_while(self.backend.s1)
                self.bf.set_const(result.bit(0), 1)
                self.bf.begin_while(self.backend.s0)
                self.bf.clear(self.backend.s0)
                self.bf.clear(result.bit(0))
                self.bf.end_while(self.backend.s0)
                if isinstance(node.ops[0], ast.NotEq):
                    self.backend._toggle_bit(result.bit(0), self.backend.s0)
                self.backend._clear_scratch()
                return result

        return super().compile_expr(node)

    def _compile_for_string_control(self, node: ast.For) -> None:
        if not isinstance(node.iter, ast.Name) or node.iter.id not in self.strings:
            raise self._error(node, "string iteration requires a string variable")
        if not isinstance(node.target, ast.Name) or node.target.id not in self.strings:
            raise self._error(
                node, "string iteration target must be a simple string variable"
            )

        source_name = node.iter.id
        target_name = node.target.id

        # Strings are immutable.  A private copy is only required if body
        # statements can rebind the source name, or if the for-target itself is
        # that source name (``for s in s``).
        if target_name == source_name or _body_rebinds(source_name, node.body):
            src = self._copy_string_new(self.strings[source_name])
        else:
            src = self.strings[source_name]
        target = self.strings[target_name]

        index = self.temps.cell()
        control = self.temps.cell()
        body_active = self.temps.cell()
        broke = self.temps.cell()
        self.bf.clear(index)
        self.bf.clear(broke)

        self._load_string_char_at(target, src, index)
        self._set_char_loop_control(control, target.char(0))

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
        tmp_flag = self.temps.cell()
        self._flag_not(proceed, broke, tmp_flag)
        self.bf.begin_while(proceed)
        self.bf.add_const(proceed, -1)
        self.bf.add_const(index, 1)
        self._load_string_char_at(target, src, index)
        self._set_char_loop_control(control, target.char(0))
        self.bf.end_while(proceed)
        self.bf.end_while(control)

        self._compile_guarded_else(node.orelse, broke)


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    tree = ast.parse(source, filename=filename)
    code = PythonToBFCompact(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
    return optimize_bf(code)


__all__ = ["CompileError", "PythonToBFCompact", "compile_source"]
