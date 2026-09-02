"""String-iteration lowering for the public Python -> Brainfuck compiler.

This layer keeps the existing compiler intact while adding Python's basic
``for ch in string`` semantics.  Loop targets are inferred as strings before
static tape allocation, and the generated Brainfuck iterates with a one-byte
runtime index.  No Python runtime support is required after compilation.
"""

from __future__ import annotations

import ast
import copy

from compiler import (
    CompileError,
    PythonToBFCompiler,
    _infer_string_list_names,
)
from transpiler_inputs import infer_split_string_names
from transpiler_v3 import infer_string_names


def _infer_scalar_string_loop_targets(tree: ast.AST) -> set[str]:
    """Return loop-target names that receive characters from scalar strings.

    ``infer_string_names`` historically forced every for-target to int because
    only ``range`` loops existed at that layer.  The final compiler now needs a
    second fixed point: once ``s`` is known to be a string, ``for c in s`` makes
    ``c`` a string, and assignments from ``c`` can in turn make more variables
    strings.
    """
    base_strings, _ = infer_string_names(tree)
    string_list_names = _infer_string_list_names(tree)
    strings = infer_split_string_names(tree, base_strings, string_list_names)
    loop_targets: set[str] = set()

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            if not isinstance(node.iter, ast.Name) or node.iter.id not in strings:
                continue
            if not isinstance(node.target, ast.Name):
                # General iterable unpacking is a later language feature.
                continue
            if node.target.id not in strings:
                strings.add(node.target.id)
                loop_targets.add(node.target.id)
                changed = True

        expanded = infer_split_string_names(tree, strings, string_list_names)
        if not expanded.issubset(strings):
            strings.update(expanded)
            changed = True

    return loop_targets


def _inference_tree_with_string_loop_targets(tree: ast.Module) -> ast.Module:
    """Add type-only assignments to a copied AST used solely for allocation.

    The synthetic statements are never compiled.  They only teach the existing
    static type/layout pass that character loop targets require string storage.
    Keeping this as an inference-only tree avoids changing Python execution
    semantics while letting the established allocator remain authoritative.
    """
    targets = _infer_scalar_string_loop_targets(tree)
    if not targets:
        return tree

    inferred = copy.deepcopy(tree)
    for name in sorted(targets):
        inferred.body.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Constant(value=""),
            )
        )
    ast.fix_missing_locations(inferred)
    return inferred


class PythonToBFStringIteration(PythonToBFCompiler):
    """Final compiler with scalar-string ``for`` iteration."""

    def _load_string_char_at(self, dst, src, index: int) -> None:
        """dst = one-character string src[index], or empty at the terminator.

        String capacity is at most 255, so a byte index is sufficient.  Dynamic
        addressing is lowered by comparing the index with each static slot;
        the loop body itself is emitted only once.
        """
        self.backend.clear_string(dst)
        match = self.temps.cell()
        for i in range(src.capacity):
            self.backend._eq_byte_const(match, index, i)
            self.bf.begin_while(match)
            self.bf.add_const(match, -1)
            self.backend.copy_cell(src.char(i), dst.char(0), self.backend.s0)
            self.bf.end_while(match)
        self.backend._clear_scratch()

    def _set_char_loop_control(self, control: int, char_cell: int) -> None:
        """control = (char_cell != NUL), preserving char_cell."""
        self.backend._eq_byte_const(control, char_cell, 0)
        self.backend._toggle_bit(control, self.backend.s0)
        self.backend._clear_scratch()

    def _compile_for_string_control(self, node: ast.For) -> None:
        if not isinstance(node.iter, ast.Name) or node.iter.id not in self.strings:
            raise self._error(node, "string iteration requires a string variable")
        if not isinstance(node.target, ast.Name) or node.target.id not in self.strings:
            raise self._error(node, "string iteration target must be a simple string variable")

        # Python obtains the iterable before assigning the loop target.  Keep a
        # private snapshot so ``for s in s`` and assignments to the source name
        # inside the body do not corrupt the iterator.
        src = self._copy_string_new(self.strings[node.iter.id])
        target = self.strings[node.target.id]

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

        from transpiler_full import _LoopContext

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

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in self.strings
        ):
            self._compile_for_string_control(node)
            return
        return super()._compile_stmt_inner(node)


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    tree = ast.parse(source, filename=filename)
    inference_tree = _inference_tree_with_string_loop_targets(tree)
    return PythonToBFStringIteration(
        inference_tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)


__all__ = ["CompileError", "PythonToBFStringIteration", "compile_source"]
