"""Public compiler layer with strict line-scoped input lowering.

This module keeps Python's ``input().split()`` boundary in the generated
Brainfuck.  Runtime execution still depends only on the eight Brainfuck
commands; Python is used only while compiling.
"""

from __future__ import annotations

import ast

from bfcore import Int64Ref
from transpiler import _is_map_int_input_split
from transpiler_full import PythonToBFFull
from transpiler_v2 import CompileError


class PythonToBFCompiler(PythonToBFFull):
    """Current production frontend.

    ``a, b = map(int, input().split())`` is lowered as one logical line.  It
    never steals tokens from the next line.  On valid Python/contest input the
    number of tokens matches the number of targets; for malformed arity, this
    fixed runtime currently zero-fills missing targets and discards extras
    instead of implementing Python's ``ValueError`` exception machinery.
    """

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and _is_map_int_input_split(node.value)
        ):
            targets = node.targets[0].elts
            if not targets or not all(isinstance(target, ast.Name) for target in targets):
                raise self._error(
                    node,
                    "map(int, input().split()) unpacking requires simple names",
                )
            self._read_int_unpack_line(targets)
            return

        super()._compile_stmt_inner(node)

    def _read_int_unpack_line(self, targets: list[ast.expr]) -> None:
        """Read one input line into a fixed number of signed-int targets."""
        bf = self.bf
        active = self.temps.cell()
        gate = self.temps.cell()
        end_gate = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()

        for cell in (active, gate, end_gate, has_token, end_line):
            bf.clear(cell)
        bf.set_const(active, 1)

        for target in targets:
            assert isinstance(target, ast.Name)
            if target.id in self.strings or target.id in self.lists:
                raise self._error(target, "integer token requires an integer variable")

            dst = self._var(target)
            self.backend._clear_word(dst)

            # Gate the read so an early newline never advances into the next
            # Python input() line for later unpacking targets.
            self.backend.copy_cell(active, gate, self.backend.s0)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            self.backend.read_s64_line_token(
                dst,
                has_token,
                end_line,
                self.workspace_base,
            )

            self.backend.copy_cell(end_line, end_gate, self.backend.s0)
            bf.begin_while(end_gate)
            bf.add_const(end_gate, -1)
            bf.clear(active)
            bf.end_while(end_gate)
            bf.end_while(gate)

        # Python input() consumes the whole line before split/unpack happens.
        # If valid input contains exactly len(targets) tokens, the last token
        # may have ended at horizontal whitespace rather than newline, so drain
        # whatever remains on this same line.  Extra tokens are discarded here
        # rather than leaking into the next input() call.
        discarded = self._new_word()
        bf.begin_while(active)
        self.backend.read_s64_line_token(
            discarded,
            has_token,
            end_line,
            self.workspace_base,
        )
        self.backend.copy_cell(end_line, end_gate, self.backend.s0)
        bf.begin_while(end_gate)
        bf.add_const(end_gate, -1)
        bf.clear(active)
        bf.end_while(end_gate)
        bf.end_while(active)


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int,
    list_capacity: int,
) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBFCompiler(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)


__all__ = ["CompileError", "PythonToBFCompiler", "compile_source"]
