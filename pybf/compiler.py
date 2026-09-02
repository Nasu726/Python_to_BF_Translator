"""Current fixed-ABI compiler frontend.

This module is the final lowering layer used by the public ``pybf`` API.  It
keeps all ``input().split()`` forms line-scoped, including the older
``map(int, ...)`` path.
"""

from __future__ import annotations

import ast

from transpiler import _is_map_int_input_split
from transpiler_inputs import PythonToBFInputs


class PythonToBFCompiler(PythonToBFInputs):
    def _finish_split_line(self, end_line: int) -> None:
        """Drain any unused bytes from the current logical input line."""
        active = self.temps.cell()
        gate = self.temps.cell()
        self.bf.set_const(active, 1)
        self.backend.copy_cell(end_line, gate, self.backend.s0)
        self.bf.begin_while(gate)
        self.bf.add_const(gate, -1)
        self.bf.clear(active)
        self.bf.end_while(gate)
        self.backend.drain_to_line_end(active, self.workspace_base)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        # The old frontend used read_s64_token(), which skips arbitrary
        # whitespace and could cross a newline.  Python input().split() is
        # explicitly one-line input, so use the line-token primitive here.
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and _is_map_int_input_split(node.value)
        ):
            targets = node.targets[0].elts
            if not targets or not all(isinstance(t, ast.Name) for t in targets):
                raise self._error(node, "map(int, input().split()) unpacking requires simple names")

            has_token = self.temps.cell()
            end_line = self.temps.cell()
            for target in targets:
                assert isinstance(target, ast.Name)
                if (
                    target.id in self.strings
                    or target.id in self.lists
                    or target.id in self.string_lists
                ):
                    raise self._error(target, "integer token requires an integer variable")
                self.backend.read_s64_line_token(
                    self._var(target), has_token, end_line, self.workspace_base
                )

            self._finish_split_line(end_line)
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
    return PythonToBFCompiler(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
