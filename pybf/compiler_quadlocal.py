"""Final-compiler layer enabling source-localized Quad operations.

Two generic source-size transformations live here:

* distant Quad arithmetic/comparisons use nearby shared snapshots; and
* fixed-arity ``map(int, input().split())`` unpacking emits one packed decimal
  parser body and repeats it at runtime instead of cloning that parser once per
  target.

Neither transformation depends on a problem identity or answer structure.
"""

from __future__ import annotations

import ast

from bfquad import WORD_CELLS
from bfquadlocal import DistanceLocalizedQuadBinaryStringListIO
from compiler_dynamic_charlist import CompileError
from compiler_dynamic_charlist import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Dynamic-charlist frontend with source-localized scalar lowering."""

    SHARED_WORKSPACE_CELLS = _BasePythonToBFStream.SHARED_WORKSPACE_CELLS + WORD_CELLS

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
        runtime_charlist_base: int | None = None,
    ) -> None:
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
            runtime_charlist_base=runtime_charlist_base,
        )
        backend = DistanceLocalizedQuadBinaryStringListIO(
            self.bf,
            scratch_base=self.scratch_base,
        )
        backend.set_quad_workspace(self.workspace_base)
        self.backend = backend

    def _read_int_unpack_line(self, targets: list[ast.AST], node: ast.AST) -> None:
        """Read a fixed-arity integer tuple with one emitted parser body.

        The established implementation statically emits the complete signed
        decimal token parser once for every target.  Here one runtime loop
        invokes that same parser repeatedly and dispatches the packed token by
        a bounded target selector.  Missing fields retain zero, extra fields are
        drained from the current line, and parsing never crosses LF/EOF.
        """
        if (
            not targets
            or len(targets) > 255
            or not all(isinstance(target, ast.Name) for target in targets)
        ):
            return super()._read_int_unpack_line(targets, node)

        destinations = []
        for target in targets:
            assert isinstance(target, ast.Name)
            if (
                target.id in self.strings
                or target.id in self.lists
                or target.id in self.string_lists
            ):
                raise self._error(target, "integer token requires an integer variable")
            dst = self._var(target)
            self.backend.set_u64(dst, 0)
            destinations.append(dst)

        line_open = self.temps.cell()
        has_token = self.temps.cell()
        end_line = self.temps.cell()
        active = self.temps.cell()
        selector = self.temps.cell()
        route = self.temps.cell()
        token_gate = self.temps.cell()
        done = self.temps.cell()
        token = self._packed_input_token()

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

        # Only one route can match the selector. Packed->Quad expansion is
        # destructive, which is safe because the matching route is unique and
        # the parser clears/reuses the token on the next runtime iteration.
        for target_index, dst in enumerate(destinations):
            self.backend._eq_byte_const(route, selector, target_index)
            self.bf.begin_while(route)
            self.bf.add_const(route, -1)
            self.backend.copy_cell(has_token, token_gate, self.backend.s0)
            self.bf.begin_while(token_gate)
            self.bf.add_const(token_gate, -1)
            self.backend.copy64(dst, token)
            self.bf.end_while(token_gate)
            self.bf.end_while(route)

        self.bf.add_const(selector, 1)
        self._close_line_if_end(line_open, end_line)

        # Rearm iff the physical line is still open and another declared target
        # remains. This also naturally handles a short/empty input line.
        self.backend.copy_cell(line_open, active, self.backend.s0)
        self.backend._eq_byte_const(done, selector, len(destinations))
        self.bf.begin_while(done)
        self.bf.add_const(done, -1)
        self.bf.clear(active)
        self.bf.end_while(done)
        self.bf.end_while(active)

        # Preserve the established fixed-arity contract: surplus tokens belong
        # to this physical line and are ignored rather than bleeding into the
        # following input statement.
        self.backend.drain_to_line_end(line_open, self.workspace_base)
        self.backend.packed64.clear(token)


__all__ = ["CompileError", "PythonToBFStream"]
