"""Final-compiler layer enabling distance-localized Quad operations.

The extra shared Quad word is reserved at class-definition time so the inherited
allocator places all later temporaries beyond it.  No AST semantics change in
this layer; it only swaps the arithmetic backend after the established frontend
has initialized its tape layout.
"""

from __future__ import annotations

import ast

from bfquad import WORD_CELLS
from bfquadlocal import DistanceLocalizedQuadBinaryStringListIO
from compiler_dynamic_charlist import CompileError
from compiler_dynamic_charlist import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Dynamic-charlist frontend with source-localized scalar arithmetic."""

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


__all__ = ["CompileError", "PythonToBFStream"]
