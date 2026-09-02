"""Final-compiler layout planning for runtime-sized tape objects.

The historical frontends allocate temporary cells from a stack-like arena and
rewind it at statement boundaries.  Runtime-sized objects cannot safely begin
at the current arena top because a later statement may temporarily grow beyond
that address.  This layer records the compile-time high-water mark so a future
second pass can place the runtime heap strictly after every compile-time temp.

Only the final public compiler uses this layer.  Older frontends intentionally
retain their existing allocator to avoid broad compatibility churn.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bfopt import optimize_bf
from bftemparena import PeakTempArena
from compiler_stream import CompileError, PythonToBFStream


@dataclass(frozen=True)
class LayoutPlan:
    """Compile-time tape boundary reserved before runtime-sized storage."""

    temp_base: int
    temp_peak: int

    @property
    def temp_cells(self) -> int:
        return self.temp_peak - self.temp_base

    def runtime_base(self, *, guard_cells: int = 1) -> int:
        if guard_cells < 1:
            raise ValueError("guard_cells must be positive")
        return self.temp_peak + guard_cells


class PythonToBFLayout(PythonToBFStream):
    """Public compact compiler with high-water-aware temp allocation."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
    ) -> None:
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
        )
        # __init__ above allocates only static/scratch/workspace regions.  All
        # expression/list/string temporaries are allocated later while lowering,
        # so replacing the arena here captures the complete temporary high-water
        # mark without touching legacy compiler classes.
        self.temps = PeakTempArena(self.temps.top)

    @property
    def layout_plan(self) -> LayoutPlan:
        return LayoutPlan(self.temps.base, self.temps.peak)


def lower_with_layout(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> tuple[str, LayoutPlan]:
    """Lower once and return both raw BF and the measured tape layout plan."""
    tree = ast.parse(source, filename=filename)
    compiler = PythonToBFLayout(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    )
    raw = compiler.compile_module(tree)
    return raw, compiler.layout_plan


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    raw, _plan = lower_with_layout(
        source,
        filename,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    )
    return optimize_bf(raw)


__all__ = [
    "CompileError",
    "LayoutPlan",
    "PythonToBFLayout",
    "lower_with_layout",
    "compile_source",
]
