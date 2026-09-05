"""Final-compiler layout planning for runtime-sized tape objects.

The historical frontends allocate temporary cells from a stack-like arena and
rewind it at statement boundaries. Runtime-sized objects cannot safely begin
at the current arena top because a later statement may temporarily grow beyond
that address. This layer records the compile-time high-water mark so the public
compiler's second pass can place runtime-sized storage strictly after every
compile-time temporary.

Only the final public compiler uses this layer.  Older frontends intentionally
retain their existing allocator to avoid broad compatibility churn.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bfstreamseq import RECORD_STRIDE
from bfopt import optimize_bf
from bftemparena import PeakTempArena
from compiler_dynamic_charlist import select_dynamic_char_list
from compiler_stream import CompileError, PythonToBFStream


@dataclass(frozen=True)
class LayoutPlan:
    """Compile-time tape boundary reserved before runtime-sized storage."""

    temp_base: int
    temp_peak: int
    dynamic_charlist_base: int | None = None

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
        runtime_charlist_base: int | None = None,
    ) -> None:
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
            runtime_charlist_base=runtime_charlist_base,
        )
        # __init__ above allocates only static/scratch/workspace regions.  All
        # expression/list/string temporaries are allocated later while lowering,
        # so replacing the arena here captures the complete temporary high-water
        # mark without touching legacy compiler classes.
        self.temps = PeakTempArena(self.temps.top)

    @property
    def layout_plan(self) -> LayoutPlan:
        return LayoutPlan(
            self.temps.base,
            self.temps.peak,
            self.runtime_charlist_base,
        )


def lower_with_layout(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> tuple[str, LayoutPlan]:
    """Lower once and return both raw BF and the measured tape layout plan."""
    tree = ast.parse(source, filename=filename)

    def lower_once(runtime_charlist_base: int | None):
        compiler = PythonToBFLayout(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
            runtime_charlist_base=runtime_charlist_base,
        )
        raw = compiler.compile_module(tree)
        return raw, compiler.layout_plan

    if select_dynamic_char_list(tree) is None:
        return lower_once(None)

    # First discover the compile-time high-water mark without placing a
    # runtime object. The dynamic lowering can use a slightly different set of
    # temporaries, so retry from its measured peak if necessary.
    _probe_raw, probe_plan = lower_once(None)
    del _probe_raw
    runtime_base = probe_plan.runtime_base(guard_cells=RECORD_STRIDE)
    for _attempt in range(3):
        raw, plan = lower_once(runtime_base)
        exact_base = plan.runtime_base(guard_cells=RECORD_STRIDE)
        if exact_base != runtime_base:
            runtime_base = exact_base
            continue
        if plan.temp_peak <= runtime_base - RECORD_STRIDE:
            return raw, plan
        runtime_base = plan.runtime_base(guard_cells=RECORD_STRIDE)
    raise CompileError("runtime character-list layout did not converge")


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
