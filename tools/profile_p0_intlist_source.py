"""Explain source cost of the ordinary P0 integer-list/alias/repr workload.

This is a compiler attribution diagnostic, not a Brainfuck optimizer.
Raw statement byte counts are *not* additive estimates of optimized code:
the optimizer can cancel instructions across statement boundaries. Always
show actual end-to-end post-optimization size separately.

Usage:
    python tools/profile_p0_intlist_source.py
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys

# Permit both "python tools/..." and "import tools.profile_..." execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pybf import compile_source  # noqa: E402
from bfopt import optimize_bf  # noqa: E402
from compiler_dynamic_intlist import (  # noqa: E402
    dynamic_int_workspace_cells,
    select_dynamic_int_list,
)
from compiler_layout import PythonToBFLayout  # noqa: E402
from tools.bench_tritium_intlist_repr import SOURCE  # noqa: E402


ATCODER_SOURCE_LIMIT = 512 * 1024


@dataclass(frozen=True)
class P0SourceProfile:
    raw_bytes: int
    final_bytes: int
    runtime_base: int
    temp_peak: int
    statement_sizes: tuple[tuple[int, str, int], ...]
    detail_sizes: tuple[tuple[int, str, int], ...]

    @property
    def headroom(self) -> int:
        return ATCODER_SOURCE_LIMIT - self.final_bytes


def profile_p0_source(source: str = SOURCE) -> P0SourceProfile:
    """Profile the same convergent dynamic-list layout as public compilation.

    The first pass discovers the maximal static temporary address. The
    following passes place the mobile frame after that boundary, exactly as
    compiler_layout.lower_with_layout does. A nonconvergent layout is an error.
    """
    tree = ast.parse(source)
    selection = select_dynamic_int_list(tree)
    if selection is None:
        raise ValueError("source does not select the runtime integer-list route")

    def lower(base: int | None) -> tuple[str, PythonToBFLayout]:
        compiler = PythonToBFLayout(
            tree, runtime_intlist_base=base,
        )
        raw = compiler.compile_module(tree)
        return raw, compiler

    _, probe = lower(None)
    guard = dynamic_int_workspace_cells(selection) + 10
    base = probe.layout_plan.runtime_base(guard_cells=guard)
    for _ in range(3):
        raw, compiler = lower(base)
        plan = compiler.layout_plan
        next_base = plan.runtime_base(guard_cells=guard)
        if next_base == base:
            final = optimize_bf(raw)
            if any(ch not in "><+-.,[]" for ch in final):
                raise AssertionError("nonstandard Brainfuck in compiler output")
            return P0SourceProfile(
                raw_bytes=len(raw),
                final_bytes=len(final),
                runtime_base=base,
                temp_peak=plan.temp_peak,
                statement_sizes=tuple(compiler.statement_sizes),
                detail_sizes=tuple(compiler.detail_sizes),
            )
        base = next_base
    raise RuntimeError("dynamic integer-list layout did not converge")


def print_profile(profile: P0SourceProfile) -> None:
    print(f"raw_source_bytes={profile.raw_bytes:,}")
    print(f"final_source_bytes={profile.final_bytes:,}")
    print(f"headroom_to_512k={profile.headroom:,}")
    print(f"temp_peak={profile.temp_peak} runtime_intlist_base={profile.runtime_base}")
    for title, rows in (
        ("top-level raw lowering attribution:", profile.statement_sizes),
        ("largest nested raw lowering attribution:", profile.detail_sizes),
    ):
        print(title)
        for lineno, kind, size in sorted(rows, key=lambda row: row[2], reverse=True)[:25]:
            print(f"  L{lineno:02d} {kind:<20} {size:>11,d} B")


def main() -> None:
    profile = profile_p0_source()
    # Check the profiler against the supported public compiler path. This
    # catches future changes to routing, layout, or final optimizer behavior.
    public = compile_source(SOURCE)
    if len(public) != profile.final_bytes:
        raise SystemExit(
            f"profiler disagrees with public compiler: "
            f"{profile.final_bytes} versus {len(public)}"
        )
    print_profile(profile)


if __name__ == "__main__":
    main()
