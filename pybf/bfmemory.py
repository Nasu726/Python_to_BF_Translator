"""Compile-time tape layout helpers.

Module-level Python variables normally live for the whole program, but many
contest-style temporaries have non-overlapping lexical lifetimes. Reusing those
blocks materially reduces pointer travel and tape footprint.

Loops need special care: a value used before another assignment in the loop
may still be live across the back edge. To stay correct without a full CFG
liveness solver, every name mentioned inside a ``for``/``while`` is pinned and
never shares storage. Non-loop names use conservative lexical intervals.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryBlock:
    base: int
    size: int


class _Intervals(ast.NodeVisitor):
    def __init__(self) -> None:
        self.clock = 0
        self.first: dict[str, int] = {}
        self.last: dict[str, int] = {}

    def visit_Name(self, node: ast.Name) -> None:
        t = self.clock
        self.clock += 1
        self.first.setdefault(node.id, t)
        self.last[node.id] = t


class _LoopNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_depth = 0
        self.names: set[str] = set()

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Name(self, node: ast.Name) -> None:
        if self.loop_depth:
            self.names.add(node.id)


def allocate_live_blocks(tree: ast.AST, sizes: dict[str, int]) -> tuple[dict[str, MemoryBlock], int]:
    """Allocate statically-sized variables with safe, conservative reuse.

    Returns ``(name -> block, static_top)``. All requested sizes must be
    positive. Names occurring in loops are pinned to unique blocks.

    The compiler keeps its hottest scratch cells immediately after the static
    region. For pinned loop variables, placing large aggregate blocks first
    leaves small scalar blocks next to that scratch boundary. This is only a
    layout heuristic -- it does not change liveness or aliasing -- but it avoids
    emitting enormous pointer runs through a fixed-capacity list every time a
    scalar bit operation touches shared scratch.
    """

    if any(size <= 0 for size in sizes.values()):
        raise ValueError("all static block sizes must be positive")

    intervals = _Intervals()
    intervals.visit(tree)
    loops = _LoopNames()
    loops.visit(tree)

    # Pinned variables are allocated first and never enter the free list.
    # Larger blocks go toward tape cell zero; small scalar blocks then sit near
    # static_top, where the compact compiler places its hot scratch/workspace.
    result: dict[str, MemoryBlock] = {}
    top = 0
    pinned = loops.names & sizes.keys()
    for name in sorted(pinned, key=lambda item: (-sizes[item], item)):
        size = sizes[name]
        result[name] = MemoryBlock(top, size)
        top += size

    # (start, end, name, size) for reusable names.
    items = []
    for name, size in sizes.items():
        if name in result:
            continue
        start = intervals.first.get(name, 0)
        end = intervals.last.get(name, start)
        items.append((start, end, name, size))
    items.sort()

    # Active reusable allocations: (end, name, block).
    active: list[tuple[int, str, MemoryBlock]] = []
    free: list[MemoryBlock] = []

    for start, end, name, size in items:
        still_active: list[tuple[int, str, MemoryBlock]] = []
        for old_end, old_name, block in active:
            if old_end < start:
                free.append(block)
            else:
                still_active.append((old_end, old_name, block))
        active = still_active

        # Best-fit reuse limits fragmentation for mixed int/string blocks.
        candidates = [(block.size, i, block) for i, block in enumerate(free) if block.size >= size]
        if candidates:
            _, i, block = min(candidates)
            free.pop(i)
            chosen = MemoryBlock(block.base, size)
            remainder = block.size - size
            if remainder:
                free.append(MemoryBlock(block.base + size, remainder))
        else:
            chosen = MemoryBlock(top, size)
            top += size

        result[name] = chosen
        active.append((end, name, chosen))

    return result, top
