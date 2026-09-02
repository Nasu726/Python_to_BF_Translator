"""Semantics-preserving Brainfuck source-size optimization.

The generated runtime targets 8-bit wrapping cells and a zero-initialized tape.
This optimizer intentionally relies only on those ABI guarantees.  It removes
redundant clears of cells proven to be zero, drops loops whose control cell is
proven zero, and canonicalizes adjacent pointer/value runs.

It does *not* introduce non-standard Brainfuck commands or require a special
interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass


BF_COMMANDS = frozenset("><+-.,[]")


@dataclass(frozen=True)
class _Loop:
    body: tuple[object, ...]


def _parse(code: str) -> tuple[object, ...]:
    filtered = [ch for ch in code if ch in BF_COMMANDS]

    def parse_from(pos: int, nested: bool) -> tuple[list[object], int]:
        out: list[object] = []
        while pos < len(filtered):
            ch = filtered[pos]
            pos += 1
            if ch == "]":
                if not nested:
                    raise ValueError("unmatched ']' in Brainfuck program")
                return out, pos
            if ch == "[":
                body, pos = parse_from(pos, True)
                out.append(_Loop(tuple(body)))
            else:
                out.append(ch)
        if nested:
            raise ValueError("unmatched '[' in Brainfuck program")
        return out, pos

    nodes, _ = parse_from(0, False)
    return tuple(nodes)


def _stringify(nodes: tuple[object, ...] | list[object]) -> str:
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, _Loop):
            parts.append("[")
            parts.append(_stringify(node.body))
            parts.append("]")
        else:
            parts.append(str(node))
    return "".join(parts)


def _canonicalize(nodes: tuple[object, ...] | list[object]) -> tuple[object, ...]:
    """Fold adjacent movement/arithmetic runs without crossing side effects."""
    out: list[object] = []
    move = 0
    add = 0

    def flush_move() -> None:
        nonlocal move
        if move > 0:
            out.extend(">" for _ in range(move))
        elif move < 0:
            out.extend("<" for _ in range(-move))
        move = 0

    def flush_add() -> None:
        nonlocal add
        # Cells wrap modulo 256.  Choose the shorter equivalent direction.
        value = add % 256
        if value <= 128:
            out.extend("+" for _ in range(value))
        else:
            out.extend("-" for _ in range(256 - value))
        add = 0

    for node in nodes:
        if node == ">":
            flush_add()
            move += 1
            continue
        if node == "<":
            flush_add()
            move -= 1
            continue
        if node == "+":
            flush_move()
            add += 1
            continue
        if node == "-":
            flush_move()
            add -= 1
            continue

        flush_move()
        flush_add()
        if isinstance(node, _Loop):
            body = _canonicalize(node.body)
            # In 8-bit wrapping Brainfuck, [+] and [-] are both clear loops.
            if body in (("+",), ("-",)):
                body = ("-",)
            out.append(_Loop(body))
        else:
            out.append(node)

    flush_move()
    flush_add()
    return tuple(out)


def _effects(nodes: tuple[object, ...], start: int = 0) -> tuple[int, set[int], bool]:
    """Return (net pointer delta, modified offsets, all loops balanced)."""
    ptr = start
    touched: set[int] = set()
    balanced = True
    for node in nodes:
        if node == ">":
            ptr += 1
        elif node == "<":
            ptr -= 1
        elif node in ("+", "-", ","):
            touched.add(ptr)
        elif isinstance(node, _Loop):
            nested_delta, nested_touched, nested_balanced = _effects(node.body, ptr)
            touched.update(nested_touched)
            if nested_delta != ptr:
                balanced = False
            balanced = balanced and nested_balanced
            # A balanced generated loop returns to the cell where it started.
        # '.' has no memory effect.
    return ptr, touched, balanced


def _all_loops_balanced(nodes: tuple[object, ...]) -> bool:
    ptr = 0
    for node in nodes:
        if node == ">":
            ptr += 1
        elif node == "<":
            ptr -= 1
        elif isinstance(node, _Loop):
            end, _touched, nested_ok = _effects(node.body, ptr)
            if end != ptr or not nested_ok:
                return False
    return True


def _eliminate_known_zero(nodes: tuple[object, ...]) -> tuple[object, ...]:
    """Optimize top-level straight-line state using zero-initialized tape facts.

    General loop bodies are left dataflow-conservative because they can execute
    multiple times.  We still use the fact that a terminating BF loop exits
    with its controlling cell equal to zero.
    """
    values: dict[int, int | None] = {}
    logical_ptr = 0
    emitted_ptr = 0
    out: list[object] = []

    def value_at(cell: int) -> int | None:
        # An untouched cell is zero by the public runtime ABI.
        return values.get(cell, 0)

    def flush_move() -> None:
        nonlocal emitted_ptr
        delta = logical_ptr - emitted_ptr
        if delta > 0:
            out.extend(">" for _ in range(delta))
        elif delta < 0:
            out.extend("<" for _ in range(-delta))
        emitted_ptr = logical_ptr

    for node in nodes:
        if node == ">":
            logical_ptr += 1
            continue
        if node == "<":
            logical_ptr -= 1
            continue

        if node in ("+", "-"):
            flush_move()
            current = value_at(logical_ptr)
            if current is not None:
                values[logical_ptr] = (current + (1 if node == "+" else -1)) & 0xFF
            out.append(node)
            continue

        if node == ",":
            flush_move()
            values[logical_ptr] = None
            out.append(node)
            continue

        if node == ".":
            flush_move()
            out.append(node)
            continue

        if not isinstance(node, _Loop):
            raise AssertionError(f"unexpected BF node: {node!r}")

        body = _canonicalize(node.body)
        clear_loop = body in (("-",), ("+",))
        current = value_at(logical_ptr)

        if current == 0:
            # The loop is skipped.  Crucially, do not flush a pointer movement
            # whose only purpose was reaching this redundant operation.
            continue

        flush_move()
        if clear_loop:
            out.append(_Loop(("-",)))
            values[logical_ptr] = 0
            continue

        # Keep the loop, but invalidate every cell it may mutate because the
        # body can run an input-dependent number of times.
        end, touched, balanced = _effects(body, logical_ptr)
        if not balanced or end != logical_ptr:
            # This optimizer is only enabled for the compiler's balanced-loop
            # output.  The caller falls back before reaching this branch.
            raise ValueError("cannot dataflow-optimize an unbalanced BF loop")
        out.append(_Loop(body))
        for cell in touched:
            values[cell] = None
        # If the loop terminates, its controlling cell is zero whether it was
        # skipped or executed one or more times.
        values[logical_ptr] = 0

    # Preserve the compiler's final pointer location.  This is not observable
    # after program termination, so omitting a trailing pure move is safe and
    # smaller.  Therefore intentionally do not flush here.
    return tuple(out)


def optimize_bf(code: str) -> str:
    """Return smaller standard Brainfuck with identical observable behavior."""
    nodes = _canonicalize(_parse(code))
    if not _all_loops_balanced(nodes):
        # Generated code should never hit this path, but do not risk changing
        # semantics if optimize_bf is used on arbitrary handwritten BF.
        return _stringify(nodes)
    nodes = _eliminate_known_zero(nodes)
    nodes = _canonicalize(nodes)
    return _stringify(nodes)


__all__ = ["optimize_bf"]
