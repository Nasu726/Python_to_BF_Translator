"""Small reusable Brainfuck runtime used by tests and the developer CLI.

The project historically had an interpreter hard-wired to ``test.bf``.  This
module keeps execution separate from file I/O so generated programs can be
compiled and exercised before merging a branch.  Straight-line runs are
collapsed into opcodes to avoid paying one Python dispatch per Brainfuck byte.
"""

from __future__ import annotations

from dataclasses import dataclass


BF_COMMANDS = set("><+-.,[]")
_LINEAR = set("><+-")


@dataclass(frozen=True)
class BFResult:
    output: str
    memory: list[int]
    pointer: int
    steps: int
    input_consumed: int


class BFExecutionError(RuntimeError):
    pass


def _compile(code: str) -> tuple[list[tuple[str, int]], dict[int, int]]:
    filtered = "".join(c for c in code if c in BF_COMMANDS)
    ops: list[tuple[str, int]] = []
    stack: list[int] = []
    jump: dict[int, int] = {}
    i = 0
    while i < len(filtered):
        c = filtered[i]
        if c in _LINEAR:
            j = i + 1
            while j < len(filtered) and filtered[j] == c:
                j += 1
            ops.append((c, j - i))
            i = j
            continue

        op_index = len(ops)
        ops.append((c, 1))
        if c == "[":
            stack.append(op_index)
        elif c == "]":
            if not stack:
                raise BFExecutionError(f"unmatched ] near command {i}")
            left = stack.pop()
            jump[left] = op_index
            jump[op_index] = left
        i += 1

    if stack:
        raise BFExecutionError("unmatched [")
    return ops, jump


def run_bf(
    code: str,
    input_data: str = "",
    *,
    memory_size: int = 300_000,
    step_limit: int | None = 500_000_000,
) -> BFResult:
    """Execute Brainfuck with wrapping 8-bit cells.

    Non-Brainfuck characters are ignored, matching common interpreters.
    Input past EOF yields a zero byte.  ``steps`` and ``step_limit`` count the
    original Brainfuck commands executed, even though consecutive linear
    commands are dispatched as one Python opcode.
    """

    ops, jump = _compile(code)
    mem = [0] * memory_size
    ptr = pc = steps = input_pos = 0
    out: list[str] = []

    while pc < len(ops):
        c, count = ops[pc]
        steps += count
        if step_limit is not None and steps > step_limit:
            raise BFExecutionError(f"step limit exceeded ({step_limit:,})")

        if c == ">":
            ptr += count
            if ptr >= memory_size:
                raise BFExecutionError("data pointer moved past allocated tape")
        elif c == "<":
            ptr -= count
            if ptr < 0:
                raise BFExecutionError("data pointer moved left of cell 0")
        elif c == "+":
            mem[ptr] = (mem[ptr] + count) & 0xFF
        elif c == "-":
            mem[ptr] = (mem[ptr] - count) & 0xFF
        elif c == ".":
            out.append(chr(mem[ptr]))
        elif c == ",":
            if input_pos < len(input_data):
                mem[ptr] = ord(input_data[input_pos]) & 0xFF
                input_pos += 1
            else:
                mem[ptr] = 0
        elif c == "[" and mem[ptr] == 0:
            pc = jump[pc]
        elif c == "]" and mem[ptr] != 0:
            pc = jump[pc]
        pc += 1

    return BFResult("".join(out), mem, ptr, steps, input_pos)
