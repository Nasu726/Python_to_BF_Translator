"""Small reusable Brainfuck runtime used by tests and the developer CLI.

The project historically had an interpreter hard-wired to ``test.bf``.  This
module keeps execution separate from file I/O so generated programs can be
compiled and exercised before merging a branch.
"""

from __future__ import annotations

from dataclasses import dataclass


BF_COMMANDS = set("><+-.,[]")


@dataclass(frozen=True)
class BFResult:
    output: str
    memory: list[int]
    pointer: int
    steps: int
    input_consumed: int


class BFExecutionError(RuntimeError):
    pass


def _jump_table(code: str) -> dict[int, int]:
    stack: list[int] = []
    jump: dict[int, int] = {}
    for i, c in enumerate(code):
        if c == "[":
            stack.append(i)
        elif c == "]":
            if not stack:
                raise BFExecutionError(f"unmatched ] at command {i}")
            j = stack.pop()
            jump[i] = j
            jump[j] = i
    if stack:
        raise BFExecutionError(f"unmatched [ at command {stack[-1]}")
    return jump


def run_bf(
    code: str,
    input_data: str = "",
    *,
    memory_size: int = 300_000,
    step_limit: int | None = 500_000_000,
) -> BFResult:
    """Execute Brainfuck with wrapping 8-bit cells.

    Non-Brainfuck characters are ignored, matching common interpreters.
    Input past EOF yields a zero byte.  Output is returned as a Python string
    whose code points are the emitted byte values 0..255.
    """

    code = "".join(c for c in code if c in BF_COMMANDS)
    jump = _jump_table(code)
    mem = [0] * memory_size
    ptr = pc = steps = input_pos = 0
    out: list[str] = []

    while pc < len(code):
        steps += 1
        if step_limit is not None and steps > step_limit:
            raise BFExecutionError(f"step limit exceeded ({step_limit:,})")

        c = code[pc]
        if c == ">":
            ptr += 1
            if ptr >= memory_size:
                raise BFExecutionError("data pointer moved past allocated tape")
        elif c == "<":
            ptr -= 1
            if ptr < 0:
                raise BFExecutionError("data pointer moved left of cell 0")
        elif c == "+":
            mem[ptr] = (mem[ptr] + 1) & 0xFF
        elif c == "-":
            mem[ptr] = (mem[ptr] - 1) & 0xFF
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
