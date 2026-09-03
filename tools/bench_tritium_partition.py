#!/usr/bin/env python3
"""Generate the public partition BF program and deterministic benchmark inputs.

The companion GitHub Actions workflow builds the exact Tritium commit used by
AtCoder's 2025-10 language image (rdebath/Brainfuck commit 14a729d) and invokes
it with the same `tritium -b -e Main.bf` command shape.

Input generation includes several distributions because raw Brainfuck work is
value-sensitive.  The expected result is computed independently with the same
fixed-width signed-int64 semantics used by the scalable compiler slice.
"""

from __future__ import annotations

import argparse
from itertools import cycle, islice
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pybf"))

from bfcontestpartition import build_partition_program  # noqa: E402


MASK64 = (1 << 64) - 1
PATTERNS = ("ones", "negative-ones", "ffff", "mixed", "deterministic")


def _s64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def _abs_s64(value: int) -> int:
    value = _s64(value)
    return _s64(-value) if value < 0 else value


def partition_reference(values: list[int], initial_ans: int = 10_000_000) -> int:
    total = 0
    for value in values:
        total = _s64(total + value)
    left = 0
    ans = _s64(initial_ans)
    for value in values:
        left = _s64(left + value)
        candidate = _abs_s64(_s64(total - _s64(2 * left)))
        if candidate < ans:
            ans = candidate
    return ans


def make_values(pattern: str, n: int) -> list[int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if pattern == "ones":
        return [1] * n
    if pattern == "negative-ones":
        return [-1] * n
    if pattern == "ffff":
        return [0xFFFF] * n
    if pattern == "mixed":
        return list(islice(cycle((5, -2, 10, 1234, -77)), n))
    if pattern == "deterministic":
        return [((i * 1_234_567) % 1_000_000) - 500_000 for i in range(n)]
    raise ValueError(f"unknown pattern: {pattern}")


def write_program(path: Path) -> int:
    code = build_partition_program()
    path.write_text(code, encoding="ascii")
    return len(code)


def write_input(path: Path, values: list[int]) -> None:
    path.write_text(
        f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, default=Path("Main.bf"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--n", type=int)
    parser.add_argument("--pattern", choices=PATTERNS, default="ones")
    parser.add_argument(
        "--no-program",
        action="store_true",
        help="only generate input/expected files; reuse an existing BF program",
    )
    args = parser.parse_args()

    if not args.no_program:
        size = write_program(args.program)
        print(f"program_bytes={size}")

    supplied = (args.input is not None, args.expected is not None, args.n is not None)
    if any(supplied) and not all(supplied):
        parser.error("--input, --expected, and --n must be supplied together")
    if args.input is not None:
        values = make_values(args.pattern, args.n)
        expected = partition_reference(values)
        write_input(args.input, values)
        args.expected.write_text(f"{expected}\n", encoding="ascii")
        print(
            f"pattern={args.pattern} input_n={args.n} "
            f"input_bytes={args.input.stat().st_size} expected={expected}"
        )


if __name__ == "__main__":
    main()
