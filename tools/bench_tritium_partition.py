#!/usr/bin/env python3
"""Generate the public partition BF program and deterministic benchmark inputs.

The companion GitHub Actions workflow builds the exact Tritium commit used by
AtCoder's 2025-10 language image (rdebath/Brainfuck commit 14a729d) and invokes
it with the same `tritium -b -e Main.bf` command shape.

This script intentionally performs no timing itself.  Keeping program/input
generation separate makes the benchmark usable locally, in Actions, or in an
AtCoder-compatible container without changing compiler behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pybf"))

from bfcontestpartition import build_partition_program  # noqa: E402


def write_program(path: Path) -> int:
    code = build_partition_program()
    path.write_text(code, encoding="ascii")
    return len(code)


def write_all_ones_input(path: Path, n: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")
    second = " ".join(["1"] * n)
    path.write_text(f"{n}\n{second}\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=Path, default=Path("Main.bf"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--n", type=int)
    args = parser.parse_args()

    size = write_program(args.program)
    print(f"program_bytes={size}")

    if (args.input is None) != (args.n is None):
        parser.error("--input and --n must be supplied together")
    if args.input is not None:
        write_all_ones_input(args.input, args.n)
        print(f"input_n={args.n} input_bytes={args.input.stat().st_size}")


if __name__ == "__main__":
    main()
