#!/usr/bin/env python3
"""Python -> Brainfuck compiler entrypoint.

Usage:
    python main.py program.py

The generated Brainfuck is written next to the source as ``program.bf``.
There are intentionally no compiler options: the runtime ABI uses fixed-size
representations defined by the compiler.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pybf import CompileError, compile_source


ATCODER_SOURCE_LIMIT = 512 * 1024


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python main.py <program.py>", file=sys.stderr)
        return 2

    source_path = Path(sys.argv[1])
    if source_path.suffix != ".py":
        print("error: input must be a .py file", file=sys.stderr)
        return 2
    if not source_path.is_file():
        print(f"error: file not found: {source_path}", file=sys.stderr)
        return 2

    try:
        source = source_path.read_text(encoding="utf-8")
        code = compile_source(source, str(source_path))
    except (CompileError, SyntaxError, ValueError) as exc:
        print(f"compile error: {exc}", file=sys.stderr)
        return 1

    output_path = source_path.with_suffix(".bf")
    output_path.write_text(code, encoding="utf-8")
    size = len(code.encode("ascii"))
    print(f"wrote {output_path} ({size:,} bytes)")
    if size > ATCODER_SOURCE_LIMIT:
        print(
            "warning: generated Brainfuck exceeds AtCoder's 512 KiB source limit "
            f"by {size - ATCODER_SOURCE_LIMIT:,} bytes",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
