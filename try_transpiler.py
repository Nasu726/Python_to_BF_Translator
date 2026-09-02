"""Developer-facing compile/run CLI for trying an unmerged branch locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bf_runtime import BFExecutionError, run_bf
from transpiler import compile_source as compile_current
from transpiler_v2 import CompileError, compile_source as compile_v2
from transpiler_v3 import compile_source as compile_v3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile Python to Brainfuck and optionally execute the generated code"
    )
    parser.add_argument("source", type=Path, help="Python source file")
    parser.add_argument("-o", "--output", type=Path, help="write generated Brainfuck here")
    parser.add_argument(
        "--backend",
        choices=("current", "v3", "v2"),
        default="current",
        help="compiler frontend; current is recommended",
    )
    parser.add_argument("--string-capacity", type=int, default=128)
    parser.add_argument("--list-capacity", type=int, default=32)
    parser.add_argument("--run", action="store_true", help="execute generated Brainfuck")
    parser.add_argument("--input-file", type=Path, help="stdin text for the Brainfuck program")
    parser.add_argument("--input-text", default=None, help="literal stdin text for the Brainfuck program")
    parser.add_argument("--memory", type=int, default=300_000)
    parser.add_argument("--step-limit", type=int, default=500_000_000)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    try:
        if args.backend == "v2":
            code = compile_v2(source, str(args.source))
        elif args.backend == "v3":
            code = compile_v3(source, str(args.source), string_capacity=args.string_capacity)
        else:
            code = compile_current(
                source,
                str(args.source),
                string_capacity=args.string_capacity,
                list_capacity=args.list_capacity,
            )
    except (CompileError, SyntaxError, ValueError) as exc:
        print(f"compile error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"generated {len(code):,} BF commands ({args.backend})", file=sys.stderr)
    if args.output:
        args.output.write_text(code, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    if not args.run:
        if not args.output:
            sys.stdout.write(code)
        return

    if args.input_file and args.input_text is not None:
        parser.error("use only one of --input-file and --input-text")
    if args.input_file:
        input_data = args.input_file.read_text(encoding="utf-8")
    elif args.input_text is not None:
        # Convenient for shell use: --input-text '7 -2\n'
        input_data = bytes(args.input_text, "utf-8").decode("unicode_escape")
    else:
        input_data = sys.stdin.read() if not sys.stdin.isatty() else ""

    try:
        result = run_bf(
            code,
            input_data,
            memory_size=args.memory,
            step_limit=args.step_limit,
        )
    except BFExecutionError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc

    sys.stdout.write(result.output)
    print(
        f"\n[BF: {result.steps:,} steps, {result.input_consumed} input bytes]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
