"""Generate real-ABC-style streaming-list Brainfuck benchmarks.

The first workload is ABC153 B (Common Raccoon vs Monster), whose official
constraint is N <= 100000 and whose ordinary explicit-loop implementation is a
single-use integer input list followed by a linear fold.  This script compiles
that Python through the public compiler; it does not call an internal BF kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compiler_layout import compile_source


ABC153_B_SOURCE = '''
h, n = map(int, input().split())
a = list(map(int, input().split()))
total = 0
for x in a:
    total += x
if total >= h:
    print("Yes")
else:
    print("No")
'''


def values_for(pattern: str, n: int) -> list[int]:
    if pattern == "ones":
        return [1] * n
    if pattern == "mixed":
        seed = (5, 17, 101, 9999, 3, 250, 42)
        return [seed[i % len(seed)] for i in range(n)]
    if pattern == "max":
        return [10_000] * n
    raise ValueError(pattern)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--pattern", choices=("ones", "mixed", "max"), default="ones")
    parser.add_argument("--program", type=Path, default=Path("Main.bf"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--no-program", action="store_true")
    args = parser.parse_args()

    if args.n < 0:
        raise SystemExit("n must be nonnegative")

    values = values_for(args.pattern, args.n)
    total = sum(values)
    # Equality deliberately exercises the >= boundary and guarantees Yes.
    h = total
    args.input.write_text(
        f"{h} {args.n}\n" + " ".join(map(str, values)) + "\n",
        encoding="ascii",
    )
    args.expected.write_text("Yes\n", encoding="ascii")

    if not args.no_program:
        code = compile_source(ABC153_B_SOURCE, list_capacity=4)
        if set(code) - set("><+-.,[]"):
            raise SystemExit("compiler emitted non-standard Brainfuck")
        args.program.write_text(code, encoding="ascii")
        print(f"program_bytes={len(code)}")


if __name__ == "__main__":
    main()
