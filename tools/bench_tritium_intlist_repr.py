"""Manual native check for the P0 repeat/alias/index-fill/list-repr prefix.

Build rdebath/Brainfuck rev 14a729d and run:
python tools/bench_tritium_intlist_repr.py --tritium /path/to/bfi --large

The source is unchanged ordinary Python except that ``a.sort()`` is omitted;
sorting and the general heap model remain separate P0 work. The large check
compares the full N=100000 output with CPython, not merely its length.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pybf import compile_source


SOURCE = '''
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
print(b)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tritium", required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    code = compile_source(SOURCE)
    limit = 512 * 1024
    print(f"source_bytes={len(code)} headroom_to_512k={limit - len(code)}",
          flush=True)
    if len(code) > limit:
        raise SystemExit("source exceeds the 512 KiB contest limit")
    with tempfile.TemporaryDirectory(prefix="pybf-intlist-repr-") as directory:
        program = Path(directory) / "repr.bf"
        program.write_text(code, encoding="ascii")
        cases = (0, 65, 1024, 4097, 100_000) if args.large else (0, 65, 1024)
        for n in cases:
            start = time.perf_counter()
            result = subprocess.run(
                [args.tritium, "-b", "-e", str(program)],
                input=f"{n}\n", text=True, capture_output=True,
                check=True, timeout=args.timeout,
            )
            elapsed = time.perf_counter() - start
            expected = f"{list(range(n))}\n"
            if result.stdout != expected:
                raise SystemExit(f"n={n}: output differs from Python")
            print(f"n={n} seconds={elapsed:.6f} output_bytes={len(result.stdout)}",
                  flush=True)


if __name__ == "__main__":
    main()
