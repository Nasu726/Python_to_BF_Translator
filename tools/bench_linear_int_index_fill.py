"""Reproduce public P0 integer-list index-fill source and execution metrics.

Run ``python tools/bench_linear_int_index_fill.py`` for raw interpreter steps.
Pass ``--tritium /path/to/bfi.out`` to run the same BF at N=65/1024/200000
with Tritium ``-b -e``. Local timing is not an AtCoder judge guarantee.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pybf"))

from bf_runtime import run_bf
from pybf import compile_source


PROGRAM = '''
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
print(b[-1], len(a))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tritium", type=Path)
    args = parser.parse_args()
    code = compile_source(PROGRAM)
    print(f"source_bytes={len(code)} headroom_to_512k={512 * 1024 - len(code)}",
          flush=True)
    for n in (8, 32, 65):
        data = f"{n}\n"
        expected = f"{n - 1} {n}\n"
        result = run_bf(code, data, memory_size=120_000,
                        step_limit=500_000_000)
        assert result.output == expected
        print(f"N={n} raw_steps={result.steps}", flush=True)

    if args.tritium is not None:
        with tempfile.TemporaryDirectory(prefix="pybf-int-index-fill-") as directory:
            program = Path(directory) / "Main.bf"
            program.write_text(code, encoding="ascii")
            for n in (65, 1024, 200_000):
                started = time.perf_counter()
                result = subprocess.run(
                    [str(args.tritium), "-b", "-e", str(program)],
                    input=f"{n}\n", text=True, capture_output=True,
                    check=True, timeout=90,
                )
                assert result.stdout == f"{n - 1} {n}\n"
                print(f"N={n} native_seconds={time.perf_counter() - started:.3f}",
                      flush=True)


if __name__ == "__main__":
    main()
