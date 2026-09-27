"""Reproduce public-compiler source bytes and raw steps for record-local loops.

Run ``PYTHONPATH=pybf python tools/bench_linear_int_updates.py``. Pass
``--tritium /path/to/bfi.out`` for the optional native N=65/256/1024 cases.
Each fallback reads ``i`` after the loop, deliberately making the induction
variable live; the programs have different output and are not exact source-size
peers. The counted example reads N on the first input line.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bf_runtime import run_bf
from pybf import compile_source


PROGRAM = '''
a = list(map(int, input().split()))
for i in range(len(a)):
    a[i] += 3
print(len(a))
'''

COUNTED_PROGRAM = '''
n = int(input())
a = list(map(int, input().split()))
for i in range(n):
    a[i] += 3
print(len(a))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tritium", type=Path)
    args = parser.parse_args()
    for label, source in (
        ("linear", PROGRAM),
        ("rooted", PROGRAM.replace("print(len(a))", "print(i, len(a))")),
        ("counted", COUNTED_PROGRAM),
        ("counted-rooted", COUNTED_PROGRAM.replace(
            "print(len(a))", "print(i, len(a))",
        )),
    ):
        code = compile_source(source)
        print(f"{label} source_bytes={len(code)}", flush=True)
        for length in (8, 32, 65):
            data = (f"{length}\n" if label.startswith("counted") else "")
            data += " ".join(str(i % 5) for i in range(length)) + "\n"
            result = run_bf(
                code, data, memory_size=120_000, step_limit=500_000_000,
            )
            expected = (f"{length - 1} {length}\n" if "rooted" in label
                        else f"{length}\n")
            if result.output != expected:
                raise AssertionError((label, length, result.output, expected))
            print(f"{label} N={length} steps={result.steps}", flush=True)

    if args.tritium is not None:
        # Reading the last item proves the native pass actually mutates the
        # far end of the sequence, while remaining below the 512 KiB limit.
        with tempfile.TemporaryDirectory(prefix="pybf-linear-int-") as directory:
            program = Path(directory) / "program.bf"
            for label, source in (("native", PROGRAM),
                                  ("native-counted", COUNTED_PROGRAM)):
                source = source.replace("print(len(a))", "print(a[-1], len(a))")
                code = compile_source(source)
                print(f"{label} source_bytes={len(code)}", flush=True)
                program.write_text(code, encoding="ascii")
                for length in (65, 256, 1024):
                    data = (f"{length}\n" if label == "native-counted" else "")
                    data += " ".join(str(i % 5) for i in range(length)) + "\n"
                    started = time.perf_counter()
                    result = subprocess.run(
                        [str(args.tritium), "-b", "-e", str(program)],
                        input=data, text=True, capture_output=True, check=True,
                        timeout=30,
                    )
                    expected = f"{(length - 1) % 5 + 3} {length}\n"
                    if result.stdout != expected:
                        raise AssertionError((label, length, result.stdout, expected))
                    elapsed = time.perf_counter() - started
                    print(f"{label} N={length} seconds={elapsed:.6f}", flush=True)


if __name__ == "__main__":
    main()
