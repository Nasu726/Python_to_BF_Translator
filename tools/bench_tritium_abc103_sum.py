"""Manual ABC103 C public-compiler benchmark; no task-specific compiler path.

Build rdebath/Brainfuck revision 14a729d, then:
python tools/bench_tritium_abc103_sum.py --tritium /path/to/tritium/bfi.out

Local timing is not an AtCoder host/runtime guarantee.
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
a = list(map(int, input().split()))
print(sum(a) - n)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tritium", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("trials and timeout must be positive")
    code = compile_source(SOURCE)
    print(f"source_bytes={len(code)} headroom_to_512k={512 * 1024 - len(code)}", flush=True)
    assert len(code) <= 512 * 1024
    n = 3000
    patterns = {
        "twos": [2] * n,
        "maximum": [100000] * n,
        "ffff": [65535] * n,
        "deterministic": [2 + (i * 48271 + 17) % 99999 for i in range(n)],
    }
    failed = False
    with tempfile.TemporaryDirectory(prefix="pybf-abc103-") as directory:
        program = Path(directory) / "Main.bf"
        program.write_text(code)
        for name, values in patterns.items():
            data = f"{n}\n" + " ".join(map(str, values)) + "\n"
            expected = f"{sum(values) - n}\n"
            for trial in range(1, args.trials + 1):
                start = time.perf_counter()
                try:
                    result = subprocess.run([args.tritium, "-b", "-e", str(program)],
                        input=data, text=True, capture_output=True, timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    failed = True
                    print(f"pattern={name} n={n} trial={trial} timeout={args.timeout}", flush=True)
                    continue
                elapsed = time.perf_counter() - start
                assert result.returncode == 0, result.stderr
                assert result.stdout == expected, (name, result.stdout, expected)
                print(f"pattern={name} n={n} trial={trial} seconds={elapsed:.6f} output={expected.strip()}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
