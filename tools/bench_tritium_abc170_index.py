"""Manual ABC170 A dynamic-index benchmark through the public compiler.

Build rdebath/Brainfuck revision 14a729d, then:
python tools/bench_tritium_abc170_index.py --tritium /path/to/tritium/bfi.out

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
x = list(map(int, input().split()))
for i in range(5):
    if x[i] == 0:
        print(i + 1)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tritium", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("trials and timeout must be positive")

    code = compile_source(SOURCE)
    limit = 512 * 1024
    print(
        f"source_bytes={len(code)} headroom_to_512k={limit - len(code)}",
        flush=True,
    )
    if len(code) > limit:
        raise SystemExit("ABC170 A source exceeded its 512 KiB gate")

    with tempfile.TemporaryDirectory(prefix="pybf-abc170-") as directory:
        program = Path(directory) / "Main.bf"
        program.write_text(code, encoding="ascii")
        for zero_index in range(5):
            values = list(range(1, 6))
            values[zero_index] = 0
            data = " ".join(map(str, values)) + "\n"
            expected = f"{zero_index + 1}\n"
            for trial in range(1, args.trials + 1):
                start = time.perf_counter()
                result = subprocess.run(
                    [args.tritium, "-b", "-e", str(program)],
                    input=data,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                    check=True,
                )
                elapsed = time.perf_counter() - start
                if result.stdout != expected:
                    raise SystemExit(
                        f"zero_index={zero_index}: unexpected output "
                        f"{result.stdout!r}"
                    )
                print(
                    f"zero_index={zero_index} trial={trial} "
                    f"seconds={elapsed:.6f} output={result.stdout.strip()}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
