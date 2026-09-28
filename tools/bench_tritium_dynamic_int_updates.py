"""Manual dynamic-int update benchmark for real ABC100/ABC136 programs.

Build rdebath/Brainfuck revision 14a729d, then:
python tools/bench_tritium_dynamic_int_updates.py \
    --tritium /path/to/tritium/bfi.out

The unchanged ABC100 C and ABC136 C sources both fit 512 KiB via mobile
record walks. The benchmark checks native correctness/runtime separately;
local timing is not an AtCoder host guarantee. Pass --large to exercise
ABC136 C at its official maximum N=100000 (rather than in every smoke run).
The large run also profiles the same list-input frontend without the loop to
separate decimal-token parsing from reverse traversal costs. The official
time limit is 2 seconds; passing output here is not a judge-time claim.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pybf import compile_source


ABC100_C_SOURCE = '''
n = int(input())
a = list(map(int, input().split()))
answer = 0
for i in range(n):
    while a[i] % 2 == 0:
        a[i] //= 2
        answer += 1
print(answer)
'''


ABC136_C_SOURCE = '''
n = int(input())
h = list(map(int, input().split()))
ok = True
for i in range(n - 2, -1, -1):
    if h[i] > h[i + 1]:
        h[i] -= 1
    if h[i] > h[i + 1]:
        ok = False
        break
if ok:
    print("Yes")
else:
    print("No")
'''


INTLIST_INPUT_SOURCE = '''
n = int(input())
h = list(map(int, input().split()))
print(len(h))
'''


def _abc100_expected(values: list[int]) -> str:
    answer = 0
    for value in values:
        while value % 2 == 0:
            value //= 2
            answer += 1
    return f"{answer}\n"


def _abc136_expected(values: list[int]) -> str:
    values = values[:]
    for index in range(len(values) - 2, -1, -1):
        if values[index] > values[index + 1]:
            values[index] -= 1
        if values[index] > values[index + 1]:
            return "No\n"
    return "Yes\n"


def _cases(*, large: bool = False):
    abc100 = [
        ("sample1", [5, 2, 4]),
        ("sample2", [631, 577, 243, 199]),
        ("sample3", [2184, 2126, 1721, 1800, 1024,
                     2528, 3360, 1945, 1280, 1776]),
        ("beyond64", [1 << (index % 8) for index in range(65)]),
        ("n256", [1 << (index % 8) for index in range(256)]),
        ("n1024", [1 << (index % 8) for index in range(1024)]),
    ]
    abc136 = [
        ("sample1", [1, 2, 1, 1, 3]),
        ("sample2", [1, 3, 2, 1]),
        ("sample3", [1, 2, 3, 4, 5]),
        ("sample4", [1_000_000_000]),
        ("beyond64", list(range(65))),
        ("n256", list(range(256))),
        ("n1024", list(range(1024))),
    ]
    if large:
        abc136.extend([
            ("n100000_equal", [1] * 100_000),
            ("n100000_failure", [3, 1] + [1] * 99_998),
            ("n100000_upper_height", [1_000_000_000] * 100_000),
        ])
    programs = [
        ("abc100", ABC100_C_SOURCE, [
            (name, f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
             _abc100_expected(values))
            for name, values in abc100
        ]),
        ("abc136", ABC136_C_SOURCE, [
            (name, f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
             _abc136_expected(values))
            for name, values in abc136
        ]),
    ]
    if large:
        programs.append(("input_only", INTLIST_INPUT_SOURCE, [
            (name, f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
             f"{len(values)}\n")
            for name, values in abc136[-3:]
        ]))
    return programs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tritium", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error("trials and timeout must be positive")

    limit = 512 * 1024
    with tempfile.TemporaryDirectory(prefix="pybf-dynamic-int-update-") as directory:
        for problem, source, cases in _cases(large=args.large):
            code = compile_source(source)
            print(
                f"problem={problem} source_bytes={len(code)} "
                f"headroom_to_512k={limit - len(code)}",
                flush=True,
            )
            program = Path(directory) / f"{problem}.bf"
            program.write_text(code, encoding="ascii")
            for case, data, expected in cases:
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
                            f"problem={problem} case={case}: unexpected output "
                            f"{result.stdout!r}, expected {expected!r}"
                        )
                    print(
                        f"problem={problem} case={case} trial={trial} "
                        f"seconds={elapsed:.6f} output={expected.strip()}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
