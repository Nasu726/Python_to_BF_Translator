"""Manual public-compiler runtime repetition benchmark (Tritium 14a729d).

python tools/bench_tritium_repeat.py --tritium /path/to/tritium/bfi.out
Local timings are not AtCoder judge acceptance. No problem-specific lowering.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pybf import compile_source

ZERO_SOURCE = 'n=int(input())\na=[0]*n\nprint(len(a),sum(a))\n'
VALUE_SOURCE = 'n,x=map(int,input().split())\na=[x]*n\nprint(len(a),sum(a))\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tritium', required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=10)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0:
        parser.error('trials and timeout must be positive')
    cases = [
        ('zero', ZERO_SOURCE, [('large', '100000\n', '100000 0\n')]),
        ('value', VALUE_SOURCE, [
            ('small', '3000 2\n', '3000 6000\n'),
            ('dense', '3000 -1\n', '3000 -3000\n'),
            ('negative_count', '-9223372036854775808 5\n', '0 0\n'),
            ('minimum_value', '1 -9223372036854775808\n', '1 -9223372036854775808\n'),
            ('maximum_value', '1 9223372036854775807\n', '1 9223372036854775807\n'),
        ]),
    ]
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / 'Main.bf'
        for name, source, patterns in cases:
            code = compile_source(source)
            if name == 'zero' and len(code) > 512 * 1024:
                raise SystemExit('zero-repeat source exceeded its 512 KiB gate')
            target.write_text(code, encoding='ascii')
            print(f'fixture={name} source_bytes={len(code)} headroom_to_512k={512*1024-len(code)}', flush=True)
            for pattern, data, expected in patterns:
                for trial in range(args.trials):
                    start = time.perf_counter()
                    result = subprocess.run([args.tritium, '-b', '-e', str(target)],
                        input=data, text=True, capture_output=True, timeout=args.timeout, check=True)
                    elapsed = time.perf_counter() - start
                    if result.stdout != expected:
                        raise SystemExit(f'{name}/{pattern}: unexpected output {result.stdout!r}')
                    print(f'fixture={name} pattern={pattern} trial={trial+1} seconds={elapsed:.6f} output={result.stdout.strip()}', flush=True)


if __name__ == '__main__':
    main()
