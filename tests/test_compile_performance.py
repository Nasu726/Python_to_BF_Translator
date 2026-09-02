import ast
import time

from bfopt import optimize_bf
from compiler_stream import PythonToBFStream


BF_COMMANDS = set("><+-.,[]")
ATCODER_SOURCE_LIMIT = 512 * 1024

CONTEST_SUM_SOURCE = '''
n = int(input())
l = list(map(int,input().split()))

s = 0
for i in range(n):
    s += l[i]

ans = 10000000
left = 0
for i in range(n):
    left += l[i]
    ans = min(ans, abs(s-2*left))

print(ans)
'''


def test_contest_list_index_program_compiles_in_practical_time():
    started = time.perf_counter()
    tree = ast.parse(CONTEST_SUM_SOURCE)

    lower_started = time.perf_counter()
    compiler = PythonToBFStream(tree, string_capacity=255, list_capacity=64)
    raw = compiler.compile_module(tree)
    lower_elapsed = time.perf_counter() - lower_started

    opt_started = time.perf_counter()
    code = optimize_bf(raw)
    opt_elapsed = time.perf_counter() - opt_started
    elapsed = time.perf_counter() - started

    raw_size = len(raw.encode("ascii"))
    final_size = len(code.encode("ascii"))
    attribution = ", ".join(
        f"L{line}:{kind}={size:,}B"
        for line, kind, size in compiler.statement_sizes
    )
    diagnostics = (
        f"total={elapsed:.2f}s lowering={lower_elapsed:.2f}s "
        f"optimizer={opt_elapsed:.2f}s raw={raw_size:,}B final={final_size:,}B "
        f"statements=[{attribution}]"
    )

    assert code
    assert set(code) <= BF_COMMANDS
    assert elapsed < 60.0, diagnostics
    assert final_size <= ATCODER_SOURCE_LIMIT, (
        diagnostics + f" limit={ATCODER_SOURCE_LIMIT:,}B"
    )
