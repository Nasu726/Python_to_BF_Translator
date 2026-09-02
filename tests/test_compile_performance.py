import time

from pybf import compile_source


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
    code = compile_source(CONTEST_SUM_SOURCE)
    elapsed = time.perf_counter() - started

    assert code
    assert set(code) <= BF_COMMANDS
    assert elapsed < 60.0, f"compile took {elapsed:.2f}s"
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"generated BF is {len(code):,} bytes; limit is {ATCODER_SOURCE_LIMIT:,}"
    )
