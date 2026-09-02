import time

from pybf import compile_source


BF_COMMANDS = set("><+-.,[]")

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
    # This is deliberately generous for a hosted CI runner.  A short ABC-style
    # program taking minutes to compile is a compiler bug, not an optimization
    # wishlist item.
    assert elapsed < 60.0, f"compile took {elapsed:.2f}s"
