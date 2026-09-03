import ast

from bf_runtime import run_bf
from compiler_partition import lower_partition_program_if_supported, match_partition_program
from pybf import compile_source


SOURCE = '''
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

RENAMED_SOURCE = '''
count = int(input())
values = list(map(int, input().split()))
total = 0
for p in range(count):
    total += values[p]
best = 7777777
prefix = 0
for q in range(count):
    prefix += values[q]
    best = min(best, abs(total - prefix * 2))
print(best)
'''


def _reference(values, initial_ans=10_000_000):
    total = sum(values)
    left = 0
    ans = initial_ans
    for value in values:
        left += value
        ans = min(ans, abs(total - 2 * left))
    return ans


def test_partition_matcher_is_structural_not_name_based():
    original = match_partition_program(ast.parse(SOURCE))
    renamed = match_partition_program(ast.parse(RENAMED_SOURCE))
    assert original is not None
    assert original.initial_ans == 10_000_000
    assert renamed is not None
    assert renamed.initial_ans == 7_777_777


def test_partition_matcher_rejects_near_misses():
    near_misses = [
        SOURCE.replace("s-2*left", "s-3*left"),
        SOURCE.replace("range(n):\n    left", "range(n - 1):\n    left"),
        SOURCE.replace("min(ans, abs", "max(ans, abs"),
        SOURCE + "print(0)\n",
        SOURCE.replace("print(ans)", "print(left)"),
    ]
    for source in near_misses:
        assert match_partition_program(ast.parse(source)) is None
        assert lower_partition_program_if_supported(ast.parse(source)) is None


def test_partition_structural_lowering_emits_standard_bf_within_limit():
    code = lower_partition_program_if_supported(ast.parse(SOURCE))
    assert code is not None
    assert set(code) <= set("><+-.,[]")
    assert len(code.encode("ascii")) <= 512 * 1024


def test_public_compiler_routes_partition_program_and_executes_correctly():
    code = compile_source(SOURCE)
    assert set(code) <= set("><+-.,[]")
    assert len(code.encode("ascii")) <= 512 * 1024

    cases = [
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        [100],
        list(range(12)),
    ]
    for values in cases:
        input_data = f"{len(values)}\n" + " ".join(map(str, values)) + "\n"
        result = run_bf(
            code,
            input_data,
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert result.output == f"{_reference(values)}\n"
