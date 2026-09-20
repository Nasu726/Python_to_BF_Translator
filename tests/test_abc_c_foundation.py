import subprocess
import sys

import pytest

from bf_runtime import run_bf
from compiler import compile_source
from pybf import compile_source as compile_public_source
from compiler_layout import compile_source as compile_layout_source


def execute(source: str, input_data: str = "") -> str:
    code = compile_source(source, string_capacity=16, list_capacity=8)
    return run_bf(
        code,
        input_data,
        memory_size=120_000,
        step_limit=500_000_000,
    ).output


def test_runtime_sized_zero_list_for_dp_style_initialization():
    source = '''
n = int(input())
a = [0] * n
for i in range(n):
    a[i] = i + 1
print(a)
'''
    assert execute(source, "5\n") == "[1, 2, 3, 4, 5]\n"


def test_general_int_list_repetition_both_operand_orders():
    source = '''
a = [1, 2]
n = 3
b = a * n
c = 2 * a
print(b)
print(c)
'''
    assert execute(source) == "[1, 2, 1, 2, 1, 2]\n[1, 2, 1, 2]\n"


def test_negative_list_repeat_count_is_empty():
    source = '''
n = -3
a = [7] * n
print(a)
'''
    assert execute(source) == "[]\n"


def test_runtime_negative_index_load_and_store():
    source = '''
a = [10, 20, 30]
i = -1
print(a[i])
a[i] = 99
j = -2
print(a[j])
print(a)
'''
    assert execute(source) == "30\n20\n[10, 20, 99]\n"


def test_abc136_c_build_stairs_official_samples_against_cpython():
    # https://atcoder.jp/contests/abc136/tasks/abc136_c
    # Correctness fixture only: the existing fixed list capacity is sufficient
    # for samples, NOT the official N<=100000. Reuse this source unchanged when
    # the dynamic-list frontend and traversal are ready; never specialize on it.
    source = '''
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
    code = compile_public_source(source)
    assert set(code) <= set("><+-.,[]")
    samples = [
        ("5\n1 2 1 1 3\n", "Yes\n"),
        ("4\n1 3 2 1\n", "No\n"),
        ("5\n1 2 3 4 5\n", "Yes\n"),
        ("1\n1000000000\n", "Yes\n"),
    ]
    for data, expected in samples:
        reference = subprocess.run([sys.executable, "-c", source], input=data,
            text=True, capture_output=True, check=True, timeout=5).stdout
        assert reference == expected
        result = run_bf(code, data, memory_size=120_000, step_limit=500_000_000)
        assert result.output == reference


@pytest.mark.parametrize("op,expected", [("+", 13), ("-", 7), ("*", 30)])
def test_int_list_augmented_assignment_evaluates_target_once_before_rhs(op, expected):
    source = f'''
a = [10, 20]
a[int(input())] {op}= int(input())
print(a)
print(input())
'''
    code = compile_layout_source(source, string_capacity=8, list_capacity=2)
    result = run_bf(code, "0\n3\nend\n", memory_size=120_000,
                    step_limit=500_000_000)
    assert result.output == f"[{expected}, 20]\nend\n"


def test_signed_range_literal_and_negative_augmented_index():
    source = '''
a = [1, 2, 3]
for i in range(2, -1, -1):
    a[i] += 1
a[-1] -= a[0]
print(a)
'''
    code = compile_layout_source(source, string_capacity=8, list_capacity=3)
    assert run_bf(code, memory_size=120_000, step_limit=500_000_000).output == "[2, 3, 2]\n"


def test_constant_list_load_preserves_payload_for_repeated_reads():
    source = '''
a = [255, -1, 256]
print(a[0], a[0], a[1], a[2])
print(a)
'''
    code = compile_layout_source(source, string_capacity=8, list_capacity=3)
    result = run_bf(code, memory_size=120_000, step_limit=500_000_000)
    assert result.output == "255 255 -1 256\n[255, -1, 256]\n"
