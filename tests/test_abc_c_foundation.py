import ast
import random
import subprocess
import sys

import pytest

from bf_runtime import run_bf
from compiler import compile_source
from compiler_dynamic_intlist import select_dynamic_int_list
from pybf import compile_source as compile_public_source
from compiler_layout import (compile_source as compile_layout_source,
                             lower_with_layout)
from bfpackedseq import ACCESS_WORKSPACE_CELLS


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


def test_abc136_c_build_stairs_official_samples_against_cpython():
    # https://atcoder.jp/contests/abc136/tasks/abc136_c
    # The unchanged source uses a guarded reverse adjacent record pass. These
    # samples and the 512 KiB gate do not alone prove judge-scale runtime.
    selection = select_dynamic_int_list(ast.parse(ABC136_C_SOURCE))
    assert selection is not None
    assert selection.needs_load and selection.needs_store
    raw, plan = lower_with_layout(ABC136_C_SOURCE)
    assert plan.dynamic_intlist_base is not None
    assert plan.dynamic_intlist_base - ACCESS_WORKSPACE_CELLS > plan.temp_peak
    assert set(raw) <= set("><+-.,[]")
    code = compile_public_source(ABC136_C_SOURCE)
    assert len(code) <= 512 * 1024
    assert set(code) <= set("><+-.,[]")
    samples = [
        ("5\n1 2 1 1 3\n", "Yes\n"),
        ("4\n1 3 2 1\n", "No\n"),
        ("5\n1 2 3 4 5\n", "Yes\n"),
        ("1\n1000000000\n", "Yes\n"),
    ]
    for data, expected in samples:
        reference = subprocess.run([sys.executable, "-c", ABC136_C_SOURCE], input=data,
            text=True, capture_output=True, check=True, timeout=5).stdout
        assert reference == expected
        result = run_bf(code, data, memory_size=120_000, step_limit=500_000_000)
        assert result.output == reference


def test_abc136_c_build_stairs_diverse_signed_inputs_against_cpython():
    code = compile_public_source(ABC136_C_SOURCE)
    generator = random.Random(136)
    for length in range(2, 10):
        for _ in range(2):
            heights = [generator.randrange(-6, 7) for _ in range(length)]
            data = f"{length}\n" + " ".join(map(str, heights)) + "\n"
            reference = subprocess.run(
                [sys.executable, "-c", ABC136_C_SOURCE], input=data,
                text=True, capture_output=True, check=True, timeout=5,
            ).stdout
            assert run_bf(code, data, memory_size=120_000,
                          step_limit=500_000_000).output == reference


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


@pytest.mark.parametrize("op", ["//", "%", "&", "|", "^"])
@pytest.mark.parametrize("left,right", [(-17, 3), (17, -3), (-17, -3), (0, 3)])
def test_integer_augmented_operators_match_python(op, left, right):
    # Both constant and runtime-negative subscripts must preserve other slots.
    # Input for the index comes before the RHS and must be consumed just once.
    source = f'''
x = {left}
x {op}= {right}
a = [{left}, 255, {left}]
a[0] {op}= {right}
a[int(input())] {op}= int(input())
print(x)
print(a)
print(input())
'''
    data = f"-1\n{right}\nnext\n"
    expected = subprocess.run([sys.executable, "-c", source], input=data,
        text=True, capture_output=True, check=True, timeout=5).stdout
    code = compile_layout_source(source, string_capacity=8, list_capacity=3)
    assert run_bf(code, data, memory_size=120_000,
                  step_limit=500_000_000).output == expected


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


def test_abc100_c_official_samples_against_cpython():
    # https://atcoder.jp/contests/abc100/tasks/abc100_c
    # Ordinary source, official samples; runtime scale remains a separate gate.
    code = compile_public_source(ABC100_C_SOURCE)
    assert len(code) <= 512 * 1024
    assert set(code) <= set("><+-.,[]")
    samples = [
        ("3\n5 2 4\n", "3\n"),
        ("4\n631 577 243 199\n", "0\n"),
        ("10\n2184 2126 1721 1800 1024 2528 3360 1945 1280 1776\n", "39\n"),
    ]
    for data, expected in samples:
        reference = subprocess.run([sys.executable, "-c", ABC100_C_SOURCE],
            input=data, text=True, capture_output=True, check=True, timeout=5).stdout
        assert reference == expected
        result = run_bf(code, data, memory_size=120_000, step_limit=500_000_000)
        assert result.output == reference


@pytest.mark.parametrize("divisor", [1, 2, 256, 1 << 62])
def test_power_of_two_division_signed_boundaries(divisor):
    from compiler_layout import PythonToBFLayout

    # Inspect generated arithmetic directly, avoiding unrelated decimal-I/O
    # cost at int64 extrema. Keep all variables live in the layout AST.
    for value in [-(1 << 63), -17, -1, 0, 17, (1 << 63) - 1]:
        tree = ast.parse(f"x = {value}\nq = x // {divisor}\nr = x % {divisor}\n"
                         f"x //= {divisor}\nr %= {divisor}\nprint(x, q, r)\n")
        compiler = PythonToBFLayout(tree, string_capacity=8, list_capacity=1)
        for statement in tree.body[:-1]:
            compiler.compile_stmt(statement)
        result = run_bf(compiler.bf.code(), memory_size=120_000,
                        step_limit=500_000_000)
        for name, expected in [("x", value // divisor), ("q", value // divisor),
                               ("r", value % divisor)]:
            ref = compiler.variables[name]
            actual = sum(result.memory[ref.bit(i)] << i for i in range(64))
            assert actual == expected & ((1 << 64) - 1)


def test_runtime_divmod_preserves_operands_and_negative_remainder_correction():
    source = '''
x = -17
y = 3
print(x // y, x % y)
print(x, y)
x = 17
y = -3
print(x // y, x % y)
print(x, y)
'''
    code = compile_layout_source(source, string_capacity=8, list_capacity=1)
    assert run_bf(code, memory_size=120_000, step_limit=500_000_000).output == (
        "-6 1\n-17 3\n-6 -1\n17 -3\n"
    )
