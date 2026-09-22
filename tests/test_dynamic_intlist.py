import ast
import subprocess
import sys

import pytest

from bf_runtime import run_bf
from compiler_dynamic_intlist import select_dynamic_int_list
from compiler_layout import lower_with_layout
from bfpackedseq import (ACCESS_WORKSPACE_CELLS, LOAD_ACCESS_WORKSPACE_CELLS,
                         REDUCTION_WORKSPACE_CELLS)
from pybf import compile_source


def reference(source, data):
    return subprocess.run([sys.executable, "-c", source], input=data, text=True,
                          capture_output=True, check=True, timeout=5).stdout


def execute(code, data):
    return run_bf(code, data, memory_size=120_000, step_limit=500_000_000)


@pytest.mark.parametrize("line", ["", "  \t", "0", "-1 255 256 -257", "1 2 3   "])
def test_alias_chain_shares_clear_and_cached_values(line):
    source = '''
a = list(map(int, input().split()))
b = a
c = b
print(len(a), sum(b), sum(c), len(c))
b.clear()
print(sum(a), len(c))
c.clear()
print(len(b), sum(c))
print(input())
'''
    data = line + "\nfollowing\n"
    code = compile_source(source)
    result = execute(code, data)
    assert result.output == reference(source, data)
    assert result.input_consumed == len(data)


def test_clear_inside_control_flow_and_repeated_reads():
    source = '''
a = list(map(int, input().split()))
b = a
for i in range(3):
    if i == 1:
        b.clear()
    print(len(a), sum(b))
print(input())
'''
    data = "5 -2 8\nnext\n"
    assert execute(compile_source(source), data).output == reference(source, data)


def test_runtime_extent_layout_and_repeated_metadata_reads_beyond_old_capacity():
    source = '''
a = list(map(int, input().split()))
b = a
print(len(b), sum(a))
x = sum(b) + len(a)
print(x, sum(a), len(b))
print(input())
'''
    code = compile_source(source)
    raw, plan = lower_with_layout(source)
    assert plan.dynamic_intlist_base is not None
    assert plan.dynamic_charlist_base is None
    assert plan.dynamic_intlist_base - REDUCTION_WORKSPACE_CELLS > plan.temp_peak
    assert set(raw) <= set("><+-.,[]")
    for n in (65, 256, 300):
        data = " ".join(str(i % 5) for i in range(n)) + "\nX\n"
        assert execute(code, data).output == reference(source, data)


@pytest.mark.parametrize("source", [
    "a=list(map(int,input().split()))\na=[1]\nprint(len(a))",
    "a=list(map(int,input().split()))\nb=a\nb=[2]\nprint(len(a))",
    "a=list(map(int,input().split()))\na[0]+=2\nprint(len(a))",
    "a=list(map(int,input().split()))\nprint(a[:])",
    "a=list(map(int,input().split()))\nb=[a]\nprint(len(a))",
    "a=list(map(int,input().split()))\nprint(sum(a, 1))",
    "a=list(map(int,input().split()))\nx=a.clear()",
    "a=list(map(int,input().split()))\na.clear(1)",
    "if True:\n a=list(map(int,input().split()))\nprint(len(a))",
    "print(len(a))\na=list(map(int,input().split()))",
    "a=list(map(int,input().split()))\nif True:\n b=a\nprint(len(b))",
    "b=1\na=list(map(int,input().split()))\nb=a\nprint(len(b))",
    "a=list(map(int,input().split()))\nb=list(map(int,input().split()))\nprint(len(a))",
    "a=list(map(int,input().split()))\ns=list(input())\nprint(len(a), len(s))",
    "sum=1\na=list(map(int,input().split()))\nprint(len(a))",
])
def test_unsupported_identity_or_use_shapes_are_not_selected(source):
    assert select_dynamic_int_list(ast.parse(source)) is None


ABC103_C_SOURCE = '''
n = int(input())
a = list(map(int, input().split()))
print(sum(a) - n)
'''


def test_abc103_c_official_samples_and_maximum_n_small_values():
    # https://atcoder.jp/contests/abc103/tasks/abc103_c
    # Maximum N with small values is a capacity/correctness test, not a claim
    # about all value patterns or the contest's Tritium wall-clock limit.
    code = compile_source(ABC103_C_SOURCE)
    assert len(code) <= 512 * 1024
    cases = [
        ("3\n3 4 6\n", "10\n"),
        ("5\n7 46 11 20 11\n", "90\n"),
        ("7\n994 518 941 851 647 2 581\n", "4527\n"),
        ("3000\n" + " ".join(["2"] * 3000) + "\n", "3000\n"),
    ]
    for data, expected in cases:
        assert reference(ABC103_C_SOURCE, data) == expected
        assert execute(code, data).output == expected


@pytest.mark.parametrize("value,count", [(0, 0), (17, -1), (2, 65), (-1, 3), (256, 256), (2, 300)])
def test_runtime_singleton_repeat_and_shared_clear(value, count):
    source = '''
x = int(input())
n = int(input())
a = [x] * n
b = a
print(len(b), sum(a), x, n)
b.clear()
print(len(a), sum(b))
print(input())
'''
    data = f"{value}\n{count}\nnext\n"
    code = compile_source(source)
    assert execute(code, data).output == reference(source, data)


@pytest.mark.parametrize("expression,data", [
    ("[int(input())] * int(input())", "7\n3\n"),
    ("int(input()) * [int(input())]", "3\n7\n"),
    ("[int(input())] * int(input())", "7\n-3\n"),
    ("int(input()) * [int(input())]", "-3\n7\n"),
])
def test_repeat_operands_evaluate_once_in_python_order_even_if_empty(expression, data):
    source = f"a = {expression}\nprint(sum(a), len(a))\nprint(input())\n"
    data += "tail\n"
    result = execute(compile_source(source), data)
    assert result.output == reference(source, data)
    assert result.input_consumed == len(data)


@pytest.mark.parametrize("source", [
    'n=3\na=["x"]*n\nprint(len(a))',
    'n=3\na=[[1]]*n\nprint(len(a))',
    'a=[1]*"2"\nprint(len(a))',
])
def test_integer_repeat_route_rejects_non_integer_operands(source):
    from pybf import CompileError
    with pytest.raises(CompileError):
        compile_source(source)


def test_repeat_count_mutation_does_not_change_constructed_list():
    source = 'n=int(input())\nx=int(input())\na=n*[x]\nn=1\nx=99\nprint(len(a),sum(a))\n'
    data = '65\n2\n'
    assert execute(compile_source(source), data).output == reference(source, data)


@pytest.mark.parametrize("value,count", [(5, -(1 << 63)), (-(1 << 63), 1), ((1 << 63) - 1, 1)])
def test_repeat_int64_boundaries_without_decimal_input_cost(value, count):
    # Direct 19-digit scalar parsing has a separate, larger existing test budget.
    # This gate isolates repetition/normalization and stays at 500M raw steps.
    source = f"x={value}\nn={count}\na=[x]*n\nprint(len(a),sum(a))\n"
    assert execute(compile_source(source), "").output == reference(source, "")


def test_repeat_operand_liveness_without_later_scalar_reads():
    source = 'x=int(input())\nn=int(input())\na=[x]*n\nprint(len(a),sum(a))\n'
    data = '2\n3\n'
    assert execute(compile_source(source), data).output == reference(source, data)


def test_zero_repeat_has_runtime_extent_and_bounded_source():
    source = 'n=int(input())\na=[0]*n\nprint(len(a),sum(a))\n'
    code = compile_source(source)
    assert len(code) <= 512 * 1024
    _, plan = lower_with_layout(source)
    assert plan.dynamic_intlist_base - REDUCTION_WORKSPACE_CELLS > plan.temp_peak
    for n in (0, -1, 65, 256, 1024):
        data = f"{n}\n"
        assert execute(code, data).output == reference(source, data)



def test_repeat_candidate_does_not_disable_established_dynamic_input_owner():
    source = 'a=list(map(int,input().split()))\nb=[0]*1\nprint(len(a))\n'
    selection = select_dynamic_int_list(ast.parse(source))
    assert selection is not None and selection.owner == 'a'
    data = ' '.join(['2'] * 65) + '\n'
    assert execute(compile_source(source), data).output == reference(source, data)


@pytest.mark.parametrize("uses,needs_load,needs_store", [
    ("print(len(a))", False, False),
    ("print(a[0])", True, False),
    ("a[0] = 1", False, True),
    ("a[0] = a[1]", True, True),
])
def test_dynamic_integer_selection_tracks_required_access_frame(
    uses, needs_load, needs_store,
):
    selection = select_dynamic_int_list(ast.parse(
        "a=list(map(int,input().split()))\n" + uses + "\n"
    ))
    assert selection is not None
    assert selection.needs_load is needs_load
    assert selection.needs_store is needs_store


@pytest.mark.parametrize("index", [0, 64, 256, -1, -65])
def test_dynamic_integer_index_load_store_alias_and_cached_sum(index):
    source = '''
a = list(map(int, input().split()))
b = a
i = int(input())
x = int(input())
b[i] = x
print(a[i], len(b), sum(a))
'''
    values = list(range(65))
    data = " ".join(map(str, values)) + f"\n{index}\n-7\n"
    code = compile_source(source)
    result = execute(code, data)
    if -len(values) <= index < len(values):
        assert result.output == reference(source, data)
    else:
        # Runtime exceptions remain deferred: invalid load is zero and store is
        # a no-op. Crucially, 256 does not wrap to element zero.
        assert result.output == f"0 65 {sum(values)}\n"


def test_dynamic_integer_store_evaluates_rhs_before_runtime_index():
    source = '''
a = [0] * 3
a[int(input())] = int(input())
print(a[0], a[1], a[2], sum(a))
'''
    data = "7\n1\n"
    # RHS consumes 7 first, then the target index consumes 1.
    assert execute(compile_source(source), data).output == reference(source, data)
    _, plan = lower_with_layout(source)
    assert plan.dynamic_intlist_base - ACCESS_WORKSPACE_CELLS > plan.temp_peak


def test_dynamic_integer_index_survives_clear_as_legacy_noop_zero_contract():
    source = '''
a = [4] * 3
b = a
b.clear()
a[0] = 9
print(a[0], len(b), sum(a))
'''
    assert execute(compile_source(source), "").output == "0 0 0\n"


ABC170_A_SOURCE = '''
x = list(map(int, input().split()))
for i in range(5):
    if x[i] == 0:
        print(i + 1)
'''


def test_abc170_a_official_samples_with_runtime_index_loop():
    # https://atcoder.jp/contests/abc170/tasks/abc170_a
    code = compile_source(ABC170_A_SOURCE)
    assert len(code) <= 512 * 1024
    raw, plan = lower_with_layout(ABC170_A_SOURCE)
    assert plan.dynamic_intlist_base is not None
    assert plan.dynamic_intlist_base - LOAD_ACCESS_WORKSPACE_CELLS > plan.temp_peak
    assert set(raw) <= set("><+-.,[]")
    for data, expected in [
        ("0 2 3 4 5\n", "1\n"),
        ("1 2 0 4 5\n", "3\n"),
    ]:
        assert reference(ABC170_A_SOURCE, data) == expected
        assert execute(code, data).output == expected
