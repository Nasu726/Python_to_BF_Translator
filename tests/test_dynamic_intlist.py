import ast
import subprocess
import sys

import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from compiler_dynamic_intlist import select_dynamic_int_list
from compiler_layout import lower_with_layout
from bfpackedseq import (ACCESS_WORKSPACE_CELLS, LOAD_ACCESS_WORKSPACE_CELLS,
                         REDUCTION_WORKSPACE_CELLS, _increment_mobile_u64)
from pybf import compile_source


def reference(source, data):
    return subprocess.run([sys.executable, "-c", source], input=data, text=True,
                          capture_output=True, check=True, timeout=5).stdout


def execute(code, data):
    return run_bf(code, data, memory_size=120_000, step_limit=500_000_000)


def test_runtime_integer_list_alias_repr_after_linear_index_fill():
    source = '''
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
print(b)
'''
    selection = select_dynamic_int_list(ast.parse(source))
    assert selection is not None and selection.needs_repr
    code = compile_source(source)
    assert len(code) <= 512 * 1024
    for n in (0, 5, 65):
        data = f"{n}\n"
        assert execute(code, data).output == reference(source, data)


def test_runtime_integer_list_input_repr_keeps_alias_and_items():
    source = '''
a = list(map(int, input().split()))
b = a
print(b)
print(a[0], len(b))
'''
    data = "-7 0 1000000000\n"
    selection = select_dynamic_int_list(ast.parse(source))
    assert selection is not None and selection.needs_repr
    assert execute(compile_source(source), data).output == reference(source, data)


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
    "a=list(map(int,input().split()))\ndel a[0]\nprint(len(a))",
    "a=list(map(int,input().split()))\nprint(a[:])",
    "a=list(map(int,input().split()))\nprint(a, a)",
    "a=list(map(int,input().split()))\nprint(a, end='!')",
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


@pytest.mark.parametrize("uses,needs_load,needs_store,needs_sum", [
    ("print(len(a))", False, False, False),
    ("print(sum(a))", False, False, True),
    ("print(a[0])", True, False, False),
    ("a[0] = 1", False, True, False),
    ("a[0] = a[1]", True, True, False),
    ("a[0] += 1", True, True, False),
])
def test_dynamic_integer_selection_tracks_required_access_frame(
    uses, needs_load, needs_store, needs_sum,
):
    selection = select_dynamic_int_list(ast.parse(
        "a=list(map(int,input().split()))\n" + uses + "\n"
    ))
    assert selection is not None
    assert selection.needs_load is needs_load
    assert selection.needs_store is needs_store
    assert selection.needs_sum is needs_sum


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


def test_dynamic_integer_augmented_assignment_evaluates_index_once_before_rhs():
    source = '''
a = [10] * 2
b = a
b[int(input())] += int(input())
print(a[0], a[1], len(b), sum(a))
print(input())
'''
    data = "0\n3\ntail\n"
    result = execute(compile_source(source), data)
    assert result.output == reference(source, data)
    assert result.input_consumed == len(data)
    _, plan = lower_with_layout(source)
    assert plan.dynamic_intlist_base - ACCESS_WORKSPACE_CELLS > plan.temp_peak


@pytest.mark.parametrize("op,rhs", [
    ("+", 3),
    ("-", 3),
    ("*", 3),
    ("//", 2),
    ("%", 2),
    ("&", 3),
    ("|", 3),
    ("^", 3),
])
def test_dynamic_integer_augmented_operators_update_alias_and_cached_sum(op, rhs):
    source = f'''
a = [10] * 1
b = a
b[-1] {op}= {rhs}
print(a[0], len(b), sum(a))
'''
    assert execute(compile_source(source), "").output == reference(source, "")


def test_dynamic_integer_augmented_assignment_beyond_old_capacity():
    source = '''
a = [2] * 65
b = a
b[64] += 3
print(a[64], len(a), sum(b))
'''
    assert execute(compile_source(source), "").output == "5 65 133\n"


@pytest.mark.parametrize("operator,operand", [
    ("+", 3), ("-", 3), ("//", 4), ("%", 4),
])
def test_linear_literal_loop_matches_cpython_and_preserves_aliases(operator, operand):
    source = f'''
a = list(map(int, input().split()))
b = a
for i in range(len(a)):
    b[i] {operator}= {operand}
print(len(a), a[0], b[1], a[2])
'''
    code = compile_source(source)
    assert len(code) < 900_000  # Three unrelated rooted print loads remain.
    data = "-17 255 -9223372036854775799\n"
    assert execute(code, data).output == reference(source, data)


def test_linear_literal_loop_scales_past_old_capacity_and_handles_empty_list():
    source = '''
a = list(map(int, input().split()))
for i in range(len(a)):
    a[i] += 3
print(len(a))
'''
    code = compile_source(source)
    assert len(code) < 250_000
    for n in (0, 1, 65):
        data = " ".join(str(i % 5) for i in range(n)) + "\n"
        result = execute(code, data)
        assert result.output == reference(source, data)
        if n == 65:
            assert result.steps < 10_000_000
            with_last = source.replace("print(len(a))", "print(a[64], len(a))")
            assert execute(compile_source(with_last), data).output == reference(
                with_last, data,
            )


def test_linear_literal_loop_falls_back_if_index_or_sum_is_observable():
    index_source = '''
a = [2] * 3
for i in range(len(a)):
    a[i] += 3
print(i, a[0], a[2])
'''
    sum_source = '''
a = [2] * 3
for i in range(len(a)):
    a[i] += 3
print(sum(a), a[2])
'''
    for source in (index_source, sum_source):
        assert execute(compile_source(source), "").output == reference(source, "")


@pytest.mark.parametrize("bound,values,initial", [
    ("len(b)", "-48 7", 7),
    ("len(b)", "", -9),
    ("n", "-16 40 3", -2),
    ("n", "4 8 16", 19),
])
def test_linear_even_halving_tallies_and_mutates_shared_records(bound, values, initial):
    source = f'''
n = 2
a = list(map(int, input().split()))
b = a
answer = {initial}
for i in range({bound}):
    while a[i] % 2 == 0:
        b[i] //= 2
        answer += 1
print(answer, b[0] if len(a) else 0, len(b))
'''
    data = values + "\n"
    code = compile_source(source)
    assert execute(code, data).output == reference(source, data)


def test_linear_even_halving_handles_signed_boundary_and_snapshot_bound():
    source = '''
a = [-(1 << 63)] * 1
answer = 1
for i in range(answer):
    while a[i] % 2 == 0:
        a[i] //= 2
        answer += 1
print(answer, a[0])
'''
    assert execute(compile_source(source), "").output == reference(source, "")


def test_linear_even_halving_keeps_suffix_and_carries_scalar_over_u32():
    source = '''
a = [8] * 3
answer = 4294967295
n = 1
for i in range(n):
    while a[i] % 2 == 0:
        a[i] //= 2
        answer += 1
print(answer, a[0], a[2])
'''
    assert execute(compile_source(source), "").output == reference(source, "")


def test_linear_even_halving_falls_back_for_observable_index_or_sum():
    source = '''
a = [8] * 2
answer = 0
for i in range(len(a)):
    while a[i] % 2 == 0:
        a[i] //= 2
        answer += 1
print(i, answer, sum(a))
'''
    assert execute(compile_source(source), "").output == reference(source, "")


def test_mobile_packed_increment_carries_across_all_eight_bytes():
    for value in (0, 255, 65535, (1 << 32) - 1, (1 << 64) - 1):
        bf = BFEmitter()
        for i in range(8):
            bf.set_const(8 + i, (value >> (8 * i)) & 255)
        _increment_mobile_u64(bf, 8, 0)
        result = execute(bf.code(), "")
        assert sum(result.memory[8 + i] << (8 * i) for i in range(8)) == (
            value + 1
        ) % (1 << 64)
        assert not any(result.memory[i] for i in range(30, 46))


def test_linear_index_fill_matches_p0_alias_shape_and_fits_submission_limit():
    source = '''
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
print(b[-1], len(a))
'''
    code = compile_source(source)
    assert len(code) <= 512 * 1024
    assert set(code) <= set("><+-.,[]")
    for n in (1, 8, 65):
        data = f"{n}\n"
        assert execute(code, data).output == reference(source, data)


@pytest.mark.parametrize("source", [
    '''a = [9] * 5
n = 3
for i in range(n):
    a[i] = i
print(a[0], a[2], a[4])''',
    '''a = [9] * 3
for i in range(-1):
    a[i] = i
print(a[0], a[2])''',
    '''a = [9] * 0
for i in range(len(a)):
    a[i] = i
print(len(a))''',
    '''n = 3
a = [9] * n
n = 1
for i in range(n):
    a[i] = i
print(a[0], a[2])''',
    '''a = [9] * 3
for i in range(len(a)):
    a[i] = i
print(a[0], a[2])''',
])
def test_linear_index_fill_count_and_list_extent_guards(source):
    assert execute(compile_source(source), "").output == reference(source, "")


@pytest.mark.parametrize("source,expected", [
    ('''a = [9] * 5
n = 7
for i in range(n):
    a[i] = i
print(a[0], a[2], a[4])''', "0 2 4\n"),
    ('''n = 3
a = [9] * n
a.clear()
for i in range(n):
    a[i] = i
print(len(a))''', "0\n"),
])
def test_linear_index_fill_preserves_restricted_miss_behavior(source, expected):
    # The existing dynamic-list route treats out-of-range stores as no-ops
    # until a raising IndexError runtime ABI is implemented.
    assert execute(compile_source(source), "").output == expected


def test_linear_index_fill_falls_back_when_index_or_sum_is_observable():
    for tail in ("print(i, a[2])", "print(sum(a), a[2])"):
        source = f'''a = [9] * 3
for i in range(len(a)):
    a[i] = i
{tail}
'''
        assert execute(compile_source(source), "").output == reference(source, "")


@pytest.mark.parametrize("extent,data", [
    ("n", "4\n100 3 2 1\n"),
    ("n", "5\n1 2 1 1 3\n"),
    ("len(h)", "4\n100 3 2 1\n"),
    ("len(h)", "1\n42\n"),
    ("n", "0\n\n"),
    ("n", "-1\n4 2\n"),
    ("n", "3\n1 2 3 10\n"),
])
def test_reverse_adjacent_decrease_preserves_alias_and_early_exit(extent, data):
    source = f'''
n = int(input())
h = list(map(int, input().split()))
alias = h
ok = True
for i in range({extent} - 2, -1, -1):
    if alias[i] > h[i + 1]:
        alias[i] -= 1
    if h[i] > alias[i + 1]:
        ok = False
        break
if len(h):
    if ok:
        print("Yes", alias[0], h[-1])
    else:
        print("No", alias[0], h[-1])
else:
    if ok:
        print("Yes", len(h))
    else:
        print("No", len(h))
'''
    code = compile_source(source)
    assert execute(code, data).output == reference(source, data)


@pytest.mark.parametrize("tail", [
    "print(i, h[0])",
    "print(sum(h), h[0])",
])
def test_reverse_adjacent_decrease_falls_back_when_index_or_sum_is_live(tail):
    source = f'''
n = 3
h = [1] * 3
ok = True
for i in range(n - 2, -1, -1):
    if h[i] > h[i + 1]:
        h[i] -= 1
    if h[i] > h[i + 1]:
        ok = False
        break
{tail}
'''
    assert execute(compile_source(source), "").output == reference(source, "")


@pytest.mark.parametrize("operator,operand", [
    ("+", 3), ("-", 3), ("//", 4), ("%", 4),
])
def test_counted_prefix_loop_matches_cpython_for_shorter_extent(operator, operand):
    source = f'''
a = list(map(int, input().split()))
b = a
n = int(input())
for i in range(n):
    b[i] {operator}= {operand}
print(a[0], b[1], a[2], n)
'''
    code = compile_source(source)
    for count in (-2, 0, 1, 2, 3):
        data = f"-17 18 20\n{count}\n"
        assert execute(code, data).output == reference(source, data)


def test_counted_prefix_loop_normalizes_literal_int64_min_to_zero():
    source = '''
a = [3] * 2
n = -9223372036854775808
for i in range(n):
    a[i] += 2
print(a[0], a[1])
'''
    assert execute(compile_source(source), "").output == reference(source, "")


def test_counted_prefix_loop_evaluates_bound_once_and_preserves_following_input():
    source = '''
a = [3] * 4
for i in range(int(input())):
    a[i] += 2
print(a[0], a[1], a[2], a[3])
print(input())
'''
    data = "2\ntail\n"
    result = execute(compile_source(source), data)
    assert result.output == reference(source, data)
    assert result.input_consumed == len(data)


def test_counted_prefix_loop_clips_legacy_out_of_range_noop_and_crosses_64():
    source = '''
a = [3] * 65
n = int(input())
for i in range(n):
    a[i] += 2
print(a[0], a[64], len(a))
'''
    code = compile_source(source)
    for count in (0, 64, 65, 67):
        output = execute(code, f"{count}\n").output
        assert output == f"{5 if count else 3} {5 if count >= 65 else 3} 65\n"


def test_counted_prefix_loop_public_source_and_steps_stay_under_budget():
    source = '''
n = int(input())
a = list(map(int, input().split()))
for i in range(n):
    a[i] += 3
print(len(a))
'''
    code = compile_source(source)
    assert len(code) <= 512 * 1024
    last_source = source.replace("print(len(a))", "print(a[-1], len(a))")
    assert len(compile_source(last_source)) <= 512 * 1024
    values = [i % 5 for i in range(65)]
    data = "65\n" + " ".join(map(str, values)) + "\n"
    result = execute(code, data)
    assert result.output == reference(source, data)
    assert result.steps < 13_000_000


def test_counted_prefix_loop_falls_back_when_induction_variable_is_live():
    source = '''
a = [3] * 3
n = 2
for i in range(n):
    a[i] += 1
print(i, a[0], a[1], a[2])
'''
    assert execute(compile_source(source), "").output == reference(source, "")


def test_dynamic_integer_augmented_index_survives_dynamic_rhs_load():
    source = '''
a = [10] * 2
b = a
b[int(input())] += a[int(input())]
print(a[0], a[1], sum(b))
'''
    data = "0\n1\n"
    assert execute(compile_source(source), data).output == reference(source, data)


def test_dynamic_integer_invalid_augmented_index_is_noop_but_evaluates_rhs():
    source = '''
a = [4] * 2
a[int(input())] += int(input())
print(a[0], a[1], sum(a))
print(input())
'''
    data = "256\n3\ntail\n"
    result = execute(compile_source(source), data)
    assert result.output == "4 4 8\ntail\n"
    assert result.input_consumed == len(data)


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
