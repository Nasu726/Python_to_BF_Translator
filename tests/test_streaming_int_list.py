import ast

from bf_runtime import run_bf
from compiler_layout import PythonToBFLayout, compile_source


STEP_LIMIT = 1_000_000_000
ABC_SOURCE_LIMIT = 512 * 1024

ABC153_B_SOURCE = """
h, n = map(int, input().split())
a = list(map(int, input().split()))
total = 0
for x in a:
    total += x
if total >= h:
    print("Yes")
else:
    print("No")
"""


def _compile(source: str) -> str:
    code = compile_source(source, string_capacity=8, list_capacity=4)
    assert set(code) <= set("><+-.,[]")
    return code


def _run(code: str, data: str):
    return run_bf(
        code,
        data,
        memory_size=120_000,
        step_limit=STEP_LIMIT,
    )


def test_streaming_int_list_passes_each_token_to_loop_target():
    source = """
a = list(map(int, input().split()))
for x in a:
    print(x, end="|")
print()
"""
    code = _compile(source)
    assert _run(code, "1 2 3\n").output == "1|2|3|\n"


def test_single_use_input_list_streams_beyond_configured_capacity():
    source = """
a = list(map(int, input().split()))
s = 0
for x in a:
    s += x
print(s)
"""
    code = _compile(source)
    values = list(range(70))
    result = _run(code, " ".join(map(str, values)) + "\n")
    assert result.output == f"{sum(values)}\n"


def test_streaming_int_list_handles_signed_values_and_empty_line():
    source = """
a = list(map(int, input().split()))
s = 0
for x in a:
    s += x
print(s)
"""
    code = _compile(source)
    signed = [7, -3, 0, 11, -20, 5]
    assert _run(code, " ".join(map(str, signed)) + "\n").output == "0\n"
    assert _run(code, "\n").output == "0\n"


def test_streaming_int_list_source_is_runtime_length_independent():
    source = """
a = list(map(int, input().split()))
s = 0
for x in a:
    s += x
print(s)
"""
    code = _compile(source)
    source_size = len(code)
    short = _run(code, "1 2 3\n")
    values = [1] * 120
    long = _run(code, " ".join(map(str, values)) + "\n")
    assert len(code) == source_size
    assert short.output == "6\n"
    assert long.output == "120\n"


def test_streaming_int_list_break_drains_rest_of_line_before_next_input():
    source = """
a = list(map(int, input().split()))
s = 0
for x in a:
    if x == 3:
        break
    s += x
y = int(input())
print(s, y)
"""
    code = _compile(source)
    middle = _run(code, "1 2 3 4 5\n9\n")
    final = _run(code, "1 2 3\n9\n")
    assert middle.output == "3 9\n"
    assert final.output == "3 9\n"


def test_streaming_int_list_continue_and_for_else_match_python():
    source = """
a = list(map(int, input().split()))
s = 0
for x in a:
    if x < 0:
        continue
    s += x
else:
    s += 100
print(s)
"""
    code = _compile(source)
    result = _run(code, "1 -10 2 -20 3\n")
    assert result.output == "106\n"


def test_fusion_does_not_cross_side_effectful_setup():
    tree = ast.parse(
        """
a = list(map(int, input().split()))
print("ready")
for x in a:
    print(x)
"""
    )
    compiler = PythonToBFLayout(tree, string_capacity=8, list_capacity=4)
    assert compiler._find_input_int_list_consumer(tree.body, 0) is None


def test_abc153_b_public_default_source_fits_abc_limit():
    code = compile_source(ABC153_B_SOURCE)
    assert set(code) <= set("><+-.,[]")
    assert len(code) <= ABC_SOURCE_LIMIT


def test_abc153_b_common_raccoon_vs_monster_samples():
    code = _compile(ABC153_B_SOURCE)
    assert _run(code, "10 3\n4 5 6\n").output == "Yes\n"
    assert _run(code, "20 3\n4 5 6\n").output == "No\n"
    assert _run(code, "210 5\n31 41 59 26 53\n").output == "Yes\n"
    assert _run(code, "211 5\n31 41 59 26 53\n").output == "No\n"


def test_abc103_c_modulo_summation_samples():
    source = """
n = int(input())
a = list(map(int, input().split()))
ans = 0
for x in a:
    ans += x - 1
print(ans)
"""
    code = _compile(source)
    assert _run(code, "3\n3 4 6\n").output == "10\n"
    assert _run(code, "5\n7 46 11 20 11\n").output == "90\n"
    assert _run(code, "7\n994 518 941 851 647 2 581\n").output == "4527\n"
