from bf_runtime import run_bf
from pybf import compile_source


def execute(source: str, input_data: str) -> str:
    code = compile_source(source)
    return run_bf(
        code,
        input_data,
        memory_size=400_000,
        step_limit=1_500_000_000,
    ).output


def test_fixed_arity_map_int_then_next_input_line():
    source = '''
a, b = map(int, input().split())
c = int(input())
print(a, b, c)
'''
    assert execute(source, '10 -2\n7\n') == '10 -2 7\n'


def test_fixed_arity_map_does_not_leak_extra_same_line_token():
    source = '''
a, b = map(int, input().split())
c = int(input())
print(a, b, c)
'''
    # CPython would raise ValueError for the extra token during unpacking.  The
    # fixed BF runtime deliberately discards extras, but it must still preserve
    # the source-level input() line boundary rather than feeding 999 to c.
    assert execute(source, '1 2 999\n7\n') == '1 2 7\n'


def test_fixed_arity_map_does_not_steal_from_next_line_when_short():
    source = '''
a, b = map(int, input().split())
c = int(input())
print(a, b, c)
'''
    # Missing values are currently zero-filled instead of raising ValueError.
    assert execute(source, '5\n8\n') == '5 0 8\n'


def test_list_map_int_input_split_and_following_line():
    source = '''
A = list(map(int, input().split()))
x = int(input())
print(A)
print(len(A), x)
'''
    assert execute(source, '1 -2 30 4\n9\n') == '[1, -2, 30, 4]\n4 9\n'


def test_mixed_line_input_sequence():
    source = '''
n = int(input())
a, b = map(int, input().split())
A = list(map(int, input().split()))
x, y = map(int, input().split())
print(n)
print(a + b)
print(A)
print(x, y)
'''
    data = '3\n10 20\n4 5 6\n-7 8\n'
    assert execute(source, data) == '3\n30\n[4, 5, 6]\n-7 8\n'


def test_int_input_consumes_its_whole_line():
    source = '''
a = int(input())
b = int(input())
print(a, b)
'''
    # Invalid for CPython int(input()), but useful to guarantee that one source
    # input() never leaks residual bytes into the next source input().
    assert execute(source, '12 999\n34\n') == '12 34\n'
