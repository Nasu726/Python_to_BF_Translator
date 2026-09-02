from bf_runtime import run_bf
from compiler import compile_source as compile_internal
from pybf import compile_source as compile_public


def run(code: str, input_data: str) -> str:
    return run_bf(
        code,
        input_data,
        memory_size=80_000,
        step_limit=500_000_000,
    ).output


def execute(source: str, input_data: str) -> str:
    # Small fixed capacities exercise exactly the same line/token lowering while
    # keeping BF execution tests fast. Public fixed-ABI routing is tested once
    # separately below.
    code = compile_internal(
        source,
        string_capacity=24,
        list_capacity=8,
    )
    return run(code, input_data)


def test_public_entrypoint_routes_fixed_arity_map_to_line_aware_compiler():
    source = '''
a, b = map(int, input().split())
c = int(input())
print(a, b, c)
'''
    assert run(compile_public(source), '10 -2\n7\n') == '10 -2 7\n'


def test_fixed_arity_map_does_not_leak_across_input_lines():
    source = '''
a, b = map(int, input().split())
c = int(input())
print(a, b, c)
'''
    # Extra values are intentionally discarded rather than implementing
    # Python's unpacking ValueError, but they must remain owned by this input().
    assert execute(source, '1 2 999\n7\n') == '1 2 7\n'

    # Missing values are currently zero-filled, and most importantly the next
    # source-level input() still starts at the following line.
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
