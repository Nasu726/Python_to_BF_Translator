from bf_runtime import run_bf
from transpiler import compile_source


def execute(source: str, input_data: str = '') -> str:
    code = compile_source(source, string_capacity=16, list_capacity=5)
    return run_bf(code, input_data, memory_size=12000, step_limit=800_000_000).output


def test_list_literal_index_assignment_append_and_print():
    source = '''
A = [1, 2, 3]
A[1] = 7
A.append(-4)
print(A, len(A), A[0], A[-1])
'''
    assert execute(source) == '[1, 7, 3, -4] 4 1 -4\n'


def test_dynamic_index_and_for_list():
    source = '''
A = [2, 4, 6]
i = 1
print(A[i])
s = 0
for x in A:
    s += x
print(s, x)
'''
    assert execute(source) == '4\n12 6\n'


def test_list_map_int_input_split():
    source = '''
A = list(map(int, input().split()))
print(A)
print(len(A), A[2])
'''
    assert execute(source, '10 -2 7 4\n') == '[10, -2, 7, 4]\n4 7\n'


def test_overlong_input_list_is_truncated_but_next_line_survives():
    source = '''
A = list(map(int, input().split()))
x = int(input())
print(A, x)
'''
    assert execute(source, '1 2 3 4 5 6 7\n9\n') == '[1, 2, 3, 4, 5] 9\n'


def test_list_assignment_is_currently_value_copy():
    source = '''
A = [1, 2]
B = A
B[0] = 9
print(A, B)
'''
    # Deliberate fixed-runtime limitation: unlike CPython, list variables are
    # currently value objects rather than references/aliases.
    assert execute(source) == '[1, 2] [9, 2]\n'
