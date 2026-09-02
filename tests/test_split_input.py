from bf_runtime import run_bf
from compiler import compile_source


def execute(source: str, input_data: str = '') -> str:
    code = compile_source(source, string_capacity=8, list_capacity=4)
    assert set(code) <= set('><+-.,[]')
    return run_bf(
        code,
        input_data,
        memory_size=30000,
        step_limit=1_000_000_000,
    ).output


def test_string_split_unpack_and_next_line_input():
    source = '''
a, b = input().split()
x = int(input())
print(a, b, x)
'''
    assert execute(source, 'hello world\n7\n') == 'hello world 7\n'


def test_string_split_unpack_does_not_read_through_newline():
    source = '''
a, b = input().split()
x = input()
print(a, b, x, sep='|')
'''
    # CPython would raise ValueError for the malformed unpack.  Until runtime
    # exceptions exist, missing split values become empty strings, but the next
    # input() must still begin on the following line.
    assert execute(source, 'hello\nworld\n') == 'hello||world\n'


def test_map_str_split_unpack():
    source = '''
a, b = map(str, input().split())
print(b, a, sep=':')
'''
    assert execute(source, 'left right\n') == 'right:left\n'


def test_string_list_input_index_len_and_iteration():
    source = '''
A = input().split()
print(A)
print(len(A), A[1])
for s in A:
    print(s, end='|')
print()
'''
    assert execute(source, 'aa bb cc\n') == "['aa', 'bb', 'cc']\n3 bb\naa|bb|cc|\n"


def test_list_map_str_input_split_and_mutation():
    source = '''
A = list(map(str, input().split()))
A[1] = 'XX'
A.append('z')
print(A, A[-1])
'''
    assert execute(source, 'a b\n') == "['a', 'XX', 'z'] z\n"


def test_int_and_string_line_inputs_can_be_mixed():
    source = '''
a, b = map(int, input().split())
S = input().split()
A = list(map(int, input().split()))
print(a + b, S[0], A[1])
'''
    assert execute(source, '3 4\nfoo bar\n8 9 10\n') == '7 foo 9\n'


def test_map_int_unpack_does_not_read_through_newline():
    source = '''
a, b = map(int, input().split())
x = int(input())
print(a, b, x)
'''
    # CPython would raise on the malformed unpack.  The fixed runtime does not
    # implement ValueError yet, but it must still preserve input()'s line
    # boundary instead of stealing x from the next line.
    assert execute(source, '5\n9\n') == '5 0 9\n'


def test_string_list_capacity_truncates_but_preserves_next_line():
    source = '''
S = input().split()
x = int(input())
print(S, x)
'''
    assert execute(source, 'a b c d e f\n42\n') == "['a', 'b', 'c', 'd'] 42\n"


def test_empty_list_stays_an_integer_list_by_default():
    source = '''
A = []
A.append(3)
A.append(-1)
print(A)
'''
    assert execute(source) == '[3, -1]\n'
