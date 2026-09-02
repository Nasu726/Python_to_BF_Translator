from bf_runtime import run_bf
from transpiler_v3 import compile_source


def execute(source: str, input_data: str = '') -> str:
    code = compile_source(source, string_capacity=16)
    return run_bf(code, input_data, memory_size=8192, step_limit=500_000_000).output


def test_string_literal_assignment_and_print_keywords():
    source = '''
s = "hello"
print("value", s, sep=":", end="!")
'''
    assert execute(source) == 'value:hello!'


def test_string_input_copy_len_and_equality():
    source = '''
s = input()
t = s
print(t, len(t), t == "nasu")
'''
    assert execute(source, 'nasu\n') == 'nasu 4 1\n'


def test_string_input_line_drain():
    source = '''
a = input()
b = input()
print(a, b, sep="|")
'''
    assert execute(source, 'abcdefghijklmnopQRST\nxy\n') == 'abcdefghijklmnop|xy\n'


def test_multiple_assignment_and_tuple_swap():
    source = '''
a = b = 3
x = 7
a, x = x, a
print(a, b, x)
'''
    assert execute(source) == '7 3 3\n'


def test_string_tuple_swap():
    source = '''
a = "left"
b = "right"
a, b = b, a
print(a, b)
'''
    assert execute(source) == 'right left\n'


def test_chained_comparison_short_circuits():
    source = '''
x = 10
print(0 < x < 20)
print(20 < x < int(input()))
'''
    # Second chain fails at 20 < x and therefore must not consume input.
    assert execute(source) == '1\n0\n'


def test_python_style_signed_right_shift():
    source = '''
print(-8 >> 2, -1 >> 100, 8 >> 2)
'''
    assert execute(source) == '-2 -1 2\n'


def test_common_scalar_builtins():
    source = '''
x = -7
print(abs(x), bool(x), min(3, 5), max(3, 5))
'''
    assert execute(source) == '7 1 3 5\n'
