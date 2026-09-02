from bf_runtime import run_bf
from transpiler import compile_source


def execute(source: str, input_data: str = '') -> str:
    code = compile_source(source, string_capacity=16)
    return run_bf(code, input_data, memory_size=8192, step_limit=500_000_000).output


def test_map_int_input_split_unpack():
    source = '''
a, b = map(int, input().split())
print(a + b, a * b)
'''
    assert execute(source, '7 -2\n') == '5 -14\n'


def test_map_int_input_split_three_values():
    source = '''
a, b, c = map(int, input().split())
print(min(a, b), max(b, c), a + b + c)
'''
    assert execute(source, '10 3 7\n') == '3 7 20\n'
