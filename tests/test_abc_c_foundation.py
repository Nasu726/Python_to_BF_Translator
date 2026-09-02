from bf_runtime import run_bf
from compiler import compile_source


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
