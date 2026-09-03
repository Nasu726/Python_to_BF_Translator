from bf_runtime import run_bf
from compiler_dynint import lower_dynamic_int_list_program_if_supported


STEP_LIMIT = 1_000_000_000


def _compile(source: str) -> str:
    code = lower_dynamic_int_list_program_if_supported(__import__("ast").parse(source))
    assert code is not None
    assert set(code) <= set("><+-.,[]")
    return code


def _run(code: str, data: str):
    return run_bf(
        code,
        data,
        memory_size=200_000,
        step_limit=STEP_LIMIT,
    )


def test_dynamic_input_list_length_and_index_beyond_old_capacity():
    source = """
a = list(map(int, input().split()))
print(len(a), a[69])
"""
    code = _compile(source)
    values = list(range(70))
    result = _run(code, " ".join(map(str, values)) + "\n")
    assert result.output == "70 69\n"


def test_dynamic_input_list_runtime_and_negative_index():
    source = """
a = list(map(int, input().split()))
i = 68
print(a[i], a[-1])
"""
    code = _compile(source)
    values = [1000 + i for i in range(70)]
    result = _run(code, " ".join(map(str, values)) + "\n")
    assert result.output == "1068 1069\n"


def test_dynamic_list_assignment_aliases_and_append_is_visible():
    source = """
a = list(map(int, input().split()))
b = a
b.append(777)
print(len(a), a[-1], len(b))
"""
    code = _compile(source)
    values = list(range(70))
    result = _run(code, " ".join(map(str, values)) + "\n")
    assert result.output == "71 777 71\n"


def test_dynamic_list_iteration_and_repeated_passes_beyond_old_capacity():
    source = """
a = list(map(int, input().split()))
s = 0
for x in a:
    s += x
t = 0
for x in a:
    t += x
print(s, t)
"""
    code = _compile(source)
    values = list(range(70))
    expected = sum(values)
    result = _run(code, " ".join(map(str, values)) + "\n")
    assert result.output == f"{expected} {expected}\n"


def test_dynamic_list_program_source_is_independent_of_runtime_line_length():
    source = """
a = list(map(int, input().split()))
print(len(a))
"""
    code = _compile(source)
    short = _run(code, "1 2 3\n")
    long_values = list(range(70))
    long = _run(code, " ".join(map(str, long_values)) + "\n")
    assert short.output == "3\n"
    assert long.output == "70\n"
