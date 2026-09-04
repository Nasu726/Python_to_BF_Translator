from bf_runtime import run_bf
from compiler_layout import compile_source


BF_COMMANDS = set("><+-.,[]")


def _compile(source: str) -> str:
    code = compile_source(source, string_capacity=32, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    return code


def test_positive_constant_index_past_logical_length_does_not_touch_suffix():
    source = '''
chars = list(input())
chars[10] = "Q"
print("".join(chars))
print(chars[10])
'''
    code = _compile(source)
    result = run_bf(code, "abc\n", memory_size=120_000, step_limit=1_000_000_000)
    # Runtime IndexError propagation is not implemented yet. The interim
    # contract is a no-op store / empty load, never hidden-suffix mutation.
    assert result.output == "abc\n\n"


def test_positive_constant_index_inside_runtime_length_still_works():
    source = '''
chars = list(input())
chars[2] = "Q"
print("".join(chars))
'''
    code = _compile(source)
    result = run_bf(code, "abcdef\n", memory_size=120_000, step_limit=1_000_000_000)
    assert result.output == "abQdef\n"
