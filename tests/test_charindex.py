from bf_runtime import run_bf
from compiler_layout import compile_source


BF_COMMANDS = set("><+-.,[]")
ATCODER_SOURCE_LIMIT = 512 * 1024


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


def test_constant_index_beyond_physical_capacity_does_not_wrap():
    source = '''
chars = list(input())
chars[256] = "Q"
print("".join(chars))
print(chars[256])
'''
    code = _compile(source)
    result = run_bf(code, "abc\n", memory_size=120_000, step_limit=1_000_000_000)
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


def test_cached_length_refreshes_when_char_list_is_read_again():
    source = '''
chars = list(input())
print(len(chars))
chars = list(input())
print(len(chars))
'''
    code = _compile(source)
    result = run_bf(
        code,
        "abcdef\nxy\n",
        memory_size=120_000,
        step_limit=1_000_000_000,
    )
    assert result.output == "6\n2\n"


def test_cached_length_guards_store_after_char_list_is_read_again():
    source = '''
chars = list(input())
chars[4] = "X"
chars = list(input())
chars[4] = "Q"
print("".join(chars))
'''
    code = _compile(source)
    result = run_bf(
        code,
        "abcdef\nxy\n",
        memory_size=120_000,
        step_limit=1_000_000_000,
    )
    assert result.output == "xy\n"


def test_join_assignment_snapshots_mutable_character_view():
    source = '''
chars = list(input())
text = "".join(chars)
chars[0] = "X"
print(text)
print("".join(chars))
'''
    code = _compile(source)
    result = run_bf(code, "abc\n", memory_size=120_000, step_limit=1_000_000_000)
    # Direct join can be a zero-copy view, but assigning the resulting Python
    # string must snapshot the value because strings are immutable and later
    # list mutation must not change ``text``.
    assert result.output == "abc\nXbc\n"


def test_direct_join_list_input_stays_below_atcoder_source_limit():
    source = '''
print("".join(list(input())))
'''
    code = compile_source(source, string_capacity=255, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"direct join(list(input())) emitted {len(code):,} bytes"
    )
    result = run_bf(
        code,
        "brainfuck\n",
        memory_size=120_000,
        step_limit=1_000_000_000,
    )
    assert result.output == "brainfuck\n"


def test_one_runtime_char_store_stays_below_atcoder_source_limit():
    source = '''
chars = list(input())
i = int(input())
chars[i] = "X"
print("".join(chars))
'''
    code = compile_source(source, string_capacity=255, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"single runtime char store emitted {len(code):,} bytes"
    )
