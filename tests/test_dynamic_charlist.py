import ast
from functools import lru_cache

import pytest

from bf_runtime import run_bf
from bfstreamseq import RECORD_STRIDE
from compiler_dynamic_charlist import select_dynamic_char_list
from compiler_layout import compile_source, lower_with_layout


BF_COMMANDS = set("><+-.,[]")
SOURCE_LIMIT = 512 * 1024
STEP_LIMIT = 1_000_000_000


@lru_cache(maxsize=None)
def _compile(source: str) -> str:
    code = compile_source(source, string_capacity=255, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    return code


def _run(code: str, data: str):
    return run_bf(code, data, memory_size=40_000, step_limit=STEP_LIMIT)


ROUNDTRIP_SOURCE = '''
chars = list(input())
print(len(chars))
print("".join(chars))
'''


INDEX_SOURCE = '''
chars = list(input())
i = int(input())
old = chars[i]
chars[i] = "X"
print(old)
print(len(chars))
print("".join(chars))
'''


def test_selector_accepts_one_owned_runtime_character_list():
    selection = select_dynamic_char_list(ast.parse(INDEX_SOURCE))
    assert selection is not None
    assert selection.name == "chars"


@pytest.mark.parametrize(
    "source",
    [
        '''
chars = list(input())
other = chars
print("".join(chars))
''',
        '''
left = list(input())
right = list(input())
print("".join(left))
''',
        '''
chars = list(input())
chars = list(input())
print("".join(chars))
''',
        '''
chars = list(input())
text = "".join(chars)
print(text)
''',
    ],
)
def test_selector_keeps_unsupported_object_shapes_on_fixed_fallback(source):
    assert select_dynamic_char_list(ast.parse(source)) is None


def test_dynamic_layout_starts_after_every_compile_time_temporary():
    raw, plan = lower_with_layout(
        INDEX_SOURCE,
        string_capacity=255,
        list_capacity=4,
    )

    assert raw
    assert plan.dynamic_charlist_base is not None
    assert plan.dynamic_charlist_base - RECORD_STRIDE == plan.temp_peak


@pytest.mark.parametrize("length", [0, 1, 8, 9, 256, 300, 1024])
def test_runtime_character_list_roundtrip_and_length_beyond_fixed_capacity(length):
    code = _compile(ROUNDTRIP_SOURCE)
    text = "".join(chr(ord("A") + i % 26) for i in range(length))
    result = _run(code, text + "\n")

    assert result.output == f"{length}\n{text}\n"


def test_runtime_character_list_iteration_beyond_configured_fixed_capacity():
    source = '''
chars = list(input())
for ch in chars:
    print(ch, end="")
print()
'''
    code = compile_source(source, string_capacity=8, list_capacity=4)
    text = "".join(chr(ord("A") + i % 26) for i in range(40))

    assert select_dynamic_char_list(ast.parse(source)) is not None
    assert _run(code, text + "\n").output == text + "\n"


def test_runtime_character_list_iteration_preserves_break_continue_else():
    source = '''
chars = list(input())
for ch in chars:
    if ch == "C":
        continue
    if ch == "E":
        break
    print(ch, end="")
else:
    print("!", end="")
print()
'''
    code = compile_source(source, string_capacity=8, list_capacity=4)

    assert _run(code, "ABCDEF\n").output == "ABD\n"
    assert _run(code, "ABCD\n").output == "ABD!\n"


@pytest.mark.parametrize("index", [255, 256, -1, -300, -301])
def test_runtime_character_list_signed_index_load_store(index):
    code = _compile(INDEX_SOURCE)
    text = "".join(chr(ord("A") + i % 26) for i in range(300))
    valid = -len(text) <= index < len(text)
    old = text[index] if valid else ""
    expected = list(text)
    if valid:
        expected[index] = "X"

    result = _run(code, f"{text}\n{index}\n")
    assert result.output == f"{old}\n300\n{''.join(expected)}\n"


def test_runtime_character_list_vertical_slice_stays_under_submission_limit():
    code = _compile(INDEX_SOURCE)
    assert len(code) < SOURCE_LIMIT
