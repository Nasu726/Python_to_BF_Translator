from functools import lru_cache

import pytest

from bf_runtime import run_bf
from compiler_layout import CompileError, compile_source


STEP_LIMIT = 1_000_000_000
BF_COMMANDS = set("><+-.,[]")


@lru_cache(maxsize=None)
def _compile(source: str, *, string_capacity: int = 32) -> str:
    code = compile_source(source, string_capacity=string_capacity, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    return code


def _run(code: str, data: str = ""):
    return run_bf(code, data, memory_size=120_000, step_limit=STEP_LIMIT)


def test_list_input_empty_join_is_identity_view():
    source = '''
chars = list(input())
print("".join(chars))
'''
    code = _compile(source)
    assert _run(code, "abcXYZ\n").output == "abcXYZ\n"
    assert _run(code, "\n").output == "\n"


def test_join_list_input_direct_expression():
    source = '''
print("".join(list(input())))
'''
    code = _compile(source)
    assert _run(code, "brainfuck\n").output == "brainfuck\n"


def test_char_list_positive_and_negative_index_assignment():
    source = '''
chars = list(input())
chars[0] = "X"
chars[-1] = "Z"
print("".join(chars))
'''
    code = _compile(source)
    assert _run(code, "abcde\n").output == "XbcdZ\n"


def test_char_list_runtime_index_assignment():
    source = '''
chars = list(input())
i = int(input())
chars[i] = "Q"
print("".join(chars))
'''
    code = _compile(source)
    assert _run(code, "abcdef\n3\n").output == "abcQef\n"
    assert _run(code, "abcdef\n-2\n").output == "abcdQf\n"


def test_char_list_large_runtime_index_does_not_wrap_to_low_byte():
    source = '''
chars = list(input())
i = int(input())
chars[i] = "Q"
print("".join(chars))
'''
    code = _compile(source)
    assert _run(code, "abcdef\n256\n").output == "abcdef\n"
    assert _run(code, "abcdef\n-257\n").output == "abcdef\n"


def test_char_list_runtime_index_load():
    source = '''
chars = list(input())
i = int(input())
print(chars[i])
'''
    code = _compile(source)
    assert _run(code, "abcdef\n2\n").output == "c\n"
    assert _run(code, "abcdef\n-1\n").output == "f\n"


def test_char_list_swap_through_one_character_temporary():
    source = '''
chars = list(input())
a = int(input())
b = int(input())
tmp = chars[a]
chars[a] = chars[b]
chars[b] = tmp
print("".join(chars))
'''
    code = _compile(source)
    assert _run(code, "ABCDE\n1\n4\n").output == "AECDB\n"


def test_char_list_iteration_reuses_string_walker():
    source = '''
chars = list(input())
for c in chars:
    print(c, end="|")
print()
'''
    code = _compile(source)
    assert _run(code, "abc\n").output == "a|b|c|\n"


def test_abc199_c_ipfl_samples_with_char_list_view():
    source = '''
n = int(input())
s = list(input())
q = int(input())
flipped = 0
for k in range(q):
    t, a, b = map(int, input().split())
    if t == 1:
        a -= 1
        b -= 1
        if flipped:
            if a < n:
                a += n
            else:
                a -= n
            if b < n:
                b += n
            else:
                b -= n
        tmp = s[a]
        s[a] = s[b]
        s[b] = tmp
    else:
        flipped = 1 - flipped
if flipped:
    for i in range(n):
        tmp = s[i]
        s[i] = s[i + n]
        s[i + n] = tmp
print("".join(s))
'''
    code = _compile(source)
    sample1 = "2\nFLIP\n2\n2 0 0\n1 1 4\n"
    sample2 = "2\nFLIP\n6\n1 1 3\n2 0 0\n1 1 2\n1 2 3\n2 0 0\n1 1 4\n"
    assert _run(code, sample1).output == "LPFI\n"
    assert _run(code, sample2).output == "ILPF\n"


def test_char_list_rejects_multi_character_assignment():
    source = '''
chars = list(input())
chars[0] = "XY"
print("".join(chars))
'''
    with pytest.raises(CompileError):
        _compile(source)


def test_char_list_rejects_direct_repr_print_until_list_repr_exists():
    source = '''
chars = list(input())
print(chars)
'''
    with pytest.raises(CompileError):
        _compile(source)


def test_char_list_rejects_alias_until_mutable_object_semantics_exist():
    source = '''
chars = list(input())
other = chars
print("".join(chars))
'''
    with pytest.raises(CompileError):
        _compile(source)


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, 9, 10, -10, 123456789, -123456789, (1 << 63) - 1, -(1 << 63)],
)
def test_dynamic_str_int64(value):
    source = '''
n = int(input())
s = str(n)
print(s)
'''
    code = _compile(source)
    assert _run(code, f"{value}\n").output == f"{value}\n"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0),
        ("42", 42),
        ("+42", 42),
        ("-42", -42),
        ("   +42   ", 42),
        ("\t-42\t", -42),
        (str((1 << 63) - 1), (1 << 63) - 1),
        (str(-(1 << 63)), -(1 << 63)),
    ],
)
def test_dynamic_int_string(text, expected):
    source = '''
s = input()
n = int(s)
print(n)
'''
    code = _compile(source)
    assert _run(code, text + "\n").output == f"{expected}\n"


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, 12345, -12345, (1 << 63) - 1, -(1 << 63)],
)
def test_int_string_roundtrip(value):
    source = '''
n = int(input())
s = str(n)
m = int(s)
print(m)
'''
    code = _compile(source)
    assert _run(code, f"{value}\n").output == f"{value}\n"


def test_constant_int_string_conversions_fold_without_input():
    source = '''
a = int("-123") + 1
s = str(-456)
print(a, s)
'''
    code = _compile(source)
    assert _run(code).output == "-122 -456\n"


def test_str_of_string_is_identity_value():
    source = '''
s = input()
t = str(s)
print(t)
'''
    code = _compile(source)
    assert _run(code, "hello\n").output == "hello\n"


def test_int_of_int_is_identity_under_fixed_int64_abi():
    source = '''
n = int(input())
m = int(n)
print(m)
'''
    code = _compile(source)
    assert _run(code, "-98765\n").output == "-98765\n"
