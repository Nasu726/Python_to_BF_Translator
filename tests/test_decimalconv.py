import ast

from bf_runtime import run_bf
from bfopt import optimize_bf
from compiler_charconv import PythonToBFStream as StaticDecimalCompiler
from compiler_decimalconv import PythonToBFStream as RuntimeDecimalCompiler
from compiler_layout import compile_source


BF_COMMANDS = set("><+-.,[]")
ATCODER_SOURCE_LIMIT = 512 * 1024


def _direct_compile(compiler_type, source: str, *, string_capacity: int = 255) -> str:
    tree = ast.parse(source)
    compiler = compiler_type(tree, string_capacity=string_capacity, list_capacity=4)
    code = optimize_bf(compiler.compile_module(tree))
    assert set(code) <= BF_COMMANDS
    return code


def test_runtime_decimal_parser_accepts_whitespace_past_old_scan_limit():
    source = '''
s = input()
n = int(s)
print(n)
'''
    code = compile_source(source, string_capacity=32, list_capacity=4)
    # The old correctness-first parser scanned only the first 21 cells. Keep
    # the sign/digits beyond that boundary to prove parsing is now driven by a
    # runtime character loop rather than a fixed compile-time scan count.
    text = " " * 24 + "-7" + " " * 4
    result = run_bf(code, text + "\n", memory_size=120_000, step_limit=1_000_000_000)
    assert result.output == "-7\n"


def test_rotation_decimal_parser_restores_named_source_exactly():
    source = '''
s = input()
a = int(s)
b = int(s)
print(s, a, b)
'''
    code = compile_source(source, string_capacity=32, list_capacity=4)
    result = run_bf(
        code,
        "  -123  \n",
        memory_size=120_000,
        step_limit=1_000_000_000,
    )
    assert result.output == "  -123   -123 -123\n"


def test_runtime_decimal_parser_emits_less_source_at_public_capacity():
    source = '''
s = input()
n = int(s)
print(n)
'''
    old_code = _direct_compile(StaticDecimalCompiler, source, string_capacity=255)
    new_code = _direct_compile(RuntimeDecimalCompiler, source, string_capacity=255)
    assert len(new_code) < len(old_code), (
        f"preserving-rotation decimal parser regressed public-capacity source size: "
        f"new={len(new_code):,} old={len(old_code):,}"
    )


def test_int_input_keeps_existing_compact_direct_reader():
    source = '''
n = int(input())
print(n)
'''
    code = compile_source(source, string_capacity=255, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"int(input()) fast path regressed to {len(code):,} bytes"
    )
    result = run_bf(
        code,
        "-9223372036854775808\n",
        memory_size=120_000,
        step_limit=1_000_000_000,
    )
    assert result.output == "-9223372036854775808\n"
