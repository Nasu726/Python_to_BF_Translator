import ast

from bf_runtime import run_bf
from bfopt import optimize_bf
from compiler_charconv import PythonToBFStream as StaticDecimalCompiler
from compiler_decimalconv import PythonToBFStream as RuntimeDecimalCompiler
from compiler_layout import compile_source


BF_COMMANDS = set("><+-.,[]")


def _direct_compile(compiler_type, source: str, *, string_capacity: int = 32) -> str:
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
    # The old correctness-first parser scanned only the first 21 cells.  Keep
    # the sign/digits beyond that boundary to prove parsing is now driven by a
    # runtime character loop rather than a fixed compile-time scan count.
    text = " " * 24 + "-7" + " " * 4
    result = run_bf(code, text + "\n", memory_size=120_000, step_limit=1_000_000_000)
    assert result.output == "-7\n"


def test_runtime_decimal_parser_emits_less_source_than_static_unroll():
    source = '''
s = input()
n = int(s)
print(n)
'''
    old_code = _direct_compile(StaticDecimalCompiler, source)
    new_code = _direct_compile(RuntimeDecimalCompiler, source)
    assert len(new_code) < len(old_code), (
        f"runtime-loop decimal parser regressed source size: "
        f"new={len(new_code):,} old={len(old_code):,}"
    )
