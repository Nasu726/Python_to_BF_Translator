from bf_runtime import run_bf
from pybf import INT_BITS, LIST_CAPACITY, STRING_CAPACITY, compile_source


def test_public_abi_is_fixed():
    assert INT_BITS == 64
    assert STRING_CAPACITY == 255
    assert LIST_CAPACITY == 64


def test_public_compile_source_generates_runnable_brainfuck():
    code = compile_source("x = 7\nprint(x)\n")
    result = run_bf(code, step_limit=50_000_000)
    assert result.output == "7\n"
