from bf_runtime import run_bf
from pybf import INT_BITS, LIST_CAPACITY, STRING_CAPACITY, compile_source


BF_COMMANDS = set("><+-.,[]")


def test_public_abi_is_fixed():
    assert INT_BITS == 64
    assert STRING_CAPACITY == 255
    assert LIST_CAPACITY == 64


def test_public_compile_source_emits_only_standard_brainfuck():
    code = compile_source(
        "x = int(input())\n"
        "A = [1, 2, 3]\n"
        "if x > 0:\n"
        "    A.append(x)\n"
        "print(A[-1])\n"
    )
    assert code
    assert set(code) <= BF_COMMANDS


def test_generated_brainfuck_executes_without_python_runtime_services():
    # Compilation happens above this boundary.  The executor receives only the
    # generated BF bytecode plus stdin; it has no access to the Python AST,
    # compiler objects, variable metadata, or helper functions.
    code = compile_source(
        "x = int(input())\n"
        "y = x * 3 + 1\n"
        "print(y)\n"
    )
    result = run_bf(code, "7\n", step_limit=100_000_000)
    assert result.output == "22\n"
