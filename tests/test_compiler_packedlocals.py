import ast

from bf_runtime import run_bf
from compiler_layout import compile_source
from compiler_packedlocals import PythonToBFStream as PackedLocalCompiler


BF_COMMANDS = set("><+-.,[]")
STEP_LIMIT = 1_000_000_000


PACKED_LOOP_SOURCE = '''
q = int(input())
bias = int(input())
total = 0
for _ in range(q):
    t, a, b = map(int, input().split())
    if t == 1:
        a += b
    else:
        a -= b
    if a < bias:
        total += a
print(total)
'''


def test_structural_selector_finds_nonescaping_loop_locals_and_readonly_cache():
    tree = ast.parse(PACKED_LOOP_SOURCE)
    compiler = PackedLocalCompiler(tree, string_capacity=8, list_capacity=4)
    loop = next(node for node in tree.body if isinstance(node, ast.For))

    candidate = compiler._packed_loop_candidate(loop)
    assert candidate is not None
    locals_, readonly = candidate
    assert locals_ == ["t", "a", "b"]
    assert "q" in readonly
    assert "bias" in readonly


def test_structural_selector_rejects_loop_local_that_escapes_after_loop():
    source = '''
q = int(input())
for _ in range(q):
    t, a, b = map(int, input().split())
print(a)
'''
    tree = ast.parse(source)
    compiler = PackedLocalCompiler(tree, string_capacity=8, list_capacity=4)
    loop = next(node for node in tree.body if isinstance(node, ast.For))

    assert compiler._packed_loop_candidate(loop) is None


def test_public_compiler_executes_packed_compare_arithmetic_and_quad_fallback():
    code = compile_source(PACKED_LOOP_SOURCE, string_capacity=8, list_capacity=4)
    assert set(code) <= BF_COMMANDS

    result = run_bf(
        code,
        "3\n10\n1 3 4\n2 20 5\n1 -5 2\n",
        memory_size=120_000,
        step_limit=STEP_LIMIT,
    )
    assert result.output == "4\n"
