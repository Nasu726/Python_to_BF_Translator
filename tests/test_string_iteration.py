from bf_runtime import run_bf
from compiler_strings import compile_source as compile_internal
from pybf import compile_source as compile_public


BF_COMMANDS = set("><+-.,[]")


def execute(source: str, input_data: str = "") -> str:
    code = compile_internal(source, string_capacity=8, list_capacity=4)
    assert set(code) <= BF_COMMANDS
    return run_bf(code, input_data, step_limit=300_000_000).output


def test_for_over_input_string_matches_user_example():
    source = '''
s = input()
for c in s:
    if c == "A":
        print("A", end="")
    else:
        print(".", end="")
'''
    assert execute(source, "ABCA\n") == "A..A"


def test_string_for_continue_break_and_else():
    source = '''
s = input()
for c in s:
    if c == "B":
        continue
    if c == "D":
        break
    print(c, end="")
else:
    print("!", end="")
'''
    assert execute(source, "ABCDX\n") == "AC"
    assert execute(source, "ABC\n") == "AC!"


def test_string_iterator_is_snapshotted_before_target_assignment():
    source = '''
s = "ABC"
for s in s:
    print(s, end="")
'''
    assert execute(source) == "ABC"


def test_public_compiler_accepts_normal_string_for_loop():
    source = '''
s = input()
for c in s:
    if c == "A":
        print("A", end="")
    else:
        print(".", end="")
'''
    code = compile_public(source)
    assert code
    assert set(code) <= BF_COMMANDS
