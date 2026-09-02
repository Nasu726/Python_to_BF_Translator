from bf_runtime import run_bf
from transpiler_full import compile_source


def execute(source: str) -> str:
    code = compile_source(source, string_capacity=16, list_capacity=5)
    return run_bf(code, memory_size=16000, step_limit=900_000_000).output


def test_range_break_continue():
    source = '''
s = 0
for i in range(10):
    if i == 2:
        continue
    if i == 5:
        break
    s += i
print(s, i)
'''
    assert execute(source) == '8 5\n'


def test_while_continue_and_break():
    source = '''
i = 0
s = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue
    if i > 5:
        break
    s += i
print(s, i)
'''
    assert execute(source) == '9 7\n'


def test_loop_else_runs_only_without_break():
    source = '''
x = 0
for i in range(3):
    x += i
else:
    x += 10
print(x)

for j in range(5):
    if j == 2:
        break
else:
    x += 100
print(x)
'''
    assert execute(source) == '13\n13\n'


def test_list_iteration_break_continue():
    source = '''
A = [1, 2, 3, 4, 5]
s = 0
for x in A:
    if x == 2:
        continue
    if x == 5:
        break
    s += x
print(s)
'''
    assert execute(source) == '8\n'


def test_nested_break_only_exits_inner_loop():
    source = '''
s = 0
for i in range(3):
    for j in range(5):
        if j == 2:
            break
        s += 1
print(s)
'''
    assert execute(source) == '6\n'
