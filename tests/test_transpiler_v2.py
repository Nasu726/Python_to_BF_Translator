from transpiler_v2 import compile_source


def run_bf(code: str, input_data: str = '', mem_size: int = 4096,
           step_limit: int = 250_000_000) -> str:
    mem = [0] * mem_size
    ptr = pc = steps = input_pos = 0
    output = []
    stack = []
    jump = {}
    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']':
            j = stack.pop()
            jump[i] = j
            jump[j] = i
    assert not stack

    while pc < len(code):
        steps += 1
        assert steps <= step_limit, 'step limit exceeded'
        c = code[pc]
        if c == '>':
            ptr += 1
        elif c == '<':
            ptr -= 1
        elif c == '+':
            mem[ptr] = (mem[ptr] + 1) & 0xFF
        elif c == '-':
            mem[ptr] = (mem[ptr] - 1) & 0xFF
        elif c == '.':
            output.append(chr(mem[ptr]))
        elif c == ',':
            mem[ptr] = ord(input_data[input_pos]) if input_pos < len(input_data) else 0
            input_pos += 1
        elif c == '[' and mem[ptr] == 0:
            pc = jump[pc]
        elif c == ']' and mem[ptr] != 0:
            pc = jump[pc]
        pc += 1
    return ''.join(output)


def execute(source: str, input_data: str = '') -> str:
    return run_bf(compile_source(source), input_data)


def test_arithmetic_and_print():
    source = '''
x = 2
y = 3
z = x * y + 4
print(z)
'''
    assert execute(source) == '10\n'


def test_signed_floor_division_and_if():
    source = '''
x = -7
y = 3
q = x // y
r = x % y
if q < 0:
    print(q, r)
else:
    print("wrong")
'''
    assert execute(source) == '-3 2\n'


def test_int_input():
    source = '''
x = int(input())
print(x * 2)
'''
    assert execute(source, '-7\n') == '-14\n'


def test_for_range_keeps_python_loop_value():
    source = '''
s = 0
for i in range(5):
    s += i
print(s, i)
'''
    assert execute(source) == '10 4\n'


def test_while_loop():
    source = '''
x = 3
s = 0
while x > 0:
    s += x
    x -= 1
print(s)
'''
    assert execute(source) == '6\n'


def test_short_circuit_avoids_rhs_input():
    source = '''
x = 0 and int(input())
y = 1 or int(input())
print(x, y)
'''
    # No input is supplied.  Correct short-circuiting must never execute ','.
    assert execute(source) == '0 1\n'
