from bfcore import BFEmitter, Int64Ref
from bfio import Binary64IO


MASK64 = (1 << 64) - 1


def to_s64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def read_u64(mem, base: int) -> int:
    return sum((mem[base + i] & 1) << i for i in range(64))


def run_bf(code: str, input_data: str = '', mem_size: int = 512,
           step_limit: int = 120_000_000):
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
    return mem, ''.join(output)


def compile_print(value: int, signed: bool):
    bf = BFEmitter()
    # source 0..63, print workspace 64..213, scratch 214..218
    core = Binary64IO(bf, scratch_base=214)
    value_ref = Int64Ref(0)
    core.set_u64(value_ref, value & MASK64)
    if signed:
        core.print_s64(value_ref, workspace_base=64)
    else:
        core.print_u64(value_ref, workspace_base=64)
    return bf.code()


def test_unsigned_decimal_print():
    values = [0, 1, 9, 10, 15, 99, 100, 12345, 2**32 - 1,
              2**63, MASK64]
    for value in values:
        _, output = run_bf(compile_print(value, signed=False))
        assert output == str(value)


def test_signed_decimal_print():
    values = [0, 1, -1, 9, -9, 10, -10, 12345, -12345,
              2**63 - 1, -(2**63)]
    for value in values:
        _, output = run_bf(compile_print(value, signed=True))
        assert output == str(value)


def test_read_signed_decimal():
    values = [0, 1, -1, 9, -9, 10, -10, 12345, -12345,
              2**31 + 17, -(2**31) - 17, 2**63 - 1, -(2**63)]
    for value in values:
        bf = BFEmitter()
        # destination 0..63, read workspace 64..195, scratch 196..200
        core = Binary64IO(bf, scratch_base=196)
        dst = Int64Ref(0)
        core.read_s64(dst, workspace_base=64)
        mem, _ = run_bf(bf.code(), input_data=f'{value}\n')
        assert to_s64(read_u64(mem, dst.base)) == value
