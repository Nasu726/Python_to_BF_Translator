import random

from bfcore import BFEmitter, Binary64Core, Int64Ref


def run_bf(code: str, mem_size: int = 512, step_limit: int = 20_000_000):
    mem = [0] * mem_size
    ptr = 0
    pc = 0
    steps = 0
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
        elif c == '[' and mem[ptr] == 0:
            pc = jump[pc]
        elif c == ']' and mem[ptr] != 0:
            pc = jump[pc]
        pc += 1
    return mem, steps


def read_u64(mem, base):
    return sum((mem[base + i] & 1) << i for i in range(64))


def compile_case(op, a, b):
    bf = BFEmitter()
    core = Binary64Core(bf, scratch_base=192)
    A = Int64Ref(0)
    B = Int64Ref(64)
    D = Int64Ref(128)
    core.set_u64(A, a)
    core.set_u64(B, b)
    if op == 'add':
        core.add64(D, A, B)
    elif op == 'sub':
        core.sub64(D, A, B)
    else:
        raise AssertionError(op)
    return bf.code(), A, B, D


def test_add_edges():
    mask = (1 << 64) - 1
    vals = [0, 1, 2, 3, 255, 256, 2**31 - 1, 2**32, 2**63 - 1, 2**63, mask]
    for a in vals:
        for b in vals:
            code, A, B, D = compile_case('add', a, b)
            mem, _ = run_bf(code)
            assert read_u64(mem, D.base) == (a + b) & mask
            assert read_u64(mem, A.base) == a
            assert read_u64(mem, B.base) == b


def test_sub_edges():
    mask = (1 << 64) - 1
    vals = [0, 1, 2, 3, 255, 256, 2**31 - 1, 2**32, 2**63 - 1, 2**63, mask]
    for a in vals:
        for b in vals:
            code, A, B, D = compile_case('sub', a, b)
            mem, _ = run_bf(code)
            assert read_u64(mem, D.base) == (a - b) & mask
            assert read_u64(mem, A.base) == a
            assert read_u64(mem, B.base) == b


def test_add_sub_randomized():
    rng = random.Random(726)
    mask = (1 << 64) - 1
    for _ in range(100):
        a = rng.getrandbits(64)
        b = rng.getrandbits(64)
        for op, expected in [('add', (a+b)&mask), ('sub', (a-b)&mask)]:
            code, A, B, D = compile_case(op, a, b)
            mem, _ = run_bf(code)
            assert read_u64(mem, D.base) == expected


def test_eq64():
    samples = [0, 1, 2, 3, 255, 256, 2**32, 2**63, (1 << 64) - 1]
    for a in samples:
        for b in samples:
            bf = BFEmitter()
            core = Binary64Core(bf, scratch_base=193)
            A, B = Int64Ref(0), Int64Ref(64)
            result = 128
            core.set_u64(A, a)
            core.set_u64(B, b)
            core.eq64(result, A, B)
            mem, _ = run_bf(bf.code())
            assert mem[result] == int(a == b)
            assert read_u64(mem, A.base) == a
            assert read_u64(mem, B.base) == b
