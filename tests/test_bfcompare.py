import random

from bfcompare import Binary64Compare
from bfcore import BFEmitter, Int64Ref


def run_bf(code: str, mem_size: int = 512, step_limit: int = 25_000_000):
    mem = [0] * mem_size
    ptr = pc = steps = 0
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
    return mem


def as_s64(value: int) -> int:
    value &= (1 << 64) - 1
    return value - (1 << 64) if value & (1 << 63) else value


def compile_compare(method: str, a: int, b: int):
    bf = BFEmitter()
    core = Binary64Compare(bf, scratch_base=193)
    A, B = Int64Ref(0), Int64Ref(64)
    result = 128
    core.set_u64(A, a)
    core.set_u64(B, b)
    getattr(core, method)(result, A, B)
    return bf.code(), result


def test_compare_edges():
    mask = (1 << 64) - 1
    vals = [0, 1, 2, 3, 255, 256, 2**31 - 1, 2**32,
            2**63 - 1, 2**63, 2**63 + 1, mask - 1, mask]
    methods = {
        'uge64': lambda a, b: a >= b,
        'ult64': lambda a, b: a < b,
        'ule64': lambda a, b: a <= b,
        'ugt64': lambda a, b: a > b,
        'sge64': lambda a, b: as_s64(a) >= as_s64(b),
        'slt64': lambda a, b: as_s64(a) < as_s64(b),
        'sle64': lambda a, b: as_s64(a) <= as_s64(b),
        'sgt64': lambda a, b: as_s64(a) > as_s64(b),
    }
    for a in vals:
        for b in vals:
            for method, expected in methods.items():
                code, result = compile_compare(method, a, b)
                mem = run_bf(code)
                assert mem[result] == int(expected(a, b)), (method, a, b)


def test_compare_randomized():
    rng = random.Random(726)
    for _ in range(40):
        a = rng.getrandbits(64)
        b = rng.getrandbits(64)
        for method, expected in (
            ('ult64', a < b),
            ('uge64', a >= b),
            ('slt64', as_s64(a) < as_s64(b)),
            ('sge64', as_s64(a) >= as_s64(b)),
        ):
            code, result = compile_compare(method, a, b)
            mem = run_bf(code)
            assert mem[result] == int(expected)
