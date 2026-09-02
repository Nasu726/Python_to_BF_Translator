import random

from bfarith import Binary64Arithmetic
from bfcore import BFEmitter, Int64Ref


MASK64 = (1 << 64) - 1


def run_bf(code: str, mem_size: int = 640, step_limit: int = 80_000_000):
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
    return mem, steps


def read_u64(mem, base: int) -> int:
    return sum((mem[base + i] & 1) << i for i in range(64))


def run_mul(a: int, b: int):
    bf = BFEmitter()
    # A/B/D occupy 0..191, mul workspace 192..384, scratch 385..389.
    core = Binary64Arithmetic(bf, scratch_base=385)
    A, B, D = Int64Ref(0), Int64Ref(64), Int64Ref(128)
    core.set_u64(A, a)
    core.set_u64(B, b)
    core.mul64(D, A, B, workspace_base=192)
    mem, steps = run_bf(bf.code())
    return read_u64(mem, D.base), read_u64(mem, A.base), read_u64(mem, B.base), steps


def run_divmod(a: int, b: int):
    bf = BFEmitter()
    # N/D/Q/R occupy 0..255, workspace 256..385, scratch 386..390.
    core = Binary64Arithmetic(bf, scratch_base=386)
    N, D = Int64Ref(0), Int64Ref(64)
    Q, R = Int64Ref(128), Int64Ref(192)
    core.set_u64(N, a)
    core.set_u64(D, b)
    core.udivmod64(Q, R, N, D, workspace_base=256)
    mem, steps = run_bf(bf.code())
    return (
        read_u64(mem, Q.base),
        read_u64(mem, R.base),
        read_u64(mem, N.base),
        read_u64(mem, D.base),
        steps,
    )


def test_mul_edges():
    cases = [
        (0, 0), (0, 1), (1, 1), (2, 3), (3, 5), (123, 456),
        (2**32, 2**32), (2**63, 2), (MASK64, 2), (MASK64, MASK64),
    ]
    for a, b in cases:
        result, got_a, got_b, _ = run_mul(a, b)
        assert result == (a * b) & MASK64
        assert got_a == a & MASK64
        assert got_b == b & MASK64


def test_mul_randomized():
    rng = random.Random(726)
    for _ in range(8):
        a = rng.getrandbits(64)
        b = rng.getrandbits(64)
        result, _, _, _ = run_mul(a, b)
        assert result == (a * b) & MASK64


def test_udivmod_edges():
    cases = [
        (0, 1), (1, 1), (2, 1), (3, 2), (123, 7),
        (2**63, 3), (MASK64, 1), (MASK64, 2), (MASK64, 2**32 + 1),
    ]
    for a, b in cases:
        q, r, got_a, got_b, _ = run_divmod(a, b)
        assert (q, r) == divmod(a, b)
        assert got_a == a
        assert got_b == b
        assert a == q * b + r
        assert 0 <= r < b


def test_udivmod_randomized():
    rng = random.Random(727)
    for _ in range(8):
        a = rng.getrandbits(64)
        b = rng.getrandbits(64) or 1
        q, r, _, _, _ = run_divmod(a, b)
        assert (q, r) == divmod(a, b)
