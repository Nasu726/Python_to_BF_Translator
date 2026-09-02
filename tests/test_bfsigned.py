from bfsigned import Binary64Signed
from bfcore import BFEmitter, Int64Ref


MASK64 = (1 << 64) - 1
INT64_MIN = -(1 << 63)


def to_u64(value: int) -> int:
    return value & MASK64


def to_s64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def run_bf(code: str, mem_size: int = 640, step_limit: int = 100_000_000):
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


def run_sdivmod(a: int, b: int):
    bf = BFEmitter()
    # Inputs/outputs: 0..255. Workspace: 256..520. Scratch: 521..525.
    core = Binary64Signed(bf, scratch_base=521)
    A, B = Int64Ref(0), Int64Ref(64)
    Q, R = Int64Ref(128), Int64Ref(192)
    core.set_u64(A, to_u64(a))
    core.set_u64(B, to_u64(b))
    core.sdivmod64(Q, R, A, B, workspace_base=256)
    mem = run_bf(bf.code())
    return to_s64(sum((mem[Q.base + i] & 1) << i for i in range(64))), \
           to_s64(sum((mem[R.base + i] & 1) << i for i in range(64)))


def test_signed_divmod_floor_semantics():
    cases = [
        (7, 3), (-7, 3), (7, -3), (-7, -3),
        (6, 3), (-6, 3), (6, -3), (-6, -3),
        (0, 3), (1, -2), (-1, 2), (-2, 5), (-2, -5),
        (2**31 + 17, -12345), (-(2**31) - 17, 12345),
    ]
    for a, b in cases:
        assert run_sdivmod(a, b) == (a // b, a % b)


def test_signed_divmod_fixed_width_overflow_case():
    # Python itself returns +2**63 here, but the project intentionally fixes
    # integers at 64 bits, so the quotient wraps to INT64_MIN.
    assert run_sdivmod(INT64_MIN, -1) == (INT64_MIN, 0)
