from bfcore import BFEmitter, Int64Ref
from bfwordops import Binary64WordOps
from tests.test_bfcore import run_bf, read_u64


def run_wordop(name, a, b=0, amount=0):
    bf = BFEmitter()
    core = Binary64WordOps(bf, scratch_base=192)
    A, B, D = Int64Ref(0), Int64Ref(64), Int64Ref(128)
    core.set_u64(A, a)
    core.set_u64(B, b)
    if name == 'and':
        core.and64(D, A, B)
    elif name == 'or':
        core.or64(D, A, B)
    elif name == 'xor':
        core.xor64(D, A, B)
    elif name == 'not':
        core.not64(D, A)
    elif name == 'shl':
        core.shl_const(D, A, amount)
    elif name == 'shr':
        core.shr_const(D, A, amount)
    mem, _ = run_bf(bf.code())
    return read_u64(mem, D.base), read_u64(mem, A.base), read_u64(mem, B.base)


def test_bitwise_edges():
    mask = (1 << 64) - 1
    vals = [0, 1, 2, 3, 0x55AA, 2**63, mask]
    for a in vals:
        got, aa, _ = run_wordop('not', a)
        assert got == (~a) & mask
        assert aa == a
        for b in vals:
            assert run_wordop('and', a, b)[0] == (a & b)
            assert run_wordop('or', a, b)[0] == (a | b)
            assert run_wordop('xor', a, b)[0] == (a ^ b)


def test_constant_shifts():
    mask = (1 << 64) - 1
    vals = [0, 1, 3, 0x123456789ABCDEF0, 2**63, mask]
    for a in vals:
        for n in [0, 1, 2, 7, 31, 63, 64, 65]:
            assert run_wordop('shl', a, amount=n)[0] == ((a << n) & mask if n < 64 else 0)
            assert run_wordop('shr', a, amount=n)[0] == (a >> n if n < 64 else 0)
