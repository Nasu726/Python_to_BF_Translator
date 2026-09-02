from bf_runtime import run_bf
from bfcore import BFEmitter
from bfopt import optimize_bf
from bfquad import Quad64Core, Quad64Ref
from bfquadbackend import QuadBinaryStringListIO


MASK64 = (1 << 64) - 1


def _u64(memory, ref: Quad64Ref) -> int:
    return sum((memory[ref.bit(i)] & 1) << i for i in range(64))


def _s64(memory, ref: Quad64Ref) -> int:
    value = _u64(memory, ref)
    return value - (1 << 64) if value & (1 << 63) else value


def test_quad_add_sub_and_unsigned_compare():
    bf = BFEmitter()
    core = Quad64Core(bf)
    a = Quad64Ref(0)
    b = Quad64Ref(120)
    add = Quad64Ref(240)
    sub = Quad64Ref(360)
    ge = 470
    tmp = Quad64Ref(480)

    core.set_u64(a, 0xFEDCBA9876543210)
    core.set_u64(b, 0x0123456789ABCDEF)
    core.add64(add, a, b)
    core.sub64(sub, a, b)
    core.uge64(ge, a, b, tmp)

    code = optimize_bf(bf.code())
    assert len(code) < 200_000
    result = run_bf(code, memory_size=900, step_limit=300_000_000)
    assert _u64(result.memory, add) == (
        0xFEDCBA9876543210 + 0x0123456789ABCDEF
    ) & MASK64
    assert _u64(result.memory, sub) == (
        0xFEDCBA9876543210 - 0x0123456789ABCDEF
    ) & MASK64
    assert result.memory[ge] == 1


def test_hybrid_quad_signed_compare_increment_and_negate():
    bf = BFEmitter()
    backend = QuadBinaryStringListIO(bf, scratch_base=700)
    backend.set_quad_workspace(400)
    a = Quad64Ref(0)
    b = Quad64Ref(120)
    lt = 300
    gt = 301

    backend.set_u64(a, -7)
    backend.set_u64(b, 5)
    backend.slt64(lt, a, b)
    backend.sgt64(gt, a, b)
    backend._inc64_inplace(a)
    backend._neg64_inplace(b)

    code = optimize_bf(bf.code())
    result = run_bf(code, memory_size=900, step_limit=300_000_000)
    assert result.memory[lt] == 1
    assert result.memory[gt] == 0
    assert _s64(result.memory, a) == -6
    assert _s64(result.memory, b) == -5


def test_hybrid_quad_alias_operands_are_well_defined():
    bf = BFEmitter()
    backend = QuadBinaryStringListIO(bf, scratch_base=700)
    backend.set_quad_workspace(400)
    x = Quad64Ref(0)
    doubled = Quad64Ref(120)
    zero = Quad64Ref(240)
    eq = 350
    lt = 351
    ge = 352

    backend.set_u64(x, -11)
    backend.add64(doubled, x, x)
    backend.sub64(zero, x, x)
    backend.eq64(eq, x, x)
    backend.slt64(lt, x, x)
    backend.sge64(ge, x, x)

    code = optimize_bf(bf.code())
    result = run_bf(code, memory_size=900, step_limit=300_000_000)
    assert _s64(result.memory, doubled) == -22
    assert _s64(result.memory, zero) == 0
    assert result.memory[eq] == 1
    assert result.memory[lt] == 0
    assert result.memory[ge] == 1
