from bf_runtime import run_bf
from bfcore import BFEmitter
from bfpacked import PackedU32Core, PackedU32Ref


def _u32(memory: list[int], ref: PackedU32Ref) -> int:
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(4))


def test_packed_u32_increment_and_decrement_carry_borrow():
    bf = BFEmitter()
    value = PackedU32Ref(0)
    rt = PackedU32Core(bf, scratch_base=16)

    rt.set_u32(value, 0x0000FFFF)
    rt.increment(value)
    rt.decrement(value)

    result = run_bf(bf.code(), memory_size=64, step_limit=10_000_000)
    assert _u32(result.memory, value) == 0x0000FFFF


def test_packed_u32_wraps_modulo_32_bits():
    bf = BFEmitter()
    value = PackedU32Ref(0)
    rt = PackedU32Core(bf, scratch_base=16)

    rt.set_u32(value, 0xFFFFFFFF)
    rt.increment(value)
    rt.decrement(value)

    result = run_bf(bf.code(), memory_size=64, step_limit=10_000_000)
    assert _u32(result.memory, value) == 0xFFFFFFFF


def test_packed_u32_copy_equal_and_zero_test_preserve_inputs():
    bf = BFEmitter()
    a = PackedU32Ref(0)
    b = PackedU32Ref(4)
    c = PackedU32Ref(8)
    eq_ab = 12
    eq_ac = 13
    zero_a = 14
    zero_c = 15
    rt = PackedU32Core(bf, scratch_base=20)

    rt.set_u32(a, 0)
    rt.set_u32(b, 0)
    rt.set_u32(c, 123456789)
    rt.equal(eq_ab, a, b)
    rt.equal(eq_ac, a, c)
    rt.is_zero(zero_a, a)
    rt.is_zero(zero_c, c)

    result = run_bf(bf.code(), memory_size=64, step_limit=20_000_000)
    assert result.memory[eq_ab] == 1
    assert result.memory[eq_ac] == 0
    assert result.memory[zero_a] == 1
    assert result.memory[zero_c] == 0
    assert _u32(result.memory, a) == 0
    assert _u32(result.memory, c) == 123456789
