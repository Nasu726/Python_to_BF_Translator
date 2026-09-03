from bf_runtime import run_bf
from bfbase4 import Base4I64Core, Base4I64Ref, MASK64, WORD_CELLS
from bfcore import BFEmitter


def _decode(memory, ref):
    value = 0
    for digit in range(32):
        lane = memory[ref.value(digit)]
        assert 0 <= lane <= 3
        value |= lane << (2 * digit)
    return value


def test_base4_word_is_smaller_than_quad_layout_target():
    assert WORD_CELLS == 66
    assert WORD_CELLS < 99


def test_base4_copy_preserves_source_and_exact_bits():
    bf = BFEmitter()
    core = Base4I64Core(bf)
    src = Base4I64Ref(20)
    dst = Base4I64Ref(140)
    value = 0xFEDCBA9876543210

    core.set_u64(src, value)
    core.copy64(dst, src)
    result = run_bf(bf.code(), memory_size=400, step_limit=20_000_000)

    assert _decode(result.memory, src) == value
    assert _decode(result.memory, dst) == value


def test_base4_add64_matches_modulo_u64_boundaries():
    cases = [
        (0, 0),
        (1, 2),
        (3, 1),
        (0xFFFFFFFFFFFFFFFF, 1),
        (0x7FFFFFFFFFFFFFFF, 1),
        (0x0123456789ABCDEF, 0x1111111111111111),
    ]

    for a_value, b_value in cases:
        bf = BFEmitter()
        core = Base4I64Core(bf)
        a = Base4I64Ref(20)
        b = Base4I64Ref(120)
        dst = Base4I64Ref(220)

        core.set_u64(a, a_value)
        core.set_u64(b, b_value)
        core.add64(dst, a, b)
        result = run_bf(bf.code(), memory_size=500, step_limit=100_000_000)

        assert _decode(result.memory, a) == (a_value & MASK64)
        assert _decode(result.memory, b) == (b_value & MASK64)
        assert _decode(result.memory, dst) == ((a_value + b_value) & MASK64)


def test_base4_mul10_is_two_fixed_lane_passes_and_matches_u64():
    cases = [
        0,
        1,
        3,
        4,
        123456789,
        0x7FFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ]

    for value in cases:
        bf = BFEmitter()
        core = Base4I64Core(bf)
        dst = Base4I64Ref(20)
        scratch = Base4I64Ref(120)

        core.set_u64(dst, value)
        core.mul10_inplace(dst, scratch)
        result = run_bf(bf.code(), memory_size=400, step_limit=150_000_000)

        assert _decode(result.memory, dst) == ((value * 10) & MASK64)

    bf = BFEmitter()
    core = Base4I64Core(bf)
    core.mul10_inplace(Base4I64Ref(20), Base4I64Ref(120))
    assert len(bf.code()) < 30_000


def test_base4_sub64_matches_modulo_u64_boundaries():
    cases = [
        (0, 0),
        (7, 3),
        (3, 7),
        (0, 1),
        (0x8000000000000000, 1),
        (0x0123456789ABCDEF, 0x1111111111111111),
    ]

    for a_value, b_value in cases:
        bf = BFEmitter()
        core = Base4I64Core(bf)
        a = Base4I64Ref(20)
        b = Base4I64Ref(120)
        dst = Base4I64Ref(220)

        core.set_u64(a, a_value)
        core.set_u64(b, b_value)
        core.sub64(dst, a, b)
        result = run_bf(bf.code(), memory_size=500, step_limit=120_000_000)

        assert _decode(result.memory, a) == (a_value & MASK64)
        assert _decode(result.memory, b) == (b_value & MASK64)
        assert _decode(result.memory, dst) == ((a_value - b_value) & MASK64)


def test_base4_unsigned_ge_uses_subtraction_final_carry():
    cases = [
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 0),
        (0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 1),
        (0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0),
    ]

    for a_value, b_value, expected in cases:
        bf = BFEmitter()
        core = Base4I64Core(bf)
        a = Base4I64Ref(20)
        b = Base4I64Ref(120)
        tmp = Base4I64Ref(220)
        flag = 310

        core.set_u64(a, a_value)
        core.set_u64(b, b_value)
        core.uge64(flag, a, b, tmp)
        result = run_bf(bf.code(), memory_size=500, step_limit=120_000_000)

        assert result.memory[flag] == expected
        assert _decode(result.memory, a) == (a_value & MASK64)
        assert _decode(result.memory, b) == (b_value & MASK64)


def test_base4_arithmetic_emits_one_lane_body_not_32_static_adders():
    bf = BFEmitter()
    core = Base4I64Core(bf)
    a = Base4I64Ref(20)
    b = Base4I64Ref(120)
    dst = Base4I64Ref(220)
    core.add64(dst, a, b)
    add_size = len(bf.code())

    bf2 = BFEmitter()
    core2 = Base4I64Core(bf2)
    core2.sub64(dst, a, b)
    sub_size = len(bf2.code())

    # Source-structure guards, not micro-optimization targets.  Regressing to
    # 32 separately emitted full adders/subtractors would be far larger.
    assert add_size < 20_000
    assert sub_size < 20_000
