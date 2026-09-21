from bf_runtime import run_bf
from bfcore import BFEmitter
from bfpacked64 import PackedI64Ref
from bfpackedops import PackedI64Ops


MASK64 = (1 << 64) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _u64(memory, ref):
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(8))


def _s64(value):
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def _run_binary(op_name, a_value, b_value):
    bf = BFEmitter()
    a = PackedI64Ref(32)
    b = PackedI64Ref(48)
    result = 64
    ops = PackedI64Ops(bf, 80)
    ops.set_u64(a, a_value)
    ops.set_u64(b, b_value)
    getattr(ops, op_name)(result, a, b)
    code = bf.code()
    run = run_bf(code, memory_size=256, step_limit=250_000_000)
    return run, a, b, result, code


def test_packed_add_wraps_and_preserves_rhs():
    cases = [
        (0, 0),
        (1, 2),
        (255, 1),
        (256, 65535),
        (INT64_MAX, 1),
        (MASK64, 1),
        (0x123456789ABCDEF0, 0x0FEDCBA987654321),
    ]
    for a_value, b_value in cases:
        bf = BFEmitter()
        a = PackedI64Ref(32)
        b = PackedI64Ref(48)
        ops = PackedI64Ops(bf, 80)
        ops.set_u64(a, a_value)
        ops.set_u64(b, b_value)
        ops.add_inplace(a, b)
        result = run_bf(bf.code(), memory_size=256, step_limit=250_000_000)
        assert _u64(result.memory, a) == (a_value + b_value) & MASK64
        assert _u64(result.memory, b) == b_value & MASK64


def test_packed_sub_wraps_and_preserves_rhs():
    cases = [
        (0, 0),
        (3, 2),
        (0, 1),
        (256, 1),
        (INT64_MIN, 1),
        (0x123456789ABCDEF0, 0x0FEDCBA987654321),
    ]
    for a_value, b_value in cases:
        bf = BFEmitter()
        a = PackedI64Ref(32)
        b = PackedI64Ref(48)
        ops = PackedI64Ops(bf, 80)
        ops.set_u64(a, a_value)
        ops.set_u64(b, b_value)
        ops.sub_inplace(a, b)
        result = run_bf(bf.code(), memory_size=256, step_limit=250_000_000)
        assert _u64(result.memory, a) == (a_value - b_value) & MASK64
        assert _u64(result.memory, b) == b_value & MASK64


def test_packed_equal_matches_u64_identity_and_preserves_operands():
    pairs = [
        (0, 0),
        (1, 1),
        (1, 2),
        (MASK64, -1),
        (INT64_MIN, INT64_MIN),
        (INT64_MIN, INT64_MAX),
        (0x0102030405060708, 0x0102030405060709),
    ]
    for a_value, b_value in pairs:
        run, a, b, flag, _code = _run_binary("equal", a_value, b_value)
        assert run.memory[flag] == int((a_value & MASK64) == (b_value & MASK64))
        assert _u64(run.memory, a) == a_value & MASK64
        assert _u64(run.memory, b) == b_value & MASK64


def test_packed_signed_less_than_matches_int64_order_and_preserves_operands():
    pairs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (-1, 0),
        (0, -1),
        (INT64_MIN, INT64_MAX),
        (INT64_MAX, INT64_MIN),
        (INT64_MIN, -1),
        (-2, -1),
        (-1, -2),
        (0x12345678, 0x12345679),
    ]
    for a_value, b_value in pairs:
        run, a, b, flag, _code = _run_binary("signed_lt", a_value, b_value)
        assert run.memory[flag] == int(_s64(a_value) < _s64(b_value))
        assert _u64(run.memory, a) == a_value & MASK64
        assert _u64(run.memory, b) == b_value & MASK64


def test_packed_ops_remain_source_compact():
    bf = BFEmitter()
    a = PackedI64Ref(32)
    b = PackedI64Ref(48)
    ops = PackedI64Ops(bf, 80)
    ops.add_inplace(a, b)
    ops.sub_inplace(a, b)
    ops.equal(64, a, b)
    ops.signed_lt(65, a, b)
    code = bf.code()
    assert set(code) <= set("><+-.,[]")
    assert len(code) < 100_000


def test_packed_add_runtime_operands_full_state_and_dense_byte_budget():
    import random

    bf = BFEmitter()
    a, b = PackedI64Ref(32), PackedI64Ref(48)
    guards = (31, 40, 47, 56, 79, 96)
    for cell in guards:
        bf.set_const(cell, 173)
    for ref in (a, b):
        for i in range(8):
            bf.move(ref.byte(i))
            bf.emit(",")
    PackedI64Ops(bf, 80).add_inplace(a, b)
    code = bf.code()
    rng = random.Random(20260921)
    cases = [(0, MASK64), (MASK64, MASK64), (MASK64, 0),
             (INT64_MAX, INT64_MAX), (INT64_MIN, INT64_MIN)]
    cases += [((1 << (8 * i)) - 1, 1) for i in range(1, 9)]
    cases += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(32)]
    for left, right in cases:
        data = ((left & MASK64).to_bytes(8, "little")
                + (right & MASK64).to_bytes(8, "little")).decode("latin1") + "X"
        result = run_bf(code, data, memory_size=128, step_limit=3_000_000)
        expected = [0] * 128
        for cell in guards:
            expected[cell] = 173
        expected[32:40] = ((left + right) & MASK64).to_bytes(8, "little")
        expected[48:56] = (right & MASK64).to_bytes(8, "little")
        assert result.memory == expected
        assert result.input_consumed == 16
        assert result.output == ""


def test_packed_add_exact_alias_doubles_and_clears_scratch():
    bf = BFEmitter()
    a = PackedI64Ref(32)
    for i in range(8):
        bf.move(a.byte(i))
        bf.emit(",")
    PackedI64Ops(bf, 80).add_inplace(a, a)
    code = bf.code()
    for value in (0, 1, 127, 128, 255, 256, INT64_MIN, INT64_MAX, MASK64):
        data = (value & MASK64).to_bytes(8, "little").decode("latin1")
        result = run_bf(code, data, memory_size=128, step_limit=3_000_000)
        expected = [0] * 128
        expected[32:40] = ((value * 2) & MASK64).to_bytes(8, "little")
        assert result.memory == expected
        assert result.input_consumed == 8
