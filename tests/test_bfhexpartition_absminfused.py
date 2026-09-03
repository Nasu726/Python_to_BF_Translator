import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpartition_absminfused import run_partition_min_pass
from bfhexseq import ANS, LEFT, TOTAL, RuntimeHexIntSequence


MASK64 = (1 << 64) - 1


def _decode_u64(memory, base):
    value = 0
    for i in range(16):
        nibble = memory[base + i]
        assert 0 <= nibble <= 15
        value |= nibble << (4 * i)
    return value


def _decode_s64(memory, base):
    value = _decode_u64(memory, base)
    return value - (1 << 64) if value & (1 << 63) else value


def _s64(value):
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def _abs_s64(value):
    value = _s64(value)
    return _s64(-value) if value < 0 else value


def _reference(values, initial_ans):
    total = 0
    for value in values:
        total = _s64(total + value)
    left = 0
    ans = _s64(initial_ans)
    for value in values:
        left = _s64(left + value)
        candidate = _abs_s64(_s64(total - _s64(left * 2)))
        if candidate < ans:
            ans = candidate
    return total, left, ans


def _assert_case(values, initial_ans=10_000_000):
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    seq.read_lf_terminated_s64s_and_sum(bf)
    seq.propagate_total_back_to_first(bf)
    run_partition_min_pass(bf, seq, initial_ans=initial_ans)
    result = run_bf(
        bf.code(),
        " ".join(map(str, values)) + "\n",
        memory_size=30_000,
        step_limit=1_000_000_000,
    )
    total, left, ans = _reference(values, initial_ans)
    sentinel = len(values)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == total
    assert _decode_s64(result.memory, seq.field(sentinel, LEFT)) == left
    assert _decode_s64(result.memory, seq.field(sentinel, ANS)) == ans
    assert result.pointer == seq.base


def test_absminfused_matches_reference_for_mixed_sign_candidates():
    for initial_ans in (0, 1, 10_000_000, (1 << 63) - 1):
        for values in (
            [1, 2, 3, 4],
            [5, -2, 10, 1234, -77],
            [15, 16, 7, 8, 255, 256],
            [-1, 0, 1, -16, 17],
            list(range(12)),
        ):
            _assert_case(values, initial_ans)


def test_absminfused_handles_negation_carry_chains_and_nibble_boundaries():
    cases = [
        [7, 1, 7, 1],
        [8, 7, 1, 15, 1],
        [15, 1, 15, 1],
        [0x0F, 0x01, 0xEF, 0x11],
        [0x7F, 0x01, 0x80, 0x80],
        [0xFF, 0x01, 0xFF, 0x01],
        [0x7FFF, 1, 0x8000, -1],
        [0xFFFF, 1, 0xFFFF, 1],
        # Long zero suffixes in a negative raw candidate keep two's-complement
        # carry alive across several low nibbles.
        [0x10000, 0, -0x20000, 0x10000],
        [0x100000000, -0x200000000, 0x100000000],
    ]
    for values in cases:
        _assert_case(values, (1 << 63) - 1)


def test_absminfused_handles_int64_min_and_sticky_negative_answer():
    cases = [
        [0, 1, (1 << 63) - 1],
        [2**62, 2**62],
        [-(1 << 63), 0, 1, -1],
        [(1 << 63) - 1, 1, 0, 0],
        [0x7FFF_FFFF_FFFF_FFF0, 15, -31],
        [-0x7000_0000_0000_0000, 0x1000_0000_0000_0000, 7],
    ]
    for values in cases:
        _assert_case(values, (1 << 63) - 1)


def test_absminfused_rejects_negative_or_out_of_range_initial_ans():
    for initial_ans in (-1, 1 << 63):
        bf = BFEmitter()
        seq = RuntimeHexIntSequence(base=128)
        with pytest.raises(ValueError):
            run_partition_min_pass(bf, seq, initial_ans=initial_ans)
