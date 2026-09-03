import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpartition_adaptiveans import run_partition_min_pass
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
        candidate = _abs_s64(_s64(total - _s64(2 * left)))
        if candidate < ans:
            ans = candidate
    return total, left, ans


def _assert_case(values, initial_ans):
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


def test_adaptive_answer_matches_reference_after_early_width_shrink():
    for values in (
        [1] * 16,
        list(range(12)),
        [5, -2, 10, 1234, -77],
        [15, 16, 7, 8, 255, 256],
        [-1, 0, 1, -16, 17],
    ):
        _assert_case(values, 10_000_000)


def test_adaptive_answer_keeps_wide_tier_when_answer_does_not_shrink():
    # Every candidate stays above the initial answer, so the high answer lanes
    # remain live and the wide branch must continue to match the reference.
    for values in (
        [1 << 40, 0],
        [1 << 44, 3, -2],
        [0x1234_000000, 0x10, -0x20],
    ):
        _assert_case(values, 10_000_000)


def test_adaptive_answer_supports_already_small_initial_extent():
    for initial_ans in (0, 1, 15, 16, 255):
        for values in ([1, 2, 3, 4], [-1, 0, 1, -16, 17], []):
            _assert_case(values, initial_ans)


def test_adaptive_answer_handles_int64_min_and_sticky_negative_answer():
    for values in (
        [2**62, 2**62],
        [0, 1, (1 << 63) - 1],
        [-(1 << 63), 0, 1, -1],
        [(1 << 63) - 1, 1, 0, 0],
        [0x7FFF_FFFF_FFFF_FFF0, 15, -31],
    ):
        _assert_case(values, 10_000_000)


def test_adaptive_answer_rejects_too_wide_initial_value():
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    with pytest.raises(ValueError):
        run_partition_min_pass(bf, seq, initial_ans=1 << 40)
