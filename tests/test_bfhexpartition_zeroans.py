import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpartition_zeroans import run_partition_min_pass
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


def test_zero_answer_fast_path_matches_reference_after_runtime_zero():
    for values in (
        [1] * 16,
        [1] * 32,
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        [-1, 0, 1, -16, 17],
    ):
        _assert_case(values, 10_000_000)


def test_zero_answer_fast_path_handles_initial_zero():
    for values in (
        [],
        [1],
        [1, 2, 3, 4],
        [-1, 0, 1],
        [1 << 40, 0],
    ):
        _assert_case(values, 0)


def test_zero_answer_fast_path_preserves_int64_min_exception():
    # With initial ANS=0, abs(INT64_MIN) wraps to INT64_MIN and is signed-smaller
    # than zero, so the fast path must still select it.
    _assert_case([-(1 << 63)], 0)
    _assert_case([2**62, 2**62], 0)
    _assert_case([-(1 << 63), 0, 1, -1], 10_000_000)


def test_zero_answer_fast_path_handles_wide_and_small_nonzero_answers():
    for initial_ans in (1, 15, 16, 255, 10_000_000):
        for values in ([1, 2, 3, 4], [1 << 24, 0], [0xFFFF, 1, 0xFFFF, 1]):
            _assert_case(values, initial_ans)


def test_zero_answer_fast_path_rejects_too_wide_initial_value():
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    with pytest.raises(ValueError):
        run_partition_min_pass(bf, seq, initial_ans=1 << 40)
