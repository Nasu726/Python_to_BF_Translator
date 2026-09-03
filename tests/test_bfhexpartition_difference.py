import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpartition_difference import run_partition_min_pass
from bfhexseq import ANS, TOTAL, RuntimeHexIntSequence


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
    d = total
    ans = _s64(initial_ans)
    for value in values:
        d = _s64(d - _s64(2 * value))
        candidate = _abs_s64(d)
        if candidate < ans:
            ans = candidate
    return d, ans


def _assert_case(values, initial_ans=(1 << 63) - 1):
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
    final_d, ans = _reference(values, initial_ans)
    sentinel = len(values)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == final_d
    assert _decode_s64(result.memory, seq.field(sentinel, ANS)) == ans
    assert result.pointer == seq.base


def test_difference_partition_matches_reference():
    for values in (
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        [15, 16, 7, 8, 255, 256],
        [-1, 0, 1, -16, 17],
        list(range(12)),
        [0xFFFF, 1, 0xFFFF, 1],
    ):
        _assert_case(values)


def test_difference_partition_handles_wrap_and_int64_edges():
    for values in (
        [2**62, 2**62],
        [(1 << 63) - 1, 1],
        [-(1 << 63), 0, 1, -1],
        [0, 1, (1 << 63) - 1],
        [0x7FFF_FFFF_FFFF_FFF0, 15, -31],
        [-0x7000_0000_0000_0000, 0x1000_0000_0000_0000, 7],
        [0x0FFF_FFFF_FFFF_FFFF, 1, 0x7000_0000_0000_0000],
    ):
        _assert_case(values)


def test_difference_partition_handles_small_answer_and_zero_length():
    for initial_ans in (0, 1, 7, 10_000_000):
        _assert_case([1, 2, 3, 4], initial_ans)
        _assert_case([], initial_ans)


def test_difference_partition_rejects_non_nonnegative_initial_answer():
    for initial_ans in (-1, 1 << 63):
        bf = BFEmitter()
        seq = RuntimeHexIntSequence(base=128)
        with pytest.raises(ValueError):
            run_partition_min_pass(bf, seq, initial_ans=initial_ans)
