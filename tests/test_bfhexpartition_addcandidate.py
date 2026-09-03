from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpartition_addcandidate import run_partition_min_pass
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


def _reference(values, initial_ans=10_000_000):
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


def _run(values, initial_ans=10_000_000):
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
    return result, seq


def _assert_case(values, initial_ans=10_000_000):
    result, seq = _run(values, initial_ans=initial_ans)
    total, left, ans = _reference(values, initial_ans)
    sentinel = len(values)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == total
    assert _decode_s64(result.memory, seq.field(sentinel, LEFT)) == left
    assert _decode_s64(result.memory, seq.field(sentinel, ANS)) == ans
    assert result.pointer == seq.base


def test_addcandidate_partition_matches_reference():
    cases = [
        [5, -2, 10, 1234, -77],
        [1, 2, 3, 4],
        [15, 16, 7, 8, 255, 256],
        [-1, 0, 1, -16, 17],
        [0, 1, (1 << 63) - 1],
    ]
    for values in cases:
        _assert_case(values)


def test_addcandidate_partition_carry_thresholds():
    # Exercise new-prefix nibble values around 7/8 and 15/0, including the
    # second-half 23/24 transition where addition and doubling carries coexist.
    cases = [
        [7, 1, 7, 1],
        [8, 7, 1, 15, 1],
        [15, 1, 15, 1],
        [0x0F, 0x01, 0xEF, 0x11],
        [0x7F, 0x01, 0x80, 0x80],
        [0xFF, 0x01, 0xFF, 0x01],
        [0x7FFF, 1, 0x8000, -1],
        [0xFFFF, 1, 0xFFFF, 1],
    ]
    for values in cases:
        _assert_case(values, initial_ans=(1 << 63) - 1)


def test_addcandidate_partition_int64_edges():
    cases = [
        [2**62, 2**62],
        [(1 << 63) - 1, 1],
        [-(1 << 63), 0],
        [0x7FFF_FFFF_FFFF_FFF0, 15, -31],
        [-0x7000_0000_0000_0000, 0x1000_0000_0000_0000, 7],
        [0x0FFF_FFFF_FFFF_FFFF, 1, 0x7000_0000_0000_0000],
    ]
    for values in cases:
        _assert_case(values, initial_ans=(1 << 63) - 1)
