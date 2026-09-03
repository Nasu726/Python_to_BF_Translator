from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexcounted_prefix import read_counted_two_line_s64s_and_sum
from bfhexpartition_prefix import run_partition_min_pass
from bfhexseq import ANS, DATA, LEFT, TOTAL, RuntimeHexIntSequence


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
    prefixes = []
    for value in values:
        left = _s64(left + value)
        prefixes.append(left)
        candidate = _abs_s64(_s64(total - _s64(2 * left)))
        if candidate < ans:
            ans = candidate
    return total, prefixes, ans


def test_prefix_reader_retains_inclusive_prefixes_and_total():
    values = [5, -2, 10, 1234, -77]
    total, prefixes, _ = _reference(values)
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    read_counted_two_line_s64s_and_sum(bf, seq)
    result = run_bf(
        bf.code(),
        f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
        memory_size=30_000,
        step_limit=1_000_000_000,
    )
    for i, prefix in enumerate(prefixes):
        assert _decode_s64(result.memory, seq.field(i, DATA)) == prefix
        # Running TOTAL was destructively split and only the sentinel keeps it.
        assert _decode_s64(result.memory, seq.field(i, TOTAL)) == 0
    assert _decode_s64(result.memory, seq.field(len(values), TOTAL)) == total
    assert result.pointer == seq.base


def test_prefix_reader_preserves_counted_short_line_and_extra_token_semantics():
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    read_counted_two_line_s64s_and_sum(bf, seq)
    short = run_bf(
        bf.code(),
        "4\n5 7\n999\n",
        memory_size=30_000,
        step_limit=1_000_000_000,
    )
    assert [_decode_s64(short.memory, seq.field(i, DATA)) for i in range(4)] == [5, 12, 12, 12]
    assert _decode_s64(short.memory, seq.field(4, TOTAL)) == 12
    assert short.input_consumed == len("4\n5 7\n")

    extra = run_bf(
        bf.code(),
        "2\n5 7 100 200\n",
        memory_size=30_000,
        step_limit=1_000_000_000,
    )
    assert [_decode_s64(extra.memory, seq.field(i, DATA)) for i in range(2)] == [5, 12]
    assert _decode_s64(extra.memory, seq.field(2, TOTAL)) == 12
    assert extra.input_consumed == len("2\n5 7 100 200\n")


def _assert_partition_case(values, initial_ans=10_000_000):
    total, prefixes, ans = _reference(values, initial_ans)
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    read_counted_two_line_s64s_and_sum(bf, seq)
    seq.propagate_total_back_to_first(bf)
    run_partition_min_pass(bf, seq, initial_ans=initial_ans)
    result = run_bf(
        bf.code(),
        f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
        memory_size=30_000,
        step_limit=1_000_000_000,
    )
    sentinel = len(values)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == total
    assert _decode_s64(result.memory, seq.field(sentinel, ANS)) == ans
    assert result.pointer == seq.base


def test_prefix_partition_matches_reference_for_signed_and_wrap_cases():
    for values in (
        [1] * 32,
        [1, 2, 3, 4],
        list(range(12)),
        [5, -2, 10, 1234, -77],
        [15, 16, 7, 8, 255, 256],
        [-1, 0, 1, -16, 17],
        [0xFFFF, 1, 0xFFFF, 1],
        [2**62, 2**62],
        [-(1 << 63), 0, 1, -1],
        [(1 << 63) - 1, 1, 0],
    ):
        _assert_partition_case(values)


def test_prefix_partition_handles_zero_answer_and_empty_sequence():
    for initial_ans in (0, 1, 255, 10_000_000):
        _assert_partition_case([], initial_ans)
        _assert_partition_case([1, 2, 3, 4], initial_ans)
    _assert_partition_case([-(1 << 63)], 0)
