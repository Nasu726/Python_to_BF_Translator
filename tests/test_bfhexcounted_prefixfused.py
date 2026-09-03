from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexcounted_prefixfused import read_counted_two_line_s64s_and_sum
from bfhexpartition_prefix import run_partition_min_pass
from bfhexseq import ANS, DATA, TOTAL, RuntimeHexIntSequence


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
    prefixes = []
    ans = _s64(initial_ans)
    for value in values:
        total = _s64(total + value)
        prefixes.append(total)
    for prefix in prefixes:
        candidate = _abs_s64(_s64(total - _s64(2 * prefix)))
        if candidate < ans:
            ans = candidate
    return total, prefixes, ans


def _reader_program():
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    read_counted_two_line_s64s_and_sum(bf, seq)
    return bf, seq


def test_fused_prefix_reader_retains_signed_prefixes_and_total():
    for values in (
        [5, -2, 10, 1234, -77],
        [15, 16, 255, 256],
        [-1, 0, 1, -16, 17],
        [2**62, 2**62],
        [-(1 << 63), 1, -1],
    ):
        total, prefixes, _ = _reference(values)
        bf, seq = _reader_program()
        result = run_bf(
            bf.code(),
            f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(len(values))] == prefixes
        assert _decode_s64(result.memory, seq.field(len(values), TOTAL)) == total
        assert result.pointer == seq.base


def test_fused_prefix_reader_keeps_short_line_and_extra_token_contract():
    bf, seq = _reader_program()
    short_input = "4\n5 7\n999\n"
    short = run_bf(bf.code(), short_input, memory_size=30_000, step_limit=1_000_000_000)
    assert [_decode_s64(short.memory, seq.field(i, DATA)) for i in range(4)] == [5, 12, 12, 12]
    assert _decode_s64(short.memory, seq.field(4, TOTAL)) == 12
    assert short.input_consumed == len("4\n5 7\n")

    extra_input = "2\n5 7 100 200\n"
    extra = run_bf(bf.code(), extra_input, memory_size=30_000, step_limit=1_000_000_000)
    assert [_decode_s64(extra.memory, seq.field(i, DATA)) for i in range(2)] == [5, 12]
    assert _decode_s64(extra.memory, seq.field(2, TOTAL)) == 12
    assert extra.input_consumed == len(extra_input)


def test_fused_prefix_reader_zero_and_negative_count():
    for data in ("0\n\n", "-3\n1 2 3\n"):
        bf, seq = _reader_program()
        result = run_bf(bf.code(), data, memory_size=30_000, step_limit=1_000_000_000)
        assert _decode_s64(result.memory, seq.field(0, TOTAL)) == 0
        assert result.pointer == seq.base


def test_fused_prefix_pipeline_matches_partition_reference():
    cases = [
        [1] * 32,
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        [0xFFFF, 1, 0xFFFF, 1],
        [2**62, 2**62],
        [-(1 << 63), 0, 1, -1],
    ]
    for values in cases:
        total, _, ans = _reference(values)
        bf = BFEmitter()
        seq = RuntimeHexIntSequence(base=128)
        read_counted_two_line_s64s_and_sum(bf, seq)
        seq.propagate_total_back_to_first(bf)
        run_partition_min_pass(bf, seq, initial_ans=10_000_000)
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
