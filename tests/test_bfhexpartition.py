from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpartition_fused import run_partition_min_pass
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
    if value < 0:
        return _s64(-value)
    return value


def _reference(values, initial_ans=10_000_000):
    total = 0
    for value in values:
        total = _s64(total + value)
    left = 0
    ans = _s64(initial_ans)
    for value in values:
        left = _s64(left + value)
        doubled = _s64(left * 2)
        candidate = _abs_s64(_s64(total - doubled))
        if candidate < ans:
            ans = candidate
    return total, left, ans


def _program(initial_ans=10_000_000):
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    seq.read_lf_terminated_s64s_and_sum(bf)
    seq.propagate_total_back_to_first(bf)
    run_partition_min_pass(bf, seq, initial_ans=initial_ans)
    return bf.code(), seq


def test_partition_min_pass_matches_reference_and_carries_state():
    values = [5, -2, 10, 1234, -77]
    code, seq = _program()
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=20_000,
        step_limit=1_000_000_000,
    )

    sentinel = len(values)
    total, left, ans = _reference(values)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == total
    assert _decode_s64(result.memory, seq.field(sentinel, LEFT)) == left
    assert _decode_s64(result.memory, seq.field(sentinel, ANS)) == ans
    assert result.pointer == seq.base


def test_partition_min_pass_handles_msb_carry_regression():
    values = [2**62, 2**62]
    code, seq = _program(initial_ans=(1 << 63) - 1)
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=12_000,
        step_limit=1_000_000_000,
    )

    sentinel = len(values)
    total, left, ans = _reference(values, initial_ans=(1 << 63) - 1)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == total
    assert _decode_s64(result.memory, seq.field(sentinel, LEFT)) == left
    assert _decode_s64(result.memory, seq.field(sentinel, ANS)) == ans


def test_partition_pass_source_is_runtime_n_independent():
    code, seq = _program()
    source_size = len(code)
    values = list(range(24))
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=20_000,
        step_limit=1_000_000_000,
    )

    assert len(code) == source_size
    assert source_size < 400_000, f"runtime partition source is {source_size:,} bytes"
    assert _decode_s64(result.memory, seq.field(len(values), ANS)) == _reference(values)[2]
