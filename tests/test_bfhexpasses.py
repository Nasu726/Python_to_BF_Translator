from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexpasses import run_prefix_state_pass
from bfhexseq import DATA, LEFT, TOTAL, RuntimeHexIntSequence


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


def _program():
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    seq.read_lf_terminated_s64s_and_sum(bf)
    seq.propagate_total_back_to_first(bf)
    run_prefix_state_pass(bf, seq)
    return bf.code(), seq


def test_prefix_state_pass_carries_total_and_running_left_to_sentinel():
    values = [5, -2, 10, 1234, -77]
    code, seq = _program()
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=10_000,
        step_limit=500_000_000,
    )

    sentinel = len(values)
    expected = sum(values)
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == expected
    assert _decode_s64(result.memory, seq.field(sentinel, LEFT)) == expected
    assert result.pointer == seq.base

    # DATA is deliberately destructive on this final-use pass. This is the
    # liveness optimization that supplies local arithmetic scratch.
    for i in range(len(values)):
        assert _decode_u64(result.memory, seq.field(i, DATA)) == 0


def test_prefix_state_pass_preserves_full_width_carry():
    values = [2**62, 2**62]
    code, seq = _program()
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=8_000,
        step_limit=500_000_000,
    )

    sentinel = len(values)
    assert _decode_u64(result.memory, seq.field(sentinel, LEFT)) == 1 << 63
    assert _decode_u64(result.memory, seq.field(sentinel, TOTAL)) == 1 << 63


def test_prefix_pass_source_is_runtime_n_independent():
    code, seq = _program()
    source_size = len(code)
    values = list(range(32))
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=16_000,
        step_limit=500_000_000,
    )

    assert len(code) == source_size
    assert source_size < 250_000
    assert _decode_s64(result.memory, seq.field(len(values), LEFT)) == sum(values)
