from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexseq import DATA, TOTAL, RECORD_STRIDE, RuntimeHexIntSequence


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


def _program(*, propagate=False):
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    seq.read_lf_terminated_s64s_and_sum(bf)
    if propagate:
        seq.propagate_total_back_to_first(bf)
    return bf.code(), seq


def test_hex_sequence_input_keeps_data_and_places_sum_on_sentinel():
    values = [5, -2, 10, 1234]
    code, seq = _program()
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=8_000,
        step_limit=500_000_000,
    )

    for i, expected in enumerate(values):
        assert result.memory[seq.marker(i)] == 1
        assert _decode_s64(result.memory, seq.field(i, DATA)) == expected

    sentinel = len(values)
    assert result.memory[seq.marker(sentinel)] == 0
    assert result.memory[seq.back(sentinel)] == 1
    assert _decode_s64(result.memory, seq.field(sentinel, TOTAL)) == sum(values)
    assert result.pointer == seq.base


def test_hex_sequence_total_can_be_carried_back_to_record_zero():
    values = [7, 11, -3, 20]
    code, seq = _program(propagate=True)
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=8_000,
        step_limit=500_000_000,
    )

    assert _decode_s64(result.memory, seq.field(0, TOTAL)) == sum(values)
    assert result.pointer == seq.base


def test_hex_sequence_empty_line_keeps_zero_total_at_record_zero():
    code, seq = _program(propagate=True)
    result = run_bf(code, "\n", memory_size=4_000, step_limit=100_000_000)

    assert result.memory[seq.marker(0)] == 0
    assert _decode_u64(result.memory, seq.field(0, TOTAL)) == 0
    assert result.pointer == seq.base


def test_hex_sequence_source_does_not_depend_on_runtime_n():
    code, seq = _program()
    source_size = len(code)
    values = list(range(24))
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=12_000,
        step_limit=500_000_000,
    )

    assert len(code) == source_size
    assert source_size < 180_000
    assert _decode_s64(result.memory, seq.field(len(values), TOTAL)) == sum(values)
    assert RECORD_STRIDE == 66
