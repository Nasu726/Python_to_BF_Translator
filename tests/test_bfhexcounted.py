from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexcounted import read_counted_two_line_s64s_and_sum
from bfhexseq import DATA, TOTAL, RuntimeHexIntSequence


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


def _program():
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=128)
    read_counted_two_line_s64s_and_sum(bf, seq)
    return bf.code(), seq


def test_counted_input_uses_n_not_second_line_token_count():
    code, seq = _program()
    input_data = "3\n5 -2 10 999 1000\n"
    result = run_bf(
        code,
        input_data,
        memory_size=20_000,
        step_limit=1_000_000_000,
    )

    assert result.pointer == seq.base
    assert result.input_consumed == len(input_data)
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(3)] == [5, -2, 10]
    assert _decode_s64(result.memory, seq.field(3, TOTAL)) == 13


def test_counted_input_exact_line_and_signed_values():
    values = [-(1 << 63), 0, (1 << 63) - 1, 7]
    code, seq = _program()
    input_data = f"{len(values)}\n" + " ".join(map(str, values)) + "\n"
    result = run_bf(
        code,
        input_data,
        memory_size=20_000,
        step_limit=1_000_000_000,
    )

    assert result.pointer == seq.base
    assert result.input_consumed == len(input_data)
    assert [_decode_u64(result.memory, seq.field(i, DATA)) for i in range(len(values))] == [
        value & MASK64 for value in values
    ]
    assert _decode_s64(result.memory, seq.field(len(values), TOTAL)) == -1 + 7


def test_counted_input_early_line_end_zero_fills_without_crossing_next_line():
    code, seq = _program()
    input_data = "5\n7 -3\n999 1000\n"
    second_line_end = len("5\n7 -3\n")
    result = run_bf(
        code,
        input_data,
        memory_size=20_000,
        step_limit=1_000_000_000,
    )

    assert result.pointer == seq.base
    assert result.input_consumed == second_line_end
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(5)] == [
        7,
        -3,
        0,
        0,
        0,
    ]
    assert _decode_s64(result.memory, seq.field(5, TOTAL)) == 4


def test_counted_input_zero_n_drains_second_line_without_records():
    code, seq = _program()
    input_data = "0\n10 20 30\n"
    result = run_bf(
        code,
        input_data,
        memory_size=10_000,
        step_limit=1_000_000_000,
    )

    assert result.pointer == seq.base
    assert result.input_consumed == len(input_data)
    assert _decode_u64(result.memory, seq.field(0, DATA)) == 0
    assert _decode_u64(result.memory, seq.field(0, TOTAL)) == 0


def test_counted_input_negative_n_behaves_like_empty_range_and_drains():
    code, seq = _program()
    input_data = "-2\n10 20 30\n"
    result = run_bf(
        code,
        input_data,
        memory_size=10_000,
        step_limit=1_000_000_000,
    )

    assert result.pointer == seq.base
    assert result.input_consumed == len(input_data)
    assert _decode_u64(result.memory, seq.field(0, DATA)) == 0
    assert _decode_u64(result.memory, seq.field(0, TOTAL)) == 0
