from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexcounted_extent import read_counted_two_line_s64s_and_sum
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


def test_extent_reader_crosses_radix16_count_boundary():
    for n in (15, 16, 17):
        code, seq = _program()
        values = [1] * n
        input_data = f"{n}\n" + " ".join(map(str, values)) + "\n"
        result = run_bf(
            code,
            input_data,
            memory_size=20_000,
            step_limit=1_000_000_000,
        )
        assert result.pointer == seq.base
        assert result.input_consumed == len(input_data)
        assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(n)] == values
        assert _decode_s64(result.memory, seq.field(n, TOTAL)) == n


def test_extent_reader_honors_n_and_drains_extra_tokens():
    code, seq = _program()
    input_data = "3\n5 -2 10 999 1000\n"
    result = run_bf(
        code,
        input_data,
        memory_size=20_000,
        step_limit=1_000_000_000,
    )
    assert result.input_consumed == len(input_data)
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(3)] == [5, -2, 10]
    assert _decode_s64(result.memory, seq.field(3, TOTAL)) == 13


def test_extent_reader_zero_fills_short_line_without_crossing():
    code, seq = _program()
    input_data = "5\n7 -3\n999 1000\n"
    second_line_end = len("5\n7 -3\n")
    result = run_bf(
        code,
        input_data,
        memory_size=20_000,
        step_limit=1_000_000_000,
    )
    assert result.input_consumed == second_line_end
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(5)] == [7, -3, 0, 0, 0]
    assert _decode_s64(result.memory, seq.field(5, TOTAL)) == 4


def test_extent_reader_signed_int64_values_and_empty_ranges():
    values = [-(1 << 63), 0, (1 << 63) - 1, 7]
    code, seq = _program()
    input_data = f"{len(values)}\n" + " ".join(map(str, values)) + "\n"
    result = run_bf(code, input_data, memory_size=20_000, step_limit=1_000_000_000)
    assert [_decode_u64(result.memory, seq.field(i, DATA)) for i in range(len(values))] == [
        value & MASK64 for value in values
    ]
    assert _decode_s64(result.memory, seq.field(len(values), TOTAL)) == 6

    for n in (0, -2):
        code, seq = _program()
        input_data = f"{n}\n10 20 30\n"
        result = run_bf(code, input_data, memory_size=10_000, step_limit=1_000_000_000)
        assert result.input_consumed == len(input_data)
        assert _decode_u64(result.memory, seq.field(0, DATA)) == 0
        assert _decode_u64(result.memory, seq.field(0, TOTAL)) == 0
