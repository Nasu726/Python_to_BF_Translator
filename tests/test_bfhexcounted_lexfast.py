from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexcounted_lexfast import read_counted_two_line_s64s_and_sum
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


def _run(n, second_line, *, suffix=""):
    code, seq = _program()
    input_data = f"{n}\n" + second_line + suffix
    result = run_bf(code, input_data, memory_size=30_000, step_limit=1_000_000_000)
    return result, seq, input_data


def test_lexfast_counted_signed_values_and_total():
    values = [-(1 << 63), 0, (1 << 63) - 1, 7]
    result, seq, input_data = _run(len(values), " ".join(map(str, values)) + "\n")
    assert result.input_consumed == len(input_data)
    assert [_decode_u64(result.memory, seq.field(i, DATA)) for i in range(len(values))] == [
        value & MASK64 for value in values
    ]
    assert _decode_s64(result.memory, seq.field(len(values), TOTAL)) == 6


def test_lexfast_counted_whitespace_and_crlf():
    result, seq, input_data = _run(4, "\t1  \t-2\r  3\t4\r\n")
    assert result.input_consumed == len(input_data)
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(4)] == [1, -2, 3, 4]
    assert _decode_s64(result.memory, seq.field(4, TOTAL)) == 6


def test_lexfast_counted_crosses_radix16_count_boundary():
    for n in (15, 16, 17):
        values = [1] * n
        result, seq, input_data = _run(n, " ".join(map(str, values)) + "\n")
        assert result.input_consumed == len(input_data)
        assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(n)] == values
        assert _decode_s64(result.memory, seq.field(n, TOTAL)) == n


def test_lexfast_counted_drains_extra_tokens():
    result, seq, input_data = _run(3, "5 -2 10 999 1000\n")
    assert result.input_consumed == len(input_data)
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(3)] == [5, -2, 10]
    assert _decode_s64(result.memory, seq.field(3, TOTAL)) == 13


def test_lexfast_counted_zero_fills_short_line_without_crossing():
    result, seq, input_data = _run(5, "7 -3\n", suffix="999 1000\n")
    second_line_end = len("5\n7 -3\n")
    assert result.input_consumed == second_line_end
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(5)] == [7, -3, 0, 0, 0]
    assert _decode_s64(result.memory, seq.field(5, TOTAL)) == 4


def test_lexfast_counted_empty_ranges_drain_second_line():
    for n in (0, -2):
        result, seq, input_data = _run(n, "10 20 30\n")
        assert result.input_consumed == len(input_data)
        assert _decode_u64(result.memory, seq.field(0, DATA)) == 0
        assert _decode_u64(result.memory, seq.field(0, TOTAL)) == 0
