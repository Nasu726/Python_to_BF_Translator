from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexseq import DATA, TOTAL, RuntimeHexIntSequence
from bfhexseq_direct import read_lf_terminated_s64s_and_sum_direct


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
    read_lf_terminated_s64s_and_sum_direct(bf, seq)
    return bf.code(), seq


def test_direct_line_reader_signed_values_and_total():
    cases = [
        [0],
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        [-(1 << 63), 0, (1 << 63) - 1, 7],
        [(1 << 63) - 1, 1, -1],
    ]
    for values in cases:
        code, seq = _program()
        input_data = " ".join(map(str, values)) + "\n"
        result = run_bf(
            code,
            input_data,
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert result.pointer == seq.base
        assert result.input_consumed == len(input_data)
        assert [_decode_u64(result.memory, seq.field(i, DATA)) for i in range(len(values))] == [
            value & MASK64 for value in values
        ]
        expected_total = sum(values) & MASK64
        assert _decode_u64(result.memory, seq.field(len(values), TOTAL)) == expected_total


def test_direct_line_reader_whitespace_and_line_boundary():
    code, seq = _program()
    input_data = "  7\t-3  10\r\n999 1000\n"
    first_line_end = len("  7\t-3  10\r\n")
    result = run_bf(
        code,
        input_data,
        memory_size=20_000,
        step_limit=1_000_000_000,
    )
    assert result.input_consumed == first_line_end
    assert [_decode_s64(result.memory, seq.field(i, DATA)) for i in range(3)] == [7, -3, 10]
    assert _decode_s64(result.memory, seq.field(3, TOTAL)) == 14


def test_direct_line_reader_empty_line_materializes_no_records():
    code, seq = _program()
    input_data = "\n123\n"
    result = run_bf(
        code,
        input_data,
        memory_size=10_000,
        step_limit=1_000_000_000,
    )
    assert result.input_consumed == 1
    assert result.pointer == seq.base
    assert _decode_u64(result.memory, seq.field(0, DATA)) == 0
    assert _decode_u64(result.memory, seq.field(0, TOTAL)) == 0
