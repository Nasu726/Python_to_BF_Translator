import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from bfpacked import PackedU32Ref
from bfpackedseq import RECORD_STRIDE, RuntimePackedIntSequence


def _program(base=64):
    bf = BFEmitter()
    seq = RuntimePackedIntSequence(base=base)
    seq.read_lf_terminated_s64s(bf)
    return bf.code(), seq


def _decode_s64(memory, ref):
    value = 0
    for i in range(8):
        value |= memory[ref.byte(i)] << (8 * i)
    if value & (1 << 63):
        value -= 1 << 64
    return value


def test_runtime_packed_sequence_reads_empty_line_without_materializing_item():
    code, seq = _program()
    result = run_bf(code, "\n", memory_size=2_000, step_limit=50_000_000)

    assert result.memory[seq.marker(0)] == 0
    assert result.memory[seq.back(0)] == 0
    assert result.pointer == seq.base


def test_runtime_packed_sequence_reads_signed_int64_boundaries():
    values = [0, 1, -2, 300, 2**63 - 1, -(2**63)]
    code, seq = _program()
    text = "  \t" + "  ".join(map(str, values)) + "\n"
    result = run_bf(code, text, memory_size=4_000, step_limit=500_000_000)

    for i, expected in enumerate(values):
        assert result.memory[seq.marker(i)] == 1
        assert result.memory[seq.back(i)] == (0 if i == 0 else 1)
        assert _decode_s64(result.memory, seq.item(i)) == expected

    # The following record is the zero marker sentinel and links back when the
    # sequence is non-empty.
    assert result.memory[seq.marker(len(values))] == 0
    assert result.memory[seq.back(len(values))] == 1
    assert result.pointer == seq.base


def test_runtime_packed_sequence_source_is_independent_of_runtime_item_count():
    code, seq = _program()
    source_size = len(code)
    values = list(range(80))
    result = run_bf(
        code,
        " ".join(map(str, values)) + "\n",
        memory_size=8_000,
        step_limit=500_000_000,
    )

    assert len(code) == source_size
    assert source_size < 120_000
    for i, expected in enumerate(values):
        assert _decode_s64(result.memory, seq.item(i)) == expected
    assert result.memory[seq.marker(len(values))] == 0
    assert result.pointer == seq.base


def test_runtime_packed_sequence_persistent_stride_is_ten_cells_per_int64():
    seq = RuntimePackedIntSequence(base=100)
    assert RECORD_STRIDE == 10
    assert seq.item(1).base - seq.item(0).base == 10


def _raw_word(value):
    return "".join(chr((value >> (8 * i)) & 255) for i in range(8))


def _two_pass_program():
    bf = BFEmitter()
    seq = RuntimePackedIntSequence(base=64)
    seq.read_lf_terminated_s64s(bf)
    seq.walk_records(bf, lambda record: record.write_value())
    seq.walk_records(bf, lambda record: record.write_value(), reverse=True)
    assert bf.ptr == seq.base
    return bf.code(), seq


@pytest.mark.parametrize("length", [0, 1, 2, 80, 256])
def test_two_physical_passes_preserve_complete_sequence_state(length):
    values = [(-1 if i % 2 else 300) for i in range(length)]
    data = " ".join(map(str, values)) + "\n"
    before, _ = _program()
    after, seq = _two_pass_program()
    baseline = run_bf(before, data, memory_size=8000, step_limit=500_000_000)
    result = run_bf(after, data, memory_size=8000, step_limit=500_000_000)
    assert result.output == "".join(_raw_word(v) for v in values + values[::-1])
    assert result.memory == baseline.memory
    assert result.pointer == baseline.pointer == seq.base
    assert result.input_consumed == baseline.input_consumed == len(data)
    # Includes both the forward pass's rewind and reverse pass's initial scan.
    assert result.steps - baseline.steps == 98 * length + 8
    assert len(after) - len(before) < 120


@pytest.mark.parametrize("reverse", [False, True])
def test_record_replacement_keeps_metadata_and_next_input_intact(reverse):
    bf = BFEmitter()
    seq = RuntimePackedIntSequence(base=64)
    seq.read_lf_terminated_s64s(bf)
    calls = []

    def replace(record):
        calls.append(1)
        record.read_value()
        record.write_value()

    seq.walk_records(bf, replace, reverse=reverse)
    seq.walk_records(bf, lambda record: record.write_value())
    values = [0, -1, -(2**63), 2**63 - 1, 256]
    packed = "".join(map(_raw_word, values))
    prefix = "1 2 3 4 5\n"
    result = run_bf(bf.code(), prefix + packed + "untouched", memory_size=2000,
                    step_limit=500_000_000)
    expected = values[::-1] if reverse else values
    assert calls == [1]  # callback builds source once, not once per runtime item
    assert result.output == packed + "".join(map(_raw_word, expected))
    assert [_decode_s64(result.memory, seq.item(i)) for i in range(5)] == expected
    assert [result.memory[seq.marker(i)] for i in range(6)] == [1] * 5 + [0]
    assert [result.memory[seq.back(i)] for i in range(6)] == [0] + [1] * 5
    assert result.input_consumed == len(prefix) + len(packed)
    assert result.pointer == seq.base


def test_repeated_physical_passes_can_overwrite_and_revisit_values():
    bf = BFEmitter()
    seq = RuntimePackedIntSequence(base=64)
    seq.read_lf_terminated_s64s(bf)
    seq.walk_records(bf, lambda record: record.set_value(-1), reverse=True)
    seq.walk_records(bf, lambda record: record.write_value())
    seq.walk_records(bf, lambda record: record.set_value(2**64 + 256))
    seq.walk_records(bf, lambda record: record.write_value(), reverse=True)
    result = run_bf(bf.code(), "0 300 -1\n", memory_size=2000,
                    step_limit=500_000_000)
    assert result.output == _raw_word(-1) * 3 + _raw_word(256) * 3
    assert [_decode_s64(result.memory, seq.item(i)) for i in range(3)] == [256] * 3
    assert result.pointer == seq.base


@pytest.mark.parametrize("length", [0, 1, 2, 80, 255, 256, 257, 1024])
@pytest.mark.parametrize("value", [0, -1, -(2**63), 2**63 - 1])
def test_runtime_constant_repeat_and_two_passes(length, value):
    bf = BFEmitter()
    count = PackedU32Ref(0)
    for i in range(4):
        bf.move(count.byte(i))
        bf.emit(",")
    bf.set_const(63, 123)  # guard against writing left of the allocated region
    seq = RuntimePackedIntSequence(base=64)
    seq.repeat_constant(bf, count, value)
    seq.walk_records(bf, lambda record: record.write_value())
    seq.walk_records(bf, lambda record: record.write_value(), reverse=True)
    code = bf.code()
    data = "".join(chr((length >> (8 * i)) & 255) for i in range(4))
    result = run_bf(code, data, memory_size=64 + (length + 2) * RECORD_STRIDE,
                    step_limit=100_000_000)
    assert result.output == _raw_word(value) * (2 * length)
    assert result.memory[:4] == list(map(ord, data))
    assert result.memory[63] == 123
    assert result.memory[seq.marker(length)] == 0
    assert result.memory[seq.back(length)] == int(length > 0)
    assert result.memory[seq.item(length).base:seq.item(length).base + 8] == [0] * 8
    assert result.pointer == seq.base
    assert result.input_consumed == 4
    assert len(code) < 5000


@pytest.mark.parametrize("count_base", [-1, 61, 64, 100])
def test_repeat_rejects_count_overlapping_dynamic_storage(count_base):
    bf = BFEmitter()
    with pytest.raises(ValueError):
        RuntimePackedIntSequence(base=64).repeat_constant(bf, PackedU32Ref(count_base), 0)
    assert bf.code() == ""
