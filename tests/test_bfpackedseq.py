import pytest

from bf_runtime import run_bf
from bfcore import BFEmitter
from bfpacked import PackedU32Core, PackedU32Ref
from bfpackedseq import ACCESS_WORKSPACE_CELLS, RECORD_STRIDE, RuntimePackedIntSequence


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


def _reduction_program(*, read_values=True, twice=False):
    from bfpacked64 import PackedI64Ref

    bf = BFEmitter()
    for i in range(4):
        bf.move(i)
        bf.emit(",")
    bf.set_const(25, 173)  # Last cell before the reserved mobile frame.
    for i in range(8, 20):
        bf.set_const(i, 211)  # Outputs must replace previous values.
    seq = RuntimePackedIntSequence(64)
    seq.repeat_constant(bf, PackedU32Ref(0), 0)
    if read_values:
        seq.walk_records(bf, lambda record: record.read_value())
    baseline = bf.code()
    seq.sum_and_length(bf, PackedI64Ref(8), PackedU32Ref(16))
    if twice:
        seq.sum_and_length(bf, PackedI64Ref(8), PackedU32Ref(16))
    assert bf.ptr == seq.base
    return baseline, bf.code(), seq


@pytest.mark.parametrize("values", [
    [], [0], [-1], [255, 1, 256, -257],
    [(1 << 63) - 1, 1], [-(1 << 63), -1],
    [-(1 << 63), (1 << 63) - 1, 1],
    [i * 104729 - 500000 for i in range(16)],
])
def test_mobile_reduction_preserves_full_sequence_and_repeats(values):
    baseline, code, seq = _reduction_program(twice=True)
    data = _raw_word(len(values))[:4] + "".join(map(_raw_word, values)) + "X"
    kwargs = dict(memory_size=64 + (len(values) + 2) * RECORD_STRIDE,
                  step_limit=500_000_000)
    before = run_bf(baseline, data, **kwargs)
    result = run_bf(code, data, **kwargs)
    expected_memory = before.memory[:]
    expected_memory[8:16] = list(map(ord, _raw_word(sum(values))))
    expected_memory[16:20] = list(map(ord, _raw_word(len(values))[:4]))
    assert result.memory == expected_memory
    assert result.pointer == before.pointer == seq.base
    assert result.input_consumed == before.input_consumed == 4 + 8 * len(values)
    assert result.output == before.output == ""
    assert set(code) <= set("><+-.,[]")


def test_mobile_reduction_runtime_length_and_scaling():
    baseline, code, seq = _reduction_program(read_values=False)
    assert len(code) - len(baseline) < 12_000
    large_steps = []
    for count in (0, 1, 2, 255, 256, 257, 512, 1024, 2048):
        data = _raw_word(count)[:4]
        kwargs = dict(memory_size=64 + (count + 2) * RECORD_STRIDE,
                      step_limit=500_000_000)
        before = run_bf(baseline, data, **kwargs)
        result = run_bf(code, data, **kwargs)
        expected_memory = before.memory[:]
        expected_memory[8:16] = [0] * 8
        expected_memory[16:20] = list(map(ord, data))
        assert result.memory == expected_memory
        assert result.pointer == before.pointer == seq.base
        assert result.input_consumed == before.input_consumed == 4
        delta = result.steps - before.steps
        # Counter transport depends on byte values (each <=255); this is a
        # measured ceiling, not an exact constant-step-per-element formula.
        assert delta <= 40_000 * count + 100_000
        if count in (256, 512, 1024, 2048):
            large_steps.append(delta)
    assert all(b < 2.2 * a for a, b in zip(large_steps, large_steps[1:]))


@pytest.mark.parametrize("base,total,length", [
    (20, 0, 8), (64, -1, 8), (64, 20, 8), (64, 0, 24),
    (64, 0, 6), (64, 30, 0), (64, 64, 0),
])
def test_mobile_reduction_rejects_invalid_layout_before_emitting(base, total, length):
    from bfpacked64 import PackedI64Ref

    bf = BFEmitter()
    with pytest.raises(ValueError):
        RuntimePackedIntSequence(base).sum_and_length(
            bf, PackedI64Ref(total), PackedU32Ref(length))
    assert bf.code() == ""


@pytest.mark.parametrize("text,values", [("  \t\n", []), ("-1 255 256  \n", [-1, 255, 256])])
def test_mobile_reduction_after_decimal_input_leaves_next_line(text, values):
    from bfpacked64 import PackedI64Ref

    bf = BFEmitter()
    seq = RuntimePackedIntSequence(64)
    seq.read_lf_terminated_s64s(bf)
    seq.sum_and_length(bf, PackedI64Ref(0), PackedU32Ref(8))
    seq.walk_records(bf, lambda record: record.write_value())
    bf.move(12)
    bf.emit(",.")
    result = run_bf(bf.code(), text + "X\n", memory_size=3000,
                    step_limit=500_000_000)
    assert result.output == "".join(map(_raw_word, values)) + "X"
    assert result.memory[:8] == list(map(ord, _raw_word(sum(values))))
    assert result.memory[8:12] == list(map(ord, _raw_word(len(values))[:4]))
    assert result.input_consumed == len(text) + 1

@pytest.mark.parametrize("count,value", [(0, -1), (1, -(1 << 63)), (3, -1),
                                           (65, 256), (256, 0), (300, 2)])
def test_runtime_value_repeat_preserves_inputs_and_cleans_future_scratch(count, value):
    from bfpacked64 import PackedI64Ref
    from bfpackedops import PackedI64Ops
    from bfpackedseq import REPEAT_FORWARD_WORKSPACE_CELLS
    bf = BFEmitter()
    n, x = PackedI64Ref(0), PackedI64Ref(8)
    ops = PackedI64Ops(bf, 16)
    ops.set_u64(n, count)
    ops.set_u64(x, value)
    seq = RuntimePackedIntSequence(80)
    seq.repeat_value(bf, n, x)
    result = run_bf(bf.code(), memory_size=8000, step_limit=500_000_000)
    assert _decode_s64(result.memory, n) == count
    assert _decode_s64(result.memory, x) == value
    for i in range(count):
        assert result.memory[seq.marker(i)] == 1
        assert result.memory[seq.back(i)] == bool(i)
        assert _decode_s64(result.memory, seq.item(i)) == value
    assert result.memory[seq.marker(count)] == 0
    assert result.memory[seq.back(count)] == bool(count)
    end = seq.marker(count)
    assert not any(result.memory[end + 2:end + REPEAT_FORWARD_WORKSPACE_CELLS])
    assert result.pointer == seq.base
    assert set(bf.code()) <= set("><+-.,[]")
    assert len(bf.code()) < 20_000


@pytest.mark.parametrize("count", [1, 256, 65536, (1 << 32), (1 << 32) + 1, (1 << 63) - 1])
def test_repeat_counter_keeps_upper_word_and_borrows_across_u32(count):
    from bfpacked64 import PackedI64Ref
    from bfpackedops import PackedI64Ops
    from bfpackedseq import _decrement_repeat_count
    bf = BFEmitter()
    ref = PackedI64Ref(8)
    PackedI64Ops(bf, 16).set_u64(ref, count)
    _decrement_repeat_count(bf)
    result = run_bf(bf.code(), memory_size=64, step_limit=1_000_000)
    assert _decode_s64(result.memory, ref) == count - 1
    assert not any(result.memory[16:32])


def test_runtime_value_repeat_linear_step_growth_and_invalid_layout():
    from bfpacked64 import PackedI64Ref
    from bfpackedops import PackedI64Ops
    seq = RuntimePackedIntSequence(80)
    steps = []
    for n in (256, 512, 1024):
        bf = BFEmitter()
        count, value = PackedI64Ref(0), PackedI64Ref(8)
        ops = PackedI64Ops(bf, 16)
        ops.set_u64(count, n)
        ops.set_u64(value, -1)
        seq.repeat_value(bf, count, value)
        result = run_bf(bf.code(), memory_size=12000, step_limit=500_000_000)
        assert _decode_s64(result.memory, seq.item(n - 1)) == -1
        steps.append(result.steps)
    assert steps[1] < 2.5 * steps[0]
    assert steps[2] < 2.5 * steps[1]
    bf = BFEmitter()
    with pytest.raises(ValueError, match="precede"):
        seq.repeat_value(bf, PackedI64Ref(74), PackedI64Ref(0))
    assert bf.code() == ""


@pytest.mark.parametrize("index,expected,found", [
    (0, 11, 1),
    (2, 11, 1),
    (64, 11, 1),
    (65, 0, 0),
    (256, 0, 0),
    (-1, 0, 0),
    (1 << 32, 0, 0),
])
def test_runtime_packed_sequence_load_value_preserves_index(index, expected, found):
    from bfpacked64 import PackedI64Ref
    from bfpackedops import PackedI64Ops
    bf = BFEmitter()
    count = PackedU32Ref(0)
    packed_index = PackedI64Ref(8)
    result = PackedI64Ref(16)
    hit = 24
    PackedU32Core(bf, 32).set_u32(count, 65)
    PackedI64Ops(bf, 32).set_u64(packed_index, index)
    seq = RuntimePackedIntSequence(128)
    seq.repeat_constant(bf, count, 11)
    seq.load_value(bf, packed_index, result, found=hit)
    execution = run_bf(bf.code(), memory_size=4_000, step_limit=500_000_000)
    assert _decode_s64(execution.memory, packed_index) == index
    assert _decode_s64(execution.memory, result) == expected
    assert execution.memory[hit] == found
    assert execution.pointer == seq.base
    assert not any(execution.memory[seq.base - ACCESS_WORKSPACE_CELLS:seq.base])
    for item in range(65):
        assert _decode_s64(execution.memory, seq.item(item)) == 11
        assert execution.memory[seq.marker(item)] == 1
        assert execution.memory[seq.back(item)] == (item != 0)
    assert execution.memory[seq.marker(65)] == 0
    assert execution.memory[seq.back(65)] == 1


@pytest.mark.parametrize("index,replacement,found", [
    (0, -7, 1),
    (64, 1 << 40, 1),
    (65, 99, 0),
    (-1, 99, 0),
])
def test_runtime_packed_sequence_exchange_value_changes_only_a_hit(
    index, replacement, found,
):
    from bfpacked64 import PackedI64Ref
    from bfpackedops import PackedI64Ops
    bf = BFEmitter()
    count = PackedU32Ref(0)
    packed_index = PackedI64Ref(8)
    value = PackedI64Ref(16)
    previous = PackedI64Ref(24)
    hit = 40
    PackedU32Core(bf, 48).set_u32(count, 65)
    ops = PackedI64Ops(bf, 48)
    ops.set_u64(packed_index, index)
    ops.set_u64(value, replacement)
    seq = RuntimePackedIntSequence(160)
    seq.repeat_constant(bf, count, 11)
    seq.exchange_value(bf, packed_index, value, previous, found=hit)
    execution = run_bf(bf.code(), memory_size=4_000, step_limit=500_000_000)
    assert _decode_s64(execution.memory, packed_index) == index
    assert _decode_s64(execution.memory, value) == replacement
    assert _decode_s64(execution.memory, previous) == (11 if found else 0)
    assert execution.memory[hit] == found
    for item in range(65):
        expected = replacement if found and item == index else 11
        assert _decode_s64(execution.memory, seq.item(item)) == expected
        assert execution.memory[seq.marker(item)] == 1
        assert execution.memory[seq.back(item)] == (item != 0)
    assert execution.memory[seq.marker(65)] == 0
    assert execution.memory[seq.back(65)] == 1
    assert not any(execution.memory[seq.base - ACCESS_WORKSPACE_CELLS:seq.base])
    assert execution.pointer == seq.base


def test_runtime_packed_sequence_access_rejects_overlapping_workspace():
    from bfpacked64 import PackedI64Ref
    bf = BFEmitter()
    seq = RuntimePackedIntSequence(ACCESS_WORKSPACE_CELLS - 1)
    with pytest.raises(ValueError, match="precede"):
        seq.load_value(bf, PackedI64Ref(0), PackedI64Ref(8))
    assert bf.code() == ""

    seq = RuntimePackedIntSequence(128)
    with pytest.raises(ValueError, match="must not overlap"):
        seq.load_value(bf, PackedI64Ref(0), PackedI64Ref(0))
    with pytest.raises(ValueError, match="hit flag must not overlap"):
        seq.load_value(bf, PackedI64Ref(0), PackedI64Ref(8), found=8)
    assert bf.code() == ""
