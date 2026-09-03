from bf_runtime import run_bf
from bfcore import BFEmitter
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
