import pytest

from bf_runtime import BF_COMMANDS, run_bf
from bfcore import BFEmitter
from bfpacked import PackedU32Ref
from bfstreamseq import (
    BACK,
    COUNT,
    LENGTH,
    LENGTH_BYTES,
    MARKER,
    PAYLOAD0,
    PAYLOAD_BYTES,
    RECORD_STRIDE,
    RuntimeByteSequence,
)


CHUNK_BOUNDARIES = (0, 1, 7, 8, 9, 15, 16, 17, 255, 256)


def _roundtrip_program():
    bf = BFEmitter()
    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.write_all_bytes(bf)
    return bf.code()


def _read_program():
    bf = BFEmitter()
    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    return bf.code(), seq


def _u32_at(memory, base):
    return sum(memory[base + i] << (8 * i) for i in range(4))


def _payload(length):
    # Printable, non-LF bytes make replay failures easy to diagnose while still
    # exercising different payload values and every chunk lane.
    return "".join(chr(33 + (index % 90)) for index in range(length))


def _expected_counts(length):
    full_chunks, tail = divmod(length, PAYLOAD_BYTES)
    return [PAYLOAD_BYTES] * full_chunks + ([tail] if tail else [])


def _raw_u32(value):
    return "".join(chr((value >> (8 * byte_index)) & 0xFF) for byte_index in range(4))


def _dynamic_load_program():
    bf = BFEmitter()
    index = PackedU32Ref(0)
    dst = 8
    for byte_index in range(4):
        bf.move(index.byte(byte_index))
        bf.emit(",")

    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.load_byte(bf, dst, index)
    seq.write_all_bytes(bf)
    return bf.code(), seq, index, dst


def _dynamic_store_program():
    bf = BFEmitter()
    index = PackedU32Ref(0)
    src = 8
    for byte_index in range(4):
        bf.move(index.byte(byte_index))
        bf.emit(",")
    bf.move(src)
    bf.emit(",")

    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.store_byte(bf, index, src)
    seq.write_all_bytes(bf)
    return bf.code(), seq, index, src


def _dynamic_swap_program():
    bf = BFEmitter()
    left_index = PackedU32Ref(0)
    right_index = PackedU32Ref(4)
    left_value = 8
    right_value = 9
    for index in (left_index, right_index):
        for byte_index in range(4):
            bf.move(index.byte(byte_index))
            bf.emit(",")

    seq = RuntimeByteSequence(base=96)
    seq.read_lf_terminated_bytes(bf)
    seq.load_byte(bf, left_value, left_index)
    seq.load_byte(bf, right_value, right_index)
    seq.store_byte(bf, left_index, right_value)
    seq.store_byte(bf, right_index, left_value)
    seq.write_all_bytes(bf)
    return bf.code(), seq, left_index, right_index


def _assert_runtime_scratch_is_clean(result, seq, length):
    record_count = len(_expected_counts(length))
    for record in range(record_count + 1):
        record_base = seq.base + record * RECORD_STRIDE
        assert result.memory[record_base + LENGTH : record_base + LENGTH + LENGTH_BYTES] == [
            0
        ] * LENGTH_BYTES


def test_runtime_sized_byte_sequence_roundtrips_empty_and_nonempty_lines():
    code = _roundtrip_program()

    assert run_bf(code, "\n", step_limit=5_000_000).output == ""
    assert run_bf(code, "ABCxyz\n", step_limit=20_000_000).output == "ABCxyz"


def test_runtime_sequence_uses_one_source_program_for_longer_runtime_input():
    code = _roundtrip_program()
    source_size = len(code)

    payload = "a" * 200
    result = run_bf(code, payload + "\n", step_limit=300_000_000)

    assert result.output == payload
    # Runtime length changes tape/step usage only. There is no capacity/N
    # parameter in code generation, so the emitted program stays compact.
    assert len(code) == source_size
    assert source_size < 5_000


def test_runtime_length_is_carried_back_to_fixed_sentinel_metadata():
    code, seq = _read_program()

    for length in (0, 1, 7, 8, 9, 255, 256):
        payload = "x" * length
        result = run_bf(
            code,
            payload + "\n",
            memory_size=30_000,
            step_limit=500_000_000,
        )
        assert _u32_at(result.memory, seq.length_ref.base) == length
        assert result.memory[seq.left_sentinel] == 0
        assert result.pointer == seq.base


def test_runtime_length_metadata_coexists_with_roundtrip_replay():
    bf = BFEmitter()
    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.write_all_bytes(bf)
    code = bf.code()

    payload = "length-carrier"
    result = run_bf(code, payload + "\n", step_limit=100_000_000)

    assert result.output == payload
    assert _u32_at(result.memory, seq.length_ref.base) == len(payload)
    assert result.pointer == seq.base


def test_runtime_sequence_layout_reserves_a_left_sentinel_record():
    seq = RuntimeByteSequence(base=3 * RECORD_STRIDE)
    assert seq.left_sentinel == 2 * RECORD_STRIDE
    assert seq.length_ref.base > seq.left_sentinel
    assert seq.length_ref.base + 3 < seq.base


@pytest.mark.parametrize("length", CHUNK_BOUNDARIES)
def test_runtime_sequence_materializes_exact_chunk_metadata(length):
    code, seq = _read_program()
    payload = _payload(length)
    result = run_bf(
        code,
        payload + "\nfollowing line\n",
        memory_size=30_000,
        step_limit=20_000_000,
    )

    expected_counts = _expected_counts(length)
    chunk_count = len(expected_counts)

    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert result.memory[seq.left_sentinel + MARKER] == 0
    assert result.pointer == seq.base
    assert result.input_consumed == length + 1

    assert [
        result.memory[seq.base + record * RECORD_STRIDE + COUNT]
        for record in range(chunk_count)
    ] == expected_counts
    assert [
        result.memory[seq.base + record * RECORD_STRIDE + MARKER]
        for record in range(chunk_count + 1)
    ] == [1] * chunk_count + [0]

    expected_backs = [0] if chunk_count == 0 else [0] + [1] * chunk_count
    assert [
        result.memory[seq.base + record * RECORD_STRIDE + BACK]
        for record in range(chunk_count + 1)
    ] == expected_backs

    if expected_counts and expected_counts[-1] < PAYLOAD_BYTES:
        final_record = seq.base + (chunk_count - 1) * RECORD_STRIDE
        unused_payload = result.memory[
            final_record + PAYLOAD0 + expected_counts[-1] :
            final_record + PAYLOAD0 + PAYLOAD_BYTES
        ]
        assert unused_payload == [0] * (PAYLOAD_BYTES - expected_counts[-1])


@pytest.mark.parametrize("length", CHUNK_BOUNDARIES)
def test_runtime_sequence_chunk_boundaries_roundtrip_exactly(length):
    code = _roundtrip_program()
    payload = _payload(length)
    result = run_bf(
        code,
        payload + "\n",
        memory_size=30_000,
        step_limit=20_000_000,
    )

    assert result.output == payload
    assert result.pointer == 64


def test_runtime_sequence_reader_stops_after_one_logical_line():
    code, _seq = _read_program()
    first_line = "eight-byte-boundary"
    result = run_bf(code, first_line + "\nsecond line\n", step_limit=20_000_000)

    assert result.input_consumed == len(first_line) + 1


@pytest.mark.parametrize(
    ("length", "index"),
    (
        (1, 0),
        (9, 1),
        (9, 7),
        (10, 8),
        (10, 9),
        (17, 16),
        (257, 255),
        (257, 256),
        (256, 256),
        (17, 17),
        (17, 0xFFFFFFFF),
        (0, 0),
    ),
)
def test_runtime_nonnegative_index_load_preserves_sequence(length, index):
    code, seq, index_ref, dst = _dynamic_load_program()
    payload = _payload(length)
    result = run_bf(
        code,
        _raw_u32(index) + payload + "\n",
        memory_size=30_000,
        step_limit=100_000_000,
    )

    expected = ord(payload[index]) if index < length else 0
    assert result.memory[dst] == expected
    assert result.output == payload
    assert _u32_at(result.memory, index_ref.base) == index
    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert result.input_consumed == 4 + length + 1
    assert result.pointer == seq.base
    assert result.memory[seq.base + BACK] == 0
    _assert_runtime_scratch_is_clean(result, seq, length)


@pytest.mark.parametrize(
    ("length", "index"),
    (
        (1, 0),
        (9, 1),
        (9, 7),
        (10, 8),
        (10, 9),
        (17, 16),
        (257, 255),
        (257, 256),
        (256, 256),
        (17, 17),
        (17, 0xFFFFFFFF),
        (0, 0),
    ),
)
def test_runtime_nonnegative_index_store_changes_exactly_one_byte(length, index):
    code, seq, index_ref, src = _dynamic_store_program()
    payload = _payload(length)
    replacement = "Z"
    result = run_bf(
        code,
        _raw_u32(index) + replacement + payload + "\n",
        memory_size=30_000,
        step_limit=100_000_000,
    )

    expected = payload
    if index < length:
        expected = payload[:index] + replacement + payload[index + 1 :]

    assert result.output == expected
    assert result.memory[src] == ord(replacement)
    assert _u32_at(result.memory, index_ref.base) == index
    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert result.input_consumed == 5 + length + 1
    assert result.pointer == seq.base
    assert result.memory[seq.base + BACK] == 0
    _assert_runtime_scratch_is_clean(result, seq, length)

    if length % PAYLOAD_BYTES:
        final_record = seq.base + (length // PAYLOAD_BYTES) * RECORD_STRIDE
        unused_payload = result.memory[
            final_record + PAYLOAD0 + (length % PAYLOAD_BYTES) :
            final_record + PAYLOAD0 + PAYLOAD_BYTES
        ]
        assert unused_payload == [0] * (PAYLOAD_BYTES - (length % PAYLOAD_BYTES))


def test_runtime_index_programs_are_source_compact_and_runtime_sized():
    load_code, *_ = _dynamic_load_program()
    store_code, *_ = _dynamic_store_program()

    assert set(load_code) <= BF_COMMANDS
    assert set(store_code) <= BF_COMMANDS
    assert len(load_code) < 32 * 1024
    assert len(store_code) < 32 * 1024


@pytest.mark.parametrize(
    ("length", "left", "right"),
    ((9, 0, 8), (17, 7, 7), (257, 255, 256)),
)
def test_runtime_index_operations_compose_into_character_swaps(length, left, right):
    code, seq, left_ref, right_ref = _dynamic_swap_program()
    payload = _payload(length)
    expected = list(payload)
    expected[left], expected[right] = expected[right], expected[left]

    result = run_bf(
        code,
        _raw_u32(left) + _raw_u32(right) + payload + "\n",
        memory_size=30_000,
        step_limit=100_000_000,
    )

    assert result.output == "".join(expected)
    assert _u32_at(result.memory, left_ref.base) == left
    assert _u32_at(result.memory, right_ref.base) == right
    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert result.pointer == seq.base
    assert set(code) <= BF_COMMANDS
    assert len(code) < 128 * 1024
    _assert_runtime_scratch_is_clean(result, seq, length)
