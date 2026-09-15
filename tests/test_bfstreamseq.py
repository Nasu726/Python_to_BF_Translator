import pytest

from bf_runtime import BF_COMMANDS, run_bf
from bfcore import BFEmitter
from bfpacked import PackedU32Ref
from bfpacked64 import PackedI64Ref
from bfstreamseq import (
    BACK,
    COUNT,
    CURSOR_VALUE,
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


def _raw_i64(value):
    value &= (1 << 64) - 1
    return "".join(chr((value >> (8 * byte_index)) & 0xFF) for byte_index in range(8))


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


def _dynamic_signed_load_program():
    bf = BFEmitter()
    index = PackedI64Ref(0)
    dst = 8
    for byte_index in range(8):
        bf.move(index.byte(byte_index))
        bf.emit(",")

    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.load_byte_signed(bf, dst, index)
    seq.write_all_bytes(bf)
    return bf.code(), seq, index, dst


def _dynamic_signed_store_program():
    bf = BFEmitter()
    index = PackedI64Ref(0)
    src = 8
    for byte_index in range(8):
        bf.move(index.byte(byte_index))
        bf.emit(",")
    bf.move(src)
    bf.emit(",")

    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.store_byte_signed(bf, index, src)
    seq.write_all_bytes(bf)
    return bf.code(), seq, index, src


def _dynamic_signed_normalization_program():
    bf = BFEmitter()
    index = PackedI64Ref(0)
    for byte_index in range(8):
        bf.move(index.byte(byte_index))
        bf.emit(",")

    seq = RuntimeByteSequence(base=64)
    for byte_index in range(4):
        bf.move(seq.length_ref.byte(byte_index))
        bf.emit(",")
    normalized = seq._normalize_signed_index(bf, index)
    return bf.code(), seq, index, normalized


def _runtime_swap_loop_program():
    bf = BFEmitter()
    query_count = 0
    left_index = PackedU32Ref(1)
    right_index = PackedU32Ref(5)
    left_value = 9
    right_value = 10
    seq = RuntimeByteSequence(base=96)

    seq.read_lf_terminated_bytes(bf)
    bf.move(query_count)
    bf.emit(",")
    bf.begin_while(query_count)
    for index in (left_index, right_index):
        for byte_index in range(4):
            bf.move(index.byte(byte_index))
            bf.emit(",")
    seq.load_byte(bf, left_value, left_index)
    seq.load_byte(bf, right_value, right_index)
    seq.store_byte(bf, left_index, right_value)
    seq.store_byte(bf, right_index, left_value)
    bf.add_const(query_count, -1)
    bf.end_while(query_count)
    seq.write_all_bytes(bf)
    return bf.code(), seq


def _cursor_load_loop_program():
    bf = BFEmitter()
    seq = RuntimeByteSequence(base=96)
    seq.read_lf_terminated_bytes(bf)
    cursor = seq.open_cursor(bf)
    cursor.begin_input_loop()
    cursor.seek_from_input()
    cursor.load_lane_from_input()
    cursor.output_value()
    cursor.end_input_loop()
    cursor.finish()
    seq.write_all_bytes(bf)
    return bf.code(), seq


def _cursor_swap_loop_program():
    bf = BFEmitter()
    seq = RuntimeByteSequence(base=96)
    seq.read_lf_terminated_bytes(bf)
    cursor = seq.open_cursor(bf)
    cursor.begin_input_loop()
    cursor.seek_from_input()
    cursor.load_lane_from_input()
    cursor.seek_from_input()
    cursor.exchange_lane_from_input()
    cursor.seek_from_input()
    cursor.exchange_lane_from_input()
    cursor.end_input_loop()
    cursor.finish()
    seq.write_all_bytes(bf)
    return bf.code(), seq


def _cursor_copy_program():
    bf = BFEmitter()
    dst = 0
    seq = RuntimeByteSequence(base=96)
    seq.read_lf_terminated_bytes(bf)
    cursor = seq.open_cursor(bf)
    cursor.seek_forward_from_input()
    cursor.load_lane_from_input()
    cursor.seek_backward_from_input()
    cursor.store_lane_from_input()
    cursor.finish(dst=dst)
    bf.move(dst)
    bf.emit(".")
    seq.write_all_bytes(bf)
    return bf.code(), seq, dst


def _cursor_transition(current_record, target_record):
    if target_record >= current_record:
        return _raw_u32(target_record - current_record) + _raw_u32(0)
    return _raw_u32(0) + _raw_u32(current_record - target_record)


def _encode_cursor_loads(indices):
    if not indices:
        return "\0"

    encoded = ["\1"]
    current_record = 0
    for offset, index in enumerate(indices):
        target_record, lane = divmod(index, PAYLOAD_BYTES)
        encoded.append(_cursor_transition(current_record, target_record))
        encoded.append(chr(lane))
        encoded.append("\1" if offset + 1 < len(indices) else "\0")
        current_record = target_record
    return "".join(encoded)


def _encode_cursor_swaps(queries):
    if not queries:
        return "\0"

    encoded = ["\1"]
    current_record = 0
    for offset, (left, right) in enumerate(queries):
        left_record, left_lane = divmod(left, PAYLOAD_BYTES)
        right_record, right_lane = divmod(right, PAYLOAD_BYTES)
        encoded.extend(
            (
                _cursor_transition(current_record, left_record),
                chr(left_lane),
                _cursor_transition(left_record, right_record),
                chr(right_lane),
                _cursor_transition(right_record, left_record),
                chr(left_lane),
                "\1" if offset + 1 < len(queries) else "\0",
            )
        )
        current_record = left_record
    return "".join(encoded)


def _assert_runtime_scratch_is_clean(result, seq, length):
    record_count = len(_expected_counts(length))
    for record in range(record_count + 1):
        record_base = seq.base + record * RECORD_STRIDE
        assert result.memory[record_base + LENGTH : record_base + LENGTH + LENGTH_BYTES] == [
            0
        ] * LENGTH_BYTES
        assert result.memory[record_base + CURSOR_VALUE] == 0


def _assert_normalization_scratch_is_clean(result, seq):
    start = seq.left_sentinel + PAYLOAD0
    assert result.memory[
        start : start + PAYLOAD_BYTES
    ] == [0] * PAYLOAD_BYTES


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


@pytest.mark.parametrize("length", (1024, 4097))
def test_runtime_sequence_roundtrips_kilobyte_scale_lines(length):
    code = _roundtrip_program()
    payload = _payload(length)
    seq = RuntimeByteSequence(base=64)
    memory_size = seq.base + ((length + 7) // 8 + 3) * RECORD_STRIDE

    result = run_bf(
        code,
        payload + "\n",
        memory_size=memory_size,
        step_limit=200_000_000,
    )

    assert result.output == payload
    assert result.pointer == seq.base
    assert len(code) < 5_000


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


SIGNED_INDEX_CASES = (
    (0, -1),
    (1, -1),
    (9, -1),
    (9, -9),
    (9, -10),
    (17, -16),
    (257, -1),
    (257, -257),
    (257, -258),
    (257, 255),
    (257, 256),
    (257, 257),
    (1, 1 << 32),
    (1, -(1 << 32)),
    (1, (1 << 63) - 1),
    (1, -(1 << 63)),
)


@pytest.mark.parametrize(("length", "index"), SIGNED_INDEX_CASES)
def test_runtime_signed_index_load_normalizes_once(length, index):
    code, seq, index_ref, dst = _dynamic_signed_load_program()
    payload = _payload(length)
    result = run_bf(
        code,
        _raw_i64(index) + payload + "\n",
        memory_size=30_000,
        step_limit=100_000_000,
    )

    expected = ord(payload[index]) if -length <= index < length and length else 0
    assert result.memory[dst] == expected
    assert result.output == payload
    assert _u32_at(result.memory, index_ref.base) == (index & 0xFFFFFFFF)
    assert _u32_at(result.memory, index_ref.base + 4) == ((index >> 32) & 0xFFFFFFFF)
    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert result.pointer == seq.base
    _assert_runtime_scratch_is_clean(result, seq, length)
    _assert_normalization_scratch_is_clean(result, seq)


@pytest.mark.parametrize(("length", "index"), SIGNED_INDEX_CASES)
def test_runtime_signed_index_store_normalizes_once(length, index):
    code, seq, index_ref, src = _dynamic_signed_store_program()
    payload = _payload(length)
    replacement = "Z"
    result = run_bf(
        code,
        _raw_i64(index) + replacement + payload + "\n",
        memory_size=30_000,
        step_limit=100_000_000,
    )

    expected = payload
    if -length <= index < length and length:
        normalized = index % length if index < 0 else index
        expected = payload[:normalized] + replacement + payload[normalized + 1 :]

    assert result.output == expected
    assert result.memory[src] == ord(replacement)
    assert _u32_at(result.memory, index_ref.base) == (index & 0xFFFFFFFF)
    assert _u32_at(result.memory, index_ref.base + 4) == ((index >> 32) & 0xFFFFFFFF)
    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert result.pointer == seq.base
    _assert_runtime_scratch_is_clean(result, seq, length)
    _assert_normalization_scratch_is_clean(result, seq)


def test_runtime_signed_index_programs_remain_source_compact():
    load_code, *_ = _dynamic_signed_load_program()
    store_code, *_ = _dynamic_signed_store_program()

    assert set(load_code) <= BF_COMMANDS
    assert set(store_code) <= BF_COMMANDS
    assert len(load_code) < 40 * 1024
    assert len(store_code) < 40 * 1024


@pytest.mark.parametrize(
    ("length", "index", "expected"),
    (
        (0, -1, 0xFFFFFFFF),
        (255, -1, 254),
        (255, -255, 0),
        (255, -256, 0xFFFFFFFF),
        (256, -1, 255),
        (256, -256, 0),
        (256, -257, 0xFFFFFFFF),
        (65_535, -65_535, 0),
        (65_536, -65_535, 1),
        (65_536, -65_536, 0),
        (65_536, -65_537, 0xFFFFFFFF),
        (0xFFFFFFFF, -0xFFFFFFFF, 0),
        (0xFFFFFFFF, -(1 << 32), 0xFFFFFFFF),
        (17, 0xFFFFFFFF, 0xFFFFFFFF),
        (17, 1 << 32, 0xFFFFFFFF),
    ),
)
def test_signed_index_normalization_handles_packed_borrow_boundaries(
    length, index, expected
):
    code, seq, index_ref, normalized = _dynamic_signed_normalization_program()
    result = run_bf(
        code,
        _raw_i64(index) + _raw_u32(length),
        memory_size=1_024,
        step_limit=100_000_000,
    )

    assert _u32_at(result.memory, normalized.base) == expected
    assert _u32_at(result.memory, seq.length_ref.base) == length
    assert _u32_at(result.memory, index_ref.base) == (index & 0xFFFFFFFF)
    assert _u32_at(result.memory, index_ref.base + 4) == ((index >> 32) & 0xFFFFFFFF)
    assert result.pointer == seq.base


def test_runtime_index_operations_are_reentrant_inside_a_query_loop():
    code, seq = _runtime_swap_loop_program()
    payload = _payload(17)
    queries = ((0, 16), (7, 8), (1, 2), (16, 16))
    expected = list(payload)
    encoded_queries = []
    for left, right in queries:
        expected[left], expected[right] = expected[right], expected[left]
        encoded_queries.append(_raw_u32(left) + _raw_u32(right))

    result = run_bf(
        code,
        payload + "\n" + chr(len(queries)) + "".join(encoded_queries),
        memory_size=30_000,
        step_limit=100_000_000,
    )

    assert result.output == "".join(expected)
    assert result.memory[0] == 0
    assert _u32_at(result.memory, seq.length_ref.base) == len(payload)
    assert result.pointer == seq.base
    assert set(code) <= BF_COMMANDS
    assert len(code) < 128 * 1024
    _assert_runtime_scratch_is_clean(result, seq, len(payload))


def test_cursor_load_loop_retains_runtime_record_and_moves_both_directions():
    code, seq = _cursor_load_loop_program()
    payload = _payload(40)
    indices = (0, 7, 8, 31, 16, 17, 1, 39)
    selected = "".join(payload[index] for index in indices)

    result = run_bf(
        code,
        payload + "\n" + _encode_cursor_loads(indices),
        memory_size=30_000,
        step_limit=100_000_000,
    )

    assert result.output == selected + payload
    assert _u32_at(result.memory, seq.length_ref.base) == len(payload)
    assert result.pointer == seq.base
    assert set(code) <= BF_COMMANDS
    assert len(code) < 32 * 1024
    _assert_runtime_scratch_is_clean(result, seq, len(payload))


def test_cursor_record_delta_crosses_packed_255_256_borrow_boundary():
    code, seq = _cursor_load_loop_program()
    payload = _payload(2057)
    indices = (2040, 2048, 0)
    selected = "".join(payload[index] for index in indices)
    memory_size = seq.base + ((len(payload) + 7) // 8 + 3) * RECORD_STRIDE

    result = run_bf(
        code,
        payload + "\n" + _encode_cursor_loads(indices),
        memory_size=memory_size,
        step_limit=200_000_000,
    )

    assert result.output == selected + payload
    assert result.pointer == seq.base
    _assert_runtime_scratch_is_clean(result, seq, len(payload))


def test_cursor_swap_loop_handles_same_cross_chunk_and_reversed_paths():
    code, seq = _cursor_swap_loop_program()
    payload = _payload(40)
    queries = ((0, 39), (7, 8), (24, 17), (31, 31), (1, 33))
    expected = list(payload)
    for left, right in queries:
        expected[left], expected[right] = expected[right], expected[left]

    result = run_bf(
        code,
        payload + "\n" + _encode_cursor_swaps(queries),
        memory_size=30_000,
        step_limit=100_000_000,
    )

    assert result.output == "".join(expected)
    assert _u32_at(result.memory, seq.length_ref.base) == len(payload)
    assert result.pointer == seq.base
    assert set(code) <= BF_COMMANDS
    assert len(code) < 64 * 1024
    _assert_runtime_scratch_is_clean(result, seq, len(payload))


def test_cursor_finish_carries_value_to_static_cell_and_store_preserves_it():
    code, seq, dst = _cursor_copy_program()
    payload = _payload(40)
    expected = list(payload)
    expected[1] = payload[39]
    cursor_input = _raw_u32(4) + chr(7) + _raw_u32(4) + chr(1)

    result = run_bf(
        code,
        payload + "\n" + cursor_input,
        memory_size=30_000,
        step_limit=100_000_000,
    )

    assert result.output == payload[39] + "".join(expected)
    assert result.memory[dst] == ord(payload[39])
    assert result.pointer == seq.base
    _assert_runtime_scratch_is_clean(result, seq, len(payload))


def test_empty_cursor_query_loop_is_a_noop_and_restores_static_pointer():
    code, seq = _cursor_swap_loop_program()
    payload = _payload(17)
    result = run_bf(
        code,
        payload + "\n\0",
        memory_size=30_000,
        step_limit=20_000_000,
    )

    assert result.output == payload
    assert result.pointer == seq.base
    _assert_runtime_scratch_is_clean(result, seq, len(payload))


def test_cursor_tail_local_swaps_avoid_repeated_root_roundtrips():
    rooted_code, _ = _runtime_swap_loop_program()
    cursor_code, seq = _cursor_swap_loop_program()
    payload = _payload(64)
    queries = tuple((55 + offset, 56 + offset) for offset in range(8))
    expected = list(payload)
    for left, right in queries:
        expected[left], expected[right] = expected[right], expected[left]

    rooted_input = payload + "\n" + chr(len(queries)) + "".join(
        _raw_u32(left) + _raw_u32(right) for left, right in queries
    )
    cursor_input = payload + "\n" + _encode_cursor_swaps(queries)
    rooted = run_bf(rooted_code, rooted_input, step_limit=100_000_000)
    cursor = run_bf(cursor_code, cursor_input, step_limit=100_000_000)

    assert rooted.output == cursor.output == "".join(expected)
    # This is an architecture regression gate, not an AtCoder wall-clock
    # prediction. Retaining the record cursor must materially beat six rooted
    # walks per swap even after including the shared input/replay baseline.
    assert cursor.steps * 5 < rooted.steps
    assert cursor.pointer == seq.base
    _assert_runtime_scratch_is_clean(cursor, seq, len(payload))
