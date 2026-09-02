from bfcore import BFEmitter, Int64Ref
from bf_runtime import run_bf
from bflists import BinaryListIO, IntListRef


def signed_bits(mem, base):
    raw = sum((mem[base + i] & 1) << i for i in range(64))
    return raw - (1 << 64) if raw & (1 << 63) else raw


def signed_packed(mem, base):
    raw = sum(mem[base + i] << (8 * i) for i in range(8))
    return raw - (1 << 64) if raw & (1 << 63) else raw


def test_literal_length_and_dynamic_index():
    bf = BFEmitter()
    backend = BinaryListIO(bf, scratch_base=900)
    values = IntListRef(0, 4)
    out = Int64Ref(300)
    idx = Int64Ref(364)
    workspace = Int64Ref(428)
    backend.set_list_literal(values, [10, -2, 30])
    backend.set_u64(idx, 1)
    backend.get_dynamic(out, values, idx, workspace, match=500)
    length = Int64Ref(501)
    backend.list_length(length, values, tmp=565)
    result = run_bf(bf.code(), memory_size=1200)
    assert signed_bits(result.memory, out.base) == -2
    assert signed_bits(result.memory, length.base) == 3


def test_append_uses_original_length_once():
    bf = BFEmitter()
    backend = BinaryListIO(bf, scratch_base=900)
    values = IntListRef(0, 4)
    x = Int64Ref(300)
    backend.set_list_literal(values, [1])
    backend.set_u64(x, 7)
    backend.append(values, x, length_copy=400, match=401)
    result = run_bf(bf.code(), memory_size=1200)
    assert result.memory[values.length_cell] == 2
    assert signed_packed(result.memory, values.item(0).base) == 1
    assert signed_packed(result.memory, values.item(1).base) == 7
    assert signed_packed(result.memory, values.item(2).base) == 0


def test_read_list_line_and_drain_overflow():
    bf = BFEmitter()
    backend = BinaryListIO(bf, scratch_base=1200)
    first = IntListRef(0, 3)
    second = IntListRef(250, 3)
    backend.read_int_list_line(first, 600, active=850, gate=851, has_token=852, end_line=853)
    backend.read_int_list_line(second, 600, active=850, gate=851, has_token=852, end_line=853)
    result = run_bf(bf.code(), '1 2 3 4 5\n-7 8\n', memory_size=1600, step_limit=500_000_000)
    assert result.memory[first.length_cell] == 3
    assert [signed_packed(result.memory, first.item(i).base) for i in range(3)] == [1, 2, 3]
    assert result.memory[second.length_cell] == 2
    assert [signed_packed(result.memory, second.item(i).base) for i in range(2)] == [-7, 8]
