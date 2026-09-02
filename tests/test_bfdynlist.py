from bf_runtime import run_bf
from bfcore import BFEmitter, Binary64Core, Int64Ref
from bfdynlist import DynamicIntListRootRuntime, DynamicIntListRuntime
from bfheap import HeapBlockArena
from bfobjects import ObjectHandleCore, ObjectHandleRef
from bfopt import optimize_bf
from bfpacked import PackedU32Core, PackedU32Ref
from bfpacked64 import PackedI64Core, PackedI64Ref


MASK64 = (1 << 64) - 1


def _u32(memory: list[int], ref: PackedU32Ref) -> int:
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(4))


def _u64_bytes(memory: list[int], ref: PackedI64Ref) -> int:
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(8))


def _run(bf: BFEmitter, *, memory_size: int, step_limit: int):
    return run_bf(optimize_bf(bf.code()), memory_size=memory_size, step_limit=step_limit)


def test_dynamic_list_root_assignment_is_alias_not_value_copy():
    bf = BFEmitter()
    next_handle = ObjectHandleRef(0)
    a = ObjectHandleRef(4)
    b = ObjectHandleRef(8)
    c = ObjectHandleRef(12)
    five = PackedU32Ref(16)
    observed_before_clear = PackedU32Ref(20)
    observed_after_clear = PackedU32Ref(24)
    separate_length = PackedU32Ref(28)
    zero = PackedU32Ref(32)
    scratch = 40
    left_sentinel = 80

    heap = HeapBlockArena(
        bf,
        left_sentinel=left_sentinel,
        next_handle=next_handle,
        scratch_base=scratch,
    )
    packed = PackedU32Core(bf, scratch)
    handles = ObjectHandleCore(bf, scratch)
    lists = DynamicIntListRootRuntime(heap, packed=packed, handles=handles)

    heap.initialize()
    lists.create_empty(a)
    lists.create_empty(c)
    lists.alias(b, a)

    packed.set_u32(five, 5)
    lists.write_length(b, five)
    lists.read_length(observed_before_clear, a)

    lists.clear(a, zero)
    lists.read_length(observed_after_clear, b)
    lists.read_length(separate_length, c)

    result = _run(bf, memory_size=512, step_limit=150_000_000)
    assert _u32(result.memory, a) == 1
    assert _u32(result.memory, b) == 1
    assert _u32(result.memory, c) == 2
    assert _u32(result.memory, observed_before_clear) == 5
    assert _u32(result.memory, observed_after_clear) == 0
    assert _u32(result.memory, separate_length) == 0


def test_dynamic_list_append_and_index_are_visible_through_alias():
    bf = BFEmitter()
    next_handle = ObjectHandleRef(0)
    a = ObjectHandleRef(4)
    b = ObjectHandleRef(8)
    value0 = Int64Ref(16)
    value1 = Int64Ref(80)
    out0 = PackedI64Ref(144)
    out1 = PackedI64Ref(152)
    index0 = PackedU32Ref(160)
    index1 = PackedU32Ref(164)
    length_out = PackedU32Ref(168)

    scratch = 180
    binary_scratch = 184
    workspace = 200
    left_sentinel = 280

    heap = HeapBlockArena(
        bf,
        left_sentinel=left_sentinel,
        next_handle=next_handle,
        scratch_base=scratch,
    )
    packed = PackedU32Core(bf, scratch)
    handles = ObjectHandleCore(bf, scratch)
    packed64 = PackedI64Core(bf, scratch)
    binary = Binary64Core(bf, scratch_base=binary_scratch)
    lists = DynamicIntListRuntime(
        heap,
        packed=packed,
        handles=handles,
        packed64=packed64,
        workspace_base=workspace,
    )

    heap.initialize()
    lists.create_empty(a)
    lists.alias(b, a)
    binary.set_u64(value0, 10)
    binary.set_u64(value1, -7)

    lists.append_int64(b, value0)
    lists.append_int64(b, value1)
    packed.set_u32(index0, 0)
    packed.set_u32(index1, 1)
    lists.get_packed(out0, a, index0)
    lists.get_packed(out1, a, index1)
    lists.read_length(length_out, a)

    result = _run(bf, memory_size=1_024, step_limit=300_000_000)
    assert _u32(result.memory, a) == _u32(result.memory, b) == 1
    assert _u32(result.memory, length_out) == 2
    assert _u64_bytes(result.memory, out0) == 10
    assert _u64_bytes(result.memory, out1) == ((-7) & MASK64)
