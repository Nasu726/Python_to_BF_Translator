from bf_runtime import run_bf
from bfcore import BFEmitter
from bfdynlist import DynamicIntListRootRuntime
from bfheap import HeapBlockArena
from bfobjects import ObjectHandleCore, ObjectHandleRef
from bfpacked import PackedU32Core, PackedU32Ref


def _u32(memory: list[int], ref: PackedU32Ref) -> int:
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(4))


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

    result = run_bf(bf.code(), memory_size=512, step_limit=150_000_000)
    assert _u32(result.memory, a) == 1
    assert _u32(result.memory, b) == 1
    assert _u32(result.memory, c) == 2
    assert _u32(result.memory, observed_before_clear) == 5
    assert _u32(result.memory, observed_after_clear) == 0
    assert _u32(result.memory, separate_length) == 0
