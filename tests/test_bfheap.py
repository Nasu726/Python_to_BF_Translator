from bf_runtime import run_bf
from bfcore import BFEmitter
from bfheap import BLOCK_STRIDE, HANDLE, MARKER, TYPE, HeapBlockArena
from bfobjects import ObjectHandleCore, ObjectHandleRef


def _u32(memory: list[int], base: int) -> int:
    return sum(memory[base + i] << (8 * i) for i in range(4))


def test_heap_allocates_blocks_at_runtime_and_returns_to_anchor():
    bf = BFEmitter()
    next_handle = ObjectHandleRef(0)
    last_handle = ObjectHandleRef(4)
    loop_count = 8
    left_sentinel = 64
    arena = HeapBlockArena(
        bf,
        left_sentinel=left_sentinel,
        next_handle=next_handle,
        scratch_base=16,
    )

    arena.initialize()
    bf.set_const(loop_count, 3)
    bf.begin_while(loop_count)
    bf.add_const(loop_count, -1)
    # One textual allocator is executed three times at BF runtime.  This is the
    # property needed for list/object creation inside Python loops later.
    arena.allocate(last_handle, type_tag=7)
    bf.end_while(loop_count)

    result = run_bf(bf.code(), memory_size=512, step_limit=20_000_000)
    assert _u32(result.memory, next_handle.base) == 4
    assert _u32(result.memory, last_handle.base) == 3

    for index, expected_handle in enumerate((1, 2, 3)):
        marker = left_sentinel + (index + 1) * BLOCK_STRIDE
        assert result.memory[marker + MARKER] == 1
        assert _u32(result.memory, marker + HANDLE) == expected_handle
        assert result.memory[marker + TYPE] == 7

    frontier = left_sentinel + 4 * BLOCK_STRIDE
    assert result.memory[frontier + MARKER] == 0


def test_heap_type_tag_is_stored_in_each_object_header():
    bf = BFEmitter()
    next_handle = ObjectHandleRef(0)
    a = ObjectHandleRef(4)
    b = ObjectHandleRef(8)
    left_sentinel = 48
    arena = HeapBlockArena(
        bf,
        left_sentinel=left_sentinel,
        next_handle=next_handle,
        scratch_base=16,
    )

    arena.initialize()
    arena.allocate(a, type_tag=1)
    arena.allocate(b, type_tag=2)

    result = run_bf(bf.code(), memory_size=256, step_limit=10_000_000)
    first = left_sentinel + BLOCK_STRIDE
    second = first + BLOCK_STRIDE
    assert result.memory[first + TYPE] == 1
    assert result.memory[second + TYPE] == 2
    assert _u32(result.memory, first + HANDLE) == 1
    assert _u32(result.memory, second + HANDLE) == 2


def test_heap_resolves_type_through_runtime_handle_and_alias():
    bf = BFEmitter()
    next_handle = ObjectHandleRef(0)
    a = ObjectHandleRef(4)
    b = ObjectHandleRef(8)
    alias = ObjectHandleRef(12)
    type_a = 20
    type_alias = 21
    type_missing = 22
    missing = ObjectHandleRef(24)
    left_sentinel = 64
    arena = HeapBlockArena(
        bf,
        left_sentinel=left_sentinel,
        next_handle=next_handle,
        scratch_base=32,
    )
    handles = ObjectHandleCore(bf, scratch_base=32)

    arena.initialize()
    arena.allocate(a, type_tag=11)
    arena.allocate(b, type_tag=22)
    handles.copy(alias, b)
    handles.set_u32(missing, 999)

    arena.read_type(type_a, a)
    arena.read_type(type_alias, alias)
    arena.read_type(type_missing, missing)

    result = run_bf(bf.code(), memory_size=512, step_limit=50_000_000)
    assert result.memory[type_a] == 11
    assert result.memory[type_alias] == 22
    assert result.memory[type_missing] == 0
    # Lookup must preserve the aliasing handle values themselves.
    assert _u32(result.memory, b.base) == 2
    assert _u32(result.memory, alias.base) == 2
