from bf_runtime import run_bf
from bfcore import BFEmitter
from bfheap import BLOCK_STRIDE, HANDLE, MARKER, TYPE, HeapBlockArena
from bfobjects import ObjectHandleRef


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
