from bf_runtime import run_bf
from bfcore import BFEmitter
from bfobjects import ObjectHandleAllocator, ObjectHandleCore, ObjectHandleRef


def _u32(memory: list[int], ref: ObjectHandleRef) -> int:
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(4))


def test_object_handle_allocator_produces_stable_aliasable_ids():
    bf = BFEmitter()
    next_handle = ObjectHandleRef(0)
    a = ObjectHandleRef(4)
    b = ObjectHandleRef(8)
    alias = ObjectHandleRef(12)
    rt = ObjectHandleAllocator(bf, next_handle, scratch_base=32)

    rt.initialize()
    rt.allocate(a)
    rt.allocate(b)
    rt.copy(alias, a)

    result = run_bf(bf.code(), memory_size=128, step_limit=5_000_000)
    assert _u32(result.memory, a) == 1
    assert _u32(result.memory, b) == 2
    assert _u32(result.memory, alias) == 1
    assert _u32(result.memory, next_handle) == 3


def test_object_handle_increment_carries_between_bytes():
    bf = BFEmitter()
    ref = ObjectHandleRef(0)
    rt = ObjectHandleCore(bf, scratch_base=16)

    rt.set_u32(ref, 0x0000FFFF)
    rt.increment(ref)

    result = run_bf(bf.code(), memory_size=64, step_limit=5_000_000)
    assert _u32(result.memory, ref) == 0x00010000


def test_object_handle_equality_is_identity_equality():
    bf = BFEmitter()
    a = ObjectHandleRef(0)
    b = ObjectHandleRef(4)
    c = ObjectHandleRef(8)
    eq_ab = 12
    eq_ac = 13
    rt = ObjectHandleCore(bf, scratch_base=20)

    rt.set_u32(a, 123456)
    rt.set_u32(b, 123456)
    rt.set_u32(c, 123457)
    rt.equal(eq_ab, a, b)
    rt.equal(eq_ac, a, c)

    result = run_bf(bf.code(), memory_size=64, step_limit=10_000_000)
    assert result.memory[eq_ab] == 1
    assert result.memory[eq_ac] == 0
    assert _u32(result.memory, a) == 123456
    assert _u32(result.memory, b) == 123456
    assert _u32(result.memory, c) == 123457
