from bf_runtime import run_bf
from bfcore import BFEmitter, Binary64Core, Int64Ref
from bfpacked64 import PackedI64Core, PackedI64Ref


MASK64 = (1 << 64) - 1


def _bits_to_u64(memory: list[int], ref: Int64Ref) -> int:
    return sum((memory[ref.bit(i)] & 1) << i for i in range(64))


def _bytes_to_u64(memory: list[int], ref: PackedI64Ref) -> int:
    return sum(memory[ref.byte(i)] << (8 * i) for i in range(8))


def _round_trip(value: int) -> tuple[int, int, int]:
    bf = BFEmitter()
    source = Int64Ref(0)
    restored = Int64Ref(64)
    packed = PackedI64Ref(128)
    binary = Binary64Core(bf, scratch_base=144)
    storage = PackedI64Core(bf, scratch_base=152)

    binary.set_u64(source, value)
    storage.from_int64(packed, source)
    storage.to_int64(restored, packed)

    result = run_bf(bf.code(), memory_size=256, step_limit=100_000_000)
    return (
        _bits_to_u64(result.memory, source),
        _bytes_to_u64(result.memory, packed),
        _bits_to_u64(result.memory, restored),
    )


def test_packed_int64_round_trip_edge_values():
    for value in (0, 1, 255, 256, 0x123456789ABCDEF0, -1, -(1 << 63)):
        source, packed, restored = _round_trip(value)
        expected = value & MASK64
        assert source == expected
        assert packed == expected
        assert restored == expected
