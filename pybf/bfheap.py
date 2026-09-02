"""Runtime-walked fixed-stride heap blocks for mutable Python objects.

Allocated blocks form one contiguous marker run. Heap allocation and lookup are
emitted as lane-walking Brainfuck loops, so generated source size is independent
of the number of objects created at runtime.

Object identities/header counters are packed u32 values.  Each block also owns
an eight-byte payload, allowing one signed int64 value to be stored compactly
and converted at the compiler/runtime boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter
from bfobjects import ObjectHandleCore, ObjectHandleRef
from bfpacked import PackedU32Ref
from bfpacked64 import PackedI64Core, PackedI64Ref


BLOCK_STRIDE = 42
MARKER = 0
HANDLE = 1          # 4 bytes: 1..4
TYPE = 5            # one byte
LENGTH = 6          # 4 packed bytes: 6..9
CAPACITY = 10       # 4 packed bytes: 10..13
NEXT = 14           # 4-byte handle: 14..17
PAYLOAD = 18        # 8 bytes: 18..25
CARRIER = 26        # 4-byte traveling query/allocated handle: 26..29
LOCAL0 = 30         # match flag
LOCAL1 = 31
LOCAL2 = 32
LOCAL3 = 33
RESULT = 34         # 8-byte traveling read/write payload: 34..41
RESULT_BYTES = 8

_U32_FIELDS = (LENGTH, CAPACITY, NEXT)


@dataclass(frozen=True)
class HeapBlockRef:
    marker: int

    @property
    def handle(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + HANDLE)

    @property
    def carrier(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + CARRIER)

    @property
    def result_carrier(self) -> PackedI64Ref:
        return PackedI64Ref(self.marker + RESULT)

    @property
    def type_cell(self) -> int:
        return self.marker + TYPE

    @property
    def length(self) -> PackedU32Ref:
        return PackedU32Ref(self.marker + LENGTH)

    @property
    def capacity(self) -> PackedU32Ref:
        return PackedU32Ref(self.marker + CAPACITY)

    @property
    def next_handle(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + NEXT)

    @property
    def payload(self) -> PackedI64Ref:
        return PackedI64Ref(self.marker + PAYLOAD)


class _RelativeBuilder:
    def __init__(self) -> None:
        self.pos = 0
        self.parts: list[str] = []

    def move(self, target: int) -> None:
        delta = target - self.pos
        if delta > 0:
            self.parts.append(">" * delta)
        elif delta < 0:
            self.parts.append("<" * -delta)
        self.pos = target

    def add(self, target: int, amount: int) -> None:
        self.move(target)
        amount %= 256
        if amount <= 128:
            self.parts.append("+" * amount)
        else:
            self.parts.append("-" * (256 - amount))

    def clear(self, target: int) -> None:
        self.move(target)
        self.parts.append("[-]")

    def transfer(self, src: int, dst: int) -> None:
        self.move(src)
        self.parts.append("[")
        self.parts.append("-")
        self.move(dst)
        self.parts.append("+")
        self.move(src)
        self.parts.append("]")

    def copy_preserved(self, src: int, dst: int, tmp: int) -> None:
        self.clear(dst)
        self.clear(tmp)
        self.move(src)
        self.parts.append("[")
        self.parts.append("-")
        self.add(dst, 1)
        self.add(tmp, 1)
        self.move(src)
        self.parts.append("]")
        self.move(tmp)
        self.parts.append("[")
        self.parts.append("-")
        self.add(src, 1)
        self.move(tmp)
        self.parts.append("]")

    def code(self) -> str:
        return "".join(self.parts)


class HeapBlockArena:
    """Monotonic heap with stable 32-bit object identity."""

    def __init__(
        self,
        bf: BFEmitter,
        *,
        left_sentinel: int,
        next_handle: ObjectHandleRef,
        scratch_base: int,
    ) -> None:
        self.bf = bf
        self.left_sentinel = left_sentinel
        self.first_block = HeapBlockRef(left_sentinel + BLOCK_STRIDE)
        self.next_handle = next_handle
        self.handles = ObjectHandleCore(bf, scratch_base)
        self.packed64 = PackedI64Core(bf, scratch_base)

    def initialize(self) -> None:
        self.bf.clear(self.left_sentinel)
        self.handles.set_u32(self.next_handle, 1)

    # ------------------------------------------------------------------
    # scan helpers
    # ------------------------------------------------------------------
    def _emit_match_query(self, r: _RelativeBuilder) -> None:
        r.clear(LOCAL0)
        r.add(LOCAL0, 1)
        for i in range(4):
            r.copy_preserved(HANDLE + i, LOCAL1, LOCAL3)
            r.copy_preserved(CARRIER + i, LOCAL2, LOCAL3)
            r.move(LOCAL2)
            r.parts.append("[")
            r.parts.append("-")
            r.add(LOCAL1, -1)
            r.move(LOCAL2)
            r.parts.append("]")
            r.move(LOCAL1)
            r.parts.append("[")
            r.clear(LOCAL1)
            r.clear(LOCAL0)
            r.move(LOCAL1)
            r.parts.append("]")

    def _emit_advance(self, r: _RelativeBuilder) -> None:
        for i in range(4):
            r.transfer(CARRIER + i, BLOCK_STRIDE + CARRIER + i)
        for i in range(RESULT_BYTES):
            r.transfer(RESULT + i, BLOCK_STRIDE + RESULT + i)
        r.move(BLOCK_STRIDE)

    def _return_result_from_frontier_body(self) -> str:
        r = _RelativeBuilder()
        for i in range(4):
            r.clear(CARRIER + i)
        for i in range(RESULT_BYTES):
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE)
        r.parts.append("[")
        for i in range(RESULT_BYTES):
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE)
        r.parts.append("]")
        return r.code()

    def _scan_start(self, handle: ObjectHandleRef) -> None:
        self.handles.copy(self.first_block.carrier, handle)
        self.packed64.clear(self.first_block.result_carrier)

    def _scan_finish(self) -> None:
        self.bf.emit(self._return_result_from_frontier_body())
        self.bf.ptr = self.left_sentinel

    def _fixed_result_i64(self) -> PackedI64Ref:
        return PackedI64Ref(self.left_sentinel + RESULT)

    def _fixed_result_u32(self) -> PackedU32Ref:
        return PackedU32Ref(self.left_sentinel + RESULT)

    def _clear_fixed_result(self) -> None:
        self.packed64.clear(self._fixed_result_i64())

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------
    def _forward_walk_body(self) -> str:
        r = _RelativeBuilder()
        for i in range(4):
            r.transfer(CARRIER + i, BLOCK_STRIDE + CARRIER + i)
        r.move(BLOCK_STRIDE)
        return r.code()

    def _allocate_at_frontier_body(self, type_tag: int) -> str:
        if not 0 < type_tag < 256:
            raise ValueError("type_tag must be in 1..255")
        r = _RelativeBuilder()
        r.add(MARKER, 1)
        for i in range(4):
            r.transfer(CARRIER + i, HANDLE + i)
        r.add(TYPE, type_tag)
        r.move(MARKER)
        r.parts.append("[")
        r.move(-BLOCK_STRIDE)
        r.parts.append("]")
        return r.code()

    def allocate(self, dst: ObjectHandleRef, *, type_tag: int) -> None:
        self.handles.copy(dst, self.next_handle)
        self.handles.copy(self.first_block.carrier, self.next_handle)
        self.bf.move(self.first_block.marker)
        self.bf.emit("[")
        self.bf.emit(self._forward_walk_body())
        self.bf.emit("]")
        self.bf.emit(self._allocate_at_frontier_body(type_tag))
        self.bf.ptr = self.left_sentinel
        self.handles.increment(self.next_handle)

    # ------------------------------------------------------------------
    # handle lookup and fields
    # ------------------------------------------------------------------
    def _lookup_read_forward_body(self, field: int, width: int) -> str:
        r = _RelativeBuilder()
        self._emit_match_query(r)
        r.move(LOCAL0)
        r.parts.append("[")
        r.parts.append("-")
        for i in range(width):
            r.copy_preserved(field + i, RESULT + i, LOCAL3)
        r.move(LOCAL0)
        r.parts.append("]")
        self._emit_advance(r)
        return r.code()

    def _lookup_write_forward_body(self, field: int, width: int) -> str:
        r = _RelativeBuilder()
        self._emit_match_query(r)
        r.move(LOCAL0)
        r.parts.append("[")
        r.parts.append("-")
        for i in range(width):
            r.copy_preserved(RESULT + i, field + i, LOCAL3)
        r.move(LOCAL0)
        r.parts.append("]")
        self._emit_advance(r)
        return r.code()

    def _run_read_scan(self, handle: ObjectHandleRef, field: int, width: int) -> None:
        self._scan_start(handle)
        self.bf.move(self.first_block.marker)
        self.bf.emit("[")
        self.bf.emit(self._lookup_read_forward_body(field, width))
        self.bf.emit("]")
        self._scan_finish()

    def _run_write_scan(self, handle: ObjectHandleRef, field: int, width: int) -> None:
        self.bf.move(self.first_block.marker)
        self.bf.emit("[")
        self.bf.emit(self._lookup_write_forward_body(field, width))
        self.bf.emit("]")
        self._scan_finish()

    def read_type(self, dst_cell: int, handle: ObjectHandleRef) -> None:
        self.bf.clear(dst_cell)
        self._run_read_scan(handle, TYPE, 1)
        fixed = self.left_sentinel + RESULT
        self.handles._copy_cell(fixed, dst_cell, self.handles.s0)
        self._clear_fixed_result()
        self.handles._clear_scratch()

    def read_u32(self, dst: PackedU32Ref, handle: ObjectHandleRef, *, field: int) -> None:
        if field not in _U32_FIELDS:
            raise ValueError("field must be LENGTH, CAPACITY, or NEXT")
        self.handles.clear(dst)
        self._run_read_scan(handle, field, 4)
        self.handles.copy(dst, self._fixed_result_u32())
        self._clear_fixed_result()

    def write_u32(self, handle: ObjectHandleRef, src: PackedU32Ref, *, field: int) -> None:
        if field not in _U32_FIELDS:
            raise ValueError("field must be LENGTH, CAPACITY, or NEXT")
        self._scan_start(handle)
        self.handles.copy(PackedU32Ref(self.first_block.marker + RESULT), src)
        self._run_write_scan(handle, field, 4)
        self._clear_fixed_result()

    def read_payload_i64(self, dst: PackedI64Ref, handle: ObjectHandleRef) -> None:
        self.packed64.clear(dst)
        self._run_read_scan(handle, PAYLOAD, 8)
        self.packed64.copy(dst, self._fixed_result_i64())
        self._clear_fixed_result()

    def write_payload_i64(self, handle: ObjectHandleRef, src: PackedI64Ref) -> None:
        self._scan_start(handle)
        self.packed64.copy(self.first_block.result_carrier, src)
        self._run_write_scan(handle, PAYLOAD, 8)
        self._clear_fixed_result()

    def read_length(self, dst: PackedU32Ref, handle: ObjectHandleRef) -> None:
        self.read_u32(dst, handle, field=LENGTH)

    def write_length(self, handle: ObjectHandleRef, src: PackedU32Ref) -> None:
        self.write_u32(handle, src, field=LENGTH)

    def read_capacity(self, dst: PackedU32Ref, handle: ObjectHandleRef) -> None:
        self.read_u32(dst, handle, field=CAPACITY)

    def write_capacity(self, handle: ObjectHandleRef, src: PackedU32Ref) -> None:
        self.write_u32(handle, src, field=CAPACITY)

    def read_next(self, dst: ObjectHandleRef, handle: ObjectHandleRef) -> None:
        self.read_u32(dst, handle, field=NEXT)

    def write_next(self, handle: ObjectHandleRef, src: ObjectHandleRef) -> None:
        self.write_u32(handle, src, field=NEXT)


__all__ = [
    "BLOCK_STRIDE",
    "MARKER",
    "HANDLE",
    "TYPE",
    "LENGTH",
    "CAPACITY",
    "NEXT",
    "PAYLOAD",
    "HeapBlockRef",
    "HeapBlockArena",
]
