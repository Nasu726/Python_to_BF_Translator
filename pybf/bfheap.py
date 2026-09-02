"""Runtime-walked fixed-stride heap blocks for mutable Python objects.

Allocated blocks form one contiguous marker run. Heap allocation and lookup are
emitted as lane-walking Brainfuck loops, so generated source size is independent
of the number of objects created at runtime.

This Phase-1 arena is monotonic. Free-list/reuse and variable-size list block
chains are layered on top later. Object identities and header counters use
packed 32-bit cells rather than the bit-per-cell int64 arithmetic layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter
from bfobjects import ObjectHandleCore, ObjectHandleRef


BLOCK_STRIDE = 38
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
RESULT = 34         # 4-byte traveling read/write payload: 34..37

_U32_FIELDS = (LENGTH, CAPACITY, NEXT)


@dataclass(frozen=True)
class HeapBlockRef:
    """Compile-time reference to a statically known block position."""

    marker: int

    @property
    def handle(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + HANDLE)

    @property
    def carrier(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + CARRIER)

    @property
    def result_carrier(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + RESULT)

    @property
    def type_cell(self) -> int:
        return self.marker + TYPE

    @property
    def length(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + LENGTH)

    @property
    def capacity(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + CAPACITY)

    @property
    def next_handle(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.marker + NEXT)

    @property
    def payload_base(self) -> int:
        return self.marker + PAYLOAD


class _RelativeBuilder:
    """Build code relative to the marker of the current runtime block."""

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

    def initialize(self) -> None:
        self.bf.clear(self.left_sentinel)
        self.handles.set_u32(self.next_handle, 1)

    # ------------------------------------------------------------------
    # relative scan helpers
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
            r.transfer(RESULT + i, BLOCK_STRIDE + RESULT + i)
        r.move(BLOCK_STRIDE)

    def _return_result_from_frontier_body(self) -> str:
        r = _RelativeBuilder()
        for i in range(4):
            r.clear(CARRIER + i)
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE)
        r.parts.append("[")
        for i in range(4):
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE)
        r.parts.append("]")
        return r.code()

    def _scan_start(self, handle: ObjectHandleRef, payload: ObjectHandleRef | None = None) -> None:
        self.handles.copy(self.first_block.carrier, handle)
        self.handles.clear(self.first_block.result_carrier)
        if payload is not None:
            self.handles.copy(self.first_block.result_carrier, payload)

    def _scan_finish(self) -> None:
        self.bf.emit(self._return_result_from_frontier_body())
        self.bf.ptr = self.left_sentinel

    def _clear_fixed_result(self) -> None:
        self.handles.clear(ObjectHandleRef(self.left_sentinel + RESULT))

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
        """Allocate one block and return its stable object handle in ``dst``."""
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
    # handle lookup and header metadata
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

    def _lookup_write_u32_forward_body(self, field: int) -> str:
        r = _RelativeBuilder()
        self._emit_match_query(r)
        r.move(LOCAL0)
        r.parts.append("[")
        r.parts.append("-")
        for i in range(4):
            # RESULT is the traveling write payload and must survive until the
            # scan reaches the frontier, so write by preserved copy.
            r.copy_preserved(RESULT + i, field + i, LOCAL3)
        r.move(LOCAL0)
        r.parts.append("]")
        self._emit_advance(r)
        return r.code()

    def read_type(self, dst_cell: int, handle: ObjectHandleRef) -> None:
        """Resolve ``handle`` at runtime and read its one-byte type tag."""
        bf = self.bf
        bf.clear(dst_cell)
        self._scan_start(handle)
        bf.move(self.first_block.marker)
        bf.emit("[")
        bf.emit(self._lookup_read_forward_body(TYPE, 1))
        bf.emit("]")
        self._scan_finish()
        fixed_result = self.left_sentinel + RESULT
        self.handles._copy_cell(fixed_result, dst_cell, self.handles.s0)
        self._clear_fixed_result()
        self.handles._clear_scratch()

    def read_u32(self, dst: ObjectHandleRef, handle: ObjectHandleRef, *, field: int) -> None:
        """Read a four-byte header field through a runtime object handle."""
        if field not in _U32_FIELDS:
            raise ValueError("field must be LENGTH, CAPACITY, or NEXT")
        self.handles.clear(dst)
        self._scan_start(handle)
        self.bf.move(self.first_block.marker)
        self.bf.emit("[")
        self.bf.emit(self._lookup_read_forward_body(field, 4))
        self.bf.emit("]")
        self._scan_finish()
        fixed_result = ObjectHandleRef(self.left_sentinel + RESULT)
        self.handles.copy(dst, fixed_result)
        self._clear_fixed_result()

    def write_u32(self, handle: ObjectHandleRef, src: ObjectHandleRef, *, field: int) -> None:
        """Write a four-byte header field through a runtime object handle."""
        if field not in _U32_FIELDS:
            raise ValueError("field must be LENGTH, CAPACITY, or NEXT")
        self._scan_start(handle, src)
        self.bf.move(self.first_block.marker)
        self.bf.emit("[")
        self.bf.emit(self._lookup_write_u32_forward_body(field))
        self.bf.emit("]")
        self._scan_finish()
        self._clear_fixed_result()

    def read_length(self, dst: ObjectHandleRef, handle: ObjectHandleRef) -> None:
        self.read_u32(dst, handle, field=LENGTH)

    def write_length(self, handle: ObjectHandleRef, src: ObjectHandleRef) -> None:
        self.write_u32(handle, src, field=LENGTH)

    def read_capacity(self, dst: ObjectHandleRef, handle: ObjectHandleRef) -> None:
        self.read_u32(dst, handle, field=CAPACITY)

    def write_capacity(self, handle: ObjectHandleRef, src: ObjectHandleRef) -> None:
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
