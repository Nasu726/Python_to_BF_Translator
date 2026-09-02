"""Runtime-walked fixed-stride heap blocks for mutable Python objects.

The current monotonic allocator gives every object a 1-based ordinal handle:
handle 1 is the first heap block, handle 2 the second, and so on. Lookup uses
that fact directly. A packed handle is carried across blocks and decremented
at runtime until it reaches zero; the old design compared four handle bytes in
every allocated block and scanned the entire heap for every access.

Handle 0 is the null sentinel and reads as zero / writes as a no-op. Arbitrary
forged handles above the allocation frontier are outside this internal runtime
contract. A future free-list allocator can replace ordinal lookup with an
indirection table without changing frontend object semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter
from bfobjects import ObjectHandleCore, ObjectHandleRef
from bfpacked import PackedU32Ref
from bfpacked64 import PackedI64Core, PackedI64Ref


BLOCK_STRIDE = 48
MARKER = 0
HANDLE = 1
TYPE = 5
LENGTH = 6
CAPACITY = 10
NEXT = 14
PAYLOAD = 18
CARRIER = 26
LOCAL0 = 30
LOCAL1 = 31
LOCAL2 = 32
LOCAL3 = 33
RESULT = 34
AUX0 = 42
AUX1 = 43
AUX2 = 44
AUX3 = 45
AUX4 = 46
AUX5 = 47
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
    def __init__(self, initial_pos: int = 0) -> None:
        self.pos = initial_pos
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
    """Monotonic heap with source-compact direct ordinal lookup."""

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
    # direct ordinal lookup
    # ------------------------------------------------------------------
    def _relative_is_zero_u32(
        self,
        r: _RelativeBuilder,
        base: int,
        result: int,
        tmp: int,
        helper: int,
    ) -> None:
        r.clear(result)
        r.add(result, 1)
        for i in range(4):
            r.copy_preserved(base + i, tmp, helper)
            r.move(tmp)
            r.parts.append("[")
            r.clear(tmp)
            r.clear(result)
            r.move(tmp)
            r.parts.append("]")

    def _relative_is_zero_u32_byte(
        self,
        r: _RelativeBuilder,
        cell: int,
        result: int,
        tmp: int,
        helper: int,
    ) -> None:
        r.clear(result)
        r.add(result, 1)
        r.copy_preserved(cell, tmp, helper)
        r.move(tmp)
        r.parts.append("[")
        r.clear(tmp)
        r.clear(result)
        r.move(tmp)
        r.parts.append("]")

    def _relative_decrement_u32(self, r: _RelativeBuilder, base: int) -> None:
        borrow, gate, tmp, helper = AUX0, AUX1, AUX2, AUX3
        r.clear(borrow)
        r.add(borrow, 1)
        for i in range(4):
            r.transfer(borrow, gate)
            r.move(gate)
            r.parts.append("[")
            r.add(gate, -1)
            self._relative_is_zero_u32_byte(r, base + i, borrow, tmp, helper)
            r.add(base + i, -1)
            r.move(gate)
            r.parts.append("]")
        r.clear(borrow)

    def _direct_lookup_body(self, field: int, width: int, *, write: bool) -> str:
        r = _RelativeBuilder(initial_pos=LOCAL0)
        r.clear(LOCAL0)
        r.clear(LOCAL1)
        r.clear(LOCAL2)

        # Current block must be allocated. A zero marker is the frontier and
        # therefore terminates an invalid internal ordinal without searching
        # further to the right.
        r.copy_preserved(MARKER, LOCAL3, AUX5)
        r.move(LOCAL3)
        r.parts.append("[")
        r.add(LOCAL3, -1)

        # Null (0) is never a target. Positive handles count allocated blocks.
        self._relative_is_zero_u32(r, CARRIER, AUX4, AUX2, AUX5)
        r.clear(AUX1)
        r.add(AUX1, 1)
        r.move(AUX4)
        r.parts.append("[")
        r.add(AUX4, -1)
        r.clear(AUX1)
        r.move(AUX4)
        r.parts.append("]")

        # Positive-handle gate.
        r.move(AUX1)
        r.parts.append("[")
        r.add(AUX1, -1)
        self._relative_decrement_u32(r, CARRIER)
        self._relative_is_zero_u32(r, CARRIER, LOCAL1, AUX4, AUX5)

        # continue = not found
        r.clear(LOCAL2)
        r.add(LOCAL2, 1)
        r.copy_preserved(LOCAL1, AUX4, AUX5)
        r.move(AUX4)
        r.parts.append("[")
        r.add(AUX4, -1)
        r.clear(LOCAL2)
        r.move(AUX4)
        r.parts.append("]")

        # Target operation.
        r.move(LOCAL1)
        r.parts.append("[")
        r.add(LOCAL1, -1)
        for i in range(width):
            if write:
                r.copy_preserved(RESULT + i, field + i, AUX4)
            else:
                r.copy_preserved(field + i, RESULT + i, AUX4)
        r.move(LOCAL1)
        r.parts.append("]")

        # Non-target: move remaining ordinal and arm next block.
        r.move(LOCAL2)
        r.parts.append("[")
        r.add(LOCAL2, -1)
        for i in range(4):
            r.transfer(CARRIER + i, BLOCK_STRIDE + CARRIER + i)
        r.add(BLOCK_STRIDE + LOCAL0, 1)
        r.move(LOCAL2)
        r.parts.append("]")

        # Close positive-handle gate before closing the marker gate.
        r.move(AUX1)
        r.parts.append("]")
        r.move(LOCAL3)
        r.parts.append("]")

        # RESULT always advances one block. On a hit, next LOCAL0 is zero so
        # the outer walker exits at that next block.
        for i in range(width):
            r.transfer(RESULT + i, BLOCK_STRIDE + RESULT + i)
        r.move(BLOCK_STRIDE + LOCAL0)
        return r.code()

    def _return_result_from_after_target(self, width: int) -> str:
        """Carry a lookup result back to the fixed left sentinel.

        The first transfer moves from the block *after* the target into the
        target block.  The following BF loop then walks one block left per
        iteration.  Its body must be expressed relative to the marker reached
        at runtime; resetting the builder origin here is essential.  Without
        that reset, the second iteration reused offsets relative to the
        original after-target block and bounced between incorrect cells.
        """
        r = _RelativeBuilder(initial_pos=LOCAL0)
        for i in range(width):
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE + MARKER)
        r.parts.append("[")

        # The loop-local origin is now the current block marker.  Rebase the
        # coordinate tracker without emitting movement so the fixed loop body
        # uses +RESULT and -BLOCK_STRIDE deltas on every runtime iteration.
        r.pos = MARKER
        for i in range(width):
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE + MARKER)
        r.parts.append("]")
        return r.code()

    def _scan_start(self, handle: ObjectHandleRef, width: int) -> None:
        self.handles.copy(self.first_block.carrier, handle)
        for i in range(width):
            self.bf.clear(self.first_block.marker + RESULT + i)

    def _run_scan(self, handle: ObjectHandleRef, field: int, width: int, *, write: bool) -> None:
        if not 1 <= width <= RESULT_BYTES:
            raise ValueError("scan width must be in 1..8")
        self.bf.set_const(self.first_block.marker + LOCAL0, 1)
        self.bf.move(self.first_block.marker + LOCAL0)
        self.bf.emit("[")
        self.bf.emit(self._direct_lookup_body(field, width, write=write))
        self.bf.emit("]")
        self.bf.emit(self._return_result_from_after_target(width))
        self.bf.ptr = self.left_sentinel

    def _fixed_result_i64(self) -> PackedI64Ref:
        return PackedI64Ref(self.left_sentinel + RESULT)

    def _fixed_result_u32(self) -> PackedU32Ref:
        return PackedU32Ref(self.left_sentinel + RESULT)

    def _clear_fixed_result(self) -> None:
        self.packed64.clear(self._fixed_result_i64())

    # ------------------------------------------------------------------
    # public lookup / metadata operations
    # ------------------------------------------------------------------
    def read_type(self, dst_cell: int, handle: ObjectHandleRef) -> None:
        self.bf.clear(dst_cell)
        self._scan_start(handle, 1)
        self._run_scan(handle, TYPE, 1, write=False)
        fixed = self.left_sentinel + RESULT
        self.handles._copy_cell(fixed, dst_cell, self.handles.s0)
        self.bf.clear(fixed)
        self.handles._clear_scratch()

    def read_u32(self, dst: PackedU32Ref, handle: ObjectHandleRef, *, field: int) -> None:
        if field not in _U32_FIELDS:
            raise ValueError("field must be LENGTH, CAPACITY, or NEXT")
        self.handles.clear(dst)
        self._scan_start(handle, 4)
        self._run_scan(handle, field, 4, write=False)
        self.handles.copy(dst, self._fixed_result_u32())
        for i in range(4):
            self.bf.clear(self.left_sentinel + RESULT + i)

    def write_u32(self, handle: ObjectHandleRef, src: PackedU32Ref, *, field: int) -> None:
        if field not in _U32_FIELDS:
            raise ValueError("field must be LENGTH, CAPACITY, or NEXT")
        self._scan_start(handle, 4)
        self.handles.copy(PackedU32Ref(self.first_block.marker + RESULT), src)
        self._run_scan(handle, field, 4, write=True)
        for i in range(4):
            self.bf.clear(self.left_sentinel + RESULT + i)

    def read_payload_i64(self, dst: PackedI64Ref, handle: ObjectHandleRef) -> None:
        self.packed64.clear(dst)
        self._scan_start(handle, 8)
        self._run_scan(handle, PAYLOAD, 8, write=False)
        self.packed64.copy(dst, self._fixed_result_i64())
        self._clear_fixed_result()

    def write_payload_i64(self, handle: ObjectHandleRef, src: PackedI64Ref) -> None:
        self._scan_start(handle, 8)
        self.packed64.copy(self.first_block.result_carrier, src)
        self._run_scan(handle, PAYLOAD, 8, write=True)
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
