"""Runtime-walked fixed-stride heap blocks for mutable Python objects.

Allocated blocks form one contiguous marker run.  Heap allocation and lookup
are emitted as small lane-walking Brainfuck loops, so generated source size is
independent of the number of objects created at runtime.

This Phase-1 arena is monotonic.  Free-list/reuse and variable-size list block
chains are layered on top later.  Object identities are packed 32-bit handles
from :mod:`bfobjects`.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter
from bfobjects import ObjectHandleCore, ObjectHandleRef


# Fixed-stride object/block layout.  QUERY and RESULT are traveling cells used
# only while a heap operation scans the marker lane; LOCAL* are per-block
# scratch and always finish zero.
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
RESULT = 34         # 4-byte traveling return payload: 34..37


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
    def length_base(self) -> int:
        return self.marker + LENGTH

    @property
    def capacity_base(self) -> int:
        return self.marker + CAPACITY

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
        """Destructively move an 8-bit cell value to a relative cell."""
        self.move(src)
        self.parts.append("[")
        self.parts.append("-")
        self.move(dst)
        self.parts.append("+")
        self.move(src)
        self.parts.append("]")

    def copy_preserved(self, src: int, dst: int, tmp: int) -> None:
        """dst = src, preserving src; tmp finishes zero."""
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
    """Monotonic heap with stable 32-bit object identity.

    ``left_sentinel`` is a permanently-zero marker.  The first real block is
    one stride to its right.  Every allocated block marker is 1 and the first
    unallocated marker is 0.  Every operation returns the data pointer to the
    left sentinel, giving the surrounding compiler a statically known pointer.
    """

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
        """Initialize heap identity metadata; tape markers rely on zero-init."""
        self.bf.clear(self.left_sentinel)
        self.handles.set_u32(self.next_handle, 1)

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------
    def _forward_walk_body(self) -> str:
        """Move a traveling handle to the next block and advance one marker."""
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
            # Carrier contains the newly issued identity.  Persist it in the
            # header; carrier is zero afterwards.
            r.transfer(CARRIER + i, HANDLE + i)
        r.add(TYPE, type_tag)
        r.move(MARKER)

        # Reverse over the contiguous allocated marker run.  The matching ']'
        # deliberately tests the previous block's marker each iteration.
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

        # Pointer is at the dynamic zero frontier.  Relative code initializes
        # that block and reverse-walks to the known left sentinel.
        self.bf.emit(self._allocate_at_frontier_body(type_tag))
        self.bf.ptr = self.left_sentinel
        self.handles.increment(self.next_handle)

    # ------------------------------------------------------------------
    # handle lookup
    # ------------------------------------------------------------------
    def _lookup_type_forward_body(self) -> str:
        """Scan one active block, carrying query and optional type result."""
        r = _RelativeBuilder()
        r.clear(LOCAL0)
        r.add(LOCAL0, 1)  # match = true

        for i in range(4):
            # LOCAL1 = header_handle[i] - query[i] mod 256.
            r.copy_preserved(HANDLE + i, LOCAL1, LOCAL3)
            r.copy_preserved(CARRIER + i, LOCAL2, LOCAL3)
            r.move(LOCAL2)
            r.parts.append("[")
            r.parts.append("-")
            r.add(LOCAL1, -1)
            r.move(LOCAL2)
            r.parts.append("]")

            # Any non-zero byte difference clears match in one iteration.
            r.move(LOCAL1)
            r.parts.append("[")
            r.clear(LOCAL1)
            r.clear(LOCAL0)
            r.move(LOCAL1)
            r.parts.append("]")

        # A successful match copies the type byte into the traveling result.
        r.move(LOCAL0)
        r.parts.append("[")
        r.parts.append("-")
        r.copy_preserved(TYPE, RESULT, LOCAL3)
        r.move(LOCAL0)
        r.parts.append("]")

        # Query and result travel with the scan to the next block.  Every
        # source carrier is consumed, keeping traversed blocks clean.
        for i in range(4):
            r.transfer(CARRIER + i, BLOCK_STRIDE + CARRIER + i)
            r.transfer(RESULT + i, BLOCK_STRIDE + RESULT + i)
        r.move(BLOCK_STRIDE)
        return r.code()

    def _return_result_from_frontier_body(self) -> str:
        """Carry RESULT from dynamic frontier back to the fixed left sentinel."""
        r = _RelativeBuilder()

        # Query is dead at the frontier.  Result first moves to the previous
        # block (or directly to left sentinel when the heap is empty).
        for i in range(4):
            r.clear(CARRIER + i)
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE)

        # While current marker is active, move result one block farther left.
        r.parts.append("[")
        for i in range(4):
            r.transfer(RESULT + i, -BLOCK_STRIDE + RESULT + i)
        r.move(-BLOCK_STRIDE)
        r.parts.append("]")
        return r.code()

    def read_type(self, dst_cell: int, handle: ObjectHandleRef) -> None:
        """Resolve ``handle`` at runtime and copy its one-byte type tag.

        Unknown/null handles currently return 0.  The operation scans active
        heap blocks at runtime with one emitted loop body and always restores
        the BF pointer to ``left_sentinel``.
        """
        bf = self.bf
        bf.clear(dst_cell)
        self.handles.copy(self.first_block.carrier, handle)

        bf.move(self.first_block.marker)
        bf.emit("[")
        bf.emit(self._lookup_type_forward_body())
        bf.emit("]")

        # Pointer is now at the dynamic frontier.  Return the result through the
        # per-block RESULT lane to the fixed sentinel.
        bf.emit(self._return_result_from_frontier_body())
        bf.ptr = self.left_sentinel

        fixed_result = self.left_sentinel + RESULT
        self.handles._copy_cell(fixed_result, dst_cell, self.handles.s0)
        bf.clear(fixed_result)
        for i in range(1, 4):
            bf.clear(self.left_sentinel + RESULT + i)
        self.handles._clear_scratch()


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
