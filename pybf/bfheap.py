"""Runtime-walked fixed-stride heap blocks for mutable Python objects.

The heap is deliberately built around Brainfuck's strengths: allocated blocks
form one contiguous run whose first cell is a marker.  A single emitted BF loop
walks those markers at runtime until the first zero marker (the bump frontier),
so generated source size does not grow with the number of objects allocated at
runtime.

This Phase-1 arena is monotonic.  Free-list/reuse and variable-size list block
chains are layered on top later.  Object identities are 32-bit packed handles
from :mod:`bfobjects`.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter
from bfobjects import ObjectHandleCore, ObjectHandleRef


# One block is intentionally small and regular.  The first payload version has
# room for one future packed int64 lane while the metadata already matches the
# planned dynamic-list/object header.
BLOCK_STRIDE = 34
MARKER = 0
HANDLE = 1          # 4 bytes: 1..4
TYPE = 5            # one byte
LENGTH = 6          # 4 packed bytes: 6..9
CAPACITY = 10       # 4 packed bytes: 10..13
NEXT = 14           # 4-byte handle: 14..17
PAYLOAD = 18        # 8 bytes: 18..25
CARRIER = 26        # 4-byte traveling handle: 26..29
LOCAL0 = 30
LOCAL1 = 31
LOCAL2 = 32
LOCAL3 = 33


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

    def transfer(self, src: int, dst: int) -> None:
        """Destructively move an 8-bit cell value to a known-relative cell."""
        self.move(src)
        self.parts.append("[")
        self.parts.append("-")
        self.move(dst)
        self.parts.append("+")
        self.move(src)
        self.parts.append("]")

    def code(self) -> str:
        return "".join(self.parts)


class HeapBlockArena:
    """Monotonic heap whose block frontier is discovered at BF runtime.

    ``left_sentinel`` is a permanently-zero marker.  The first real block is
    one stride to its right.  Every allocated block marker is 1 and the first
    unallocated marker is 0.  The allocator walks right to that zero marker,
    initializes it, then walks left over the allocated-marker run to restore the
    Brainfuck data pointer to ``left_sentinel``.  Therefore callers regain a
    statically known pointer after every allocation.
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

    def _forward_walk_body(self) -> str:
        """Move a traveling handle to the next block and advance one marker."""
        r = _RelativeBuilder()
        for i in range(4):
            r.transfer(CARRIER + i, BLOCK_STRIDE + CARRIER + i)
        r.move(BLOCK_STRIDE)
        return r.code()

    def _allocate_at_frontier_body(self, type_tag: int) -> str:
        """Initialize the zero frontier block, then return to left sentinel."""
        if not 0 < type_tag < 256:
            raise ValueError("type_tag must be in 1..255")

        r = _RelativeBuilder()
        r.add(MARKER, 1)
        for i in range(4):
            # The carrier contains the newly issued identity.  Move it into the
            # persistent block header; the carrier is zero afterwards.
            r.transfer(CARRIER + i, HANDLE + i)
        r.add(TYPE, type_tag)
        r.move(MARKER)

        # All allocated markers, including this one, are exactly 1.  Leaving
        # them untouched lets one textual loop walk left across the whole run.
        r.parts.append("[")
        r.move(-BLOCK_STRIDE)
        r.parts.append("]")
        return r.code()

    def allocate(self, dst: ObjectHandleRef, *, type_tag: int) -> None:
        """Allocate one block and return its stable object handle in ``dst``."""
        # Return identity is available before the dynamic pointer walk.
        self.handles.copy(dst, self.next_handle)
        self.handles.copy(self.first_block.carrier, self.next_handle)

        # Runtime walk over the contiguous active-marker run.  The body is
        # emitted once regardless of how many blocks have already been used.
        self.bf.move(self.first_block.marker)
        self.bf.emit("[")
        self.bf.emit(self._forward_walk_body())
        self.bf.emit("]")

        # Pointer is now at the dynamic zero frontier.  Emit only relative code
        # until the reverse marker-walk reaches the known left sentinel again.
        self.bf.emit(self._allocate_at_frontier_body(type_tag))
        self.bf.ptr = self.left_sentinel

        self.handles.increment(self.next_handle)


__all__ = [
    "BLOCK_STRIDE",
    "HeapBlockRef",
    "HeapBlockArena",
]
