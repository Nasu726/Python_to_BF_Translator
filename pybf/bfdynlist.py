"""Heap-backed dynamic-list root objects.

This is the first mutable Python container using the Phase-1 object model.  A
list variable stores only a 32-bit object handle.  The list's mutable metadata
lives in the heap header, so copying a handle creates Python-style aliasing
rather than copying list contents.

Element data blocks are the next layer; this module intentionally establishes
identity/length/capacity semantics first.
"""

from __future__ import annotations

from bfheap import HeapBlockArena
from bfobjects import ObjectHandleCore, ObjectHandleRef
from bfpacked import PackedU32Core, PackedU32Ref


TYPE_LIST = 1


class DynamicIntListRootRuntime:
    """Identity and header operations for future dynamic ``list[int]`` objects."""

    def __init__(
        self,
        heap: HeapBlockArena,
        *,
        packed: PackedU32Core,
        handles: ObjectHandleCore,
    ) -> None:
        self.heap = heap
        self.packed = packed
        self.handles = handles

    def create_empty(self, dst: ObjectHandleRef) -> None:
        """Allocate a new empty list object and return its handle."""
        self.heap.allocate(dst, type_tag=TYPE_LIST)
        # Fresh bump-allocated headers are zero-initialized, so length,
        # capacity and NEXT already represent an empty list.

    def alias(self, dst: ObjectHandleRef, src: ObjectHandleRef) -> None:
        """Python assignment semantics: dst and src point to the same list."""
        self.handles.copy(dst, src)

    def read_length(self, dst: PackedU32Ref, ref: ObjectHandleRef) -> None:
        self.heap.read_length(dst, ref)

    def write_length(self, ref: ObjectHandleRef, src: PackedU32Ref) -> None:
        self.heap.write_length(ref, src)

    def read_capacity(self, dst: PackedU32Ref, ref: ObjectHandleRef) -> None:
        self.heap.read_capacity(dst, ref)

    def write_capacity(self, ref: ObjectHandleRef, src: PackedU32Ref) -> None:
        self.heap.write_capacity(ref, src)

    def clear(self, ref: ObjectHandleRef, zero: PackedU32Ref) -> None:
        """Logical list clear: length becomes zero; storage can be reused later."""
        self.packed.clear(zero)
        self.heap.write_length(ref, zero)


__all__ = ["TYPE_LIST", "DynamicIntListRootRuntime"]
