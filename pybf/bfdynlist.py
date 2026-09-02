"""Heap-backed dynamic ``list[int]`` primitives.

A list variable stores only a 32-bit object handle. The root object owns
length/capacity/head/tail metadata; each current correctness-first element uses
one heap block containing an eight-byte packed int64 payload. Emitted Brainfuck
source size is independent of runtime list length and Python assignment aliases
the same mutable object.

Root layout conventions:
- header ``NEXT``: first element-block handle (head)
- header ``LENGTH``: runtime element count
- header ``CAPACITY``: currently allocated element count
- first four bytes of root ``PAYLOAD``: last element-block handle (tail)

Element blocks use ``NEXT`` as the linked-list pointer and ``PAYLOAD`` as the
packed int64 value. A later chunked-block optimization can replace the
one-element-per-block representation without changing frontend semantics.
"""

from __future__ import annotations

from bfheap import HeapBlockArena
from bfobjects import ObjectHandleCore, ObjectHandleRef
from bfpacked import PackedU32Core, PackedU32Ref
from bfpacked64 import PackedI64Core, PackedI64Ref


TYPE_LIST = 1
TYPE_LIST_INT_ELEMENT = 2


class DynamicIntListRootRuntime:
    """Identity and root-header operations for heap-backed ``list[int]``."""

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
        self.heap.allocate(dst, type_tag=TYPE_LIST)

    def alias(self, dst: ObjectHandleRef, src: ObjectHandleRef) -> None:
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
        """Logical clear; old element blocks are unreachable until reuse exists."""
        self.packed.clear(zero)
        self.heap.write_length(ref, zero)
        self.heap.write_capacity(ref, zero)
        self.heap.write_next(ref, ObjectHandleRef(zero.base))
        # The old tail payload may remain stale: LENGTH==0 makes it unreachable,
        # and the next append replaces both head and tail before exposing data.


class DynamicIntListRuntime(DynamicIntListRootRuntime):
    """Append and indexed access for linked heap-backed integer lists.

    ``workspace_base`` reserves a reusable private region. Public operands and
    destinations must live outside this region. No workspace state escapes an
    operation, so the same cells can be reused by calls emitted inside BF loops.
    """

    WORKSPACE_CELLS = 48

    def __init__(
        self,
        heap: HeapBlockArena,
        *,
        packed: PackedU32Core,
        handles: ObjectHandleCore,
        packed64: PackedI64Core,
        workspace_base: int,
    ) -> None:
        super().__init__(heap, packed=packed, handles=handles)
        self.packed64 = packed64
        self.workspace_base = workspace_base

    @property
    def _node(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.workspace_base)

    @property
    def _current(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.workspace_base + 4)

    @property
    def _next(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.workspace_base + 8)

    @property
    def _counter(self) -> PackedU32Ref:
        return PackedU32Ref(self.workspace_base + 12)

    @property
    def _length(self) -> PackedU32Ref:
        return PackedU32Ref(self.workspace_base + 16)

    @property
    def _tail_payload(self) -> PackedI64Ref:
        return PackedI64Ref(self.workspace_base + 20)

    @property
    def _tail(self) -> ObjectHandleRef:
        return ObjectHandleRef(self.workspace_base + 20)

    @property
    def _value_tmp(self) -> PackedI64Ref:
        return PackedI64Ref(self.workspace_base + 28)

    @property
    def _is_zero(self) -> int:
        return self.workspace_base + 36

    @property
    def _empty_gate(self) -> int:
        return self.workspace_base + 37

    @property
    def _nonempty_gate(self) -> int:
        return self.workspace_base + 38

    @property
    def _loop(self) -> int:
        return self.workspace_base + 39

    @property
    def _flag_gate(self) -> int:
        return self.workspace_base + 40

    def _clear_workspace(self) -> None:
        for cell in range(self.workspace_base, self.workspace_base + self.WORKSPACE_CELLS):
            self.heap.bf.clear(cell)

    def _copy_flag(self, src: int, dst: int) -> None:
        self.packed._copy_cell(src, dst, self.packed.s0)

    def _set_not_flag(self, dst: int, src: int) -> None:
        """dst = not src for a preserved Boolean src."""
        bf = self.heap.bf
        bf.set_const(dst, 1)
        self._copy_flag(src, self._flag_gate)
        bf.begin_while(self._flag_gate)
        bf.add_const(self._flag_gate, -1)
        bf.clear(dst)
        bf.end_while(self._flag_gate)

    def _append_packed_body(self, ref: ObjectHandleRef, value: PackedI64Ref) -> None:
        bf = self.heap.bf
        self.heap.allocate(self._node, type_tag=TYPE_LIST_INT_ELEMENT)
        self.heap.write_payload_i64(self._node, value)

        self.heap.read_length(self._length, ref)
        self.packed.is_zero(self._is_zero, self._length)
        self.heap.read_payload_i64(self._tail_payload, ref)
        self._copy_flag(self._is_zero, self._empty_gate)
        self._set_not_flag(self._nonempty_gate, self._is_zero)

        bf.begin_while(self._empty_gate)
        bf.add_const(self._empty_gate, -1)
        self.heap.write_next(ref, self._node)
        bf.end_while(self._empty_gate)

        bf.begin_while(self._nonempty_gate)
        bf.add_const(self._nonempty_gate, -1)
        self.heap.write_next(self._tail, self._node)
        bf.end_while(self._nonempty_gate)

        self.packed64.clear(self._tail_payload)
        self.handles.copy(self._tail, self._node)
        self.heap.write_payload_i64(ref, self._tail_payload)

        self.packed.increment(self._length)
        self.heap.write_length(ref, self._length)
        self.heap.write_capacity(ref, self._length)

    def append_packed(self, ref: ObjectHandleRef, value: PackedI64Ref) -> None:
        """Append an external packed int64 while preserving list identity."""
        self._clear_workspace()
        self._append_packed_body(ref, value)
        self._clear_workspace()

    def append_int64(self, ref: ObjectHandleRef, value) -> None:
        """Pack a normal compiler int64 and append it."""
        self._clear_workspace()
        self.packed64.from_int64(self._value_tmp, value)
        self._append_packed_body(ref, self._value_tmp)
        self._clear_workspace()

    def _get_packed_body(
        self,
        dst: PackedI64Ref,
        ref: ObjectHandleRef,
        index: PackedU32Ref,
    ) -> None:
        bf = self.heap.bf
        self.packed64.clear(dst)
        self.heap.read_next(self._current, ref)
        self.packed.copy(self._counter, index)
        self.packed.is_zero(self._is_zero, self._counter)
        self._set_not_flag(self._loop, self._is_zero)

        bf.begin_while(self._loop)
        bf.add_const(self._loop, -1)
        self.heap.read_next(self._next, self._current)
        self.handles.copy(self._current, self._next)
        self.packed.decrement(self._counter)
        self.packed.is_zero(self._is_zero, self._counter)
        self._set_not_flag(self._loop, self._is_zero)
        bf.end_while(self._loop)

        self.heap.read_payload_i64(dst, self._current)

    def get_packed(
        self,
        dst: PackedI64Ref,
        ref: ObjectHandleRef,
        index: PackedU32Ref,
    ) -> None:
        """Load a non-negative index; missing/out-of-range currently yields 0."""
        self._clear_workspace()
        self._get_packed_body(dst, ref, index)
        self._clear_workspace()

    def get_int64(self, dst, ref: ObjectHandleRef, index: PackedU32Ref) -> None:
        self._clear_workspace()
        self._get_packed_body(self._value_tmp, ref, index)
        self.packed64.to_int64(dst, self._value_tmp)
        self._clear_workspace()


__all__ = [
    "TYPE_LIST",
    "TYPE_LIST_INT_ELEMENT",
    "DynamicIntListRootRuntime",
    "DynamicIntListRuntime",
]
