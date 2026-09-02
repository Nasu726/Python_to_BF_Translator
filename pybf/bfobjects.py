"""Compact runtime object handles for the Brainfuck heap/object layer.

This is the first Phase-1 primitive for moving mutable Python values away from
"variable storage == object storage".  Object identity uses a packed 32-bit
little-endian handle (four ordinary 8-bit Brainfuck cells) rather than the
bit-per-cell int64 arithmetic representation.

Handle value 0 is reserved for null / no-object.  A bump ID allocator starts at
1 and returns monotonically increasing identities.  Actual object storage and
handle-to-object traversal are layered on top of this module in later Phase-1
steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter


HANDLE_BYTES = 4
HANDLE_MASK = (1 << 32) - 1


@dataclass(frozen=True)
class ObjectHandleRef:
    """Four-byte little-endian object identity."""

    base: int

    def byte(self, index: int) -> int:
        if not 0 <= index < HANDLE_BYTES:
            raise IndexError(index)
        return self.base + index

    @property
    def cells(self) -> int:
        return HANDLE_BYTES


class ObjectHandleCore:
    """Small BF primitives for object identities.

    Four scratch cells are used and restored to zero by every public operation.
    The operations deliberately work on byte-packed handles so reference
    manipulation stays compact even while ordinary Python integers still use
    the correctness-first bit-per-cell backend.
    """

    SCRATCH_CELLS = 4

    def __init__(self, bf: BFEmitter, scratch_base: int) -> None:
        self.bf = bf
        self.s0 = scratch_base
        self.s1 = scratch_base + 1
        self.s2 = scratch_base + 2
        self.s3 = scratch_base + 3

    def _clear_scratch(self) -> None:
        for cell in (self.s0, self.s1, self.s2, self.s3):
            self.bf.clear(cell)

    def _copy_cell(self, src: int, dst: int, tmp: int) -> None:
        """dst = src, preserving src; tmp finishes zero."""
        bf = self.bf
        bf.clear(dst)
        bf.clear(tmp)
        bf.begin_while(src)
        bf.add_const(src, -1)
        bf.add_const(dst, 1)
        bf.add_const(tmp, 1)
        bf.end_while(src)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        bf.add_const(src, 1)
        bf.end_while(tmp)

    def _is_zero_byte(self, result: int, src: int, tmp: int) -> None:
        """result = 1 iff src == 0, preserving src."""
        bf = self.bf
        bf.set_const(result, 1)
        self._copy_cell(src, tmp, self.s3)
        bf.begin_while(tmp)
        bf.clear(tmp)
        bf.clear(result)
        bf.end_while(tmp)

    def clear(self, ref: ObjectHandleRef) -> None:
        for i in range(HANDLE_BYTES):
            self.bf.clear(ref.byte(i))
        self._clear_scratch()

    def set_u32(self, ref: ObjectHandleRef, value: int) -> None:
        value &= HANDLE_MASK
        for i in range(HANDLE_BYTES):
            self.bf.set_const(ref.byte(i), (value >> (8 * i)) & 0xFF)
        self._clear_scratch()

    def copy(self, dst: ObjectHandleRef, src: ObjectHandleRef) -> None:
        if dst == src:
            return
        for i in range(HANDLE_BYTES):
            self._copy_cell(src.byte(i), dst.byte(i), self.s0)
        self._clear_scratch()

    def increment(self, ref: ObjectHandleRef) -> None:
        """ref = ref + 1 modulo 2**32."""
        bf = self.bf
        carry = self.s0
        gate = self.s1
        tmp = self.s2
        bf.set_const(carry, 1)

        for i in range(HANDLE_BYTES):
            self._copy_cell(carry, gate, self.s3)
            bf.clear(carry)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.add_const(ref.byte(i), 1)
            self._is_zero_byte(carry, ref.byte(i), tmp)
            bf.end_while(gate)

        self._clear_scratch()

    def equal(self, result: int, a: ObjectHandleRef, b: ObjectHandleRef) -> None:
        """result = 1 iff the two handles identify the same object."""
        bf = self.bf
        bf.set_const(result, 1)

        for i in range(HANDLE_BYTES):
            # s0 = a_i - b_i modulo 256.  It is zero exactly when the bytes
            # match.  Work only on copies so both handles are preserved.
            self._copy_cell(a.byte(i), self.s0, self.s2)
            self._copy_cell(b.byte(i), self.s1, self.s2)
            bf.begin_while(self.s1)
            bf.add_const(self.s1, -1)
            bf.add_const(self.s0, -1)
            bf.end_while(self.s1)
            bf.begin_while(self.s0)
            bf.clear(self.s0)
            bf.clear(result)
            bf.end_while(self.s0)

        self._clear_scratch()


class ObjectHandleAllocator(ObjectHandleCore):
    """Monotonic object-identity allocator.

    This allocates *identity numbers*, not object payload blocks yet.  The
    separation is intentional: frontend alias semantics can depend on stable
    identities while the following heap step decides how handles locate and
    resize physical object storage.
    """

    def __init__(
        self,
        bf: BFEmitter,
        next_handle: ObjectHandleRef,
        scratch_base: int,
    ) -> None:
        super().__init__(bf, scratch_base)
        self.next_handle = next_handle

    def initialize(self) -> None:
        # 0 is the null sentinel, so the first real object receives handle 1.
        self.set_u32(self.next_handle, 1)

    def allocate(self, dst: ObjectHandleRef) -> None:
        self.copy(dst, self.next_handle)
        self.increment(self.next_handle)


__all__ = [
    "HANDLE_BYTES",
    "ObjectHandleRef",
    "ObjectHandleCore",
    "ObjectHandleAllocator",
]
