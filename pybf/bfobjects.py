"""Compact runtime object handles for the Brainfuck heap/object layer.

Object identity is a packed 32-bit little-endian value.  The byte arithmetic is
shared with heap/list metadata through :mod:`bfpacked`; this module adds only
the semantic distinction that 0 means null/no-object and real identities start
at 1.
"""

from __future__ import annotations

from bfcore import BFEmitter
from bfpacked import PackedU32Core, PackedU32Ref, U32_BYTES


HANDLE_BYTES = U32_BYTES


class ObjectHandleRef(PackedU32Ref):
    """Packed u32 interpreted as a stable mutable-object identity."""


class ObjectHandleCore(PackedU32Core):
    """Packed-u32 operations with object-identity semantics."""


class ObjectHandleAllocator(ObjectHandleCore):
    """Monotonic object-identity allocator; handle 0 is reserved for null."""

    def __init__(
        self,
        bf: BFEmitter,
        next_handle: ObjectHandleRef,
        scratch_base: int,
    ) -> None:
        super().__init__(bf, scratch_base)
        self.next_handle = next_handle

    def initialize(self) -> None:
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
