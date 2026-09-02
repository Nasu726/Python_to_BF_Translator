"""Fixed-capacity byte-string primitives for the Brainfuck backend.

Strings are NUL-terminated byte arrays.  This is intentionally a compact
contest/runtime representation rather than CPython's Unicode object model.
Source literals must fit in bytes 1..255; NUL is reserved as the terminator.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import Int64Ref
from bfio import Binary64IO


@dataclass(frozen=True)
class StringRef:
    base: int
    capacity: int

    def char(self, i: int) -> int:
        if not 0 <= i < self.capacity:
            raise IndexError(i)
        return self.base + i

    @property
    def terminator(self) -> int:
        return self.base + self.capacity

    @property
    def cells(self) -> int:
        return self.capacity + 1


class BinaryStringIO(Binary64IO):
    STRING_IO_SCRATCH = 5

    def clear_string(self, ref: StringRef) -> None:
        for i in range(ref.cells):
            self.bf.clear(ref.base + i)

    def set_string_literal(self, ref: StringRef, text: str) -> None:
        values = [ord(ch) for ch in text]
        if any(v == 0 or v > 255 for v in values):
            raise ValueError("string literals currently require non-NUL byte characters")
        if len(values) > ref.capacity:
            raise ValueError("string literal exceeds allocated capacity")
        self.clear_string(ref)
        for i, value in enumerate(values):
            self.bf.set_const(ref.char(i), value)

    def copy_string(self, dst: StringRef, src: StringRef) -> None:
        if dst.capacity < src.capacity:
            raise ValueError("destination string capacity is smaller than source")
        self.clear_string(dst)
        for i in range(src.capacity):
            self.copy_cell(src.char(i), dst.char(i), self.s0)
        # src's explicit terminator and dst's already-cleared suffix guarantee
        # NUL termination even when the runtime string fills src.capacity.
        self._clear_scratch()

    def print_string(self, src: StringRef, control: int) -> None:
        """Print a NUL-terminated string without losing pointer tracking.

        The loop is statically unrolled across the capacity.  Each nonzero byte
        executes the output body once (``control`` is cleared in that body), so
        zero and all cleared suffix cells emit nothing.
        """
        bf = self.bf
        bf.clear(control)
        for i in range(src.capacity):
            self.copy_cell(src.char(i), control, self.s0)
            bf.begin_while(control)
            bf.clear(control)
            bf.move(src.char(i))
            bf.emit(".")
            bf.end_while(control)
        self._clear_scratch()

    def _set_line_end(self, result: int, ch: int, tmp: int) -> None:
        """result = (ch == '\\n' or ch == EOF-zero), preserving ch."""
        bf = self.bf
        self._eq_byte_const(result, ch, ord("\n"))
        self._eq_byte_const(tmp, ch, 0)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        bf.set_const(result, 1)
        bf.end_while(tmp)

    def read_line(self, dst: StringRef, workspace_base: int) -> None:
        """Read one input line into ``dst`` and remove the trailing newline.

        A line longer than ``capacity`` is truncated, but the excess input is
        drained through the newline so the next input operation starts at the
        next line instead of in the middle of this one.
        """
        bf = self.bf
        active = workspace_base
        control = workspace_base + 1
        is_end = workspace_base + 2
        tmp = workspace_base + 3
        drain = workspace_base + 4

        self.clear_string(dst)
        bf.set_const(active, 1)
        for c in (control, is_end, tmp, drain):
            bf.clear(c)

        for i in range(dst.capacity):
            cell = dst.char(i)
            self.copy_cell(active, control, self.s0)
            bf.begin_while(control)
            bf.add_const(control, -1)
            bf.move(cell)
            bf.emit(",")
            self._set_line_end(is_end, cell, tmp)
            bf.begin_while(is_end)
            bf.add_const(is_end, -1)
            bf.clear(cell)
            bf.clear(active)
            bf.end_while(is_end)
            bf.end_while(control)

        # If the buffer filled before newline, consume the remainder safely.
        bf.begin_while(active)
        bf.move(drain)
        bf.emit(",")
        self._set_line_end(is_end, drain, tmp)
        bf.begin_while(is_end)
        bf.add_const(is_end, -1)
        bf.clear(active)
        bf.end_while(is_end)
        bf.end_while(active)

        bf.clear(dst.terminator)
        for c in (control, is_end, tmp, drain, active):
            bf.clear(c)
        self._clear_scratch()

    def string_length(self, dst: Int64Ref, src: StringRef, control: int) -> None:
        self._clear_word(dst)
        bf = self.bf
        bf.clear(control)
        for i in range(src.capacity):
            self.copy_cell(src.char(i), control, self.s0)
            bf.begin_while(control)
            bf.clear(control)
            self._inc64_inplace(dst)
            bf.end_while(control)
        self._clear_scratch()

    def string_truth(self, result: int, src: StringRef, control: int) -> None:
        bf = self.bf
        bf.clear(result)
        bf.clear(control)
        for i in range(src.capacity):
            self.copy_cell(src.char(i), control, self.s0)
            bf.begin_while(control)
            bf.clear(control)
            bf.set_const(result, 1)
            bf.end_while(control)
        self._clear_scratch()

    def eq_string(self, result: int, a: StringRef, b_ref: StringRef) -> None:
        """Bytewise equality, preserving both strings."""
        bf = self.bf
        bf.set_const(result, 1)
        n = max(a.capacity, b_ref.capacity) + 1
        for i in range(n):
            if i <= a.capacity:
                self.copy_cell(a.base + i, self.s0, self.s2)
            else:
                bf.clear(self.s0)
            if i <= b_ref.capacity:
                self.copy_cell(b_ref.base + i, self.s1, self.s2)
            else:
                bf.clear(self.s1)
            bf.begin_while(self.s1)
            bf.add_const(self.s1, -1)
            bf.add_const(self.s0, -1)
            bf.end_while(self.s1)
            bf.begin_while(self.s0)
            bf.clear(self.s0)
            bf.clear(result)
            bf.end_while(self.s0)
        self._clear_scratch()
