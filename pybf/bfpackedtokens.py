"""Source-compact decimal parsing directly into packed int64 storage.

The packed parser deliberately spends Brainfuck runtime work to keep emitted
source small.  Decimal accumulation uses eight base-256 bytes and a parallel
marker/carry lane, so ``x = x*10 + digit`` needs one repeated byte-lane body
instead of eight Python-unrolled copies.
"""

from __future__ import annotations

from bfcore import WORD_BITS
from bfpacked64 import PackedI64Ref
from bftokens import BinaryTokenIO


class _Rel:
    """Tiny relative Brainfuck builder used by the packed byte walker."""

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

    def emit(self, code: str) -> None:
        self.parts.append(code)

    def clear(self, target: int) -> None:
        self.move(target)
        self.emit("[-]")

    def add(self, target: int, value: int) -> None:
        self.move(target)
        value %= 256
        if value <= 128:
            self.emit("+" * value)
        else:
            self.emit("-" * (256 - value))

    def set_const(self, target: int, value: int) -> None:
        self.clear(target)
        self.add(target, value)

    def code(self) -> str:
        return "".join(self.parts)


class PackedBinaryTokenIO(BinaryTokenIO):
    """BinaryTokenIO plus packed signed-decimal token primitives."""

    def _packed_is_zero_byte(
        self,
        result: int,
        src: int,
        tmp: int,
        helper: int,
    ) -> None:
        """Set result=1 iff src is zero, preserving src."""
        bf = self.bf
        bf.set_const(result, 1)
        self.copy_cell(src, tmp, helper)
        bf.begin_while(tmp)
        bf.clear(tmp)
        bf.clear(result)
        bf.end_while(tmp)

    @staticmethod
    def _rel_zero_preserved(
        r: _Rel,
        result: int,
        src: int,
        tmp: int,
        helper: int,
    ) -> None:
        """Relative equivalent of _packed_is_zero_byte."""
        r.set_const(result, 1)
        r.clear(tmp)
        r.clear(helper)
        r.move(src)
        r.emit("[")
        r.emit("-")
        r.add(tmp, 1)
        r.add(helper, 1)
        r.move(src)
        r.emit("]")
        r.move(helper)
        r.emit("[")
        r.emit("-")
        r.add(src, 1)
        r.move(helper)
        r.emit("]")
        r.move(tmp)
        r.emit("[")
        r.clear(tmp)
        r.clear(result)
        r.move(tmp)
        r.emit("]")

    def _packed_mul10_add_ascii_digit(
        self,
        dst: PackedI64Ref,
        ch: int,
        workspace_base: int,
    ) -> None:
        """dst = dst*10 + (ch-'0') modulo 2**64; consume ch.

        Eight byte lanes execute one emitted body.  Parallel control arrays are
        intentionally placed in the otherwise-dead low portion of the shared
        arithmetic workspace; marker[i] and byte[i] therefore have a constant
        relative offset even though the packed value bytes are contiguous.
        """
        bf = self.bf

        # ``workspace_base`` here is the token reader's arithmetic sub-workspace
        # (global shared workspace + 144).  Negative offsets remain inside that
        # same reserved shared region and are dead between backend calls.
        marker_base = workspace_base - 120  # 9 cells, last is sentinel
        carry_base = marker_base + 9        # 9 cells, last is overflow sink
        ten_base = carry_base + 9           # 8 parallel scratch cells
        zero_base = ten_base + 8
        tmp_base = zero_base + 8
        helper_base = tmp_base + 8

        # Arm eight lanes and a zero sentinel.  Contiguous initialization is
        # tiny; the expensive per-byte arithmetic lives only in the runtime body.
        for i in range(8):
            bf.set_const(marker_base + i, 1)
            bf.clear(carry_base + i)
        bf.clear(marker_base + 8)
        bf.clear(carry_base + 8)

        bf.add_const(ch, -ord("0"))
        bf.begin_while(ch)
        bf.add_const(ch, -1)
        bf.add_const(carry_base, 1)
        bf.end_while(ch)

        byte_delta = dst.base - marker_base
        carry_delta = carry_base - marker_base
        ten_delta = ten_base - marker_base
        zero_delta = zero_base - marker_base
        tmp_delta = tmp_base - marker_base
        helper_delta = helper_base - marker_base

        r = _Rel()
        marker = 0
        byte = byte_delta
        carry = carry_delta
        next_carry = carry_delta + 1
        ten = ten_delta
        zero = zero_delta
        tmp = tmp_delta
        helper = helper_delta

        # Consume this lane's one-shot outer marker, then reuse it as the old
        # byte count while reconstructing the destination byte from zero.
        r.clear(marker)
        r.clear(next_carry)
        r.move(byte)
        r.emit("[")
        r.emit("-")
        r.add(marker, 1)
        r.move(byte)
        r.emit("]")

        r.move(marker)
        r.emit("[")
        r.emit("-")
        r.set_const(ten, 10)
        r.move(ten)
        r.emit("[")
        r.emit("-")
        r.add(byte, 1)
        self._rel_zero_preserved(r, zero, byte, tmp, helper)
        r.move(zero)
        r.emit("[")
        r.emit("-")
        r.add(next_carry, 1)
        r.move(zero)
        r.emit("]")
        r.move(ten)
        r.emit("]")
        r.move(marker)
        r.emit("]")

        # Add incoming base-256 carry (digit for lane zero, at most nine).
        r.move(carry)
        r.emit("[")
        r.emit("-")
        r.add(byte, 1)
        self._rel_zero_preserved(r, zero, byte, tmp, helper)
        r.move(zero)
        r.emit("[")
        r.emit("-")
        r.add(next_carry, 1)
        r.move(zero)
        r.emit("]")
        r.move(carry)
        r.emit("]")

        # The matching outer ']' tests marker[i+1].
        r.move(1)

        bf.move(marker_base)
        bf.emit("[" + r.code() + "]")
        bf.ptr = marker_base + 8
        bf.clear(carry_base + 8)  # discard overflow past bit 63

    def _packed_negate_inplace(self, dst: PackedI64Ref, workspace_base: int) -> None:
        """Two's-complement negate one packed 64-bit word in place."""
        bf = self.bf
        count = workspace_base
        carry = workspace_base + 1
        gate = workspace_base + 2
        zero = workspace_base + 3
        tmp = workspace_base + 4
        helper = workspace_base + 5

        for byte_index in range(8):
            byte = dst.byte(byte_index)
            bf.clear(count)
            bf.begin_while(byte)
            bf.add_const(byte, -1)
            bf.add_const(count, 1)
            bf.end_while(byte)
            bf.set_const(byte, 255)
            bf.begin_while(count)
            bf.add_const(count, -1)
            bf.add_const(byte, -1)
            bf.end_while(count)

        bf.set_const(carry, 1)
        for byte_index in range(8):
            byte = dst.byte(byte_index)
            self.copy_cell(carry, gate, helper)
            bf.clear(carry)
            bf.begin_while(gate)
            bf.add_const(gate, -1)
            bf.add_const(byte, 1)
            self._packed_is_zero_byte(zero, byte, tmp, helper)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.add_const(carry, 1)
            bf.end_while(zero)
            bf.end_while(gate)

        for cell in (count, carry, gate, zero, tmp, helper):
            bf.clear(cell)

    def read_packed_s64_line_token(
        self,
        dst: PackedI64Ref,
        has_token: int,
        end_line: int,
        workspace_base: int,
    ) -> None:
        """Read one signed integer from the current line directly into 8 bytes."""
        bf = self.bf
        control_base = workspace_base + WORD_BITS * 2
        ch = control_base
        sign = control_base + 1
        is_minus = control_base + 2
        skip = control_base + 3
        tmp = control_base + 4
        active = control_base + 5
        delimiter = control_base + 6
        gate = control_base + 7
        line_tmp = control_base + 8
        arithmetic_workspace = control_base + 16

        self.packed64.clear(dst)
        for cell in (
            has_token,
            end_line,
            ch,
            sign,
            is_minus,
            skip,
            tmp,
            active,
            delimiter,
            gate,
            line_tmp,
        ):
            bf.clear(cell)

        bf.move(ch)
        bf.emit(",")
        self._is_hspace(skip, ch, tmp)
        bf.begin_while(skip)
        bf.add_const(skip, -1)
        bf.move(ch)
        bf.emit(",")
        self._is_hspace(skip, ch, tmp)
        bf.end_while(skip)

        self._is_line_end(end_line, ch, tmp)
        bf.set_const(has_token, 1)
        self.copy_cell(end_line, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.clear(has_token)
        bf.end_while(gate)

        self.copy_cell(has_token, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)

        self._eq_byte_const(is_minus, ch, ord("-"))
        bf.begin_while(is_minus)
        bf.add_const(is_minus, -1)
        bf.set_const(sign, 1)
        bf.move(ch)
        bf.emit(",")
        bf.end_while(is_minus)

        bf.set_const(active, 1)
        self._is_hspace(delimiter, ch, tmp)
        self._is_line_end(line_tmp, ch, tmp)
        bf.begin_while(line_tmp)
        bf.add_const(line_tmp, -1)
        bf.set_const(delimiter, 1)
        bf.end_while(line_tmp)
        bf.begin_while(delimiter)
        bf.add_const(delimiter, -1)
        bf.clear(active)
        bf.end_while(delimiter)

        bf.begin_while(active)
        bf.add_const(active, -1)
        self._packed_mul10_add_ascii_digit(dst, ch, arithmetic_workspace)
        bf.move(ch)
        bf.emit(",")

        self._is_line_end(line_tmp, ch, tmp)
        bf.begin_while(line_tmp)
        bf.add_const(line_tmp, -1)
        bf.set_const(end_line, 1)
        bf.end_while(line_tmp)

        bf.set_const(active, 1)
        self._is_hspace(delimiter, ch, tmp)
        self._is_line_end(line_tmp, ch, tmp)
        bf.begin_while(line_tmp)
        bf.add_const(line_tmp, -1)
        bf.set_const(delimiter, 1)
        bf.end_while(line_tmp)
        bf.begin_while(delimiter)
        bf.add_const(delimiter, -1)
        bf.clear(active)
        bf.end_while(delimiter)
        bf.end_while(active)

        bf.begin_while(sign)
        bf.add_const(sign, -1)
        self._packed_negate_inplace(dst, arithmetic_workspace)
        bf.end_while(sign)
        bf.end_while(gate)
        self._clear_scratch()


__all__ = ["PackedBinaryTokenIO"]
