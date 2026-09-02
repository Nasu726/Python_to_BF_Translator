"""Source-compact decimal parsing directly into packed int64 storage.

The legacy scalar parser accumulates every decimal digit in the 64 Boolean-cell
arithmetic representation.  That is correct but very expensive in generated
Brainfuck source because ``x = x * 10 + digit`` expands several full-word
copies/shifts/additions.

This module keeps the same modulo-2**64 integer ABI while accumulating directly
in eight little-endian byte cells.  It is primarily used by
``list(map(int, input().split()))`` so input can travel directly from decimal
text to the packed list slot without an intermediate 64-cell word.
"""

from __future__ import annotations

from bfcore import WORD_BITS
from bfpacked64 import PackedI64Ref
from bftokens import BinaryTokenIO


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

    def _packed_mul10_add_ascii_digit(
        self,
        dst: PackedI64Ref,
        ch: int,
        workspace_base: int,
    ) -> None:
        """dst = dst*10 + (ch-'0') modulo 2**64; consume ch.

        Runtime work is intentionally traded for compact source.  Each byte is
        multiplied by ten with nested BF loops, carrying overflow to the next
        byte.  Only eight byte lanes are emitted statically instead of several
        64-bit Boolean-cell primitives per input digit.
        """
        bf = self.bf
        count = workspace_base
        ten = workspace_base + 1
        carry = workspace_base + 2
        next_carry = workspace_base + 3
        zero = workspace_base + 4
        tmp = workspace_base + 5
        helper = workspace_base + 6

        for cell in (count, ten, carry, next_carry, zero, tmp, helper):
            bf.clear(cell)

        # Valid contest input guarantees an ASCII decimal digit here.
        bf.add_const(ch, -ord("0"))
        bf.begin_while(ch)
        bf.add_const(ch, -1)
        bf.add_const(carry, 1)
        bf.end_while(ch)

        for byte_index in range(8):
            byte = dst.byte(byte_index)

            # Move the old byte value to count, leaving the output byte at zero.
            bf.clear(count)
            bf.begin_while(byte)
            bf.add_const(byte, -1)
            bf.add_const(count, 1)
            bf.end_while(byte)
            bf.clear(next_carry)

            # byte = old_byte * 10 (mod 256), recording every wrap.
            bf.begin_while(count)
            bf.add_const(count, -1)
            bf.set_const(ten, 10)
            bf.begin_while(ten)
            bf.add_const(ten, -1)
            bf.add_const(byte, 1)
            self._packed_is_zero_byte(zero, byte, tmp, helper)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.add_const(next_carry, 1)
            bf.end_while(zero)
            bf.end_while(ten)
            bf.end_while(count)

            # Add the carry from the previous byte (the decimal digit for lane
            # zero).  This is at most nine, and can overflow this byte at most
            # once.
            bf.begin_while(carry)
            bf.add_const(carry, -1)
            bf.add_const(byte, 1)
            self._packed_is_zero_byte(zero, byte, tmp, helper)
            bf.begin_while(zero)
            bf.add_const(zero, -1)
            bf.add_const(next_carry, 1)
            bf.end_while(zero)
            bf.end_while(carry)

            # Carry becomes the next lane's input.
            bf.begin_while(next_carry)
            bf.add_const(next_carry, -1)
            bf.add_const(carry, 1)
            bf.end_while(next_carry)

        # Overflow past byte seven is discarded by the fixed-width ABI.
        for cell in (count, ten, carry, next_carry, zero, tmp, helper):
            bf.clear(cell)

    def _packed_negate_inplace(self, dst: PackedI64Ref, workspace_base: int) -> None:
        """Two's-complement negate one packed 64-bit word in place."""
        bf = self.bf
        count = workspace_base
        carry = workspace_base + 1
        gate = workspace_base + 2
        zero = workspace_base + 3
        tmp = workspace_base + 4
        helper = workspace_base + 5

        # Bitwise NOT byte-wise: byte = 255 - byte.
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

        # Add one, propagating a one-shot carry across byte lanes.
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
        """Read one signed integer from the current line directly into 8 bytes.

        The delimiter is consumed.  Leading horizontal whitespace is skipped,
        but the operation never crosses a newline.  ``has_token`` and
        ``end_line`` match ``read_s64_line_token``.
        """
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
