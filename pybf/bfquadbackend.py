"""Hybrid compiler backend with source-compact hot int64 operations.

The public compiler still needs strings, packed lists, decimal I/O and the
correctness-first binary primitives.  Replacing that entire stack at once would
be unnecessarily risky.  This adapter keeps ``BinaryStringListIO`` as the
fallback implementation but dispatches hot scalar operations to ``Quad64Core``
when all participating values are ``Quad64Ref`` instances.
"""

from __future__ import annotations

from bfpacked64 import PackedI64Ref
from bfquad import (
    DIGITS,
    STRIDE,
    WORD_CELLS,
    Quad64Core,
    Quad64Ref,
    _RelativeBuilder,
)
from bfstringlists import BinaryStringListIO


class QuadBinaryStringListIO(BinaryStringListIO):
    """String/list backend plus runtime-lane scalar copy/add/sub/comparison."""

    def __init__(self, bf, scratch_base: int) -> None:
        super().__init__(bf, scratch_base=scratch_base)
        self.quad = Quad64Core(bf)
        self._quad_workspace_base: int | None = None

    def set_quad_workspace(self, base: int) -> None:
        self._quad_workspace_base = base

    def _qtmp(self, index: int = 0) -> Quad64Ref:
        if self._quad_workspace_base is None:
            raise RuntimeError("quad workspace was not configured")
        return Quad64Ref(self._quad_workspace_base + index * WORD_CELLS)

    @staticmethod
    def _all_quad(*refs) -> bool:
        return all(isinstance(ref, Quad64Ref) for ref in refs)

    @staticmethod
    def _same(a, b) -> bool:
        return isinstance(a, Quad64Ref) and isinstance(b, Quad64Ref) and a.base == b.base

    def _packed_to_quad_destructive(self, dst: Quad64Ref, src: PackedI64Ref) -> None:
        """Expand eight packed bytes into Quad bits with only eight long moves.

        ``src`` is traversal scratch when this fast path is used, so consuming
        it is intentional.  One packed byte is moved directly beside the four
        Quad lanes that own its eight output bits.  Those four lane-marker cells
        temporarily act as byte/quotient/parity/gate scratch, so the repeated
        divmod-by-two work stays local instead of bouncing through the global
        scratch area once per output bit.  All four markers finish zero.
        """
        bf = self.bf

        for byte_index in range(8):
            first_digit = byte_index * 4
            byte = dst.marker(first_digit)
            quotient = dst.marker(first_digit + 1)
            parity = dst.marker(first_digit + 2)
            gate = dst.marker(first_digit + 3)

            for cell in (byte, quotient, parity, gate):
                bf.clear(cell)

            # This is the only long-distance transfer for the packed byte.
            source = src.byte(byte_index)
            bf.begin_while(source)
            bf.add_const(source, -1)
            bf.add_const(byte, 1)
            bf.end_while(source)

            for within in range(8):
                bf.clear(quotient)
                bf.clear(parity)

                # quotient, parity = divmod(byte, 2), consuming byte.
                bf.begin_while(byte)
                bf.add_const(byte, -1)
                bf.set_const(gate, 1)

                bf.begin_while(parity)
                bf.add_const(parity, -1)
                bf.clear(gate)
                bf.add_const(quotient, 1)
                bf.end_while(parity)

                bf.begin_while(gate)
                bf.add_const(gate, -1)
                bf.add_const(parity, 1)
                bf.end_while(gate)
                bf.end_while(byte)

                out = dst.bit(byte_index * 8 + within)
                bf.clear(out)
                bf.begin_while(parity)
                bf.add_const(parity, -1)
                bf.add_const(out, 1)
                bf.end_while(parity)

                bf.begin_while(quotient)
                bf.add_const(quotient, -1)
                bf.add_const(byte, 1)
                bf.end_while(quotient)

        # Eight divisions consume every byte completely.  The local markers
        # are therefore already zero and global scratch was never touched.

    def copy64(self, dst, src) -> None:
        if self._all_quad(dst, src):
            if not self._same(dst, src):
                self.quad.copy64(dst, src)
            return
        if isinstance(dst, Quad64Ref) and isinstance(src, PackedI64Ref):
            self._packed_to_quad_destructive(dst, src)
            return
        super().copy64(dst, src)

    def set_u64(self, dst, value: int) -> None:
        if isinstance(dst, Quad64Ref):
            self.quad.set_u64(dst, value)
            return
        super().set_u64(dst, value)

    def add64(self, dst, a, b) -> None:
        if self._all_quad(dst, a, b):
            if self._same(a, b):
                rhs = self._qtmp(0)
                self.copy64(rhs, b)
                self.quad.add64(dst, a, rhs)
            else:
                self.quad.add64(dst, a, b)
            return
        super().add64(dst, a, b)

    def sub64(self, dst, a, b) -> None:
        if self._all_quad(dst, a, b):
            if self._same(a, b):
                self.quad.set_u64(dst, 0)
            else:
                self.quad.sub64(dst, a, b)
            return
        super().sub64(dst, a, b)

    def uge64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.set_const(result, 1)
            else:
                self.quad.uge64(result, a, b, self._qtmp(0))
            return
        super().uge64(result, a, b)

    def ult64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.clear(result)
            else:
                self.uge64(result, a, b)
                self._toggle_bit(result, self.s0)
                self._clear_scratch()
            return
        super().ult64(result, a, b)

    def ule64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.set_const(result, 1)
            else:
                self.uge64(result, b, a)
            return
        super().ule64(result, a, b)

    def ugt64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.clear(result)
            else:
                self.uge64(result, b, a)
                self._toggle_bit(result, self.s0)
                self._clear_scratch()
            return
        super().ugt64(result, a, b)

    def sge64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.set_const(result, 1)
                return
            self._toggle_bit(a.bit(63), self.s0)
            self._toggle_bit(b.bit(63), self.s0)
            self._clear_scratch()
            self.quad.uge64(result, a, b, self._qtmp(0))
            self._toggle_bit(a.bit(63), self.s0)
            self._toggle_bit(b.bit(63), self.s0)
            self._clear_scratch()
            return
        super().sge64(result, a, b)

    def slt64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.clear(result)
            else:
                self.sge64(result, a, b)
                self._toggle_bit(result, self.s0)
                self._clear_scratch()
            return
        super().slt64(result, a, b)

    def sle64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.set_const(result, 1)
            else:
                self.sge64(result, b, a)
            return
        super().sle64(result, a, b)

    def sgt64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.clear(result)
            else:
                self.sge64(result, b, a)
                self._toggle_bit(result, self.s0)
                self._clear_scratch()
            return
        super().sgt64(result, a, b)

    def eq64(self, result: int, a, b) -> None:
        if self._all_quad(a, b):
            if self._same(a, b):
                self.bf.set_const(result, 1)
                return
            other = self.s1
            gate = self.s2
            self.quad.uge64(result, a, b, self._qtmp(0))
            self.quad.uge64(other, b, a, self._qtmp(0))
            self.copy_cell(result, gate, self.s0)
            self.bf.clear(result)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.copy_cell(other, result, self.s0)
            self.bf.end_while(gate)
            self._clear_scratch()
            return
        super().eq64(result, a, b)

    @staticmethod
    def _force_seen_from_bit(
        r: _RelativeBuilder,
        bit: int,
        seen: int,
        scratch: int,
    ) -> None:
        """seen |= bit while preserving the Boolean source bit."""
        r.clear(scratch)
        r.move(bit)
        r.emit("[")
        r.emit("-")
        r.clear(seen)
        r.add(seen, 1)
        r.add(scratch, 1)
        r.move(bit)
        r.emit("]")
        r.move(scratch)
        r.emit("[")
        r.emit("-")
        r.add(bit, 1)
        r.move(scratch)
        r.emit("]")

    def _quad_is_nonzero(self, result: int, word: Quad64Ref) -> None:
        """Compute bool(word) with one emitted repeated 32-lane body."""
        bf = self.bf
        q0 = self._qtmp(0)
        tmp = self._qtmp(1) if word.base == q0.base else q0
        delta = tmp.base - word.base

        for digit in range(DIGITS):
            bf.set_const(word.marker(digit), 1)
        bf.clear(word.marker(DIGITS))
        bf.clear(tmp.marker(0))

        r = _RelativeBuilder()
        marker = 0
        bit0, bit1 = 1, 2
        seen_in = delta
        scratch = delta + 1
        seen_out = delta + STRIDE

        r.clear(marker)
        r.clear(seen_out)
        r.transfer(seen_in, seen_out)
        self._force_seen_from_bit(r, bit0, seen_out, scratch)
        self._force_seen_from_bit(r, bit1, seen_out, scratch)
        r.move(STRIDE)

        bf.move(word.marker(0))
        bf.emit("[" + r.code() + "]")
        bf.ptr = word.marker(DIGITS)

        bf.clear(result)
        carry = tmp.marker(DIGITS)
        bf.move(carry)
        bf.emit("[-")
        bf.move(result)
        bf.emit("+")
        bf.move(carry)
        bf.emit("]")
        self._clear_scratch()

    def _is_nonzero64(self, result: int, word) -> None:
        if isinstance(word, Quad64Ref):
            self._quad_is_nonzero(result, word)
            return
        super()._is_nonzero64(result, word)

    def _inc64_inplace(self, word) -> None:
        if isinstance(word, Quad64Ref):
            one = self._qtmp(0)
            tmp = self._qtmp(1)
            self.quad.set_u64(one, 1)
            self.quad.add64(tmp, word, one)
            self.copy64(word, tmp)
            return
        super()._inc64_inplace(word)

    def _neg64_inplace(self, word) -> None:
        if isinstance(word, Quad64Ref):
            zero = self._qtmp(0)
            tmp = self._qtmp(1)
            self.quad.set_u64(zero, 0)
            self.quad.sub64(tmp, zero, word)
            self.copy64(word, tmp)
            return
        super()._neg64_inplace(word)

    def print_u64(self, src, workspace_base: int) -> None:
        if not isinstance(src, Quad64Ref):
            return super().print_u64(src, workspace_base)

        magnitude = self._qtmp(0)
        bcd_base = workspace_base + WORD_CELLS
        counter = bcd_base + 80
        ge5_flag = counter + 1
        started = ge5_flag + 1
        ascii_cell = started + 1
        control = ascii_cell + 1

        self.copy64(magnitude, src)
        self._bcd_from_magnitude(magnitude, bcd_base, counter, ge5_flag)
        self._print_bcd_digits(bcd_base, started, ascii_cell, control)

    def print_s64(self, src, workspace_base: int) -> None:
        if not isinstance(src, Quad64Ref):
            return super().print_s64(src, workspace_base)

        magnitude = self._qtmp(0)
        bcd_base = workspace_base + WORD_CELLS
        counter = bcd_base + 80
        ge5_flag = counter + 1
        sign = ge5_flag + 1
        started = sign + 1
        ascii_cell = started + 1
        control = ascii_cell + 1

        self.copy64(magnitude, src)
        self.copy_cell(src.bit(63), sign, self.s0)
        self.bf.begin_while(sign)
        self.bf.add_const(sign, -1)
        self.bf.set_const(ascii_cell, ord("-"))
        self.bf.move(ascii_cell)
        self.bf.emit(".")
        self._neg64_inplace(magnitude)
        self.bf.end_while(sign)

        self._bcd_from_magnitude(magnitude, bcd_base, counter, ge5_flag)
        self._print_bcd_digits(bcd_base, started, ascii_cell, control)


__all__ = ["QuadBinaryStringListIO"]
