"""Hybrid compiler backend with source-compact hot int64 operations.

The public compiler still needs strings, packed lists, decimal I/O and the
correctness-first binary primitives. Replacing that entire stack at once would
be unnecessarily risky. This adapter keeps ``BinaryStringListIO`` as the
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


_DECIMAL_DIGITS = 20
_DECIMAL_STRIDE = 7


class QuadBinaryStringListIO(BinaryStringListIO):
    """String/list backend plus runtime-lane scalar operations."""

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

    # ------------------------------------------------------------------
    # packed list payload -> Quad scalar
    # ------------------------------------------------------------------
    def _packed_to_quad_destructive(self, dst: Quad64Ref, src: PackedI64Ref) -> None:
        """Expand eight packed bytes into Quad bits with only eight long moves.

        ``src`` is traversal scratch when this fast path is used, so consuming
        it is intentional. One packed byte is moved directly beside the four
        Quad lanes that own its eight output bits. Those four lane-marker cells
        temporarily act as byte/quotient/parity/gate scratch.
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

            source = src.byte(byte_index)
            bf.begin_while(source)
            bf.add_const(source, -1)
            bf.add_const(byte, 1)
            bf.end_while(source)

            for within in range(8):
                bf.clear(quotient)
                bf.clear(parity)

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

    # ------------------------------------------------------------------
    # basic dispatch
    # ------------------------------------------------------------------
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

    def shl1_inplace(self, word) -> None:
        if isinstance(word, Quad64Ref):
            self.quad.shl1_inplace(word)
            return
        super().shl1_inplace(word)

    def shr1_inplace(self, word) -> None:
        if isinstance(word, Quad64Ref):
            self.quad.shr1_inplace(word)
            return
        super().shr1_inplace(word)

    def shl_const(self, dst, src, amount: int) -> None:
        if self._all_quad(dst, src):
            amount = max(0, amount)
            if amount >= DIGITS * 2:
                self.quad.set_u64(dst, 0)
                return
            if not self._same(dst, src):
                self.quad.copy64(dst, src)
            for _ in range(amount):
                self.quad.shl1_inplace(dst)
            return
        super().shl_const(dst, src, amount)

    def shr_const(self, dst, src, amount: int) -> None:
        if self._all_quad(dst, src):
            amount = max(0, amount)
            if amount >= DIGITS * 2:
                self.quad.set_u64(dst, 0)
                return
            if not self._same(dst, src):
                self.quad.copy64(dst, src)
            for _ in range(amount):
                self.quad.shr1_inplace(dst)
            return
        super().shr_const(dst, src, amount)

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

    # ------------------------------------------------------------------
    # compact Quad truthiness
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # alias-safe increment/negate
    # ------------------------------------------------------------------
    def _quad_local_increment(self, word: Quad64Ref) -> None:
        """Increment a Quad word using nearby global carry scratch.

        This path is mainly for shared-workspace Quad temporaries, where a
        second full Quad temporary would overlap the compiler temp arena.
        """
        bf = self.bf
        self._clear_scratch()
        bf.set_const(self.carry0, 1)
        carry_in, carry_out = self.carry0, self.carry1
        for bit_index in range(64):
            bf.clear(carry_out)
            bf.begin_while(carry_in)
            bf.add_const(carry_in, -1)
            self._add_one(word.bit(bit_index), carry_out, self.s2)
            bf.end_while(carry_in)
            carry_in, carry_out = carry_out, carry_in
        self._clear_scratch()

    def _quad_local_negate(self, word: Quad64Ref) -> None:
        for bit_index in range(64):
            self._toggle_bit(word.bit(bit_index), self.s0)
        self._clear_scratch()
        self._quad_local_increment(word)

    def _inc64_inplace(self, word) -> None:
        if isinstance(word, Quad64Ref):
            q0 = self._qtmp(0)
            q1 = self._qtmp(1)
            if word.base in (q0.base, q1.base):
                self._quad_local_increment(word)
                return
            self.quad.set_u64(q0, 1)
            self.quad.add64(q1, word, q0)
            self.copy64(word, q1)
            return
        super()._inc64_inplace(word)

    def _neg64_inplace(self, word) -> None:
        if isinstance(word, Quad64Ref):
            q0 = self._qtmp(0)
            q1 = self._qtmp(1)
            if word.base in (q0.base, q1.base):
                self._quad_local_negate(word)
                return
            self.quad.set_u64(q0, 0)
            self.quad.sub64(q1, q0, word)
            self.copy64(word, q1)
            return
        super()._neg64_inplace(word)

    # ------------------------------------------------------------------
    # compact decimal output
    # ------------------------------------------------------------------
    @staticmethod
    def _decimal_is_zero_preserved(
        r: _RelativeBuilder,
        result: int,
        src: int,
        tmp: int,
        helper: int,
    ) -> None:
        """result = (src == 0), preserving src; all offsets are lane-local."""
        r.clear(result)
        r.add(result, 1)
        r.clear(tmp)
        r.clear(helper)

        r.move(src)
        r.emit("[")
        r.add(src, -1)
        r.add(tmp, 1)
        r.add(helper, 1)
        r.move(src)
        r.emit("]")

        r.move(helper)
        r.emit("[")
        r.add(helper, -1)
        r.add(src, 1)
        r.move(helper)
        r.emit("]")

        r.move(tmp)
        r.emit("[")
        r.clear(tmp)
        r.clear(result)
        r.move(tmp)
        r.emit("]")

    def _decimal_lane_body(self) -> str:
        """One base-10 lane for ``digit = digit*2 + carry``.

        Lane layout::

            marker, digit, carry, gate, remainder, zero, tmp

        ``marker`` is consumed by the outer walker and then reused as helper.
        Since an existing digit is in 0..9, one five-count wrap is enough to
        determine the outgoing carry, and the residual count is ``digit % 5``.
        """
        r = _RelativeBuilder()
        marker = 0
        digit = 1
        carry = 2
        gate = 3
        remainder = 4
        zero = 5
        tmp = 6
        helper = marker
        next_carry = _DECIMAL_STRIDE + carry

        r.clear(marker)
        r.clear(remainder)
        r.clear(zero)
        r.clear(tmp)
        r.clear(gate)
        r.add(gate, 5)
        r.clear(next_carry)

        r.move(digit)
        r.emit("[")
        r.add(digit, -1)
        r.add(remainder, 1)
        r.add(gate, -1)
        self._decimal_is_zero_preserved(r, zero, gate, tmp, helper)
        r.move(zero)
        r.emit("[")
        r.add(zero, -1)
        r.clear(remainder)
        r.clear(next_carry)
        r.add(next_carry, 1)
        r.clear(gate)
        r.add(gate, 5)
        r.move(zero)
        r.emit("]")
        r.move(digit)
        r.emit("]")

        # digit = 2 * (old_digit % 5)
        r.move(remainder)
        r.emit("[")
        r.add(remainder, -1)
        r.add(digit, 2)
        r.move(remainder)
        r.emit("]")

        # + incoming carry; carry is consumed.
        r.move(carry)
        r.emit("[")
        r.add(carry, -1)
        r.add(digit, 1)
        r.move(carry)
        r.emit("]")

        r.move(_DECIMAL_STRIDE)
        return r.code()

    def _decimal_push_bit(self, magnitude: Quad64Ref, bit_index: int, base: int) -> None:
        """digits = digits*2 + selected magnitude bit."""
        bf = self.bf
        first_carry = base + 2
        bf.clear(first_carry)

        # magnitude is a private print copy, so the source bit can be consumed.
        source = magnitude.bit(bit_index)
        bf.begin_while(source)
        bf.add_const(source, -1)
        bf.add_const(first_carry, 1)
        bf.end_while(source)

        for digit in range(_DECIMAL_DIGITS):
            bf.set_const(base + digit * _DECIMAL_STRIDE, 1)
        sentinel = base + _DECIMAL_DIGITS * _DECIMAL_STRIDE
        bf.clear(sentinel)

        bf.move(base)
        bf.emit("[" + self._decimal_lane_body() + "]")
        bf.ptr = sentinel

    def _quad_to_decimal_bytes(self, magnitude: Quad64Ref, base: int) -> None:
        # Twenty lanes plus one sentinel lane fit inside the existing 265-cell
        # signed-divmod shared workspace: WORD_CELLS + 21*7 = 246 cells.
        for offset in range((_DECIMAL_DIGITS + 1) * _DECIMAL_STRIDE):
            self.bf.clear(base + offset)

        for bit_index in range(63, -1, -1):
            self._decimal_push_bit(magnitude, bit_index, base)

    def _print_decimal_bytes(self, base: int) -> None:
        bf = self.bf
        sentinel = base + _DECIMAL_DIGITS * _DECIMAL_STRIDE
        started = sentinel
        control = sentinel + 1
        ascii_cell = sentinel + 2
        tmp = sentinel + 3
        helper = sentinel + 4

        for cell in (started, control, ascii_cell, tmp, helper):
            bf.clear(cell)

        for digit_index in range(_DECIMAL_DIGITS - 1, -1, -1):
            digit = base + digit_index * _DECIMAL_STRIDE + 1
            if digit_index == 0:
                bf.set_const(started, 1)
            else:
                self.copy_cell(digit, tmp, helper)
                bf.begin_while(tmp)
                bf.clear(tmp)
                bf.set_const(started, 1)
                bf.end_while(tmp)

            self.copy_cell(started, control, helper)
            bf.begin_while(control)
            bf.add_const(control, -1)
            bf.set_const(ascii_cell, ord("0"))
            # Decimal digits are dead after output, so consume directly.
            bf.begin_while(digit)
            bf.add_const(digit, -1)
            bf.add_const(ascii_cell, 1)
            bf.end_while(digit)
            bf.move(ascii_cell)
            bf.emit(".")
            bf.end_while(control)

    def _print_quad_magnitude(self, magnitude: Quad64Ref, workspace_base: int) -> None:
        decimal_base = workspace_base + WORD_CELLS
        self._quad_to_decimal_bytes(magnitude, decimal_base)
        self._print_decimal_bytes(decimal_base)

    def print_u64(self, src, workspace_base: int) -> None:
        if not isinstance(src, Quad64Ref):
            return super().print_u64(src, workspace_base)

        magnitude = self._qtmp(0)
        self.copy64(magnitude, src)
        self._print_quad_magnitude(magnitude, workspace_base)

    def print_s64(self, src, workspace_base: int) -> None:
        if not isinstance(src, Quad64Ref):
            return super().print_s64(src, workspace_base)

        magnitude = self._qtmp(0)
        self.copy64(magnitude, src)

        sign = self.s0
        self.copy_cell(src.bit(63), sign, self.s2)
        self.bf.begin_while(sign)
        self.bf.add_const(sign, -1)
        self.bf.set_const(self.s1, ord("-"))
        self.bf.move(self.s1)
        self.bf.emit(".")
        self._neg64_inplace(magnitude)
        self.bf.end_while(sign)
        self._clear_scratch()

        self._print_quad_magnitude(magnitude, workspace_base)


__all__ = ["QuadBinaryStringListIO"]
