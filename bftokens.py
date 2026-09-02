"""Whitespace-delimited scalar input on top of the typed BF backend."""

from bfcore import Int64Ref, WORD_BITS
from bfstrings import BinaryStringIO


class BinaryTokenIO(BinaryStringIO):
    TOKEN_WORKSPACE_CELLS = WORD_BITS * 2 + 12

    def _is_ws(self, result: int, ch: int, tmp: int) -> None:
        bf = self.bf
        bf.clear(result)
        for value in (ord(' '), ord('\t'), ord('\n'), ord('\r')):
            self._eq_byte_const(tmp, ch, value)
            bf.begin_while(tmp)
            bf.add_const(tmp, -1)
            bf.set_const(result, 1)
            bf.end_while(tmp)

    def _is_zero_byte(self, result: int, ch: int) -> None:
        self._eq_byte_const(result, ch, 0)

    def read_s64_token(self, dst: Int64Ref, workspace_base: int) -> None:
        """Read one signed decimal token separated by ASCII whitespace.

        Leading whitespace is skipped.  The delimiter after the token is
        consumed, making repeated calls suitable for common patterns such as
        ``a, b = map(int, input().split())`` on valid contest input.
        """
        bf = self.bf
        original = Int64Ref(workspace_base)
        summed = Int64Ref(workspace_base + WORD_BITS)
        ch = workspace_base + WORD_BITS * 2
        sign = ch + 1
        is_minus = ch + 2
        skip = ch + 3
        tmp = ch + 4
        active = ch + 5
        is_ws = ch + 6
        is_zero = ch + 7

        self._clear_word(dst)
        for c in (ch, sign, is_minus, skip, tmp, active, is_ws, is_zero):
            bf.clear(c)

        # Read the first non-whitespace byte. EOF (zero) is not treated as
        # skippable whitespace, so an exhausted stream deterministically
        # returns zero instead of looping forever.
        bf.move(ch)
        bf.emit(',')
        self._is_ws(skip, ch, tmp)
        bf.begin_while(skip)
        bf.add_const(skip, -1)
        bf.move(ch)
        bf.emit(',')
        self._is_ws(skip, ch, tmp)
        bf.end_while(skip)

        self._eq_byte_const(is_minus, ch, ord('-'))
        bf.begin_while(is_minus)
        bf.add_const(is_minus, -1)
        bf.set_const(sign, 1)
        bf.move(ch)
        bf.emit(',')
        bf.end_while(is_minus)

        # active = current byte is neither whitespace nor EOF.
        bf.set_const(active, 1)
        self._is_ws(is_ws, ch, tmp)
        bf.begin_while(is_ws)
        bf.add_const(is_ws, -1)
        bf.clear(active)
        bf.end_while(is_ws)
        self._is_zero_byte(is_zero, ch)
        bf.begin_while(is_zero)
        bf.add_const(is_zero, -1)
        bf.clear(active)
        bf.end_while(is_zero)

        bf.begin_while(active)
        bf.add_const(active, -1)

        # dst *= 10 using 2*x + 8*x.
        self.copy64(original, dst)
        self.shl1_inplace(dst)
        self.shl1_inplace(original)
        self.shl1_inplace(original)
        self.shl1_inplace(original)
        self.add64(summed, dst, original)
        self.copy64(dst, summed)

        # ASCII digit to repeated +1. Valid contest input keeps this in 0..9.
        bf.add_const(ch, -ord('0'))
        bf.begin_while(ch)
        bf.add_const(ch, -1)
        self._inc64_inplace(dst)
        bf.end_while(ch)

        bf.move(ch)
        bf.emit(',')
        bf.set_const(active, 1)
        self._is_ws(is_ws, ch, tmp)
        bf.begin_while(is_ws)
        bf.add_const(is_ws, -1)
        bf.clear(active)
        bf.end_while(is_ws)
        self._is_zero_byte(is_zero, ch)
        bf.begin_while(is_zero)
        bf.add_const(is_zero, -1)
        bf.clear(active)
        bf.end_while(is_zero)
        bf.end_while(active)

        bf.begin_while(sign)
        bf.add_const(sign, -1)
        self._neg64_inplace(dst)
        bf.end_while(sign)
        self._clear_scratch()
