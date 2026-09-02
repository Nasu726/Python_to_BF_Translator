"""Whitespace-delimited scalar input on top of the typed BF backend."""

from bfcore import Int64Ref, WORD_BITS
from bfstrings import BinaryStringIO


class BinaryTokenIO(BinaryStringIO):
    TOKEN_WORKSPACE_CELLS = WORD_BITS * 2 + 16

    def _is_ws(self, result: int, ch: int, tmp: int) -> None:
        bf = self.bf
        bf.clear(result)
        for value in (ord(' '), ord('\t'), ord('\n'), ord('\r')):
            self._eq_byte_const(tmp, ch, value)
            bf.begin_while(tmp)
            bf.add_const(tmp, -1)
            bf.set_const(result, 1)
            bf.end_while(tmp)

    def _is_hspace(self, result: int, ch: int, tmp: int) -> None:
        """Space accepted inside ``str.split()`` before the line terminator."""
        bf = self.bf
        bf.clear(result)
        for value in (ord(' '), ord('\t'), ord('\r')):
            self._eq_byte_const(tmp, ch, value)
            bf.begin_while(tmp)
            bf.add_const(tmp, -1)
            bf.set_const(result, 1)
            bf.end_while(tmp)

    def _is_zero_byte(self, result: int, ch: int) -> None:
        self._eq_byte_const(result, ch, 0)

    def _is_line_end(self, result: int, ch: int, tmp: int) -> None:
        bf = self.bf
        self._eq_byte_const(result, ch, ord('\n'))
        self._eq_byte_const(tmp, ch, 0)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        bf.set_const(result, 1)
        bf.end_while(tmp)

    def _mul10_add_ascii_digit(
        self,
        dst: Int64Ref,
        ch: int,
        original: Int64Ref,
        summed: Int64Ref,
    ) -> None:
        bf = self.bf
        self.copy64(original, dst)
        self.shl1_inplace(dst)
        self.shl1_inplace(original)
        self.shl1_inplace(original)
        self.shl1_inplace(original)
        self.add64(summed, dst, original)
        self.copy64(dst, summed)

        # Valid contest input guarantees '0'..'9'.
        bf.add_const(ch, -ord('0'))
        bf.begin_while(ch)
        bf.add_const(ch, -1)
        self._inc64_inplace(dst)
        bf.end_while(ch)

    def read_s64_token(self, dst: Int64Ref, workspace_base: int) -> None:
        """Read one signed decimal token separated by arbitrary whitespace.

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
        self._mul10_add_ascii_digit(dst, ch, original, summed)
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

    def read_s64_line_token(
        self,
        dst: Int64Ref,
        has_token: int,
        end_line: int,
        workspace_base: int,
    ) -> None:
        """Read at most one signed integer from the current input line.

        ``has_token`` is 1 when a value was read. ``end_line`` is 1 when the
        token (or leading horizontal whitespace) reached newline/EOF.  Unlike
        ``read_s64_token``, this never skips across a newline.  It is the
        primitive used to implement ``list(map(int, input().split()))``.
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
        delimiter = ch + 6
        gate = ch + 7
        line_tmp = ch + 8

        self._clear_word(dst)
        for c in (has_token, end_line, ch, sign, is_minus, skip, tmp, active, delimiter, gate, line_tmp):
            bf.clear(c)

        # Skip horizontal whitespace only; newline terminates this logical line.
        bf.move(ch)
        bf.emit(',')
        self._is_hspace(skip, ch, tmp)
        bf.begin_while(skip)
        bf.add_const(skip, -1)
        bf.move(ch)
        bf.emit(',')
        self._is_hspace(skip, ch, tmp)
        bf.end_while(skip)

        self._is_line_end(end_line, ch, tmp)
        bf.set_const(has_token, 1)
        self.copy_cell(end_line, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.clear(has_token)
        bf.end_while(gate)

        # Everything below is dynamically gated so an empty/end-of-line read
        # performs no digit work and leaves dst == 0.
        self.copy_cell(has_token, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)

        self._eq_byte_const(is_minus, ch, ord('-'))
        bf.begin_while(is_minus)
        bf.add_const(is_minus, -1)
        bf.set_const(sign, 1)
        bf.move(ch)
        bf.emit(',')
        bf.end_while(is_minus)

        # active = not delimiter. A delimiter is horizontal whitespace,
        # newline, or EOF-zero.
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
        self._mul10_add_ascii_digit(dst, ch, original, summed)
        bf.move(ch)
        bf.emit(',')

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
        self._neg64_inplace(dst)
        bf.end_while(sign)
        bf.end_while(gate)
        self._clear_scratch()
