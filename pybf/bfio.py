"""Decimal I/O for the 64-bit Brainfuck backend.

Output deliberately does not use repeated general division by ten.  Instead it
uses double-dabble with twenty 4-bit BCD digits.  All BCD state remains Boolean,
which maps much better to Brainfuck than repeatedly comparing byte-valued
0..19 decimal digits.
"""

from bfcore import Int64Ref, WORD_BITS
from bfsigned import Binary64Signed


class Binary64IO(Binary64Signed):
    PRINT_WORKSPACE_CELLS = 150
    READ_WORKSPACE_CELLS = 132

    def _set_bcd_ge5(self, result: int, nibble: int) -> None:
        """result = 1 iff the 4-bit BCD nibble is at least five."""
        bf = self.bf
        bf.clear(result)

        # d >= 5 iff b3 OR (b2 AND (b1 OR b0)).
        self.copy_cell(nibble + 3, self.s0, self.s1)
        bf.begin_while(self.s0)
        bf.add_const(self.s0, -1)
        bf.set_const(result, 1)
        bf.end_while(self.s0)

        self.copy_cell(nibble + 2, self.s0, self.s1)
        bf.begin_while(self.s0)
        bf.add_const(self.s0, -1)

        self.copy_cell(nibble + 1, self.s1, self.s2)
        bf.begin_while(self.s1)
        bf.add_const(self.s1, -1)
        bf.set_const(result, 1)
        bf.end_while(self.s1)

        self.copy_cell(nibble, self.s1, self.s2)
        bf.begin_while(self.s1)
        bf.add_const(self.s1, -1)
        bf.set_const(result, 1)
        bf.end_while(self.s1)

        bf.end_while(self.s0)

    def _ripple_increment_bits(self, base: int, first_bit: int) -> None:
        """Increment a four-bit nibble by 1<<first_bit."""
        bf = self.bf
        bf.clear(self.carry0)
        bf.clear(self.carry1)
        bf.set_const(self.carry0, 1)
        carry_in, carry_out = self.carry0, self.carry1
        for bit in range(first_bit, 4):
            bf.clear(carry_out)
            bf.begin_while(carry_in)
            bf.add_const(carry_in, -1)
            self._add_one(base + bit, carry_out, self.s2)
            bf.end_while(carry_in)
            carry_in, carry_out = carry_out, carry_in
        bf.clear(self.carry0)
        bf.clear(self.carry1)

    def _add3_bcd_if(self, nibble: int, control: int) -> None:
        bf = self.bf
        bf.begin_while(control)
        bf.add_const(control, -1)
        self._ripple_increment_bits(nibble, 0)  # +1
        self._ripple_increment_bits(nibble, 1)  # +2
        bf.end_while(control)

    def _bcd_from_magnitude(
        self,
        magnitude: Int64Ref,
        bcd_base: int,
        counter: int,
        ge5_flag: int,
    ) -> None:
        """Convert a mutable unsigned word to twenty BCD nibbles."""
        bf = self.bf
        for i in range(80):
            bf.clear(bcd_base + i)
        bf.set_const(counter, WORD_BITS)
        bf.clear(ge5_flag)

        # One emitted body, executed 64 times at runtime.
        bf.begin_while(counter)
        for digit in range(20):
            nibble = bcd_base + digit * 4
            self._set_bcd_ge5(ge5_flag, nibble)
            self._add3_bcd_if(nibble, ge5_flag)

        # Shift the complete 20-nibble BCD register and feed the next MSB.
        for i in range(79, 0, -1):
            self.copy_cell(bcd_base + i - 1, bcd_base + i, self.s0)
        self.copy_cell(magnitude.bit(WORD_BITS - 1), bcd_base, self.s0)
        self.shl1_inplace(magnitude)

        bf.add_const(counter, -1)
        bf.end_while(counter)
        self._clear_scratch()

    def _print_bcd_digits(
        self,
        bcd_base: int,
        started: int,
        ascii_cell: int,
        control: int,
    ) -> None:
        bf = self.bf
        bf.clear(started)
        bf.clear(ascii_cell)
        bf.clear(control)

        for digit in range(19, -1, -1):
            nibble = bcd_base + digit * 4
            if digit == 0:
                # Always print the ones digit so zero becomes "0".
                bf.set_const(started, 1)
            else:
                # Any set nibble bit starts zero-unsuppressed output.
                for bit in range(4):
                    self.copy_cell(nibble + bit, control, self.s0)
                    bf.begin_while(control)
                    bf.add_const(control, -1)
                    bf.set_const(started, 1)
                    bf.end_while(control)

            self.copy_cell(started, control, self.s0)
            bf.begin_while(control)
            bf.add_const(control, -1)
            bf.set_const(ascii_cell, ord('0'))
            for bit, weight in enumerate((1, 2, 4, 8)):
                self.copy_cell(nibble + bit, self.s0, self.s1)
                bf.begin_while(self.s0)
                bf.add_const(self.s0, -1)
                bf.add_const(ascii_cell, weight)
                bf.end_while(self.s0)
            bf.move(ascii_cell)
            bf.emit('.')
            bf.end_while(control)

        self._clear_scratch()

    def print_u64(self, src: Int64Ref, workspace_base: int) -> None:
        bf = self.bf
        magnitude = Int64Ref(workspace_base)
        bcd_base = workspace_base + WORD_BITS
        counter = bcd_base + 80
        ge5_flag = counter + 1
        started = ge5_flag + 1
        ascii_cell = started + 1
        control = ascii_cell + 1

        self.copy64(magnitude, src)
        self._bcd_from_magnitude(magnitude, bcd_base, counter, ge5_flag)
        self._print_bcd_digits(bcd_base, started, ascii_cell, control)

    def print_s64(self, src: Int64Ref, workspace_base: int) -> None:
        bf = self.bf
        magnitude = Int64Ref(workspace_base)
        bcd_base = workspace_base + WORD_BITS
        counter = bcd_base + 80
        ge5_flag = counter + 1
        sign = ge5_flag + 1
        started = sign + 1
        ascii_cell = started + 1
        control = ascii_cell + 1

        self.copy64(magnitude, src)
        self.copy_cell(src.bit(WORD_BITS - 1), sign, self.s0)
        bf.begin_while(sign)
        bf.add_const(sign, -1)
        bf.set_const(ascii_cell, ord('-'))
        bf.move(ascii_cell)
        bf.emit('.')
        self._neg64_inplace(magnitude)
        bf.end_while(sign)

        self._bcd_from_magnitude(magnitude, bcd_base, counter, ge5_flag)
        self._print_bcd_digits(bcd_base, started, ascii_cell, control)

    def print_char(self, value: int, cell: int) -> None:
        self.bf.set_const(cell, value)
        self.bf.move(cell)
        self.bf.emit('.')

    def print_space(self, cell: int) -> None:
        self.print_char(ord(' '), cell)

    def print_newline(self, cell: int) -> None:
        self.print_char(ord('\n'), cell)

    def _eq_byte_const(self, result: int, cell: int, value: int) -> None:
        """result = (cell == value), preserving cell."""
        bf = self.bf
        self.copy_cell(cell, self.s0, self.s1)
        bf.add_const(self.s0, -value)
        bf.set_const(result, 1)
        bf.begin_while(self.s0)
        bf.clear(self.s0)
        bf.clear(result)
        bf.end_while(self.s0)

    def read_s64(self, dst: Int64Ref, workspace_base: int) -> None:
        """Read one newline-terminated decimal integer into dst.

        Supports an optional leading '-'.  The intended contest input grammar
        is ``-?[0-9]+\\n``; general Python whitespace handling is deliberately
        left to the frontend/runtime policy layer.
        """
        bf = self.bf
        original = Int64Ref(workspace_base)
        summed = Int64Ref(workspace_base + WORD_BITS)
        ch = workspace_base + WORD_BITS * 2
        sign = ch + 1
        is_minus = ch + 2
        loop = ch + 3

        self._clear_word(dst)
        bf.clear(ch)
        bf.clear(sign)
        bf.clear(is_minus)
        bf.clear(loop)

        bf.move(ch)
        bf.emit(',')
        self._eq_byte_const(is_minus, ch, ord('-'))
        bf.begin_while(is_minus)
        bf.add_const(is_minus, -1)
        bf.set_const(sign, 1)
        bf.move(ch)
        bf.emit(',')
        bf.end_while(is_minus)

        # loop = (ch != '\n')
        self._eq_byte_const(loop, ch, ord('\n'))
        self._toggle_bit(loop, self.s0)

        bf.begin_while(loop)
        bf.add_const(loop, -1)

        # dst *= 10 as 2*dst + 8*dst.  The mutable copy means the emitted
        # multiply-by-ten body is compact and is reused for every input digit.
        self.copy64(original, dst)
        self.shl1_inplace(dst)
        self.shl1_inplace(original)
        self.shl1_inplace(original)
        self.shl1_inplace(original)
        self.add64(summed, dst, original)
        self.copy64(dst, summed)

        # ASCII digit -> at most nine increments.  ch is dead afterwards.
        bf.add_const(ch, -ord('0'))
        bf.begin_while(ch)
        bf.add_const(ch, -1)
        self._inc64_inplace(dst)
        bf.end_while(ch)

        bf.move(ch)
        bf.emit(',')
        self._eq_byte_const(loop, ch, ord('\n'))
        self._toggle_bit(loop, self.s0)
        bf.end_while(loop)

        bf.begin_while(sign)
        bf.add_const(sign, -1)
        self._neg64_inplace(dst)
        bf.end_while(sign)
        self._clear_scratch()
