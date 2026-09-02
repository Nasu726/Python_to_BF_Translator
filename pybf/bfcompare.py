"""64-bit comparison operations for the Brainfuck backend.

The comparator does not materialize a 64-bit subtraction result.  It streams
one full-adder bit at a time for ``a + ~b + 1`` and keeps only the final carry.
For unsigned words that carry is exactly ``a >= b``.

Signed two's-complement ordering is reduced to unsigned ordering by flipping
the sign bit of both operands.  We fold that transform into the MSB step, so
no temporary 64-bit words are needed.
"""

from bfcore import Binary64Core, Int64Ref, WORD_BITS


class Binary64Compare(Binary64Core):
    def _ge64(self, result: int, a: Int64Ref, b_ref: Int64Ref, *, signed: bool) -> None:
        """Set result to one iff a >= b in the selected ordering."""
        bf = self.bf
        self._clear_scratch()
        bf.set_const(self.carry0, 1)  # +1 in a + ~b + 1
        carry_in, carry_out = self.carry0, self.carry1

        # result doubles as the throw-away difference bit.  Only the final
        # carry survives, so comparison needs no temporary 64-bit word.
        for i in range(WORD_BITS):
            bf.clear(result)
            bf.clear(carry_out)

            if signed and i == WORD_BITS - 1:
                # Signed ordering equals unsigned ordering after XORing both
                # operands with 1<<63.  For a' + ~b' + 1 at the MSB:
                #   a'_63  = ~a_63
                #   ~b'_63 =  b_63
                self.copy_cell(a.bit(i), self.s0, self.s1)
                self._toggle_bit(self.s0, self.s2)
                bf.begin_while(self.s0)
                bf.add_const(self.s0, -1)
                self._add_one(result, carry_out, self.s2)
                bf.end_while(self.s0)

                self._add_preserved_source_bit(b_ref.bit(i), result, carry_out)
            else:
                self._add_preserved_source_bit(a.bit(i), result, carry_out)

                # Add ~b_i without mutating b.
                self.copy_cell(b_ref.bit(i), self.s0, self.s1)
                self._toggle_bit(self.s0, self.s2)
                bf.begin_while(self.s0)
                bf.add_const(self.s0, -1)
                self._add_one(result, carry_out, self.s2)
                bf.end_while(self.s0)

            bf.begin_while(carry_in)
            bf.add_const(carry_in, -1)
            self._add_one(result, carry_out, self.s2)
            bf.end_while(carry_in)

            carry_in, carry_out = carry_out, carry_in

        self.copy_cell(carry_in, result, self.s0)
        self._clear_scratch()

    def uge64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self._ge64(result, a, b, signed=False)

    def ult64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self.uge64(result, a, b)
        self._toggle_bit(result, self.s0)
        self._clear_scratch()

    def ule64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self.uge64(result, b, a)

    def ugt64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self.uge64(result, b, a)
        self._toggle_bit(result, self.s0)
        self._clear_scratch()

    def sge64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self._ge64(result, a, b, signed=True)

    def slt64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self.sge64(result, a, b)
        self._toggle_bit(result, self.s0)
        self._clear_scratch()

    def sle64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self.sge64(result, b, a)

    def sgt64(self, result: int, a: Int64Ref, b: Int64Ref) -> None:
        self.sge64(result, b, a)
        self._toggle_bit(result, self.s0)
        self._clear_scratch()
