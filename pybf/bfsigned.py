"""Signed two's-complement helpers, including Python-style // and %.

The project deliberately uses fixed-width 64-bit integers, so ordinary
addition/subtraction/multiplication already have the correct two's-complement
bit patterns.  Division is the operation where Python semantics need extra
work: Python floors toward -infinity and the remainder has the divisor's sign.
"""

from bfarith import Binary64Arithmetic
from bfcore import Int64Ref, WORD_BITS


class Binary64Signed(Binary64Arithmetic):
    """Signed operations on 64-bit two's-complement words."""

    # abs(a), abs(b), unsigned-divmod workspace, then seven one-byte flags.
    SDIVMOD_WORKSPACE_CELLS = WORD_BITS * 2 + Binary64Arithmetic.UDIVMOD_WORKSPACE_CELLS + 7

    def _not64_inplace(self, word: Int64Ref) -> None:
        for i in range(WORD_BITS):
            self._toggle_bit(word.bit(i), self.s0)
        self._clear_scratch()

    def _inc64_inplace(self, word: Int64Ref) -> None:
        """word = word + 1 modulo 2**64."""
        bf = self.bf
        self._clear_scratch()
        bf.set_const(self.carry0, 1)
        carry_in, carry_out = self.carry0, self.carry1
        for i in range(WORD_BITS):
            bf.clear(carry_out)
            bf.begin_while(carry_in)
            bf.add_const(carry_in, -1)
            self._add_one(word.bit(i), carry_out, self.s2)
            bf.end_while(carry_in)
            carry_in, carry_out = carry_out, carry_in
        self._clear_scratch()

    def _neg64_inplace(self, word: Int64Ref) -> None:
        self._not64_inplace(word)
        self._inc64_inplace(word)

    def _is_nonzero64(self, result: int, word: Int64Ref) -> None:
        """result = bool(word), preserving word."""
        bf = self.bf
        bf.clear(result)
        for i in range(WORD_BITS):
            self.copy_cell(word.bit(i), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            bf.set_const(result, 1)
            bf.end_while(self.s0)
        self._clear_scratch()

    def _abs64_into(self, dst: Int64Ref, src: Int64Ref, control: int) -> None:
        """Copy |src| as an unsigned magnitude into dst."""
        bf = self.bf
        self.copy64(dst, src)
        self.copy_cell(src.bit(WORD_BITS - 1), control, self.s0)
        bf.begin_while(control)
        bf.add_const(control, -1)
        self._neg64_inplace(dst)
        bf.end_while(control)

    def sdivmod64(
        self,
        quotient: Int64Ref,
        remainder: Int64Ref,
        dividend: Int64Ref,
        divisor: Int64Ref,
        workspace_base: int,
    ) -> None:
        """Python-style signed floor division and modulo.

        Inputs and outputs are 64-bit two's-complement bit patterns.
        ``divisor`` must be nonzero.  Results wrap to 64 bits; consequently
        the one mathematical overflow case, INT64_MIN // -1, produces the
        wrapped INT64_MIN bit pattern rather than Python's unbounded +2**63.

        For nonzero remainder, opposite operand signs require floor correction:

            trunc quotient  -> one more negative
            |remainder|      -> |divisor| - |remainder|

        The final remainder is then given the divisor's sign, matching Python.
        """
        bf = self.bf
        abs_a = Int64Ref(workspace_base)
        abs_b = Int64Ref(workspace_base + WORD_BITS)
        inner_workspace = workspace_base + WORD_BITS * 2
        flags = inner_workspace + self.UDIVMOD_WORKSPACE_CELLS

        sign_a = flags
        sign_b = flags + 1
        signs_differ = flags + 2
        rem_nonzero = flags + 3
        rem_zero = flags + 4
        control0 = flags + 5
        control1 = flags + 6

        self.copy_cell(dividend.bit(WORD_BITS - 1), sign_a, self.s0)
        self.copy_cell(divisor.bit(WORD_BITS - 1), sign_b, self.s0)
        self._abs64_into(abs_a, dividend, control0)
        self._abs64_into(abs_b, divisor, control0)

        self.udivmod64(
            quotient,
            remainder,
            abs_a,
            abs_b,
            workspace_base=inner_workspace,
        )

        # signs_differ = sign_a XOR sign_b
        self.copy_cell(sign_a, signs_differ, self.s0)
        self.copy_cell(sign_b, control0, self.s0)
        bf.begin_while(control0)
        bf.add_const(control0, -1)
        self._toggle_bit(signs_differ, self.s0)
        bf.end_while(control0)

        self._is_nonzero64(rem_nonzero, remainder)
        self.copy_cell(rem_nonzero, rem_zero, self.s0)
        self._toggle_bit(rem_zero, self.s0)
        self._clear_scratch()

        # Quotient correction.
        # signs differ, r != 0: floor(-q_abs - fraction) = -(q_abs+1) = ~q_abs
        # signs differ, r == 0: exact quotient = -q_abs
        self.copy_cell(signs_differ, control0, self.s0)
        bf.begin_while(control0)
        bf.add_const(control0, -1)

        self.copy_cell(rem_nonzero, control1, self.s0)
        bf.begin_while(control1)
        bf.add_const(control1, -1)
        self._not64_inplace(quotient)
        bf.end_while(control1)

        self.copy_cell(rem_zero, control1, self.s0)
        bf.begin_while(control1)
        bf.add_const(control1, -1)
        self._neg64_inplace(quotient)
        bf.end_while(control1)

        bf.end_while(control0)

        # If signs differ and the unsigned remainder was nonzero, change its
        # magnitude from r to |b|-r.  The second word of udivmod's workspace
        # is dead now and can be reused as the subtraction destination.
        adjust_tmp = Int64Ref(inner_workspace + WORD_BITS)
        self.copy_cell(rem_nonzero, control0, self.s0)
        bf.begin_while(control0)
        bf.add_const(control0, -1)
        self.copy_cell(signs_differ, control1, self.s0)
        bf.begin_while(control1)
        bf.add_const(control1, -1)
        self.sub64(adjust_tmp, abs_b, remainder)
        self.copy64(remainder, adjust_tmp)
        bf.end_while(control1)
        bf.end_while(control0)

        # Python remainder has the divisor's sign.  Negating zero is harmless.
        self.copy_cell(sign_b, control0, self.s0)
        bf.begin_while(control0)
        bf.add_const(control0, -1)
        self._neg64_inplace(remainder)
        bf.end_while(control0)

        self._clear_scratch()
