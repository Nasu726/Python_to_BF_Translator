"""Higher 64-bit arithmetic built from the verified primitive backend.

The important code-size rule here is that 64 iterations are *runtime*
Brainfuck loops, not 64 copies emitted by Python.  A naive shift/add multiply
that emits one adder per multiplier bit makes multi-megabyte BF programs.  We
instead copy the operands into mutable workspace words and reuse one emitted
adder while shifting them at runtime.
"""

from bfcompare import Binary64Compare
from bfcore import Int64Ref, WORD_BITS
from bfwordops import Binary64WordOps


class Binary64Arithmetic(Binary64Compare, Binary64WordOps):
    """General 64-bit integer arithmetic.

    Workspace passed to the methods below must be disjoint from operands,
    destinations, and the five scratch cells owned by Binary64Core.
    """

    MUL_WORKSPACE_CELLS = WORD_BITS * 3 + 1
    UDIVMOD_WORKSPACE_CELLS = WORD_BITS * 2 + 2

    def _clear_word(self, word: Int64Ref) -> None:
        for i in range(WORD_BITS):
            self.bf.clear(word.bit(i))

    def mul64(
        self,
        dst: Int64Ref,
        a: Int64Ref,
        b_ref: Int64Ref,
        workspace_base: int,
    ) -> None:
        """dst = a*b modulo 2**64, preserving a and b.

        Workspace layout::

            [multiplicand:64][multiplier:64][tmp:64][counter]

        The loop is classic shift/add multiplication.  The low multiplier bit
        is deliberately consumed by the conditional because it is discarded
        by the following right shift anyway.
        """
        bf = self.bf
        multiplicand = Int64Ref(workspace_base)
        multiplier = Int64Ref(workspace_base + WORD_BITS)
        tmp = Int64Ref(workspace_base + WORD_BITS * 2)
        counter = workspace_base + WORD_BITS * 3

        self.copy64(multiplicand, a)
        self.copy64(multiplier, b_ref)
        self._clear_word(dst)
        self._clear_word(tmp)
        bf.set_const(counter, WORD_BITS)

        bf.begin_while(counter)

        # multiplier.bit(0) is 0/1.  If it is one, consume that bit and add
        # the current multiplicand.  Since the multiplier is shifted right
        # immediately afterwards this destructive test costs no restoration.
        bf.begin_while(multiplier.bit(0))
        bf.add_const(multiplier.bit(0), -1)
        self.add64(tmp, dst, multiplicand)
        self.copy64(dst, tmp)
        bf.end_while(multiplier.bit(0))

        self.shl1_inplace(multiplicand)
        self.shr1_inplace(multiplier)
        bf.add_const(counter, -1)
        bf.end_while(counter)

        self._clear_scratch()

    def udivmod64(
        self,
        quotient: Int64Ref,
        remainder: Int64Ref,
        dividend: Int64Ref,
        divisor: Int64Ref,
        workspace_base: int,
    ) -> None:
        """Unsigned restoring division, preserving dividend and divisor.

        Precondition: divisor != 0.  Division-by-zero policy belongs to the
        frontend/runtime layer; keeping it out of this primitive avoids baking
        an exception convention into the arithmetic backend.

        Workspace layout::

            [mutable_dividend:64][tmp:64][counter][compare_flag]

        One emitted loop body performs all 64 restoring-division iterations.
        The mutable dividend is shifted left so its current MSB supplies input
        bits from most-significant to least-significant order.
        """
        bf = self.bf
        moving_dividend = Int64Ref(workspace_base)
        tmp = Int64Ref(workspace_base + WORD_BITS)
        counter = workspace_base + WORD_BITS * 2
        ge_flag = counter + 1

        self.copy64(moving_dividend, dividend)
        self._clear_word(quotient)
        self._clear_word(remainder)
        self._clear_word(tmp)
        bf.set_const(counter, WORD_BITS)
        bf.clear(ge_flag)

        bf.begin_while(counter)

        # remainder = (remainder << 1) | next_dividend_bit
        self.shl1_inplace(remainder)
        self.copy_cell(moving_dividend.bit(WORD_BITS - 1), remainder.bit(0), self.s0)
        self.shl1_inplace(moving_dividend)

        # Building Q left-to-right avoids addressing a different quotient bit
        # on each runtime iteration.
        self.shl1_inplace(quotient)

        self.uge64(ge_flag, remainder, divisor)
        bf.begin_while(ge_flag)
        bf.add_const(ge_flag, -1)
        self.sub64(tmp, remainder, divisor)
        self.copy64(remainder, tmp)
        bf.set_const(quotient.bit(0), 1)
        bf.end_while(ge_flag)

        bf.add_const(counter, -1)
        bf.end_while(counter)

        self._clear_scratch()
