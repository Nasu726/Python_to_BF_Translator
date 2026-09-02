"""Additional word operations built only from bfcore primitives."""

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS


class Binary64WordOps(Binary64Core):
    def not64(self, dst: Int64Ref, src: Int64Ref) -> None:
        for i in range(WORD_BITS):
            self.copy_cell(src.bit(i), dst.bit(i), self.s0)
            self._toggle_bit(dst.bit(i), self.s0)
        self._clear_scratch()

    def xor64(self, dst: Int64Ref, a: Int64Ref, b: Int64Ref) -> None:
        bf = self.bf
        for i in range(WORD_BITS):
            self.copy_cell(a.bit(i), dst.bit(i), self.s0)
            self.copy_cell(b.bit(i), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            self._toggle_bit(dst.bit(i), self.s2)
            bf.end_while(self.s0)
        self._clear_scratch()

    def and64(self, dst: Int64Ref, a: Int64Ref, b: Int64Ref) -> None:
        bf = self.bf
        for i in range(WORD_BITS):
            bf.clear(dst.bit(i))
            self.copy_cell(b.bit(i), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            self.copy_cell(a.bit(i), dst.bit(i), self.s1)
            bf.end_while(self.s0)
        self._clear_scratch()

    def or64(self, dst: Int64Ref, a: Int64Ref, b: Int64Ref) -> None:
        bf = self.bf
        for i in range(WORD_BITS):
            self.copy_cell(a.bit(i), dst.bit(i), self.s0)
            self.copy_cell(b.bit(i), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            bf.set_const(dst.bit(i), 1)
            bf.end_while(self.s0)
        self._clear_scratch()

    def shl1_inplace(self, word: Int64Ref) -> None:
        """word = word << 1 modulo 2**64."""
        for i in range(WORD_BITS - 1, 0, -1):
            self.copy_cell(word.bit(i - 1), word.bit(i), self.s0)
        self.bf.clear(word.bit(0))
        self._clear_scratch()

    def shr1_inplace(self, word: Int64Ref) -> None:
        """Unsigned logical right shift."""
        for i in range(WORD_BITS - 1):
            self.copy_cell(word.bit(i + 1), word.bit(i), self.s0)
        self.bf.clear(word.bit(WORD_BITS - 1))
        self._clear_scratch()

    def shl_const(self, dst: Int64Ref, src: Int64Ref, amount: int) -> None:
        amount = max(0, amount)
        if amount >= WORD_BITS:
            for i in range(WORD_BITS):
                self.bf.clear(dst.bit(i))
            return
        for i in range(WORD_BITS):
            if i < amount:
                self.bf.clear(dst.bit(i))
            else:
                self.copy_cell(src.bit(i - amount), dst.bit(i), self.s0)
        self._clear_scratch()

    def shr_const(self, dst: Int64Ref, src: Int64Ref, amount: int) -> None:
        amount = max(0, amount)
        if amount >= WORD_BITS:
            for i in range(WORD_BITS):
                self.bf.clear(dst.bit(i))
            return
        for i in range(WORD_BITS):
            j = i + amount
            if j >= WORD_BITS:
                self.bf.clear(dst.bit(i))
            else:
                self.copy_cell(src.bit(j), dst.bit(i), self.s0)
        self._clear_scratch()
