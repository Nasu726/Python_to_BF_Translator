"""Hybrid compiler backend with source-compact hot int64 operations.

The public compiler still needs strings, packed lists, decimal I/O and the
correctness-first binary primitives.  Replacing that entire stack at once would
be unnecessarily risky.  This adapter keeps ``BinaryStringListIO`` as the
fallback implementation but dispatches hot scalar operations to ``Quad64Core``
when all participating values are ``Quad64Ref`` instances.
"""

from __future__ import annotations

from bfquad import WORD_CELLS, Quad64Core, Quad64Ref
from bfstringlists import BinaryStringListIO


class QuadBinaryStringListIO(BinaryStringListIO):
    """String/list backend plus runtime-lane scalar add/sub/comparison."""

    def __init__(self, bf, scratch_base: int) -> None:
        super().__init__(bf, scratch_base=scratch_base)
        self.quad = Quad64Core(bf)
        self._quad_workspace_base: int | None = None

    def set_quad_workspace(self, base: int) -> None:
        # Two Quad words fit inside the existing shared signed-divmod workspace.
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

    def copy64(self, dst, src) -> None:
        if isinstance(dst, Quad64Ref) and isinstance(src, Quad64Ref) and dst.base == src.base:
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
                # Quad64Core uses B's marker cells as lane-local scratch, so
                # aliased operands must be separated physically first.
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
            # Signed ordering is unsigned ordering after XORing the sign bit of
            # both operands with 1<<63.  Mutate only for the duration of the
            # comparison and restore before returning.
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
            # Equality iff both unsigned >= directions hold.
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


__all__ = ["QuadBinaryStringListIO"]
