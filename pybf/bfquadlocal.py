"""Distance-localized Quad64 operations for source-size control.

Quad64 arithmetic already executes one repeated lane body at runtime, but the
body contains literal tape moves between its operands, destination and scratch
word.  When ordinary scalar variables are far apart on the static tape, those
moves can dominate emitted Brainfuck source even though the arithmetic itself
is compact.

This adapter keeps the established Quad backend semantics and only changes the
physical lowering of sufficiently distant operations:

    distant operands -> nearby shared Quad snapshots -> operation -> writeback

The snapshots cost a few long copies, while the arithmetic/compare body then
uses only short deltas.  Nearby operations retain the original backend path.
The optimization is generic; it has no knowledge of any AtCoder task or AST
shape.
"""

from __future__ import annotations

from bfquad import WORD_CELLS, Quad64Ref
from bfquadbackend import QuadBinaryStringListIO


class DistanceLocalizedQuadBinaryStringListIO(QuadBinaryStringListIO):
    """Quad backend that snapshots distant hot operations into shared workspace."""

    # Two Quad-word distances is large enough that repeated operand/scratch
    # travel in arithmetic bodies normally exceeds the cost of snapshot copies.
    # Keep this deliberately conservative; profile data decides whether it
    # should change, not task-specific matching.
    LOCALIZE_DISTANCE = 2 * WORD_CELLS

    def _local_words(self) -> tuple[Quad64Ref, Quad64Ref, Quad64Ref]:
        return self._qtmp(0), self._qtmp(1), self._qtmp(2)

    def _is_local_word(self, ref) -> bool:
        if not isinstance(ref, Quad64Ref):
            return False
        return any(ref.base == local.base for local in self._local_words())

    def _should_localize(self, *refs) -> bool:
        if not refs or not self._all_quad(*refs):
            return False
        # Internal workspace calls must never recursively borrow the same words.
        if any(self._is_local_word(ref) for ref in refs):
            return False

        q0, _q1, _q2 = self._local_words()
        bases = [ref.base for ref in refs]
        spread = max(bases) - min(bases)
        workspace_distance = max(abs(base - q0.base) for base in bases)
        return (
            spread >= self.LOCALIZE_DISTANCE
            or workspace_distance >= self.LOCALIZE_DISTANCE
        )

    def _snapshot_pair(self, a: Quad64Ref, b: Quad64Ref):
        q0, q1, q2 = self._local_words()
        self.quad.copy64(q0, a)
        self.quad.copy64(q1, b)
        return q0, q1, q2

    # ------------------------------------------------------------------
    # arithmetic
    # ------------------------------------------------------------------
    def add64(self, dst, a, b) -> None:
        if self._all_quad(dst, a, b) and self._should_localize(dst, a, b):
            q0, q1, q2 = self._snapshot_pair(a, b)
            self.quad.add64(q2, q0, q1)
            self.quad.copy64(dst, q2)
            return
        super().add64(dst, a, b)

    def sub64(self, dst, a, b) -> None:
        if self._all_quad(dst, a, b) and self._should_localize(dst, a, b):
            q0, q1, q2 = self._snapshot_pair(a, b)
            self.quad.sub64(q2, q0, q1)
            self.quad.copy64(dst, q2)
            return
        super().sub64(dst, a, b)

    # ------------------------------------------------------------------
    # comparisons
    # ------------------------------------------------------------------
    def uge64(self, result: int, a, b) -> None:
        if self._all_quad(a, b) and self._should_localize(a, b):
            q0, q1, q2 = self._snapshot_pair(a, b)
            self.quad.uge64(result, q0, q1, q2)
            return
        super().uge64(result, a, b)

    def sge64(self, result: int, a, b) -> None:
        if self._all_quad(a, b) and self._should_localize(a, b):
            q0, q1, q2 = self._snapshot_pair(a, b)
            # Signed order is unsigned order after biasing the sign bit.  The
            # snapshots are disposable, so unlike the base path they need not
            # be restored afterwards.
            self._toggle_bit(q0.bit(63), self.s0)
            self._toggle_bit(q1.bit(63), self.s0)
            self._clear_scratch()
            self.quad.uge64(result, q0, q1, q2)
            return
        super().sge64(result, a, b)

    def eq64(self, result: int, a, b) -> None:
        if self._all_quad(a, b) and self._should_localize(a, b):
            if self._same(a, b):
                self.bf.set_const(result, 1)
                return
            q0, q1, q2 = self._snapshot_pair(a, b)
            other = self.s1
            gate = self.s2
            self.quad.uge64(result, q0, q1, q2)
            self.quad.uge64(other, q1, q0, q2)
            self.copy_cell(result, gate, self.s0)
            self.bf.clear(result)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.copy_cell(other, result, self.s0)
            self.bf.end_while(gate)
            self._clear_scratch()
            return
        super().eq64(result, a, b)


__all__ = ["DistanceLocalizedQuadBinaryStringListIO"]
