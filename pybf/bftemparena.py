"""Temporary arena variants used by size/scalability-oriented compilers."""

from __future__ import annotations

from bfcore import Int64Ref, WORD_BITS


class PeakTempArena:
    """Stack-like compile-time tape allocator with a high-water mark.

    The historical frontends expose ``temps.top`` publicly and several layers
    increment it directly when allocating strings/lists/Quad words.  Keeping a
    read/write property makes this class drop-in compatible while recording the
    maximum live address reached between rewinds.
    """

    def __init__(self, top: int) -> None:
        self.base = top
        self._top = top
        self.peak = top

    @property
    def top(self) -> int:
        return self._top

    @top.setter
    def top(self, value: int) -> None:
        self._top = value
        if value > self.peak:
            self.peak = value

    @property
    def cells_used_peak(self) -> int:
        return self.peak - self.base

    def mark(self) -> int:
        return self.top

    def rewind(self, mark: int) -> None:
        self.top = mark

    def cell(self) -> int:
        result = self.top
        self.top += 1
        return result

    def word(self) -> Int64Ref:
        result = Int64Ref(self.top)
        self.top += WORD_BITS
        return result


__all__ = ["PeakTempArena"]
