"""Partition minimum specialized by the compile-time upper bound on ANS.

For the common nonnegative-initial-answer path, ANS can only decrease. If the
initial constant fits in ``k`` hexadecimal nibbles, every later nonnegative ANS
also fits in those ``k`` nibbles. After wrapped abs, a candidate is either:

- nonnegative, in which case any nonzero nibble above ``k`` proves it cannot
  beat ANS; or
- INT64_MIN, the only negative result of fixed-width abs, which wins against a
  nonnegative ANS and then remains sticky.

This allows the expensive digitwise candidate/ANS comparison and transport to
run over only ``k`` low nibbles instead of all sixteen. The current prototype
uses this lowering for k <= 9, leaving high LEFT lanes for local flags.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _nibble_ge8, _set_hex_const
from bfhexpartition_totalcandidate import _abs_total_inplace
from bfhexradixfast import map_total_base16_threshold
from bfhexseq import (
    ANS,
    BACK,
    DATA,
    HEX_DIGITS,
    LEFT,
    MARKER,
    RECORD_STRIDE,
    TOTAL,
    RuntimeHexIntSequence,
    _RelativeBuilder,
)


MAX_BOUNDED_NIBBLES = 9
SIGN_CANDIDATE = LEFT + 9
SIGN_ANS = LEFT + 10
TOO_LARGE = LEFT + 11
TMP = LEFT + 12
RESTORE = LEFT + 13
CHOOSE = LEFT + 14


def answer_extent(initial_ans: int) -> int:
    """Return the number of low hex nibbles needed for a nonnegative answer."""
    if not 0 <= initial_ans < (1 << 63):
        raise ValueError("bounded-answer partition requires signed nonnegative initial_ans")
    return max(1, (initial_ans.bit_length() + 3) // 4)


def _consume_high_candidate_into_nonzero_flag(
    r: _RelativeBuilder,
    *,
    extent: int,
) -> None:
    """Consume candidate lanes above extent and set TOO_LARGE iff any was nonzero."""
    r.clear(TOO_LARGE)
    for i in range(extent, HEX_DIGITS):
        cell = TOTAL + i
        r.move(cell)
        r.emit("[")
        # One outer iteration is enough: clear the whole nibble locally, then
        # record that a high lane was nonzero.
        r.clear(cell)
        r.set_const(TOO_LARGE, 1)
        r.move(cell)
        r.emit("]")


def _min_and_move_ans_bounded(r: _RelativeBuilder, *, extent: int) -> None:
    """Set next.ANS=min_signed(ANS,TOTAL), comparing only low bounded lanes."""
    if not 1 <= extent <= MAX_BOUNDED_NIBBLES:
        raise ValueError("bounded answer extent outside supported scratch layout")

    # Preserve sign facts before candidate high lanes are destructively checked.
    _nibble_ge8(
        r,
        SIGN_CANDIDATE,
        TOTAL + HEX_DIGITS - 1,
        TMP,
        RESTORE,
    )
    _nibble_ge8(r, SIGN_ANS, ANS + HEX_DIGITS - 1, TMP, RESTORE)

    _consume_high_candidate_into_nonzero_flag(r, extent=extent)

    # High next-ANS lanes are not transported by the bounded comparison. Fresh
    # records are normally zero, but clear explicitly because parser/runtime
    # scratch is allowed to alias future record cells.
    for i in range(extent, HEX_DIGITS):
        r.clear(RECORD_STRIDE + ANS + i)

    # Compare the only low lanes that can matter for a nonnegative answer.
    r.set_const(MARKER, 1)
    for i in range(extent):
        candidate = TOTAL + i
        saved = LEFT + i
        acc = DATA + i
        ans = ANS + i
        next_ans = RECORD_STRIDE + ANS + i

        r.clear(saved)
        r.clear(acc)
        r.transfer(MARKER, acc)

        r.move(candidate)
        r.emit("[")
        r.add(candidate, -1)
        r.add(saved, 1)
        r.add(acc, 1)
        r.move(candidate)
        r.emit("]")

        r.add(acc, 15)
        r.move(ans)
        r.emit("[")
        r.add(ans, -1)
        r.add(next_ans, 1)
        r.add(acc, -1)
        r.move(ans)
        r.emit("]")

        map_total_base16_threshold(r, acc, candidate, MARKER)

    # Unsigned low-part candidate < ans is !final radix carry.
    ult = DATA
    r.set_const(ult, 1)
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    r.clear(ult)
    r.move(MARKER)
    r.emit("]")
    r.copy_preserved(ult, CHOOSE, TMP)

    # A nonnegative candidate with any high lane set exceeds the compile-time
    # upper bound and therefore cannot improve the carried nonnegative answer.
    r.move(TOO_LARGE)
    r.emit("[")
    r.add(TOO_LARGE, -1)
    r.clear(CHOOSE)
    r.move(TOO_LARGE)
    r.emit("]")

    # Wrapped abs is signed-negative only for exactly INT64_MIN. It beats every
    # nonnegative answer; represent it directly as 0x8000... instead of copying
    # sixteen candidate lanes.
    r.move(SIGN_CANDIDATE)
    r.emit("[")
    r.add(SIGN_CANDIDATE, -1)
    r.set_const(CHOOSE, 1)
    r.set_const(RECORD_STRIDE + ANS + HEX_DIGITS - 1, 8)
    r.move(SIGN_CANDIDATE)
    r.emit("]")

    # Once ANS is INT64_MIN it is sticky and no later abs candidate can win.
    r.move(SIGN_ANS)
    r.emit("[")
    r.add(SIGN_ANS, -1)
    r.clear(CHOOSE)
    r.set_const(RECORD_STRIDE + ANS + HEX_DIGITS - 1, 8)
    r.move(SIGN_ANS)
    r.emit("]")

    # Old ANS is already in next.ANS. Overwrite only the low live lanes when
    # the candidate wins. For INT64_MIN those low lanes are all zero.
    r.move(CHOOSE)
    r.emit("[")
    r.add(CHOOSE, -1)
    for i in range(extent):
        r.clear(RECORD_STRIDE + ANS + i)
        r.transfer(LEFT + i, RECORD_STRIDE + ANS + i)
    r.move(CHOOSE)
    r.emit("]")

    for cell in (SIGN_CANDIDATE, SIGN_ANS, TOO_LARGE, TMP, RESTORE, CHOOSE):
        r.clear(cell)


def _partition_body(extent: int) -> str:
    r = _RelativeBuilder()
    add_data_and_move_state_total_minus_double_left_into_total(r)
    _abs_total_inplace(r)
    _min_and_move_ans_bounded(r, extent=extent)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=None)
def partition_body(extent: int) -> str:
    if not 1 <= extent <= MAX_BOUNDED_NIBBLES:
        raise ValueError("bounded answer extent outside supported scratch layout")
    return _partition_body(extent)


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    extent = answer_extent(initial_ans)
    if extent > MAX_BOUNDED_NIBBLES:
        raise ValueError("initial_ans is too wide for bounded-answer prototype")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body(extent) + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["MAX_BOUNDED_NIBBLES", "answer_extent", "partition_body", "run_partition_min_pass"]
