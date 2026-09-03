"""Partition experiment fusing wrapped abs with minimum comparison.

After the fused prefix/candidate kernel, current TOTAL contains the raw signed
candidate and next TOTAL/LEFT already hold carried state.  The public path first
normalizes a negative TOTAL with a full sixteen-nibble two's-complement pass and
then scans the resulting magnitude again for the minimum comparison.

This variant instead writes the wrapped absolute-value candidate directly into
next.ANS while it performs the comparison.  For a negative raw candidate, each
nibble computes ``15 - raw + negate_carry`` and contributes that magnitude to
the unsigned subtraction against the old answer in the same local pass.
Positive raw candidates are copied directly.  The old answer is saved in dead
current LEFT scratch and restored only when the candidate does not win.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexaddcandidate import add_data_and_move_state_total_minus_double_left_into_total
from bfhexpartition import MASK64, _nibble_ge8, _set_hex_const
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


# Fixed scratch cells in dead current DATA.  Per-nibble compare accumulators use
# DATA[0..9], then reuse those already-consumed cells for nibbles 10..15.
SIGN = DATA + 10
NEG_CARRY = DATA + 11
NEG_GATE = DATA + 12
POS_GATE = DATA + 13
TMP = DATA + 14
RESTORE = DATA + 15


def _copy_flag(r: _RelativeBuilder, dst: int, src: int, tmp: int) -> None:
    r.copy_preserved(src, dst, tmp)


def _make_sign_gates(r: _RelativeBuilder) -> None:
    """Build one-shot NEG_GATE/POS_GATE from preserved SIGN."""
    _copy_flag(r, NEG_GATE, SIGN, TMP)
    r.set_const(POS_GATE, 1)
    _copy_flag(r, TMP, SIGN, RESTORE)
    r.move(TMP)
    r.emit("[")
    r.clear(TMP)
    r.clear(POS_GATE)
    r.move(TMP)
    r.emit("]")


def _emit_positive_magnitude(
    r: _RelativeBuilder,
    *,
    raw: int,
    out: int,
    acc: int,
) -> None:
    """Consume a nonnegative raw nibble into magnitude output and compare acc."""
    r.clear(out)
    r.move(raw)
    r.emit("[")
    r.add(raw, -1)
    r.add(out, 1)
    r.add(acc, 1)
    r.move(raw)
    r.emit("]")


def _emit_negative_magnitude(
    r: _RelativeBuilder,
    *,
    raw: int,
    out: int,
    acc: int,
    count: int,
) -> None:
    """Consume a negative raw nibble into its two's-complement magnitude.

    ``count`` is dead current LEFT for this nibble.  It temporarily stores
    ``15 - raw + NEG_CARRY`` in 0..16.  A sixteen-level bounded decoder writes
    the low nibble directly to ``out`` and contributes the same value to the
    comparison accumulator.  At exactly 16 the low nibble wraps to zero and the
    outgoing negation carry becomes one.
    """
    r.clear(out)
    r.set_const(count, 15)
    r.move(raw)
    r.emit("[")
    r.add(raw, -1)
    r.add(count, -1)
    r.move(raw)
    r.emit("]")
    r.transfer(NEG_CARRY, count)

    for step in range(1, 17):
        r.move(count)
        r.emit("[")
        r.add(count, -1)
        if step == 16:
            # Fifteen units were tentatively emitted.  A count of sixteen is
            # radix zero with carry one, so undo those contributions.
            r.clear(out)
            r.add(acc, -15)
            r.set_const(NEG_CARRY, 1)
        else:
            r.add(out, 1)
            r.add(acc, 1)
    for _ in range(16):
        r.move(count)
        r.emit("]")


def _abs_min_and_move_ans(r: _RelativeBuilder) -> None:
    """Set next.ANS=min_signed(ANS, abs_s64(TOTAL)) in one candidate scan."""
    _nibble_ge8(r, SIGN, TOTAL + HEX_DIGITS - 1, TMP, RESTORE)
    _copy_flag(r, NEG_CARRY, SIGN, TMP)

    # Radix-complement subtraction carry for magnitude - old_ans.
    r.set_const(MARKER, 1)

    for i in range(HEX_DIGITS):
        raw = TOTAL + i
        saved_ans = LEFT + i
        ans = ANS + i
        next_ans = RECORD_STRIDE + ANS + i
        acc = DATA + i if i < 10 else DATA + (i - 10)

        r.clear(saved_ans)
        r.clear(acc)
        r.transfer(MARKER, acc)
        r.add(acc, 15)

        _make_sign_gates(r)

        r.move(POS_GATE)
        r.emit("[")
        r.add(POS_GATE, -1)
        _emit_positive_magnitude(r, raw=raw, out=next_ans, acc=acc)
        r.move(POS_GATE)
        r.emit("]")

        r.move(NEG_GATE)
        r.emit("[")
        r.add(NEG_GATE, -1)
        _emit_negative_magnitude(
            r,
            raw=raw,
            out=next_ans,
            acc=acc,
            count=saved_ans,
        )
        r.move(NEG_GATE)
        r.emit("]")

        # Consume old ans into dead current LEFT while subtracting it from the
        # comparison accumulator.  next.ANS currently holds the candidate.
        r.move(ans)
        r.emit("[")
        r.add(ans, -1)
        r.add(saved_ans, 1)
        r.add(acc, -1)
        r.move(ans)
        r.emit("]")

        map_total_base16_threshold(r, acc, raw, MARKER)

    # Unsigned magnitude < old answer is !final carry.
    ult = DATA
    r.set_const(ult, 1)
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    r.clear(ult)
    r.move(MARKER)
    r.emit("]")

    sign_candidate = DATA + 1
    sign_ans = DATA + 2
    choose = DATA + 3
    tmp = DATA + 4
    restore = DATA + 5

    _nibble_ge8(
        r,
        sign_candidate,
        RECORD_STRIDE + ANS + HEX_DIGITS - 1,
        tmp,
        restore,
    )
    _nibble_ge8(r, sign_ans, LEFT + HEX_DIGITS - 1, tmp, restore)

    _copy_flag(r, choose, ult, tmp)

    # Wrapped abs can be negative only for INT64_MIN.  Against a nonnegative
    # carried answer it must win.
    r.move(sign_candidate)
    r.emit("[")
    r.add(sign_candidate, -1)
    r.set_const(choose, 1)
    r.move(sign_candidate)
    r.emit("]")

    # Once INT64_MIN is already the carried answer, no later candidate can be
    # smaller.  Keeping the old answer also handles equality.
    r.move(sign_ans)
    r.emit("[")
    r.add(sign_ans, -1)
    r.clear(choose)
    r.move(sign_ans)
    r.emit("]")

    # Candidate is already in next.ANS.  Restore the old answer only when the
    # candidate did not win.
    keep_old = DATA + 6
    r.set_const(keep_old, 1)
    _copy_flag(r, tmp, choose, restore)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(keep_old)
    r.move(tmp)
    r.emit("]")

    r.move(keep_old)
    r.emit("[")
    r.add(keep_old, -1)
    for i in range(HEX_DIGITS):
        r.clear(RECORD_STRIDE + ANS + i)
        r.transfer(LEFT + i, RECORD_STRIDE + ANS + i)
    r.move(keep_old)
    r.emit("]")

    for cell in (SIGN, NEG_CARRY, NEG_GATE, POS_GATE, TMP, RESTORE):
        r.clear(cell)


def _partition_body() -> str:
    r = _RelativeBuilder()
    add_data_and_move_state_total_minus_double_left_into_total(r)
    _abs_min_and_move_ans(r)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=1)
def partition_body() -> str:
    return _partition_body()


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    if not 0 <= initial_ans < (1 << 63):
        raise ValueError("abs/min fused partition requires signed nonnegative initial_ans")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
