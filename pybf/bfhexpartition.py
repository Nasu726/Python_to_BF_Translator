"""Final-use partition/min pass over carried-state hexadecimal records.

This is an algorithmic vertical slice for the user-reported ABC-B program::

    left += a[i]
    ans = min(ans, abs(total - 2*left))

The outer sequence length is entirely runtime-defined.  One emitted BF loop
body processes every record.  DATA is destructively consumed because this pass
is explicitly the array's final sequential use; TOTAL/LEFT/ANS are moved only
one record to the right.

All arithmetic is exact modulo 2**64, matching the translator's fixed int64 ABI.
Signed comparison follows two's-complement ordering.  As with the existing
fixed-width backend, ``abs(INT64_MIN)`` remains INT64_MIN after wraparound.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
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
    _add_preserved_small,
    _map_total_base16,
    _transfer_word,
)


MASK64 = (1 << 64) - 1


def _add_double_preserved(r: _RelativeBuilder, src: int, total: int, tmp: int) -> None:
    """total += 2*src for a hex nibble, preserving src."""
    r.move(src)
    r.emit("[")
    r.add(src, -1)
    r.add(total, 2)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.add(tmp, -1)
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


def _subtract_preserved_from_total(
    r: _RelativeBuilder,
    src: int,
    total: int,
    tmp: int,
) -> None:
    """total -= src, preserving one 0..15 source nibble."""
    r.clear(tmp)
    r.move(src)
    r.emit("[")
    r.add(src, -1)
    r.add(total, -1)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.add(tmp, -1)
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


def _nibble_ge8(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """result = (src >= 8), preserving a 0..15 nibble."""
    r.clear(result)
    r.copy_preserved(src, tmp, restore)
    for step in range(1, 9):
        r.move(tmp)
        r.emit("[")
        r.add(tmp, -1)
        if step == 8:
            r.set_const(result, 1)
            r.clear(tmp)
    for _ in range(8):
        r.move(tmp)
        r.emit("]")


def _flag_not(
    r: _RelativeBuilder,
    dst: int,
    src: int,
    gate: int,
    restore: int,
) -> None:
    r.set_const(dst, 1)
    r.copy_preserved(src, gate, restore)
    r.move(gate)
    r.emit("[")
    r.add(gate, -1)
    r.clear(dst)
    r.move(gate)
    r.emit("]")


def _copy_flag(
    r: _RelativeBuilder,
    dst: int,
    src: int,
    tmp: int,
) -> None:
    r.copy_preserved(src, dst, tmp)


def _consume_data_into_left(r: _RelativeBuilder) -> None:
    """LEFT += DATA, consuming DATA and using MARKER as radix carry."""
    r.add(MARKER, -1)
    for i in range(HEX_DIGITS):
        r.transfer(LEFT + i, DATA + i)
        r.transfer(MARKER, DATA + i)
        _map_total_base16(r, DATA + i, LEFT + i, MARKER)
    r.clear(MARKER)


def _move_primary_state_right(r: _RelativeBuilder) -> None:
    """Move TOTAL/LEFT into next record, leaving current words zero scratch."""
    _transfer_word(r, TOTAL, RECORD_STRIDE + TOTAL)
    _transfer_word(r, LEFT, RECORD_STRIDE + LEFT)


def _double_next_left_into_data(r: _RelativeBuilder) -> None:
    """DATA = 2*next.LEFT, preserving next.LEFT."""
    preserve_tmp = LEFT
    r.clear(MARKER)
    r.clear(preserve_tmp)
    for i in range(HEX_DIGITS):
        total = TOTAL + i
        r.clear(total)
        r.transfer(MARKER, total)
        _add_double_preserved(
            r,
            RECORD_STRIDE + LEFT + i,
            total,
            preserve_tmp,
        )
        _map_total_base16(r, total, DATA + i, MARKER)
    r.clear(MARKER)
    r.clear(preserve_tmp)


def _total_minus_data_into_data(r: _RelativeBuilder) -> None:
    """DATA = next.TOTAL - DATA modulo 2**64, preserving next.TOTAL."""
    preserve_tmp = LEFT
    r.set_const(MARKER, 1)  # radix-complement +1
    r.clear(preserve_tmp)

    for i in range(HEX_DIGITS):
        total = TOTAL + i
        r.clear(total)
        r.transfer(MARKER, total)
        _add_preserved_small(
            r,
            RECORD_STRIDE + TOTAL + i,
            total,
            preserve_tmp,
        )

        # + (15 - DATA[i]) for radix-16 two's-complement subtraction. DATA is
        # scratch after the doubled-left value is consumed, so no restore copy.
        r.add(total, 15)
        r.move(DATA + i)
        r.emit("[")
        r.add(DATA + i, -1)
        r.add(total, -1)
        r.move(DATA + i)
        r.emit("]")

        _map_total_base16(r, total, DATA + i, MARKER)

    r.clear(MARKER)  # discard no-borrow/overflow carry
    r.clear(preserve_tmp)


def _abs_data_inplace(r: _RelativeBuilder) -> None:
    """Two's-complement absolute value of DATA, modulo the fixed int64 ABI."""
    sign = LEFT
    tmp = LEFT + 1
    restore = LEFT + 2
    _nibble_ge8(r, sign, DATA + HEX_DIGITS - 1, tmp, restore)

    r.move(sign)
    r.emit("[")
    r.add(sign, -1)
    r.set_const(MARKER, 1)
    for i in range(HEX_DIGITS):
        total = TOTAL + i
        r.set_const(total, 15)
        r.move(DATA + i)
        r.emit("[")
        r.add(DATA + i, -1)
        r.add(total, -1)
        r.move(DATA + i)
        r.emit("]")
        r.transfer(MARKER, total)
        _map_total_base16(r, total, DATA + i, MARKER)
    r.clear(MARKER)
    r.move(sign)
    r.emit("]")

    for cell in (sign, tmp, restore):
        r.clear(cell)


def _unsigned_lt_data_ans(r: _RelativeBuilder) -> int:
    """Return a cell holding DATA < ANS unsigned; operands are preserved."""
    # Current TOTAL/LEFT are scratch because live state already moved right.
    copy_tmp = RECORD_STRIDE + ANS + HEX_DIGITS - 1
    r.clear(copy_tmp)
    r.set_const(MARKER, 1)

    for i in range(HEX_DIGITS):
        total = LEFT + i
        diff = TOTAL + i
        r.clear(total)
        r.clear(diff)
        r.transfer(MARKER, total)
        _add_preserved_small(r, DATA + i, total, copy_tmp)
        r.add(total, 15)
        _subtract_preserved_from_total(r, ANS + i, total, copy_tmp)
        _map_total_base16(r, total, diff, MARKER)

    # Final carry == DATA >= ANS. ULT is its Boolean negation.
    ult = LEFT
    r.set_const(ult, 1)
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    r.clear(ult)
    r.move(MARKER)
    r.emit("]")
    r.clear(copy_tmp)
    return ult


def _signed_lt_data_ans(r: _RelativeBuilder) -> int:
    """Return Boolean cell for signed DATA < ANS, preserving both words."""
    ult = _unsigned_lt_data_ans(r)
    sign_data = LEFT + 1
    sign_ans = LEFT + 2
    choose = LEFT + 3
    gate1 = LEFT + 4
    gate2 = LEFT + 5
    not_ans = LEFT + 6
    not_data = LEFT + 7
    tmp = LEFT + 8
    restore = LEFT + 9

    # TOTAL diff is dead now and supplies nibble-test workspace.
    _nibble_ge8(r, sign_data, DATA + HEX_DIGITS - 1, TOTAL, TOTAL + 1)
    _nibble_ge8(r, sign_ans, ANS + HEX_DIGITS - 1, TOTAL, TOTAL + 1)
    _flag_not(r, not_ans, sign_ans, gate1, restore)
    _flag_not(r, not_data, sign_data, gate1, restore)
    r.clear(choose)

    # Negative candidate:
    #   positive ans -> candidate is smaller;
    #   negative ans -> unsigned order is the signed order.
    _copy_flag(r, gate1, sign_data, tmp)
    r.move(gate1)
    r.emit("[")
    r.add(gate1, -1)
    _copy_flag(r, gate2, not_ans, tmp)
    r.move(gate2)
    r.emit("[")
    r.add(gate2, -1)
    r.set_const(choose, 1)
    r.move(gate2)
    r.emit("]")
    _copy_flag(r, gate2, sign_ans, tmp)
    r.move(gate2)
    r.emit("[")
    r.add(gate2, -1)
    _copy_flag(r, choose, ult, tmp)
    r.move(gate2)
    r.emit("]")
    r.move(gate1)
    r.emit("]")

    # Non-negative candidate is smaller only when ans is also non-negative and
    # the ordinary unsigned comparison says so.
    _copy_flag(r, gate1, not_data, tmp)
    r.move(gate1)
    r.emit("[")
    r.add(gate1, -1)
    _copy_flag(r, gate2, not_ans, tmp)
    r.move(gate2)
    r.emit("[")
    r.add(gate2, -1)
    _copy_flag(r, choose, ult, tmp)
    r.move(gate2)
    r.emit("]")
    r.move(gate1)
    r.emit("]")

    for cell in (
        ult,
        sign_data,
        sign_ans,
        gate1,
        gate2,
        not_ans,
        not_data,
        tmp,
        restore,
    ):
        if cell != choose:
            r.clear(cell)
    return choose


def _min_ans_with_data(r: _RelativeBuilder) -> None:
    choose = _signed_lt_data_ans(r)
    r.move(choose)
    r.emit("[")
    r.add(choose, -1)
    for i in range(HEX_DIGITS):
        r.clear(ANS + i)
        r.transfer(DATA + i, ANS + i)
    r.move(choose)
    r.emit("]")
    r.clear(choose)


def _partition_body() -> str:
    r = _RelativeBuilder()
    _consume_data_into_left(r)
    _move_primary_state_right(r)
    _double_next_left_into_data(r)
    _total_minus_data_into_data(r)
    _abs_data_inplace(r)
    _min_ans_with_data(r)

    # Current ANS is now final for this iteration. Move it beside TOTAL/LEFT in
    # the next record, then advance to that record's marker.
    _transfer_word(r, ANS, RECORD_STRIDE + ANS)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@lru_cache(maxsize=1)
def partition_body() -> str:
    return _partition_body()


def _set_hex_const(bf: BFEmitter, base: int, value: int) -> None:
    value &= MASK64
    for i in range(HEX_DIGITS):
        bf.set_const(base + i, (value >> (4 * i)) & 0xF)


def run_partition_min_pass(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    """Run ``left += data; ans=min(ans,abs(total-2*left))`` over all records.

    Precondition: full TOTAL is on record zero (use
    ``seq.propagate_total_back_to_first``). DATA is consumed by this final-use
    pass. The zero-marker sentinel receives final TOTAL/LEFT/ANS.
    """
    _set_hex_const(bf, seq.base + ANS, initial_ans)

    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body() + "]")

    # Restore the known static anchor without a runtime length counter.
    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["partition_body", "run_partition_min_pass"]
