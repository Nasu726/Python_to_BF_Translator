"""Sequential passes over ``RuntimeHexIntSequence`` carried-state records.

These passes deliberately exploit liveness.  Once the final sequential pass of
an array consumes ``DATA`` at a record, that data cell range is dead and can be
reused as arithmetic scratch.  ``TOTAL``/``LEFT`` state is then transferred only
one record to the right.  No operation returns to a fixed scalar area whose
distance grows with runtime N.
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
    _map_total_base16,
    _transfer_word,
)


@lru_cache(maxsize=1)
def _consume_data_into_left_body() -> str:
    """One final-use record step: LEFT += DATA, then carry state right.

    Entry is a materialized record marker (1).  DATA is consumed permanently.
    TOTAL and LEFT are moved into the following record.  Exit is the following
    record marker, allowing one BF loop body to process arbitrary runtime N.
    """
    r = _RelativeBuilder()

    # Consume the outer-loop marker and reuse it as the radix-16 carry bit.
    r.add(MARKER, -1)
    for i in range(HEX_DIGITS):
        # DATA[i] is dead after this pass, so make it the local total cell.
        r.transfer(LEFT + i, DATA + i)
        r.transfer(MARKER, DATA + i)
        _map_total_base16(r, DATA + i, LEFT + i, MARKER)

    # Fixed-width overflow is modulo 2**64.
    r.clear(MARKER)

    # Following state fields are zero by construction/previous transfer.  Move,
    # rather than copy, because current state is dead after the cursor advances.
    _transfer_word(r, TOTAL, RECORD_STRIDE + TOTAL)
    _transfer_word(r, LEFT, RECORD_STRIDE + LEFT)
    _transfer_word(r, ANS, RECORD_STRIDE + ANS)
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


def run_prefix_state_pass(bf: BFEmitter, seq: RuntimeHexIntSequence) -> None:
    """Run the final-use prefix pass and leave final state on the sentinel.

    Precondition: ``seq.propagate_total_back_to_first`` has put the full TOTAL on
    record zero and cleared TOTAL in later records. LEFT/ANS must be zero unless
    the caller intentionally seeded them on record zero.

    On return the compiler pointer is restored to ``seq.base``.  The zero-marker
    end sentinel owns the final TOTAL/LEFT/ANS state.
    """
    bf.move(seq.base + MARKER)
    bf.emit("[" + _consume_data_into_left_body() + "]")

    # We are at the zero-marker sentinel. Follow its BACK chain to record zero;
    # this works for both empty and non-empty sequences and needs no length.
    bf.emit(">" * BACK)
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["run_prefix_state_pass"]
