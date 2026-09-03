"""Final-output helpers for runtime hexadecimal carried-state records.

The sequential hex passes deliberately consume record markers.  The BACK lane
remains intact, so a final state word can still be transported from the sentinel
back to record zero without knowing runtime N.  The transported sixteen-nibble
word is then converted once to the public Boolean-bit int64 representation for
existing decimal output.
"""

from __future__ import annotations

from bfcore import BFEmitter, Binary64Core, Int64Ref, WORD_BITS
from bfhexseq import (
    BACK,
    HEX_DIGITS,
    RECORD_STRIDE,
    RuntimeHexIntSequence,
    _RelativeBuilder,
    _transfer_word,
)


def propagate_field_back_after_consumed_markers(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    field_base: int,
) -> None:
    """Move a sixteen-nibble sentinel field back to record zero.

    This is intended for final carried state after a pass that consumed all
    materialized MARKER cells.  Record zero has BACK=0; every later materialized
    record and the sentinel have BACK=1.  We first walk one record beyond the
    sentinel to the first zero BACK, step left once, and then move the field
    backward until record zero is reached.
    """
    if not 0 <= field_base <= RECORD_STRIDE - HEX_DIGITS:
        raise ValueError("field must fit inside one runtime hex record")

    # Record 1 is either the first successor (BACK=1) or, for an empty
    # sequence, already the first unmaterialized record (BACK=0).
    bf.move(seq.base + RECORD_STRIDE + BACK)
    bf.emit("[" + ">" * RECORD_STRIDE + "]")
    bf.emit("<" * RECORD_STRIDE)

    # We are at sentinel BACK (or record-zero BACK for an empty sequence).
    r = _RelativeBuilder(initial_pos=BACK)
    _transfer_word(r, field_base, field_base - RECORD_STRIDE)
    r.move(BACK - RECORD_STRIDE)
    bf.emit("[" + r.code() + "]")

    bf.emit("<" * BACK)
    bf.ptr = seq.base


def consume_hex_word_to_binary64(
    bf: BFEmitter,
    *,
    hex_base: int,
    dst: Int64Ref,
    scratch_base: int,
) -> None:
    """Consume sixteen 0..15 nibbles into a Boolean-bit int64 word.

    Each nibble drives at most fifteen iterations of a four-bit incrementer, so
    runtime is bounded independently of the represented 64-bit magnitude.  This
    avoids the rejected value-proportional decimal arithmetic pattern.
    """
    core = Binary64Core(bf, scratch_base=scratch_base)

    for bit in range(WORD_BITS):
        bf.clear(dst.bit(bit))
    core._clear_scratch()

    for nibble in range(HEX_DIGITS):
        src = hex_base + nibble
        low_bit = nibble * 4

        bf.begin_while(src)
        bf.add_const(src, -1)
        bf.set_const(core.carry0, 1)
        bf.clear(core.carry1)

        carry_in, carry_out = core.carry0, core.carry1
        for offset in range(4):
            bf.clear(carry_out)
            bf.begin_while(carry_in)
            bf.add_const(carry_in, -1)
            core._add_one(dst.bit(low_bit + offset), carry_out, core.s2)
            bf.end_while(carry_in)
            carry_in, carry_out = carry_out, carry_in

        # Four-bit overflow is impossible for an input nibble <= 15, but clear
        # both lanes so the scratch contract remains explicit even on bad tape.
        bf.clear(core.carry0)
        bf.clear(core.carry1)
        bf.end_while(src)

    core._clear_scratch()


__all__ = [
    "consume_hex_word_to_binary64",
    "propagate_field_back_after_consumed_markers",
]
