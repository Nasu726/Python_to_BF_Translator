"""Direct final output from a runtime-sequence sentinel record.

A sequential carried-state pass leaves its final ANS word on the zero-marker
sentinel and then rewinds the BF pointer to record zero.  The established output
path first transports that ANS word back through every record before invoking
the compact decimal printer.

This helper instead walks once to the sentinel, temporarily makes that record a
local record-zero by clearing its BACK cell, and runs the existing compact
printer as relative Brainfuck.  After output it restores the sentinel BACK link
and rewinds to the real record zero.  No runtime-N answer-word transport is
needed and the existing decimal conversion implementation remains canonical.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexio import print_record_hex_s64_compact
from bfhexseq import ANS, BACK, MARKER, RECORD_STRIDE, RuntimeHexIntSequence


@lru_cache(maxsize=None)
def _relative_compact_printer(field_base: int) -> str:
    """Return compact-printer BF whose logical record zero is current pointer."""
    local = BFEmitter()
    local_seq = RuntimeHexIntSequence(base=0)
    print_record_hex_s64_compact(local, local_seq, field_base=field_base)
    if local.ptr != 0:
        raise AssertionError("relative compact printer must return to its origin")
    return local.code()


def print_sentinel_hex_s64_compact(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    field_base: int = ANS,
) -> None:
    """Print the final sentinel field directly and return to real record zero.

    Preconditions match ``propagate_field_back_after_consumed_markers``:
    materialized record markers have been consumed, each successor/sentinel has
    BACK=1, record zero has BACK=0, and the final carried word is on the
    sentinel.  The runtime records are dead after this call except for their
    use as compact-printer scratch.
    """
    # Starting from record 1 BACK, walk to the first unmaterialized BACK=0 and
    # step one record left.  We are now on the sentinel BACK.
    bf.move(seq.base + RECORD_STRIDE + BACK)
    bf.emit("[" + ">" * RECORD_STRIDE + "]")
    bf.emit("<" * RECORD_STRIDE)

    # Make the sentinel a temporary local origin for the existing printer.
    bf.emit("[-]")
    bf.emit("<" * BACK)
    bf.emit(_relative_compact_printer(field_base))

    # The relative printer returns to sentinel MARKER.  Restore the original
    # BACK=1 invariant, then follow BACK links to the actual record-zero BACK.
    bf.emit(">" * BACK + "+")
    bf.emit("[" + "<" * RECORD_STRIDE + "]")
    bf.emit("<" * BACK)
    bf.ptr = seq.base


__all__ = ["print_sentinel_hex_s64_compact"]
