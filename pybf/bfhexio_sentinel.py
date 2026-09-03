"""Terminal compact output from the current runtime-sequence sentinel.

A carried-state partition loop exits with the runtime pointer on its zero-marker
sentinel.  The reusable pass normally rewinds to record zero, transports ANS
backward through every record, and starts the compact decimal printer there.
For a terminal program tail none of those relocations are required: the
sentinel itself can serve as the printer's logical record zero.

The helper below therefore assumes the runtime pointer is already on the final
sentinel MARKER.  It clears that sentinel's BACK cell so the existing reverse
printer terminates locally, then emits the existing compact printer as relative
Brainfuck.  The runtime pointer finishes on the same sentinel MARKER.  Because
that location depends on input N, callers must treat this as a terminal tail and
must not emit further pointer-sensitive code through BFEmitter afterwards.
"""

from __future__ import annotations

from functools import lru_cache

from bfcore import BFEmitter
from bfhexio import print_record_hex_s64_compact
from bfhexseq import ANS, BACK, RuntimeHexIntSequence


@lru_cache(maxsize=None)
def relative_compact_printer_code(field_base: int = ANS) -> str:
    """Return compact-printer BF rooted at the current runtime pointer."""
    local = BFEmitter()
    local_seq = RuntimeHexIntSequence(base=0)
    print_record_hex_s64_compact(local, local_seq, field_base=field_base)
    if local.ptr != 0:
        raise AssertionError("relative compact printer must return to its origin")
    return local.code()


def emit_terminal_sentinel_print(bf: BFEmitter, *, field_base: int = ANS) -> None:
    """Print a final sentinel field when runtime pointer is on its MARKER.

    The sentinel BACK is 1 for non-empty sequences and 0 when record zero is
    itself the sentinel.  Clearing it is correct in both cases and makes the
    sentinel a local origin for the compact printer's reverse decimal walk.
    This function is intentionally terminal: BFEmitter cannot statically track
    the input-dependent sentinel address after the call.
    """
    bf.emit(">" * BACK + "[-]" + "<" * BACK)
    bf.emit(relative_compact_printer_code(field_base))


__all__ = ["emit_terminal_sentinel_print", "relative_compact_printer_code"]
