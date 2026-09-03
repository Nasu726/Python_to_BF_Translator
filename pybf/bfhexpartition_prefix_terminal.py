"""Terminal stored-prefix partition pass with direct sentinel output."""

from __future__ import annotations

from bfcore import BFEmitter
from bfhexio_sentinel import emit_terminal_sentinel_print
from bfhexpartition import MASK64, _set_hex_const
from bfhexpartition_boundedans import MAX_BOUNDED_NIBBLES, answer_extent
from bfhexpartition_prefix import partition_body
from bfhexseq import ANS, MARKER, RuntimeHexIntSequence


def run_partition_min_pass_and_print_terminal(
    bf: BFEmitter,
    seq: RuntimeHexIntSequence,
    *,
    initial_ans: int = 10_000_000,
) -> None:
    """Run the bounded stored-prefix pass and print from its final sentinel.

    Unlike the reusable partition runner, this function deliberately does not
    rewind through BACK links after the forward pass.  The runtime pointer exits
    the marker loop on the sentinel MARKER, where the final ANS already lives;
    the compact decimal printer is attached there directly.

    This must be the final pointer-sensitive operation in the generated program.
    """
    extent = answer_extent(initial_ans)
    if extent > MAX_BOUNDED_NIBBLES:
        raise ValueError("initial_ans is too wide for terminal stored-prefix path")

    _set_hex_const(bf, seq.base + ANS, initial_ans & MASK64)
    bf.move(seq.base + MARKER)
    bf.emit("[" + partition_body(extent) + "]")
    emit_terminal_sentinel_print(bf, field_base=ANS)


__all__ = ["run_partition_min_pass_and_print_terminal"]
