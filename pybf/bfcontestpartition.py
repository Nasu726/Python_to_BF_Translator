"""Standalone scalable lowering for the ABC-style partition vertical slice.

This module proves that the runtime-sized hex sequence can carry the complete
algorithm with source size independent of N.  It is deliberately separate from
the public frontend until structural recognition and semantics guards are wired
in there.
"""

from __future__ import annotations

from bf_runtime import BFResult
from bfcore import BFEmitter, Int64Ref
from bfhexio import consume_hex_word_to_binary64, propagate_field_back_after_consumed_markers
from bfhexpartition import run_partition_min_pass
from bfhexseq import ANS, RuntimeHexIntSequence
from bfio import Binary64IO
from bfopt import optimize_bf


BINARY_BASE = 0
IO_SCRATCH_BASE = 64
PRINT_WORKSPACE_BASE = 80
NEWLINE_CELL = 240
SEQUENCE_BASE = 320


def _drain_required_first_line(bf: BFEmitter, cell: int = 0) -> None:
    """Consume a required LF-terminated first line without materializing it."""
    bf.move(cell)
    bf.emit(",----------[,----------]")
    bf.ptr = cell


def build_partition_program(*, initial_ans: int = 10_000_000) -> str:
    """Build standard BF for the two-line ABC partition program.

    The first line (N) is consumed and the second line determines the runtime
    record count.  This vertical slice therefore assumes the contest input
    contract that N equals the number of integers on the second line.  The
    eventual public lowering must make that precondition explicit in its
    structural/semantic guard rather than applying this path generally.
    """
    bf = BFEmitter()
    _drain_required_first_line(bf)

    seq = RuntimeHexIntSequence(base=SEQUENCE_BASE)
    seq.read_lf_terminated_s64s_and_sum(bf)
    seq.propagate_total_back_to_first(bf)
    run_partition_min_pass(bf, seq, initial_ans=initial_ans)

    propagate_field_back_after_consumed_markers(bf, seq, ANS)
    result = Int64Ref(BINARY_BASE)
    consume_hex_word_to_binary64(
        bf,
        hex_base=seq.base + ANS,
        dst=result,
        scratch_base=IO_SCRATCH_BASE,
    )

    io = Binary64IO(bf, scratch_base=IO_SCRATCH_BASE)
    io.print_s64(result, PRINT_WORKSPACE_BASE)
    io.print_newline(NEWLINE_CELL)
    return optimize_bf(bf.code())


__all__ = ["build_partition_program"]
