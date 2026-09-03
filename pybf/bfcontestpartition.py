"""Standalone scalable lowering for the ABC-style partition vertical slice.

This module proves that the runtime-sized hex sequence can carry the complete
algorithm with source size independent of N.  It is deliberately separate from
the public frontend until structural recognition and semantics guards are wired
in there.
"""

from __future__ import annotations

from bfcore import BFEmitter
from bfhexio import (
    print_record_hex_s64_compact,
    propagate_field_back_after_consumed_markers,
)
from bfhexpartition import run_partition_min_pass
from bfhexseq import ANS, RuntimeHexIntSequence
from bfopt import optimize_bf


SEQUENCE_BASE = 320


def _drain_required_first_line(bf: BFEmitter, cell: int = 0) -> None:
    """Consume a required LF-terminated first line without materializing it."""
    bf.move(cell)
    bf.emit(",----------[,----------]")
    bf.ptr = cell


def _raw_size(bf: BFEmitter) -> int:
    return sum(len(part) for part in bf.parts)


def _build_partition_program(
    *,
    initial_ans: int,
) -> tuple[str, dict[str, int]]:
    bf = BFEmitter()
    cumulative: dict[str, int] = {}

    _drain_required_first_line(bf)
    cumulative["drain_n"] = _raw_size(bf)

    seq = RuntimeHexIntSequence(base=SEQUENCE_BASE)
    seq.read_lf_terminated_s64s_and_sum(bf)
    cumulative["read_and_sum"] = _raw_size(bf)

    seq.propagate_total_back_to_first(bf)
    cumulative["reverse_total"] = _raw_size(bf)

    run_partition_min_pass(bf, seq, initial_ans=initial_ans)
    cumulative["partition"] = _raw_size(bf)

    propagate_field_back_after_consumed_markers(bf, seq, ANS)
    cumulative["reverse_ans"] = _raw_size(bf)

    print_record_hex_s64_compact(bf, seq, field_base=ANS)
    raw = bf.code()
    cumulative["decimal_print"] = len(raw)

    code = optimize_bf(raw)
    cumulative["optimized"] = len(code)
    return code, cumulative


def build_partition_program(*, initial_ans: int = 10_000_000) -> str:
    """Build standard BF for the two-line ABC partition program.

    The first line (N) is consumed and the second line determines the runtime
    record count.  This vertical slice therefore assumes the contest input
    contract that N equals the number of integers on the second line.  The
    eventual public lowering must make that precondition explicit in its
    structural/semantic guard rather than applying this path generally.
    """
    code, _ = _build_partition_program(initial_ans=initial_ans)
    return code


def partition_program_size_breakdown(
    *,
    initial_ans: int = 10_000_000,
) -> dict[str, int]:
    """Return cumulative emitted-byte checkpoints for source-budget work."""
    _, cumulative = _build_partition_program(initial_ans=initial_ans)
    return cumulative


__all__ = ["build_partition_program", "partition_program_size_breakdown"]
