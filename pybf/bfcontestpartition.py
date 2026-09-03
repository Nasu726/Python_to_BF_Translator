"""Standalone scalable lowering for the ABC-style partition vertical slice.

This module carries the complete recognized program on a runtime-sized hex
sequence.  The first-line N is parsed and carried into the second-line reader,
so the number of participating array elements is determined by ``range(n)``
rather than inferred from the number of tokens present on the input line.
"""

from __future__ import annotations

from bfcore import BFEmitter
from bfhexcounted_direct import read_counted_two_line_s64s_and_sum
from bfhexio import (
    print_record_hex_s64_compact,
    propagate_field_back_after_consumed_markers,
)
from bfhexpartition_minfused import run_partition_min_pass
from bfhexseq import ANS, RuntimeHexIntSequence
from bfopt import optimize_bf


SEQUENCE_BASE = 320


def _raw_size(bf: BFEmitter) -> int:
    return sum(len(part) for part in bf.parts)


def _build_partition_program(
    *,
    initial_ans: int,
) -> tuple[str, dict[str, int]]:
    bf = BFEmitter()
    cumulative: dict[str, int] = {}

    seq = RuntimeHexIntSequence(base=SEQUENCE_BASE)
    read_counted_two_line_s64s_and_sum(bf, seq)
    cumulative["read_n_and_values"] = _raw_size(bf)

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
    """Build standard BF for the recognized two-line partition program."""
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
