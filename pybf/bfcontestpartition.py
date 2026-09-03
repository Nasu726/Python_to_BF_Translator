"""Standalone scalable lowering for the ABC-style partition vertical slice.

This module carries the complete recognized program on a runtime-sized hex
sequence. The first-line N is parsed and carried into the second-line reader,
so the number of participating array elements is determined by ``range(n)``
rather than inferred from the number of tokens present on the input line.
"""

from __future__ import annotations

from bfcore import BFEmitter
from bfhexcounted_lexfast import read_counted_two_line_s64s_and_sum as read_counted_canonical
from bfhexcounted_prefix import read_counted_two_line_s64s_and_sum as read_counted_prefix
from bfhexio import (
    print_record_hex_s64_compact,
    propagate_field_back_after_consumed_markers,
)
from bfhexpartition_addcandidate import run_partition_min_pass as run_partition_min_pass_general
from bfhexpartition_boundedans import MAX_BOUNDED_NIBBLES, answer_extent
from bfhexpartition_nonnegans import run_partition_min_pass as run_partition_min_pass_nonnegative
from bfhexpartition_prefix_terminal import run_partition_min_pass_and_print_terminal
from bfhexseq import ANS, RuntimeHexIntSequence
from bfopt import optimize_bf


SEQUENCE_BASE = 320


def _raw_size(bf: BFEmitter) -> int:
    return sum(len(part) for part in bf.parts)


def _select_partition_lowering(initial_ans: int):
    """Choose a reader/pass pair whose record ABI matches the minimum lowering."""
    if 0 <= initial_ans < (1 << 63):
        if answer_extent(initial_ans) <= MAX_BOUNDED_NIBBLES:
            # Store inclusive prefix sums during input. The terminal pass exits
            # on the final sentinel and prints ANS there directly, avoiding both
            # the partition rewind and the runtime-N ANS backward transport.
            return read_counted_prefix, run_partition_min_pass_and_print_terminal, True
        return read_counted_canonical, run_partition_min_pass_nonnegative, False
    return read_counted_canonical, run_partition_min_pass_general, False


def _build_partition_program(
    *,
    initial_ans: int,
) -> tuple[str, dict[str, int]]:
    bf = BFEmitter()
    cumulative: dict[str, int] = {}

    seq = RuntimeHexIntSequence(base=SEQUENCE_BASE)
    reader, partition_runner, terminal_output = _select_partition_lowering(initial_ans)
    reader(bf, seq)
    cumulative["read_n_and_values"] = _raw_size(bf)

    seq.propagate_total_back_to_first(bf)
    cumulative["reverse_total"] = _raw_size(bf)

    partition_runner(bf, seq, initial_ans=initial_ans)
    if terminal_output:
        # The bounded stored-prefix runner includes final decimal output while
        # the runtime pointer is already on the sentinel. It must be terminal.
        cumulative["partition_and_output"] = _raw_size(bf)
    else:
        cumulative["partition"] = _raw_size(bf)
        propagate_field_back_after_consumed_markers(bf, seq, ANS)
        cumulative["reverse_ans"] = _raw_size(bf)
        print_record_hex_s64_compact(bf, seq, field_base=ANS)
        cumulative["decimal_print"] = _raw_size(bf)

    raw = bf.code()
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
