#!/usr/bin/env python3
"""Profile S1 runtime-byte storage with the literal Brainfuck interpreter.

This is deterministic architecture telemetry, not an AtCoder wall-clock
benchmark. It separates sequential replay, one indexed load at increasing
distances, and repeated swap distributions while compiling each operation only
once. The swap fixture reads a runtime query count and packed-u32 index pairs,
so generated BF source does not grow with query count or sequence length.

The cursor fixture measures the storage/session primitive in isolation. Its
absolute valid byte indices are encoded by this harness as forward/backward
record deltas plus a lane. A future frontend must perform the equivalent
normalization at runtime; these numbers are not end-to-end Python results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pybf"))

from bf_runtime import BF_COMMANDS, BFResult, run_bf  # noqa: E402
from bfcore import BFEmitter  # noqa: E402
from bfpacked import PackedU32Ref  # noqa: E402
from bfstreamseq import PAYLOAD_BYTES, RECORD_STRIDE, RuntimeByteSequence  # noqa: E402


@dataclass(frozen=True)
class Program:
    code: str
    sequence: RuntimeByteSequence
    result_cell: int | None = None


def _payload(length: int) -> str:
    return "".join(chr(33 + (index % 90)) for index in range(length))


def _raw_u32(value: int) -> str:
    return "".join(chr((value >> (8 * byte)) & 0xFF) for byte in range(4))


def _memory_size(sequence: RuntimeByteSequence, length: int) -> int:
    chunks = (length + 7) // 8
    return max(4_096, sequence.base + (chunks + 3) * RECORD_STRIDE)


def _check_code(code: str) -> None:
    extra = set(code) - BF_COMMANDS
    if extra:
        raise AssertionError(f"non-standard Brainfuck commands: {sorted(extra)!r}")


def build_replay_program() -> Program:
    bf = BFEmitter()
    sequence = RuntimeByteSequence(base=96)
    sequence.read_lf_terminated_bytes(bf)
    sequence.write_all_bytes(bf)
    code = bf.code()
    _check_code(code)
    return Program(code, sequence)


def build_load_program() -> Program:
    bf = BFEmitter()
    index = PackedU32Ref(0)
    result_cell = 8
    sequence = RuntimeByteSequence(base=96)
    sequence.read_lf_terminated_bytes(bf)
    for byte in range(4):
        bf.move(index.byte(byte))
        bf.emit(",")
    sequence.load_byte(bf, result_cell, index)
    sequence.write_all_bytes(bf)
    code = bf.code()
    _check_code(code)
    return Program(code, sequence, result_cell)


def build_swap_loop_program() -> Program:
    bf = BFEmitter()
    query_count = 0
    left_index = PackedU32Ref(1)
    right_index = PackedU32Ref(5)
    left_value = 9
    right_value = 10
    sequence = RuntimeByteSequence(base=96)

    sequence.read_lf_terminated_bytes(bf)
    bf.move(query_count)
    bf.emit(",")
    bf.begin_while(query_count)
    for index in (left_index, right_index):
        for byte in range(4):
            bf.move(index.byte(byte))
            bf.emit(",")
    sequence.load_byte(bf, left_value, left_index)
    sequence.load_byte(bf, right_value, right_index)
    sequence.store_byte(bf, left_index, right_value)
    sequence.store_byte(bf, right_index, left_value)
    bf.add_const(query_count, -1)
    bf.end_while(query_count)
    sequence.write_all_bytes(bf)

    code = bf.code()
    _check_code(code)
    return Program(code, sequence)


def build_cursor_swap_loop_program() -> Program:
    bf = BFEmitter()
    sequence = RuntimeByteSequence(base=96)
    sequence.read_lf_terminated_bytes(bf)
    cursor = sequence.open_cursor(bf)
    cursor.begin_input_loop()
    cursor.seek_from_input()
    cursor.load_lane_from_input()
    cursor.seek_from_input()
    cursor.exchange_lane_from_input()
    cursor.seek_from_input()
    cursor.exchange_lane_from_input()
    cursor.end_input_loop()
    cursor.finish()
    sequence.write_all_bytes(bf)

    code = bf.code()
    _check_code(code)
    return Program(code, sequence)


def _run(program: Program, input_data: str, length: int, step_limit: int) -> BFResult:
    return run_bf(
        program.code,
        input_data,
        memory_size=_memory_size(program.sequence, length),
        step_limit=step_limit,
    )


def swap_queries(pattern: str, length: int, count: int) -> list[tuple[int, int]]:
    if length <= 0:
        return [(0, 0)] * count
    if pattern in {"head-adjacent", "middle-adjacent", "tail-adjacent"}:
        if length == 1:
            return [(0, 0)] * count
        if pattern == "head-adjacent":
            origin = 0
        elif pattern == "middle-adjacent":
            origin = max(0, length // 2 - count // 2)
        else:
            origin = max(0, length - count - 1)
        return [
            (
                min(length - 2, origin + query),
                min(length - 2, origin + query) + 1,
            )
            for query in range(count)
        ]
    if pattern == "alternating":
        return [(0, length - 1)] * count
    if pattern == "random":
        state = 0xC0FFEE
        result = []
        for _ in range(count):
            state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
            left = state % length
            state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
            right = state % length
            result.append((left, right))
        return result
    raise ValueError(pattern)


def _apply_swaps(payload: str, queries: list[tuple[int, int]]) -> str:
    chars = list(payload)
    for left, right in queries:
        if left < len(chars) and right < len(chars):
            chars[left], chars[right] = chars[right], chars[left]
    return "".join(chars)


def _cursor_transition(current_record: int, target_record: int) -> str:
    if target_record >= current_record:
        return _raw_u32(target_record - current_record) + _raw_u32(0)
    return _raw_u32(0) + _raw_u32(current_record - target_record)


def encode_cursor_swaps(queries: list[tuple[int, int]]) -> str:
    """Encode absolute byte swaps for the relative-record cursor harness."""
    if not queries:
        return "\0"

    encoded = ["\1"]
    current_record = 0
    for offset, (left, right) in enumerate(queries):
        left_record, left_lane = divmod(left, PAYLOAD_BYTES)
        right_record, right_lane = divmod(right, PAYLOAD_BYTES)
        encoded.extend(
            (
                _cursor_transition(current_record, left_record),
                chr(left_lane),
                _cursor_transition(left_record, right_record),
                chr(right_lane),
                _cursor_transition(right_record, left_record),
                chr(left_lane),
                "\1" if offset + 1 < len(queries) else "\0",
            )
        )
        current_record = left_record
    return "".join(encoded)


def profile(lengths: list[int], query_count: int, step_limit: int) -> None:
    replay = build_replay_program()
    load = build_load_program()
    swaps = build_swap_loop_program()
    cursor_swaps = build_cursor_swap_loop_program()

    print(
        "kind,length,queries,pattern_or_index,source_bytes,total_steps,"
        "incremental_steps,steps_per_query"
    )
    for length in lengths:
        payload = _payload(length)
        replay_result = _run(replay, payload + "\n", length, step_limit)
        if replay_result.output != payload:
            raise AssertionError("sequential replay mismatch")
        print(
            f"replay,{length},0,sequential,{len(replay.code)},"
            f"{replay_result.steps},0,0"
        )

        load_indices = sorted({0, length // 4, length // 2, max(0, length - 1), length})
        for index in load_indices:
            result = _run(load, payload + "\n" + _raw_u32(index), length, step_limit)
            expected = ord(payload[index]) if index < length else 0
            if result.output != payload or result.memory[load.result_cell] != expected:
                raise AssertionError(f"indexed load mismatch at length={length}, index={index}")
            incremental = result.steps - replay_result.steps
            print(
                f"load,{length},1,{index},{len(load.code)},{result.steps},"
                f"{incremental},{incremental}"
            )

        baseline = _run(swaps, payload + "\n" + chr(0), length, step_limit)
        if baseline.output != payload:
            raise AssertionError("zero-query swap baseline mismatch")
        cursor_baseline = None
        if length:
            cursor_baseline = _run(
                cursor_swaps,
                payload + "\n\0",
                length,
                step_limit,
            )
            if cursor_baseline.output != payload:
                raise AssertionError("zero-query cursor-swap baseline mismatch")
        for pattern in (
            "head-adjacent",
            "middle-adjacent",
            "tail-adjacent",
            "alternating",
            "random",
        ):
            queries = swap_queries(pattern, length, query_count)
            encoded = "".join(_raw_u32(a) + _raw_u32(b) for a, b in queries)
            result = _run(
                swaps,
                payload + "\n" + chr(query_count) + encoded,
                length,
                step_limit,
            )
            expected = _apply_swaps(payload, queries)
            if result.output != expected:
                raise AssertionError(
                    f"swap mismatch at length={length}, pattern={pattern}"
                )
            incremental = result.steps - baseline.steps
            per_query = incremental / query_count if query_count else 0
            print(
                f"root-swap,{length},{query_count},{pattern},{len(swaps.code)},"
                f"{result.steps},{incremental},{per_query:.1f}"
            )

            # The cursor ABI deliberately accepts only prevalidated in-range
            # coordinates, whereas the rooted primitive owns its range check.
            if not length:
                continue
            assert cursor_baseline is not None
            cursor_result = _run(
                cursor_swaps,
                payload + "\n" + encode_cursor_swaps(queries),
                length,
                step_limit,
            )
            if cursor_result.output != expected:
                raise AssertionError(
                    f"cursor swap mismatch at length={length}, pattern={pattern}"
                )
            cursor_incremental = cursor_result.steps - cursor_baseline.steps
            cursor_per_query = (
                cursor_incremental / query_count if query_count else 0
            )
            print(
                f"cursor-swap,{length},{query_count},{pattern},"
                f"{len(cursor_swaps.code)},{cursor_result.steps},"
                f"{cursor_incremental},{cursor_per_query:.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        default="32,64,128,256",
        help="comma-separated non-negative runtime sequence lengths",
    )
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--step-limit", type=int, default=1_000_000_000)
    args = parser.parse_args()

    lengths = [int(item) for item in args.lengths.split(",") if item]
    if not lengths or any(length < 0 for length in lengths):
        parser.error("--lengths must contain non-negative integers")
    if not 0 <= args.queries <= 255:
        parser.error("--queries must be in 0..255 for the byte-count fixture")
    profile(lengths, args.queries, args.step_limit)


if __name__ == "__main__":
    main()
