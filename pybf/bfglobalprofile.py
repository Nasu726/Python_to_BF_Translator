"""Near-linear, read-only redundancy profile for executable Brainfuck source.

This does NOT rewrite the program. In particular, equal loops are not
automatically deduplicable: standard Brainfuck has no subroutine opcode.

Time: O(n + k log k) expected for n commands and k distinct loop fingerprints
(the final sort); space: O(n + k + max_nesting_depth). Each input byte is
visited O(1) times. No substring all-pairs comparisons or materialized suffixes.
Hash collisions can only change a *diagnostic estimate*, never program output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

COMMANDS = frozenset("><+-.,[]")
_BYTES = {ch: ch.encode("ascii") for ch in COMMANDS}


@dataclass
class _Frame:
    start: int
    depth: int
    digest: Any


def analyze(code: str, *, top: int = 12) -> dict[str, Any]:
    """Profile a BF text without modifying it or assuming compiler metadata.

    A loop fingerprint hashes its canonical nested structure rather than
    copying entire bodies. Nested regions are collapsed to their digests in
    the parent frame. Fingerprints are probabilistic diagnostics only.
    """
    if top < 0:
        raise ValueError("top must be non-negative")
    source_bytes = len(code)
    filtered = "".join(ch for ch in code if ch in COMMANDS)
    n = len(filtered)
    frequencies = Counter(filtered)
    frames: list[_Frame] = []
    root = hashlib.blake2b(digest_size=16)
    loops: dict[tuple[int, bytes], list[int]] = {}
    max_depth = 0
    pointer_run = arithmetic_run = 0
    max_pointer_run = max_arithmetic_run = 0
    pointer_run_count = arithmetic_run_count = 0

    pointer_run_bytes_by_distance = {
        "1-7": 0, "8-31": 0, "32-127": 0, "128-511": 0, "512+": 0,
    }

    def record_pointer_run(length: int) -> None:
        if length < 8:
            band = "1-7"
        elif length < 32:
            band = "8-31"
        elif length < 128:
            band = "32-127"
        elif length < 512:
            band = "128-511"
        else:
            band = "512+"
        pointer_run_bytes_by_distance[band] += length
    # Maximal contiguous groups, not semantic cancellations across loops.
    for i, ch in enumerate(filtered):
        if ch in "><":
            pointer_run += 1
            if arithmetic_run:
                max_arithmetic_run = max(max_arithmetic_run, arithmetic_run)
                arithmetic_run_count += 1
                arithmetic_run = 0
        elif ch in "+-":
            arithmetic_run += 1
            if pointer_run:
                max_pointer_run = max(max_pointer_run, pointer_run)
                pointer_run_count += 1
                record_pointer_run(pointer_run)
                pointer_run = 0
        else:
            if pointer_run:
                max_pointer_run = max(max_pointer_run, pointer_run)
                pointer_run_count += 1
                record_pointer_run(pointer_run)
                pointer_run = 0
            if arithmetic_run:
                max_arithmetic_run = max(max_arithmetic_run, arithmetic_run)
                arithmetic_run_count += 1
                arithmetic_run = 0

        if ch == "[":
            frames.append(_Frame(i, len(frames) + 1, hashlib.blake2b(digest_size=16)))
            max_depth = max(max_depth, len(frames))
        elif ch == "]":
            if not frames:
                raise ValueError(f"unmatched ']' at command offset {i}")
            frame = frames.pop()
            size = i - frame.start + 1
            hash_value = frame.digest.digest()
            key = (size, hash_value)
            # Store only the count and one offset; never copy each loop body.
            if key not in loops:
                loops[key] = [0, frame.start]
            loops[key][0] += 1
            parent = frames[-1].digest if frames else root
            parent.update(b"[")
            parent.update(size.to_bytes(8, "little"))
            parent.update(hash_value)
            parent.update(b"]")
        else:
            (frames[-1].digest if frames else root).update(_BYTES[ch])

    if frames:
        raise ValueError(f"unmatched '[' at command offset {frames[-1].start}")
    if pointer_run:
        pointer_run_count += 1
        record_pointer_run(pointer_run)
        max_pointer_run = max(max_pointer_run, pointer_run)
    if arithmetic_run:
        arithmetic_run_count += 1
        max_arithmetic_run = max(max_arithmetic_run, arithmetic_run)

    ranked = sorted(
        ((size * (count - 1), size, count, loc, digest)
         for (size, digest), (count, loc) in loops.items() if count > 1),
        reverse=True,
    )
    largest = sorted(
        ((size, count, loc, digest) for (size, digest), (count, loc) in loops.items()),
        reverse=True,
    )

    return {
        "original_bytes": source_bytes,
        "bf_bytes": n,
        "ignored_bytes": source_bytes - n,
        "commands": {ch: frequencies[ch] for ch in "><+-.,[]"},
        "pointer_run_count": pointer_run_count,
        "pointer_max_run": max_pointer_run,
        "pointer_run_bytes_by_distance": pointer_run_bytes_by_distance,
        "arithmetic_run_count": arithmetic_run_count,
        "arithmetic_max_run": max_arithmetic_run,
        "loop_count": frequencies["["],
        "distinct_loop_fingerprints": len(loops),
        "max_loop_depth": max_depth,
        "repeated_loop_groups": len(ranked),
        # Not a compression estimate: nested loops overlap, and BF has no
        # reusable macro-call instruction.
        "repeat_motif_bytes_upper_bound_with_overlap": sum(row[0] for row in ranked),
        "top_repeated_loops": [
            {"bytes_per_loop": size, "occurrences": count,
             "duplicate_bytes_with_overlap": excess, "example_offset": loc,
             "fingerprint": digest.hex()}
            for excess, size, count, loc, digest in ranked[:top]
        ],
        "top_largest_loops": [
            {"bytes_per_loop": size, "occurrences": count,
             "example_offset": loc, "fingerprint": digest.hex()}
            for size, count, loc, digest in largest[:top]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Existing BF file")
    group.add_argument("--abc199", action="store_true", help="Compile ordinary ABC199 C source using public compiler")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    if args.abc199:
        from compiler_layout import compile_source
        from profile_abc199_source import ABC199_SOURCE
        source = compile_source(ABC199_SOURCE)
    else:
        source = args.file.read_text(encoding="ascii")
    print(json.dumps(analyze(source, top=args.top), indent=2))


if __name__ == "__main__":
    main()
