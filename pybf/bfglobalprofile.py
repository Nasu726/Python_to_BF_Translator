"""Near-linear, read-only profiling of a whole standard Brainfuck program.

This module does NOT compress BF or estimate achievable source savings. Identical
loops are only evidence of redundancy: Brainfuck has no call/macro instruction.

Complexity: O(n + u log n + g log min(g, top)), where n is filtered source
length, u is the number of distinct loop spans, and g is the number of distinct
loop fingerprints. Memory is O(n) in the worst case (normally much less).
No loop bodies are sliced/copied and no pairwise substring comparisons occur.
"""

from __future__ import annotations

from collections import Counter
from heapq import nlargest


COMMANDS = "><+-.,[]"
_COMMAND_SET = frozenset(COMMANDS)
_MASK = (1 << 64) - 1
_BASE = 1_000_000_007


def profile_bf(code: str, *, top: int = 12) -> dict[str, object]:
    """Summarize BF source structure without modifying its semantics.

    Duplicate fingerprints are probabilistic evidence of identical *text*, not
    evidence that any such bytes can be removed from executable BF. Offsets
    refer to the filtered eight-command source. Overlapping/nested occurrences
    are counted independently; reported duplicate bytes are NOT savings.
    """
    if top < 1:
        raise ValueError("top must be positive")

    filtered = "".join(c for c in code if c in _COMMAND_SET)
    counts = Counter(filtered)
    stack: list[tuple[int, int]] = []
    motifs: dict[tuple[int, int], list[int]] = {}
    powers: dict[int, int] = {}
    fingerprint = 0
    max_depth = 0
    max_span = 0
    loops = 0

    run_kind = ""
    run_length = 0
    run_delta = 0
    pointer_runs = 0
    arithmetic_runs = 0
    cancelled_pointer_bytes = 0
    reducible_arithmetic_bytes = 0

    def finish_run() -> None:
        nonlocal run_kind, run_length, run_delta
        nonlocal pointer_runs, arithmetic_runs
        nonlocal cancelled_pointer_bytes, reducible_arithmetic_bytes
        if run_kind == "pointer":
            pointer_runs += 1
            cancelled_pointer_bytes += run_length - abs(run_delta)
        elif run_kind == "arithmetic":
            arithmetic_runs += 1
            mod = run_delta % 256
            shortest = min(mod, 256 - mod)
            reducible_arithmetic_bytes += run_length - shortest
        run_kind = ""
        run_length = 0
        run_delta = 0

    for offset, command in enumerate(filtered):
        if command in "><":
            kind = "pointer"
        elif command in "+-":
            kind = "arithmetic"
        else:
            kind = ""
        if kind != run_kind:
            finish_run()
        if kind:
            run_kind = kind
            run_length += 1
            run_delta += 1 if command in ">+" else -1

        previous_fingerprint = fingerprint
        fingerprint = (fingerprint * _BASE + ord(command)) & _MASK

        if command == "[":
            stack.append((offset, previous_fingerprint))
            if len(stack) > max_depth:
                max_depth = len(stack)
        elif command == "]":
            if not stack:
                raise ValueError(f"unmatched ']' at BF command offset {offset}")
            start, start_fingerprint = stack.pop()
            loops += 1
            span = offset + 1 - start
            if span > max_span:
                max_span = span
            power = powers.get(span)
            if power is None:
                power = pow(_BASE, span, 1 << 64)
                powers[span] = power
            digest = (fingerprint - start_fingerprint * power) & _MASK
            key = (span, digest)
            entry = motifs.get(key)
            if entry is None:
                motifs[key] = [1, start]
            else:
                entry[0] += 1

    finish_run()
    if stack:
        raise ValueError(f"unmatched '[' at BF command offset {stack[-1][0]}")

    repeated = [
        (span, digest, hits, sample)
        for (span, digest), (hits, sample) in motifs.items()
        if hits > 1
    ]
    frequent = nlargest(
        top, repeated, key=lambda item: ((item[2] - 1) * item[0], item[2])
    )
    top_motifs = [
        {
            "length": span,
            "instances": hits,
            "first_offset": sample,
            "fingerprint": f"{digest:016x}",
            "duplicate_text_bytes": (hits - 1) * span,
        }
        for span, digest, hits, sample in frequent
    ]

    return {
        "source_bytes": len(filtered),
        "ignored_non_command_chars": len(code) - len(filtered),
        "commands": {command: counts[command] for command in COMMANDS},
        "runs": {
            "pointer_runs": pointer_runs,
            "arithmetic_runs": arithmetic_runs,
            "locally_cancellable_pointer_bytes": cancelled_pointer_bytes,
            "locally_reducible_arithmetic_bytes": reducible_arithmetic_bytes,
        },
        "loops": {
            "instances": loops,
            "max_depth": max_depth,
            "max_span": max_span,
            "distinct_fingerprints": len(motifs),
            "repeated_fingerprints": len(repeated),
            "duplicate_text_bytes_not_savings": sum(
                (hits - 1) * span for span, _digest, hits, _sample in repeated
            ),
        },
        "top_loop_motifs": top_motifs,
    }


__all__ = ["profile_bf"]
