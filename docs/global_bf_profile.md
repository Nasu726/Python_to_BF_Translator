# Global Brainfuck redundancy profiling (Issue #10)

The proposed whole-program BF optimizer must emit executable standard Brainfuck
directly, without a decoder, macro calls, or altered VM semantics. It cannot
assume that repeated text can be factored like gzip: standard BF has no call
instruction. This stage intentionally measures redundancy **without rewriting
the input**.

## Usage

Run with the repository's existing PYTHONPATH convention:

    PYTHONPATH=pybf python tools/profile_bf_global.py --bf Main.bf
    PYTHONPATH=pybf python tools/profile_bf_global.py --python solution.py --json
    PYTHONPATH=pybf python tools/profile_bf_global.py --abc199 --top 20
    PYTHONPATH=pybf python tools/bench_bf_global_profile.py

The ABC199 C case compiles a **normal Python solution** using the public
compiler. It is not a special-purpose BF kernel.

## What the profiler measures

- Command histogram, ignoring non-command comments as the runtime does.
- Contiguous pointer-motion and arithmetic runs; how many bytes could be
  eliminated just by canonicalizing each run in isolation.
- Loop count, maximum syntactic depth, maximum span, and repeated fingerprints
  of **textually identical complete loop spans**.
- Sample position, length, occurrence count, and textual duplicate bytes for
  the highest-repetition loops.

Fingerprints are 64-bit rolling hashes; their equality is probabilistic, not a
formal proof of byte equality. The metrics are a diagnostic, not justification
for a rewrite. Loop spans may overlap because nested loops are counted, so
"duplicate_text_bytes_not_savings" is **neither additive compression potential
nor achievable source-size reduction**.

## Complexity

The source is filtered once, then scanned once. At each matched closing bracket
the hash of its complete span is obtained from a prefix fingerprint and a power
of the base (cached by span length), without slicing or comparing loop text.

Let n be source length, u the number of distinct loop span lengths and g the
number of distinct fingerprints. The cost is

    O(n + u log n + g log(top))

and space is O(n) in the worst case. Because u and g are at most n, this
satisfies the required O(n log n) ceiling; repeated fixed-sized loops are
essentially linear. Top results use a heap selection, not a full all-pairs or
substring search. The benchmark evaluates 64 KiB, 512 KiB, 1 MiB, and 2 MiB
BF without making assumptions about compressibility.

## Next research gates

1. Profile real compiler outputs before choosing a rewrite. Compare the
   partition vertical slice, ABC199 C, and the current dynamic-int-list path.
2. Separate raw output attributable to repeated code emission from pointer
   traffic and from compiler ABI conversions. Existing statement attribution
   in tools/profile_abc199_source.py provides a complementary view.
3. Propose **one** semantics-proven BF rewrite with explicit preconditions;
   update differential tests for I/O, pointer motion, loop nesting,
   8-bit wraparound and EOF behavior.
4. Benchmark output bytes, optimizer wall time/peak RSS, and raw-BF and Tritium
   execution cost separately. Keep source <= 512 KiB where already enforced.
5. Do not accept a textual deduplication mechanism requiring runtime
   decompression, a custom interpreter, or quadratic whole-program search.

This profiler deliberately does not edit emitted BF or modify the current
public compile route. It is safe to land independently of PR #29.
