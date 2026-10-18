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


## First measurements (GitHub Actions, PR #31)

The following are measurements of **current main**, not of PR #29. The
source-size numbers describe public compiler output, not runtime cost:

| Workload | Standard BF bytes | Loop instances | Notes |
| --- | ---: | ---: | --- |
| Ordinary ABC199 C | 1,344,281 | 18,210 | 819,993 B above 512 KiB |
| Existing partition-specialized output | 363,109 | 6,934 | Under 512 KiB |

For ABC199 C, 634,471 `>` plus 637,127 `<` consume **1,271,598
bytes (94.6%)** of the generated source. The local adjacent-run canonicalization
opportunity measured on this already optimized BF is **zero** (both pointer
cancellations and arithmetic reductions). Of its 18,210 complete loop spans,
576 distinct fingerprints occur more than once. The total 853,640 bytes of
textually repeated loop spans **cannot be interpreted as removable bytes**;
these include nested/overlapping text and BF has no macro call.

Synthetic benchmark (Ubuntu GitHub-hosted runner, Python 3.12):

| BF source | Profiler time | Process peak RSS |
| ---: | ---: | ---: |
| 65,536 B | 0.052 s | 13,360 KiB |
| 524,288 B | 0.407 s | 18,432 KiB |
| 1,048,576 B | 0.820 s | 28,816 KiB |
| 2,097,152 B | 1.642 s | 47,432 KiB |

The synthetic pattern consists of repeated small loops; distinct-span counts
are small, making this a linear-path measurement rather than a proof of optimal
behavior for adversarial distinct-span programs. The algorithmic bound above
also covers the latter.

**Rejected as the first optimization target:** straightforward pointer-path
rescheduling *within* straight-line regions. Of the 1,271,598 pointer-movement
bytes in ABC199 C, the sum of mathematically minimal pointer paths over 34,277
independent regions is **1,271,596 bytes**. That is a maximum reduction of just
**2 bytes** under this restricted reordering model, despite pointer movement
accounting for 94.6% of source bytes. Only one region has a two-byte excess.

The source profiler now derives this bound by separating every loop/I/O
barrier and solving the two possible directions for visiting the low/high
written cells inside each region. This result rules out a useful whole-program
gain from *block-local scheduling alone*; it does not rule out changing loop
algorithms, runtime data layout, mobile-pointer ABI, or proven loop-level
rewrites. Those are the next hypotheses to investigate. Do not move BF
operations across loop or I/O boundaries without separate semantic proof.
