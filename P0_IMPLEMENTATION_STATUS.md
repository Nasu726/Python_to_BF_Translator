# Feature / optimization / ABC acceptance track

Updated 2026-09-21. `IMPLEMENTATION_PLAN.md` defines the minimum feature scope.
Every feature must eventually have all three: implementation, optimization, and
validation using real ABC programs. The order can vary; none substitutes for
the others. Keep ordinary Python source unchanged instead of specializing by
problem identity or rewriting away unsupported syntax.

## Current increment: public single-owner integer-list views

The public compiler now selects a statically proven ownership slice:

```python
a = list(map(int, input().split()))
b = a
print(len(b), sum(a))
b.clear()
print(len(a), sum(b))  # 0 0: one shared mutable object
```

There must be exactly one unconditional top-level input-list construction.
Alias declarations must also be unconditional/top-level, and each use must
follow its binding. Allowed uses are `len`, one-argument `sum`, alias binding,
and statement-form `clear`. Rebinding, escaping, indexing, other mutations,
multiple input-list owners, builtin shadowing and simultaneous dynamic character
storage reject this selection and retain the previous route/diagnostics.
This is **statically resolved shared identity**, not general runtime handles,
heap allocation or complete Python list semantics. Unsupported programs do not
acquire these alias/scalability guarantees through the old fallback.

Input materializes uncapped 10-cell records. A persistent 12-cell header caches
int64 sum and u32 length, computed with the mobile reduction. Only `clear` can
mutate selected objects, so these caches stay valid; repeated queries preserve
the header and never rescan the list. All aliases address the same header and
sequence. `clear` is logical: payload becomes unreachable, with no allocator
reuse. The public two-pass layout puts both records and the 38-cell mobile frame
after the measured scalar/temp high-water mark. The input line is consumed at
the original construction statement, without deferral across other input.

New real acceptance fixture: [ABC103 C — Modulo Summation](https://atcoder.jp/contests/abc103/tasks/abc103_c),
using the ordinary `n=input; a=list(map(...)); print(sum(a)-n)` solution through
`pybf.compile_source` (the existing loop-based streaming fixture is retained).
Generated source: **306,652 B**, **217,636 B below 512 KiB**. Official sample
raw steps: **1,129,077 / 1,757,464 / 3,122,768**. N=3000 with all values 2
matches CPython in **412,523,451** raw steps under the unchanged 500-million
budget. This exercises runtime storage, not a raised fixed list capacity.

Manual native measurement: built `rdebath/Brainfuck` revision `14a729d`
(Tritium 1.2.73) and ran `bfi.out -b -e Main.bf`, 3 trials per N=3000 pattern:

| Pattern | Correct output | Local elapsed range |
| --- | ---: | ---: |
| all 2 | 3000 | 0.032529–0.039071 s |
| all 100000 | 299997000 | 0.040804–0.041731 s |
| all 65535 | 196602000 | 0.044776–0.045180 s |
| deterministic mixed | 150317490 | 0.043763–0.047519 s |

Reproduce with `python tools/bench_tritium_abc103_sum.py --tritium /path/to/tritium/bfi.out`.
These are local timings on four maximum-N distributions, not AtCoder judge
acceptance or an exhaustive worst-case timing proof. The build used the local
Makefile's native optimization flags and disabled unavailable optional engines.

Validation: **22 new tests + 58 existing regression tests passed**. Coverage
includes alias chains, clear inside loops, repeated metadata queries, following
input, negative values, empty lines, 65/256/300-element arrays, unsafe-selection
rejections, layout separation, all official samples and the maximum-N fixture.
New tests are included in the normal frontend CI shard; existing size and step
gates remain unchanged. Full runtime identity/allocation, indexed mutation,
repeat, nested containers, copying and sorting remain P0 work.

## Previous increment: packed addition without repeated byte-overflow scans

PR #15's mobile reduction is merged at
`475e65b308807aa874ea33b89920f72ce035d558` (head
`21f122d40ddb122e25d0ba15d6db6a6da758ecb1`, all four shards green in run
`35517568434`). Its dense-byte bottleneck is addressed by changing
`PackedI64Ops.add_inplace` to extract bits by repeated halving. A byte's bit
positions share one BF loop whose weight goes 1,2,...,128 and wraps to zero.
Carry is maintained across bytes. The RHS is preserved for disjoint operands;
exact operand alias now explicitly supports doubling. Partial/scratch overlap
is unsupported. Scratch is cleared, and an all-zero RHS bypasses arithmetic.

The 38-cell mobile frame and 10-cell records remain unchanged. Re-running
`tools/profile_packed_sequence_reduction.py` gives:

- extra reduction source: **11,829 B** (was 10,569), still below the existing
  **12,000 B** regression gate;
- zero-repeat plus reduction: **15,010 B**, independent of runtime N;
- zero-list N=256/512/1024/2048 reduction steps:
  **2,338,832 / 4,694,983 / 9,460,376 / 19,203,526**;
- `--value=-1 --counts 1 8 16`:
  **1,331,609 / 12,344,933 / 24,920,405** steps, compared with the previous
  **17,534,205 / 139,584,294 / 279,076,950**. N=16 improves about **11.2x**.

This is a trade-off, not universal speedup: isolated `1234 + 1` at operand bases
32/48 and scratch base 80 takes 122,294 steps versus 48,159 before. At that same
layout, `1234 + 123` improves 3,030,816 -> 158,695; `MASK64 + MASK64` improves
54,726,636 -> 1,701,023. Source grows 9,559 -> 10,597 B for the isolated add.
Small increment lowering remains a possible separate improvement. All figures
are raw BF steps, not Tritium wall-clock acceptance.

Validation: **76 tests passed** for packed operations, sequences and the public
packed-local frontend. New runtime-fed tests cover 45 boundary/random operand
pairs and 9 exact-alias values, with full-tape/RHS/scratch/input checks and a
3-million-step bound. `test_bfpackedops.py` is now included in normal runtime CI
rather than only the optional experiment workflow. Existing size/step gates
remain unchanged. General dynamic-list identity/allocation/frontend routing
remains incomplete; this arithmetic change does not change that boundary.

## Previous increment: mobile sum/length workspace

`RuntimePackedIntSequence.sum_and_length` adds a preserving reduction over
runtime-sized contiguous records. One 38-cell frame, initially reserved directly
before the sequence, contains an int64 total, u32 count and scratch. Each forward
step rotates `[frame][record]` into `[record][frame]`. An inverse return walk
restores every record and carries the results back; fixed output addresses are
accessed only after the traversal. Empty sequences and repeated calls work.
Outputs must be disjoint and precede the reserved frame; invalid layouts fail
before any BF is emitted. The frame is exclusively caller-reserved scratch and
is zero on return. Sum/count wrap modulo 2**64 / 2**32 respectively.

This closes the specific *fixed scalar return on every item* gap for sum/count.
It does not close general object allocation/alias routing, arbitrary loop bodies
or public `sum(list)` support. Sequential traversal needs **10*N + O(1)** tape,
including just one frame rather than padding every record. There is one forward
reduction and one restoring return pass. Each record incurs bounded fixed-width
byte operations and constant-distance moves, so runtime is O(N) in the byte-tape
ABI; byte-value-dependent constants remain significant. A future object header
should cache length instead of rescanning for each public `len` call.

Reproduce with `PYTHONPATH=pybf python tools/profile_packed_sequence_reduction.py`:

- reduction adds **10,569 BF bytes** at sequence base 64 / output bases 8 and 16;
- runtime zero-repeat plus reduction: **13,750 B**, independent of runtime N;
- additional workspace: **38 cells**, independent of N;
- zero-list reduction steps for N=256/512/1024/2048:
  **2,462,736 / 4,942,791 / 9,955,992 / 20,194,758**.

The default profile is not a worst-case arithmetic benchmark. The same tool
with `--value=-1 --counts 1 8 16` reports **17,534,205 / 139,584,294 /
279,076,950** reduction steps. Existing packed addition repeatedly detects
byte overflow and is costly for dense bytes. This is a remaining arithmetic
bottleneck, not evidence of maximum-constraint practicality. No step limit was
raised to hide it, and no public ABC acceptance milestone is claimed here.

Validation: **70 tests passed** in packed sequence/u32/int64 suites, including
18 new cases. Full-memory comparison checks exact record/sentinel restoration,
frame cleanup and untouched outside cells; tests cover empty/singleton lists,
negative values, int64 overflow, repeated calls, decimal-input integration,
following-input preservation and runtime lengths through 2048. The same emitted
program is reused across lengths, with source/step scaling gates. All new tests
are in the existing runtime CI shard.

Next P0 boundary remains runtime allocation + identity/header routing. The
mobile-frame technique now has a concrete reduction proof/test; extending it to
arbitrary live scalars and loop control is separate work. Retain compact records
and avoid reinserting rooted heap lookup inside the sequential loop.

## Previous increment: integer augmented assignment and ABC100 C

Scalar and integer-list targets now support `//=`, `%=`, `&=`, `|=`, `^=`
as well as `+=`, `-=`, `*=`. Subscript target evaluation and loading precede
RHS evaluation; the index is evaluated once. These additions use the existing
fixed-capacity list frontend, not the pending dynamic object model.

A differential regression exposed an existing Quad signed-divmod bug:
boolean/negation kernels borrowed shared Quad temporaries while that workspace
still held division magnitudes and sign flags. Negative remainder correction
could produce zero (for example `-17 % 3`). Signed division now uses the binary
bit-addressed kernel throughout its owned workspace; Quad operands need no
conversion or extra tape. Both ordinary expressions and updates are covered.

Positive literal power-of-two divisors through 2**62 use sign-extended bit
copies for floor division and low bits for modulo. The transformation applies
to expressions and updates, preserves negative-dividend floor semantics, and
does not skip evaluation of an effectful RHS. Other divisors retain the generic
kernel. Zero-divisor error handling remains part of the unfinished error ABI.

[ABC100 C — *3 or /2](https://atcoder.jp/contests/abc100/tasks/abc100_c)
is a new public-API fixture: ordinary indexed list mutation with `a[i] //= 2`.
All three official samples match CPython and the published output. The public
source is **2,485,322 B**, versus **29,415,994 B** with the power-of-two
transformation disabled and the same corrected general division kernel.
Sample raw steps: **12,894,431 / 23,979,296 / 290,989,246**.
This is still above 512 KiB. The official N<=10000 bound is not established:
this sample fixture still uses fixed-capacity storage, and no maximum-scale
Tritium benchmark was performed.

Local validation: 26 new focused cases passed (including the three official
samples and 24 arithmetic boundary executions), plus 29 public-API, layout,
source-size, compile-performance and older frontend tests. The new cases are
in the existing contest CI shard; no test/step limit was raised.

The preceding #7/#8/#9/#12/#13 stack is now merged; see the handoff for the
exact tested tree and successful CI runs. The allocation/identity/mobile
workspace work below remains the next P0 architectural boundary.

## Previous increment: contiguous repeat and physical traversal

`bfpackedseq.RuntimePackedIntSequence` now has two additional runtime primitives:

- `repeat_constant`: `[constant] * runtime_u32` into a fresh zero-initialized
  contiguous region, preserving the external count. Remaining count travels in
  not-yet-materialized payload, then is consumed; no heap-origin lookup occurs
  per element. Signed Python repeat normalization and allocation ownership are
  still frontend/heap responsibilities.
- `walk_records`: a forward or reverse pass with the physical BF head retained
  on the current record. A restricted relative body can read/write a packed
  value as eight raw bytes or assign a constant. Bodies cannot address fixed
  scalars through this interface. Each body is emitted once and both traversal
  directions restore the base, including for an empty sequence.

The persistent representation remains 10 cells per int64. Traversals do not
borrow scratch from adjacent payload. Arbitrary Python loop bodies, decimal
formatting, arithmetic workspaces, break/continue and object alias routing are
NOT implemented by this restricted body API.

`PYTHONPATH=pybf python tools/profile_packed_sequence_walk.py` reproduces:

- zero-repeat constructor: **3,181 B**, independent of runtime N;
- constructor plus forward/reverse raw-output passes: **3,287 B**;
- both passes together: **106 extra source bytes**, **98*N + 8 raw steps**;
- N=1024: creation 10,472,733 steps; both traversals 100,360 extra steps.

Destructive count transport reduces the initial preserving-copy prototype from
3,343 to 3,181 source bytes and N=1024 creation from 12,423,501 to 10,472,733
steps. Creation cost varies with packed counter byte values (bounded by 255),
so its small-N timing is not exactly proportional to N. The exact linear step
formula above applies only to the two raw-output passes.

Validation: **52 tests passed** across packed sequence/u32/int64 primitives.
Coverage includes full-state preservation across forward/reverse passes, 80/256
items, runtime repeats through 1024 items and the 255/256 counter boundary,
int64 extrema, writes and rereads, untouched following input, and count-region
overlap rejection. These are runtime primitive results, not public dynamic-list
support or a new ABC acceptance milestone.

Next: attach identity/length/allocation metadata and define a mobile scalar
workspace before using these walkers in ordinary Python loops. A caller that
returns to fixed scalar storage on every item can reintroduce quadratic tape
travel even with this walker, so that bridge requires its own scaling test.

## Previous increment: integer-list mutation

Implemented in PR #12, now merged:

- `DynamicIntListRuntime.set_packed`: indexed mutation through object handles,
  visible through aliases and isolated from other list objects; preserves index,
  value, identity and length. External operands must not overlap its workspace.
- Shared get/set locator stops on null. It traverses at most the list's links,
  instead of decrementing an arbitrary invalid u32 index billions of times.
- Ordinary integer-list `a[i] += x`, `-=`, `*=` lowering. Evaluate the index once,
  load before evaluating the RHS, and reuse the same normalized index for store.
- Signed literal range steps, including `range(n-2, -1, -1)`. Runtime-valued
  steps are still not implemented.

The last two gaps were exposed by compiling the real ABC136 C solution below,
not by changing that solution to accommodate the compiler.

An additional regression exposed a pre-existing destructive constant-index
read: Quad conversion consumed the persistent packed list payload. The Quad
backend now snapshots constant-index reads into the disposable slot result lane,
matching the dynamic-read contract. Repeated reads and subsequent list output
must preserve all elements, including 255, -1 and 256.

## Acceptance anchor

[ABC136 C — Build Stairs](https://atcoder.jp/contests/abc136/tasks/abc136_c),
official N <= 100000. `tests/test_abc_c_foundation.py` compiles the ordinary
backward, in-place greedy Python solution through the public API and compares
all four official samples with both CPython and expected output.

This currently exercises the existing **fixed-capacity** list frontend. It does
NOT exercise the new heap mutation primitive and does NOT establish scalable
ABC136 support. Reuse the same source for the future dynamic-list route, then
test beyond 64 elements and benchmark the official maximum separately.

The public-default ABC136 fixture emits **5,746,608 BF bytes**, above 512 KiB.
Sample raw steps are 37,136,657 / 24,597,246 / 34,056,141 / 123,241,308.
These measurements document the remaining optimization work; passing the
samples is a functional milestone, not a contest-ready claim.

Local validation: 49 tests covering the ABC fixture, evaluation order,
control flow, legacy list frontend and heap/handle/packed primitives passed;
15 additional public-API/layout/source-size/compile-performance gates passed.
Both edited test modules are already included in the normal CI matrix.

## What is still missing before P0 is complete

1. General public object-handle routing: ordinary list variables still do not
   have Python alias semantics. Low-level handle tests are not frontend support.
2. Scalable allocation and traversal: the retained linked-list heap is a
   correctness prototype. Each handle lookup scans from heap origin; stopping
   at null does not remove this compounded traversal cost. Do not restore the
   previously rejected public route or raise its billion-step guard.
3. Runtime-sized repeat and nested containers on the reference model.
4. Shallow copy and basic deep copy on that model.
5. Stable bottom-up merge sort and reverse sorting.
6. Temporary allocation reuse; current heap allocation is monotonic.

Invalid indexes still use legacy zero/no-op behavior in these primitives.
The final error-state contract remains required, not waived.

## Next implementation boundary

Design chunk/physical-cursor traversal and heap object routing together so a
sequential list pass does not repeatedly perform indexed lookup from the root.
Expose alias + runtime repeat + mutation through ordinary Python only after
that boundary is validated. Then proceed through nested lists, copying and
sorting in the plan's P0 order. Keep separate ledger entries for:

- semantics (CPython differential cases, alias/rebinding and operand order);
- source bytes, steps and tape use (no weakened gates);
- real ABC samples, beyond-old-capacity tests and maximum-size benchmarks.
