# Feature / optimization / ABC acceptance track

Updated 2026-09-28. `IMPLEMENTATION_PLAN.md` defines the minimum feature scope.
Every feature must eventually have all three: implementation, optimization, and
validation using real ABC programs. The order can vary; none substitutes for
the others. Keep ordinary Python source unchanged instead of specializing by
problem identity or rewriting away unsupported syntax.

## Current increment: reverse adjacent signed comparisons (ABC136 C)

The unchanged [ABC136 C source](https://atcoder.jp/contests/abc136/tasks/abc136_c)
in `tests/test_abc_c_foundation.py` now emits **193,217 B**, **331,071 B below
512 KiB**, down from 1,724,276 B. Four official samples match CPython. One
mobile reverse pass positions the frame between adjacent records, compares
signed 64-bit values, optionally decrements the left record, checks again,
and suppresses subsequent mutations after failure while still rewinding and
restoring the physical list. The compiler recognizes this specific *source
shape*, not an AtCoder problem name: an unshadowed descending `range` with
two exact adjacent tests, `-= 1`, `break`, preceding `ok = True`, a dead
index and no observable `sum(a)`. Other loops retain the existing lowering.

`python tools/bench_tritium_dynamic_int_updates.py --tritium /path/to/bfi
--trials 1 --large` verifies the unchanged source at N=65/256/1024 and
N=100,000 on portable Tritium rev `14a729d`. Equal heights return Yes in
1.16 seconds; an early-failing left pair returns No in 1.19 seconds on this
machine. Critically, an official-constraint upper-height input with
100,000 copies of 10^9 returns Yes but takes **6.31 seconds**, beyond the
problem's **2-second** limit even on this machine. The `--large` diagnostic's
input-only version takes **4.37 seconds** for those values; the decimal token
parser contributes most of the time, with the reverse pass also costing time.
These are separate local runs, not a proof that subtraction gives an exact
operation-by-operation profile. No ABC136 C judge acceptance is claimed.
Correctness at different runtime bounds, alias
mutations, signed extremal record values and compiler fallback is covered by
focused tests. The restricted int-list ABI still uses zero reads/no-op stores
out of range and fixed signed-int64 arithmetic; P0's general heap handles,
copies, nesting and stable sorting remain open.

## Previous increment: linear index fill for the P0 repeat/alias program

The unchanged first half of `IMPLEMENTATION_PLAN.md`'s P0 example now takes
one mobile pass:

```python
n = int(input())
a = [0] * n
b = a
for i in range(n):
    a[i] = i
print(b[-1], len(a))
```

The source in `tools/bench_linear_int_index_fill.py` emits **514,197 B**,
**10,091 B below 512 KiB**. Raw steps at N=8/32/65 are **1,387,704 /
2,749,414 / 6,118,629**. Before this pass, exactly the same source emitted
679,088 B and took 5,297,050 / 24,979,324 / 88,520,932 steps. The new
Tritium rev `14a729d` portable-build test checks the last element and length
at N=65/1024/200,000 (0.074/0.189/0.287 seconds in one local run). These
timings do not prove performance on the judge or for other statement shapes.

The source-level proof accepts only `for i in range(n): alias[i] = i` or
`range(len(alias))`, with no other use of the induction variable, no
observable `sum(a)`, and an unshadowed `range`. The mobile 56-cell frame
carries the current packed 64-bit index and stores it into each record; the
runtime extent controls the whole-list walk. `range(n)` uses this shorter
walk only when `n` is exactly the unchanged name used by a preceding singleton
repeat and the intervening top-level statements only bind aliases. All
other bounds use a separate 64-bit prefix count, preserving short ranges,
negative bounds and the restricted route's current out-of-range no-op stores.
Tests cover aliases, empty/negative bounds, changed `n`, `clear()`, observed
index/sum fallback, and carry across the low 32 bits and 64-bit wrap.

This does **not** implement `a.sort()` or `print(b)` for arbitrary runtime
lists; the full example and general heap/reference requirements remain open.
The ABC100 C regression still passes official samples after the shared 64-bit
carry code was shortened: it now emits **356,781 B**, with raw steps
**1,032,857 / 2,257,631 / 5,741,459**. The ABC136 C size gap was closed
by the subsequent reverse adjacent walk documented above.

## Previous increment: record-local even/halve/tally loop (ABC100 C)

The ordinary [ABC100 C source](https://atcoder.jp/contests/abc100/tasks/abc100_c)
in `tests/test_abc_c_foundation.py` originally compiled to **357,519 B**, leaving
**166,769 B** below 512 KiB (previously 1,068,376 B). The three official
samples matched CPython; raw steps were **1,034,210 / 2,257,631 /
5,818,718**, down from 8,181,784 / 8,466,202 / 113,901,836. The unchanged
source also returns the correct count at N=65/256/1024 under Tritium rev
`14a729d` on a portable local build. A separate N=10,000/200,000 test using
repeated positive powers of two produced the expected results in 0.243/4.024
seconds locally. This is evidence of linear traversal and a practical large
case on this machine, not an AtCoder judge acceptance or timing guarantee.

The proof recognizes a `range(n)` or `range(len(alias))` loop whose dead index
only appears in `alias[i] % 2 == 0` and `alias[i] //= 2`, followed by one
`scalar += 1` inside the `while`. The list has one statically proven owner and
no observable cached `sum(a)`; a shadowed `range`, observable loop index,
different divisor/body, or other effects retain general lowering. A signed
packed count and initial scalar tally travel inside the existing 56-cell
frame. Each current record is divided arithmetically by two until odd, the
tally increments modulo 2**64, and the frame rewinds before copying the tally
back to the scalar. Negative range bounds perform zero iterations; a zero
payload or an out-of-bounds read of the current restricted route's zero value
continues to loop as the source does. Tests cover aliases, initial tally,
negative values including INT64_MIN, empty input, short bounds, and fallback.
The interpreter remains 8-bit wrapping standard Brainfuck and the program's
BF source length does not depend on N.

At this earlier checkpoint ABC136 C emitted **1,724,276 B** and used rooted
accesses; its later reverse pass is documented above. The runtime heap/object
features in `IMPLEMENTATION_PLAN.md` remain outstanding. Next improve
observable sums and general dynamic list operations.

## Previous increment: runtime-bounded prefix updates

The same statically proven, one-statement pure literal update loop now accepts
`for i in range(n): a[i] op= literal`, even if N differs from `len(a)`. The
range expression is evaluated exactly once before the loop; negative signed
int64 bounds become zero. A mobile int64 count travels with the 56-cell frame,
which visits only `min(max(n, 0), len(a))` records, then rewinds. Count, other
scalars, all suffix elements and list metadata are preserved. Invalid indexes
beyond the list keep the restricted route's documented no-op store behavior
until the runtime error ABI is implemented. The induction variable must be
dead outside the update, `range` unshadowed and `sum(a)` unobserved. All other
loop shapes retain the general rooted route.

The ordinary two-input-line public source in `tools/bench_linear_int_updates.py`
emits **320,326 B** and takes **1,605,592 / 5,113,667 / 10,822,760** raw
steps at N=8/32/65. Its fallback variant reads `i` afterward, emits 540,548 B
and takes 4,894,483 / 22,854,989 / 69,960,058 steps. The outputs differ;
compare growth rather than claiming equivalent whole-program source savings.
The optional native test prints the mutated last element: **517,238 B** (7,050
B under 512 KiB) and exact output at N=65/256/1024 on locally rebuilt
Tritium rev `14a729d` with `-b -e` (0.070–0.088 seconds locally). The original
binary failed with SIGILL on the present host; the rebuild used GCC `-O2
-fwrapv` with DynASM/GNU Lightning/TCC/dlopen/GMP disabled. These timing
samples are not a maximum-N or judge acceptance claim.

The standalone pre-existing decimal `int(input())` path exceeds the unchanged
500-million raw-step guard on INT64_MIN input even without a list or loop.
Negative bound normalization is covered here with smaller runtime inputs and
the literal INT64_MIN case; do not raise the guard or misattribute that parser
cost to the new prefix traversal.

At this earlier checkpoint ABC100 C's inner `while` and answer accumulation,
and ABC136 C's backward adjacent compare and `break`, prevented the
pure-literal loop lowering. Both have subsequent guarded mobile passes above.

## Previous increment: one linear pass for pure literal indexed loops

For the restricted single-owner runtime integer list, the compiler now proves
`for i in range(len(a)): alias[i] op= constant` can visit each current record
exactly once. This proof requires the body to consist solely of one augmented
update, a literal add/sub or positive power-of-two floor-div/mod, no observable
`sum(a)`, and an induction variable used nowhere else in the module. It also
rejects a shadowed `range`. The list extent cannot change in this body, and
the alias shares the same list. Zero-length input is valid. All other loops
retain the existing general lowering. ABC100 C and ABC136 C were subsequently
handled by their own guarded loop shapes above.

The low-level operation carries its 56-cell arithmetic workspace across
consecutive 10-cell records, then rewinds it to its fixed position. It emits
constant-size BF source, uses O(N) runtime record work and 10*N+O(1) tape,
and restores marker/back links and scratch. For the reproducible ordinary
source in `tools/bench_linear_int_updates.py`, the specialized form emits
**214,670 B**, below 512 KiB, and takes **1,085,058 / 4,121,801 /
8,912,966** raw steps at N=8/32/65. The benchmark's fallback variant reads
`i` after the loop, so it has different output; it emits 536,081 B and takes
4,726,420 / 23,518,318 / 72,732,263 raw steps. Compare their growth rather
than interpreting them as equivalent whole-program byte counts. The 65-item
test checks the actual mutated last element through a subsequent indexed read.
An optional native Tritium rev `14a729d` run (`-b -e`) with that last-element
read emits **405,240 B**, checks the mutation at N=65/256/1024, and took
0.055–0.085 seconds per run locally; startup dominates these small timings.
This does not establish maximum-N behavior for any ABC problem.

This is a narrowly proven sequential update, not a general physical cursor:
At this earlier checkpoint conditionals, side effects, observable sums,
nested indexing and the backward ABC136 C pass still used rooted accesses.
The later guarded passes above added mobile loop control for two specific
shapes; arbitrary indexed loops still need locality-preserving routing.

## Previous increment: dynamic integer-list updates and real ABC loops

The restricted single-owner runtime integer-list route now supports ordinary
`a[i] op= rhs` for all existing integer augmented operators: `+=`, `-=`, `*=`,
`//=`, `%=`, `&=`, `|=` and `^=`. This applies to uncapped input lists and
runtime singleton repetition, including statically proven aliases. The target
index is evaluated exactly once; its old value is loaded before the RHS, then
the same normalized packed index is reused for the store. A successful generic
exchange updates an observable cached sum as `sum -= old; sum += new`.

The selector separately records whether `sum(a)` can ever be observed. If not,
simple stores omit cached-sum maintenance and singleton repetition omits its
initial sum reduction. More importantly, pure literal `+=` / `-=` and positive
power-of-two `//=` / `%=` are applied to the matching packed payload inside the
same mobile access scan. Packed division uses an arithmetic right shift, so
negative floor division and positive modulo retain Python semantics. Programs
which do observe `sum(a)`, effectful RHS expressions, and the other operators
keep the fully general load-before-RHS plus exchange path.

Loads use a 48-cell mobile frame; stores and fused literal updates use 56 cells.
The complete 64-bit normalized index, value/result state and scratch move across
10-cell records, then inverse swaps restore the list and fixed base. One access
therefore has source independent of N, `10*N + O(1)` tape and O(N) record work.
Fusing a literal update removes its second full traversal, but N indexed loop
iterations remain O(N²). A locality-preserving cursor/batched lowering is still
required before either ABC loop can claim official maximum-N scalability.

All eight index bytes participate; `2**32` cannot wrap to item zero. Negative
indexes add the cached int64 length once. Until the error ABI exists, invalid
loads return zero and invalid updates are no-ops. Slices, rebinding, escaping,
multiple dynamic owners, append/capacity growth, general heap handles and
nested lists remain outside this restricted route.

Historical public source telemetry for this earlier increment:

| Ordinary source | Generated BF | 512 KiB headroom |
| --- | ---: | ---: |
| ABC170 A indexed reads | 518,124 B | 6,164 B |
| ABC100 C indexed `//= 2` (historical) | 1,068,376 B | -544,088 B |
| ABC136 C indexed compare/`-= 1` | 1,724,276 B | -1,199,988 B |

[ABC100 C — *3 or /2](https://atcoder.jp/contests/abc100/tasks/abc100_c)
now uses the runtime-sized route without changing its ordinary Python source.
It is down from the earlier fixed-list 2,485,322 B. Official-sample raw steps
are **8,181,784 / 8,466,202 / 113,901,836**; all match CPython under the
unchanged 500-million guard.

[ABC136 C — Build Stairs](https://atcoder.jp/contests/abc136/tasks/abc136_c)
also selects runtime-sized storage from the unchanged backward greedy source.
The old 64-item capacity boundary is removed. Source fell from the fixed-route
5,746,608 B to 1,724,276 B; official-sample raw steps are
**15,910,034 / 8,424,538 / 15,884,453 / 4,151,172**. This proves samples and
runtime extent, not the official N<=100000 bound or 512 KiB submission limit.

`tools/bench_tritium_dynamic_int_updates.py` reproduces both programs with
Tritium revision `14a729d` and `-b -e`. Three local trials of every official
sample plus one 65-element case returned exact output. ABC100 runs were
0.090–0.178 seconds and ABC136 runs were 0.115–0.296 seconds after compilation.
These local timings are not an AtCoder-host guarantee. The existing ABC170 A
benchmark remains 0.033–0.045 seconds across all zero positions.

No size or step limit was raised: at this historical checkpoint ABC100/136 were above 512 KiB,
and the first two-pass implementation exposed a real ABC100 sample regression
over 500M steps. The fused packed update replaced that implementation for pure
literals and brought the same sample to 113.9M steps. Tests cover all eight
operators, aliases, cached sums, nested dynamic RHS loads, index/RHS input
order, invalid misses, positions beyond 64, int64 wrap, negative floor/modulo,
frame restoration and layout separation. The three directly affected suites
pass **209 tests**; the complete repository passes **713 tests** with four local
workers.

## Previous increment: public runtime singleton integer repetition

The single-owner route now accepts ordinary `[x] * n` and `n * [x]`, where both
operands are integer expressions. For example:

```python
n = int(input())
a = [0] * n
b = a
print(len(b), sum(a))
b.clear()
print(len(a), sum(b))
```

Repeat selection requires one unconditional top-level repeat construction,
no input-list owner, unconditional aliases and only len/sum/statement-clear uses. General indexed mutation,
append, nested lists, copying, rebinding, multiple dynamic owners and allocator
reuse are **not** implemented by this increment. The pre-existing input-owner
route takes precedence over new repeat candidates: an unrelated fixed repeat
must not silently restore the input list's old capacity bound. Such mixed
programs keep the earlier behavior; the second list does not gain dynamic
repeat support. This does not complete P0's
repeat-plus-indexed-update-and-sort acceptance program.

Both operands evaluate once, in Python order, even for an empty result.
Negative signed-int64 counts normalize to zero (including INT64_MIN). Positive
counts retain all 64 bits: no u32/byte truncation. The cached length is now an
8-byte value alongside the 8-byte cached sum. Input-list construction still
uses its existing u32 reduction count, zero-extended into this header. Repeated
metadata reads preserve their cache; all aliases share logical clear. Sum uses
the existing modulo-2**64 arithmetic ABI.

The runtime constructor carries only remaining count and repeated value through
fresh future cells. Each step shifts this 16-byte carrier forward, fills one
10-cell record, and advances; one marker walk returns to the static base.
It never looks up each element from the list origin. Peak tape is 10*N+O(1),
emitted source is independent of N, and scratch at the final sentinel is
scrubbed. The generic mobile-frame reduction then computes the initial sum.
A prototype that rotated finished records with a larger frame exceeded the
500M-step budget on dense 1024-element input; it was replaced, not retained
behind an increased limit.

Low-level runtime-value constructor (including input initialization), value -1:

| N | Raw BF bytes | Executed raw commands |
| ---: | ---: | ---: |
| 256 | 17,141 | 58,174,912 |
| 512 | 17,142 | 115,793,292 |
| 1024 | 17,144 | 231,227,839 |

The tiny byte differences are only count initialization literals. The loop is
emitted once; doubling N doubles work. Tests preserve every element, both
inputs, markers/backlinks, scratch cleanup and the final pointer. Separate
bounded counter tests verify borrowing at 2**32 and the upper word at INT64_MAX,
without attempting infeasible billion-element allocations.

Public source sizes:

- `n=int(input()); a=[0]*n; print(len(a),sum(a))`: **498,860 B**, below 512 KiB.
- `n,x=map(int,input().split()); a=[x]*n; print(len(a),sum(a))`:
  **543,644 B**, still **19,356 B over 512 KiB**. Runtime support is not a claim
  that every repeated-value program is ready for submission.
- Unchanged ABC103 C retained-list solution: **307,892 B**, still below 512 KiB.
  The 1,240-byte increase from PR #17 is the wider persistent length header.

`tools/bench_tritium_repeat.py` is the reproducible manual benchmark using
Tritium revision 14a729d with `-b -e`. All 18 local trials returned exact output:
zero-repeat N=100000 (0.082–0.117 s), runtime value 2 and -1 at N=3000,
INT64_MIN count, and both int64 value endpoints at N=1 (0.071–0.239 s across the
runtime-value cases). These are local timings, not judge acceptance.
The ABC103 C native benchmark also passes all four maximum-N distributions,
three trials each. Its official samples and N=3000/all-2 retain the unchanged
500M-step regression guard.

Local validation: **178 focused tests passed** across the two affected suites
and existing layout, public API, code-size, compile-performance, character-list
and streaming-list regressions. Both affected suites already run in normal CI.
Regression tests cover operand evaluation/input order, empty results, negative
counts, 65/256/300/1024 extents, aliases and clear, scalar preservation, and
layout separation. Boundary semantics use scalar literals to isolate repetition
from the pre-existing costly 19-digit decimal reader; the native benchmark
additionally verifies those same boundaries through input. Existing parser
budgets are unchanged. An early lifetime-analysis error (rewriting a repeat to
zero let x's storage be reused for n) is fixed by preserving operand reads in
the inference tree and has a dedicated regression case. An additional mixed input/repeat regression
verifies that the new selector preserves the established input-owner route.

## Previous increment: public single-owner integer-list views

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
RHS evaluation; the index is evaluated once. At that historical milestone
these additions used the existing fixed-capacity list frontend. The current
restricted runtime-sized route is documented above.

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
This historical result was still above 512 KiB and used fixed-capacity storage.
The current dynamic-route measurements superseding it are documented above;
the official N<=10000 bound remains unestablished.

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

At that checkpoint this exercised the **fixed-capacity** list frontend. The
unchanged source now selects the restricted runtime-sized route; the current
reverse pass above checks N=100,000 locally. This does not establish judge
time limits or the complete P0 object model.

The historical fixed-route fixture emitted **5,746,608 BF bytes**, above
512 KiB. Sample raw steps were 37,136,657 / 24,597,246 / 34,056,141 /
123,241,308. Current dynamic-route measurements are listed above. The old
numbers document the previous bottleneck, not current source-size readiness.

Local validation: 49 tests covering the ABC fixture, evaluation order,
control flow, legacy list frontend and heap/handle/packed primitives passed;
15 additional public-API/layout/source-size/compile-performance gates passed.
Both edited test modules are already included in the normal CI matrix.

## What is still missing before P0 is complete

1. General public object-handle routing: ordinary list variables still do not
   have Python alias semantics beyond the one statically proven owner/alias
   slice. Low-level handle tests are not general frontend support.
2. Scalable repeated access/traversal: the new contiguous access primitive is
   O(N) per access, so an indexed pass is O(N²). The retained linked-list heap
   is also only a correctness prototype because each handle lookup scans from
   heap origin. Do not hide either issue behind a larger step limit.
3. Append/capacity growth, multiple runtime-sized objects and nested containers
   on the reference model. Singleton repetition currently exists only in the
   restricted single-owner slice.
4. Shallow copy and basic deep copy on that model.
5. Stable bottom-up merge sort and reverse sorting.
6. Temporary allocation reuse; current heap allocation is monotonic.

Invalid indexes still use legacy zero/no-op behavior in these primitives.
The final error-state contract remains required, not waived.

## Next implementation boundary

The ABC136 C reverse pass now validates one additional bounded batched
indexed-loop shape at N=100,000. General indexed loops still need reusable
locality-preserving routing; do not infer a universal cursor from two guarded
patterns. Join the proven single-owner semantics to general heap object
routing and proceed through append, nested lists, copying and sorting in the
plan's P0 order. Keep separate
ledger entries for:

- semantics (CPython differential cases, alias/rebinding and operand order);
- source bytes, steps and tape use (no weakened gates);
- real ABC samples, beyond-old-capacity tests and maximum-size benchmarks.
