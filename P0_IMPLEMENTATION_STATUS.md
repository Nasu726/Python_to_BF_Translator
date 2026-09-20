# Feature / optimization / ABC acceptance track

Updated 2026-09-20. `IMPLEMENTATION_PLAN.md` defines the minimum feature scope.
Every feature must eventually have all three: implementation, optimization, and
validation using real ABC programs. The order can vary; none substitutes for
the others. Keep ordinary Python source unchanged instead of specializing by
problem identity or rewriting away unsupported syntax.

## Current increment: contiguous repeat and physical traversal

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

Implemented in a separate branch stacked on runtime-byte-sequence-1:

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
