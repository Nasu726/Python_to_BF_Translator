# Feature / optimization / ABC acceptance track

Updated 2026-09-20. `IMPLEMENTATION_PLAN.md` defines the minimum feature scope.
Every feature must eventually have all three: implementation, optimization, and
validation using real ABC programs. The order can vary; none substitutes for
the others. Keep ordinary Python source unchanged instead of specializing by
problem identity or rewriting away unsupported syntax.

## Current increment: integer-list mutation

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
