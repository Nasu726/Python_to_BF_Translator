# Engineering State / Decision Log

This file is a compact memory aid for long-running compiler work. Keep only information that can change future implementation decisions or prevent a repeated failure.

## Retention policy

- **PERMANENT** — keep while the project exists unless the design is explicitly reversed.
- **MILESTONE** — keep until the named milestone is completed, then replace it with the resulting invariant/decision.
- **PR-LIFETIME** — keep until the named PR is merged/closed and the regression is protected on `main`.

Do not record routine green CI runs. Record only baselines or results that affect architecture or acceptance criteria.

---

## Product target

### [PERMANENT] Public contract

- Public entry point remains ordinary Python source -> standalone standard Brainfuck (`><+-.,[]`).
- Python is compile-time only; generated BF must not require Python/runtime services.
- Primary practical target is competitive-programming style programs, especially workloads with N around 200,000.
- Source size is an acceptance constraint. Current contest regression must remain <= 512 KiB.
- Runtime complexity matters independently of source size. Do not trade a small source reduction for value-proportional or otherwise explosive BF execution.

### [PERMANENT] Runtime / BF ABI

- 8-bit wrapping tape cells.
- Tape is zero-initialized before execution.
- Standard byte I/O via `,` / `.`.
- Generated code may require substantial rightward tape. AtCoder uses Tritium; large virtual tape is plausible, but an actual submission benchmark is still required before treating wall-clock feasibility as proven.

### [PERMANENT] Python semantics

- Signed integer work in the scalable contest slice uses fixed-width int64 wrap semantics.
- `b = a` for lists is alias/reference assignment, not a copy.
- `a.copy()` / `a[:]` are shallow copies.
- Python list repetition must preserve shared nested references.
- Standard `list.sort()` target is stable bottom-up merge sort; reverse sorting must preserve stability.

---

## Current public scalable contest slice

### [PR-LIFETIME: PR #6] Whole-program structural specialization

`pybf/compiler_partition.py` recognizes the complete data-flow shape:

```python
n = int(input())
a = list(map(int, input().split()))
total = 0
for i in range(n):
    total += a[i]
ans = C
left = 0
for i in range(n):
    left += a[i]
    ans = min(ans, abs(total - 2 * left))
print(ans)
```

Variable names are irrelevant; data-flow relationships are required. Near misses fall back to the generic compiler.

The specialization is no longer dependent on “number of tokens == N”. First-line N is explicitly parsed and carried into the second-line reader:

- exactly N values participate in the algorithm;
- extra second-line tokens are drained but ignored;
- a short second line zero-fills the missing participating values and does not read the next line;
- N=0 is supported and still drains the list line;
- negative N is normalized to zero by the counted reader.

`bfcontestpartition.py` selects the faster nonnegative-answer minimum path when `0 <= initial_ans < 2**63`; otherwise it uses the general signed-min path.

### [PR-LIFETIME: PR #6] Current measured baseline

Latest stable public program:

- optimized BF source: **438,702 bytes**;
- 512 KiB headroom: **85,586 bytes**;
- N=32 all-ones: **2,220,724** raw interpreter steps;
- N=64 all-ones: **4,263,655** raw interpreter steps;
- measured slope: **63,841.6 steps / added record**;
- linear N=200,000 projection: **12,768,496,543 raw steps**.

Raw interpreter steps are a regression/complexity proxy, not AtCoder wall-clock time. Tritium performs optimized/JIT execution. Do not claim N=200,000 is fast enough until tested with Tritium/AtCoder.

N=64 phase telemetry on the same public construction:

- counted lexical reader: **1,182,801** cumulative steps;
- reverse TOTAL propagation: **1,231,061** cumulative;
- partition pass: **4,079,669** cumulative;
- reverse ANS propagation: **4,097,595** cumulative;
- decimal output/full program: **4,263,655**.

The partition pass remains the dominant optimization target.

### [PR-LIFETIME: PR #6] Runtime record design

Current scalable numeric sequence uses fixed-stride hexadecimal int64 records. Core carried fields are DATA, TOTAL, LEFT and ANS plus marker/back links. The emitted source is independent of runtime N; tape use and execution scale with N.

The current public reader is `bfhexcounted_lexfast.py`:

- carries N with the record cursor;
- uses direct decimal -> fixed-width hexadecimal accumulation;
- preserves signed int64 wrap;
- explicitly clears parser scratch that aliases future-record cells.

Current public partition arithmetic uses:

1. fused `LEFT += DATA` and `TOTAL - 2*LEFT` construction;
2. destructive state transport to the next record;
3. two's-complement absolute value;
4. minimum propagation;
5. compact decimal output.

The canonical add-candidate decoder uses thresholds 8/16/24 rather than one 31-level source-unrolled decoder. This reduced public source from **458,094** to **438,702 bytes** with identical measured runtime.

---

## General runtime object/list foundation

### [MILESTONE: general scalable Python lists]

PR #6 also contains reusable but still experimental pieces:

- compact packed-u32 primitives;
- 32-bit object handles, with 0 reserved for null;
- monotonic runtime identity allocation;
- marker-walk heap blocks with type/length/capacity/next metadata;
- packed 8-byte int64 payloads;
- heap-backed list root and alias identity;
- dynamic append/index experiments;
- runtime-sized byte and packed sequence prototypes;
- high-water-aware layout planning via `compiler_layout.py` / `PeakTempArena`.

These are **not yet** the general public Python-list backend. The narrow partition specialization must not be mistaken for completion of scalable arbitrary list semantics.

Desired memory ordering remains:

```text
static scalars / small objects
hot scratch
shared arithmetic workspace
compile-time temporary high-water region
----------------------------------------
runtime heap / large sequences
```

Do not place huge runtime arrays between hot scalars or scratch; BF tape distance directly becomes `<` / `>` traffic and source.

---

## Permanent source/runtime lessons

### [PERMANENT] Prefer runtime repetition over static expansion

If an operation logically repeats at runtime, first ask whether BF can contain one loop body instead of the compiler emitting one body per slot/lane/item. Major reductions came from runtime string iteration, list walkers, numeric lane loops, direct token parsing, and runtime record traversal.

### [PERMANENT] Measure source and runtime separately

A smaller or apparently simpler BF lowering can be slower. Promote an optimization only after correctness and measured source/runtime evaluation.

Known examples from PR #6:

- first fused candidate experiment reduced source but worsened partition runtime;
- `absfast` reduced apparent arithmetic structure but was much slower;
- fused abs+min was correct but increased both source and runtime;
- destructive state transport was a genuine runtime win;
- tiered 8/16/24 candidate decoding was a genuine source win with no runtime loss.

### [PERMANENT] Bounded nested BF loops have strict control-flow semantics

A rejected 31 -> 16+8+residual split assumed execution would fall through after a nonzero nested guard. It does not: `]` jumps back to its matching `[` while the control cell is nonzero. Sequentially splitting a bounded decoder is unsafe unless the residual is fully consumed inside the deepest active guard.

### [PERMANENT] Parser scratch may alias future record state

The runtime parser intentionally borrows cells in future-record workspace. A SIGN/BACK alias bug showed that these cells cannot be assumed zero on subsequent records. Explicitly clear scratch/flags at the point their invariant requires zero.

### [PERMANENT] Runtime count must travel with the cursor

For `range(n)` semantics, N cannot remain only at a fixed anchor while the parser advances through runtime-created records. Carry the count/extent with the cursor. Count extent can avoid a full 16-nibble nonzero rescan after each decrement.

### [PERMANENT] Avoid value-proportional decimal arithmetic

A packed decimal parser using value-proportional byte `*10` loops saved little source and exceeded enormous step budgets. Decimal kernels should have work bounded by representation width/digit count, not numeric byte value.

### [PERMANENT] Repository editing safety

Never perform `partial fetch -> full replacement` on a foundational file. A previous partial replacement truncated `transpiler_v2.py`. Fetch the full current blob before replacement or isolate work in a new module.

### [PERMANENT] Runtime walker coordinate discipline

Repeated BF-loop builders that move to another record/block must rebase relative coordinates. Reusing coordinates relative to the previous block caused nontermination in an earlier heap return path.

### [PERMANENT] Sentinels must be physical storage, not neighboring variables

Any walker that probes the next slot needs a reserved sentinel. Treating the next compiler allocation as an implicit sentinel caused overflow/corruption.

### [PERMANENT] Do not mask bugs with step-limit increases

Step limits are diagnostic and regression guards. Increase temporarily only to distinguish slowness from nontermination; fix the root cause and restore meaningful limits.

---

## CI / acceptance gates

### [PR-LIFETIME: PR #6]

CI is sharded into `arithmetic`, `runtime`, `frontend`, and `contest`, with xdist inside each job.

Do not relax these gates merely to land the PR:

- standard Brainfuck characters only;
- <= 512 KiB contest source;
- compile-performance guard;
- end-to-end output correctness;
- runtime-N source independence;
- signed int64 boundary tests;
- extra-token / short-line / N=0 counted-input semantics;
- linear raw-step growth guard.

Latest head before this document refresh had all four shards green.

---

## Next implementation order

1. Keep the current specialization green and source <=512 KiB.
2. Benchmark the generated program with actual Tritium/AtCoder rather than inferring wall-clock performance from the Python interpreter step count.
3. Continue partition-pass optimization only when it produces a measured win; current phase telemetry says this is the dominant cost.
4. Generalize reusable runtime numeric/list primitives rather than adding more one-off whole-program patterns where a shared lowering is practical.
5. Wire scalable storage into ordinary Python `list(map(int, input().split()))` semantics when aliasing/indexing/multi-pass behavior is ready.
6. Resume broader heap/list/deepcopy/sort work after the scalable numeric list ABI is stable.

---

## Maintenance rule

At each meaningful design change:

1. update the relevant retained decision instead of appending a duplicate;
2. for a failed experiment, record the reason and the general prevention rule;
3. update a numeric baseline only when it changes an acceptance or architectural conclusion;
4. delete obsolete MILESTONE/PR-LIFETIME details when their retention condition expires.
