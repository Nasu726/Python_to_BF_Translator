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
- Source size and runtime are independent acceptance axes. The contest regression must remain <= 512 KiB and runtime work must remain linear in runtime N.
- Do not trade a small source reduction for value-proportional or otherwise explosive BF execution.

### [PERMANENT] Runtime / BF ABI

- 8-bit wrapping tape cells, zero-initialized tape, standard byte I/O via `,` / `.`.
- Scalable numeric records are 66-cell fixed-stride hexadecimal int64 records with MARKER/BACK plus DATA/TOTAL/LEFT/ANS words.
- Generated code may require substantial rightward tape. AtCoder uses Tritium; raw interpreter steps are only a regression/complexity proxy. An actual Tritium/AtCoder benchmark is still required before claiming N=200,000 wall-clock feasibility.

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

First-line N is explicitly parsed and carried with the runtime cursor:

- exactly N values participate;
- extra second-line tokens are drained but ignored;
- a short second line zero-fills missing participating values without crossing into the next line;
- N=0 is supported and still drains the list line;
- negative N is normalized to zero.

### [PR-LIFETIME: PR #6] Current measured baseline

Latest public bounded-prefix program on CI:

- optimized BF source: **363,109 bytes**;
- 512 KiB headroom: **161,179 bytes**;
- N=32 all-ones: **1,600,192** raw interpreter steps;
- N=64 all-ones: **2,998,259** raw interpreter steps;
- measured slope: **43,689.6 steps / added record**;
- linear N=200,000 projection: **8,738,120,875 raw steps**.

For comparison, the retained pre-prefix baseline was 438,702 bytes / 4,263,655 N64 steps / 63,841.6 steps per record. The current path is therefore about 17% smaller in source and about 30% lower in the N64 raw-step proxy.

N=64 all-ones component telemetry:

- prefix-retaining counted reader: **1,370,133** cumulative steps, 242,650 optimized source bytes;
- reverse TOTAL propagation: **1,418,393** cumulative;
- stored-prefix candidate only: **2,439,261** cumulative, so candidate construction adds about **1,020,868** steps;
- candidate + abs: **2,788,205** cumulative, so abs adds about **348,944** steps;
- reusable stored-prefix partition pass: **2,836,565** cumulative;
- terminal direct-sentinel full program: **2,998,259** steps.

The two largest remaining runtime targets are now the counted reader and stored-prefix candidate construction. Minimum/gating work has been reduced to a comparatively small remainder.

### [PR-LIFETIME: PR #6] Bounded-prefix fast path

For a nonnegative initial answer that fits the bounded-answer scratch layout, `bfcontestpartition.py` uses a specialized reader/pass/output ABI:

1. `bfhexcounted_prefix.py` parses signed decimal int64 values and retains each inclusive prefix sum in the current record while carrying TOTAL forward.
2. TOTAL is propagated back once after input.
3. `bfhexpartition_prefix.py` computes `TOTAL - 2*stored_prefix` directly; it no longer recomputes `LEFT += DATA` during the second pass.
4. The minimum lowering uses bounded answer width, adaptive narrowing when ANS becomes small, and an ANS=0 fast path. The zero path still handles the fixed-width `abs(INT64_MIN) == INT64_MIN` exception correctly.
5. `bfhexpartition_prefix_terminal.py` leaves the runtime pointer on the final sentinel and prints ANS there directly. This removes the ordinary partition rewind plus runtime-N ANS backward transport.

The terminal output change reduced the public source from 365,746 to 363,109 bytes. On N64 all-ones it reduced full execution from 3,020,551 to 2,998,259 steps; savings are larger when the final ANS has more nonzero digits because backward word transport is avoided.

Wide nonnegative answers and general signed answers retain the canonical reader/reusable partition/output path. The stored-prefix record ABI must not leak into those fallbacks.

### [PR-LIFETIME: PR #6] Input and arithmetic evolution

Important promoted changes, in order:

- direct decimal -> fixed-width hexadecimal parsing;
- lexical single-pass character classification;
- counted N extent carried with the record cursor;
- destructive candidate/state transport;
- thresholded 8/16/24 candidate decoding;
- bounded/adaptive answer-width minimum;
- zero-answer fast path;
- input-time prefix retention;
- terminal direct-sentinel decimal output.

The emitted source remains independent of runtime N.

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

If an operation logically repeats at runtime, first ask whether BF can contain one loop body instead of the compiler emitting one body per slot/lane/item. Major reductions came from runtime string iteration, list walkers, numeric lane loops, direct token parsing, runtime record traversal, and carried decimal-output rounds.

### [PERMANENT] Move reusable work to the earliest pass when it removes a later pass

Input-time prefix retention is a strong example. The reader became more expensive, but storing information already available during input removed a larger `LEFT += DATA` state computation from the partition pass and won across all tested value distributions.

### [PERMANENT] Terminal state need not be normalized back to a compile-time anchor

If no pointer-sensitive code follows, a runtime-dependent sentinel can be the origin for final output. Direct sentinel printing avoids a full carried-state backward transport. Keep reusable helpers normalized, but allow explicit terminal variants when the program ends there.

### [PERMANENT] Measure source and runtime separately and across value distributions

A smaller or apparently simpler BF lowering can be slower. Promote only after correctness plus source/runtime measurement on more than one easy distribution.

Known rejected experiments from PR #6:

- first fused candidate experiment reduced source but worsened runtime;
- `absfast` was much slower;
- fused abs+min increased source and runtime;
- DATA->LEFT candidate merge improved all-ones but regressed mixed/negative inputs;
- fused dual-output prefix-sum reader improved all-ones but regressed mixed/negative inputs and increased source substantially.

The last experiment also exposed an ABI bug: LEFT[15] is the live count extent. Scratch-lane optimizations must respect count/parser fields that share the same runtime record.

### [PERMANENT] Bounded nested BF loops have strict control-flow semantics

A rejected 31 -> 16+8+residual split assumed execution would fall through after a nonzero nested guard. It does not: `]` jumps back to its matching `[` while the control cell is nonzero. Sequentially splitting a bounded decoder is unsafe unless the residual is fully consumed inside the deepest active guard.

### [PERMANENT] Parser scratch may alias future record state

The runtime parser intentionally borrows cells in future-record workspace. A SIGN/BACK alias bug showed that these cells cannot be assumed zero on subsequent records. Explicitly clear scratch/flags at the point their invariant requires zero.

### [PERMANENT] Runtime count must travel with the cursor

For `range(n)` semantics, N cannot remain only at a fixed anchor while the parser advances through runtime-created records. Carry count/extent with the cursor. Count extent avoids a full 16-nibble nonzero rescan after each decrement.

### [PERMANENT] Avoid value-proportional decimal arithmetic

Decimal kernels must have work bounded by representation width/digit count, not numeric byte value.

### [PERMANENT] Repository editing safety

Never perform `partial fetch -> full replacement` on a foundational file. Fetch the full current blob before replacement or isolate work in a new module.

### [PERMANENT] Runtime walker coordinate discipline

Repeated BF-loop builders that move to another record/block must rebase relative coordinates. Reusing coordinates relative to the previous block caused nontermination in an earlier heap return path.

### [PERMANENT] Sentinels must be physical storage, not neighboring variables

Any walker that probes the next slot needs reserved sentinel storage.

### [PERMANENT] Do not mask bugs with step-limit increases

Step limits are diagnostic and regression guards. Increase temporarily only to distinguish slowness from nontermination; fix the root cause and restore meaningful limits.

---

## CI / acceptance gates

### [PR-LIFETIME: PR #6]

CI is sharded into `arithmetic`, `runtime`, `frontend`, and `contest`, with xdist inside each job. Runtime telemetry is part of the main test workflow. Experimental benchmark workflow is manual-only.

Do not relax these gates merely to land the PR:

- standard Brainfuck characters only;
- <= 512 KiB contest source;
- compile-performance guard;
- end-to-end output correctness;
- runtime-N source independence;
- signed int64 boundary tests;
- extra-token / short-line / N=0 counted-input semantics;
- terminal sentinel output including empty sequence and `INT64_MIN` behavior;
- linear raw-step growth guard.

The current runtime shard executes 114 tests including the terminal sentinel regression.

---

## Next implementation order

1. Keep the current specialization green and <=512 KiB.
2. Benchmark the generated program with actual Tritium/AtCoder; do not infer wall-clock feasibility from Python raw-step telemetry.
3. Optimize the two remaining dominant components: counted lexical/prefix reader and stored-prefix candidate construction. Require wins across mixed, large-positive, and negative distributions, not only all-ones.
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
