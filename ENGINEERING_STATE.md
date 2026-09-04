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
- Generated code may require substantial rightward tape.
- AtCoder's 2025-10 Brainfuck environment is Tritium 1.2.73. Its published install recipe clones `rdebath/Brainfuck`, checks out commit `14a729d`, runs `make` / `make install` in `tritium`, and executes `tritium -b -e Main.bf`.
- GitHub Actions can reproduce that source revision and command, but its host packages/CPU are not identical to AtCoder. Tritium's build auto-detects optional backends, so Actions wall time is strong evidence of practicality rather than an exact AtCoder timing prediction.

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
- linear N=200,000 raw-step projection: **8,738,120,875 raw steps**.

For comparison, the retained pre-prefix baseline was 438,702 bytes / 4,263,655 N64 steps / 63,841.6 steps per record. The current path is therefore about 17% smaller in source and about 30% lower in the N64 raw-step proxy.

N=64 all-ones component telemetry:

- prefix-retaining counted reader: **1,370,133** cumulative steps, 242,650 optimized source bytes;
- reverse TOTAL propagation: **1,418,393** cumulative;
- stored-prefix candidate only: **2,439,261** cumulative, so candidate construction adds about **1,020,868** steps;
- candidate + abs: **2,788,205** cumulative, so abs adds about **348,944** steps;
- reusable stored-prefix partition pass: **2,836,565** cumulative;
- terminal direct-sentinel full program: **2,998,259** steps.

Raw Python-interpreter steps remain useful for deterministic regression profiling, but they dramatically overstate optimized Tritium execution cost.

### [PR-LIFETIME: PR #6] Exact Tritium-revision benchmark

`tools/bench_tritium_partition.py` plus the manual `atcoder-tritium-benchmark` workflow reproduce AtCoder's published Tritium source revision (`14a729d`, reporting version 1.2.73) and execution command. On a GitHub-hosted Ubuntu runner, N=200,000 was executed three times for each deterministic distribution and every run produced the independently computed expected answer.

Median elapsed wall times:

- all `1`: **0.09 s**;
- all `-1`: **0.16 s**;
- all `65535`: **0.18 s**;
- repeating `(5, -2, 10, 1234, -77)`: **0.14 s**;
- deterministic values in roughly `[-500000, 500000)`: **0.27 s**.

Peak RSS was about **19 MB** for all five N=200,000 cases. A separate size-scaling run observed N=64/1,000/10,000/50,000/200,000 all-ones at approximately 0.03/0.03/0.05/0.09/0.26 s on another hosted runner.

Conclusion: practical N=200,000 execution under Tritium is no longer the main blocker for this vertical slice. The exact AtCoder host may differ, so this is not a promise of identical contest timing, but it is sufficient to deprioritize further narrow micro-optimization relative to generalizing the compiler backend.

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

### [MILESTONE: PR #9 dynamic byte sequence]

`bfstreamseq.RuntimeByteSequence` now stores eight payload bytes per 15-cell
runtime record, carries a packed-u32 length back once after input, and supports
source-size-independent packed-u32 load/store. Signed packed-int64 wrappers
normalize Python-style negative indices once against runtime length and reject
values outside u32/range without low-byte wrapping. S1e cursor/telemetry remains
before public scalable character-list routing.

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

### [MILESTONE: general scalable Python lists] Deferred generic route experiment

A generic heap-backed `list[int]` AST route was prototyped near the end of PR #6 and intentionally removed from the merge candidate after measurement. The prototype remains recoverable from branch history (`e79a12c` introduced `compiler_dynint.py`; `5417e27` added its tests).

What worked with runtime-sized input beyond the old 64-element fixed list capacity:

- runtime `len` and index 69 on a 70-element list;
- runtime and Python-style negative indexing;
- `b = a` alias identity plus `append` visibility through both names;
- emitted-source independence from runtime line length.

What failed the acceptance bar:

- iterating a 70-element list twice exceeded the existing **1,000,000,000 raw-step guard**.

This was not treated as a reason to raise the guard. The current correctness-first representation compounds two expensive traversals: each list operation starts from the list head, and each object-handle dereference performs ordinal heap lookup from the heap origin. Repeated sequential access therefore acquires O(n²)-style work with very large BF constants before the list is remotely contest-sized.

Required architectural direction before public dynamic-list lowering:

1. make sequential iteration a true forward walk rather than repeated indexed lookup;
2. prefer chunked/contiguous list storage so nearby elements are physically nearby on tape;
3. avoid resolving every element handle by rescanning from the heap origin;
4. only then reconnect the generic AST route and restore >64-element repeated-pass tests.

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

### [PERMANENT] Use problem-specialized BF as a laboratory, not a compiler shortcut

For difficult real ABC workloads, a separately maintained code-golfed or
problem-specialized BF implementation can reveal useful tape-native structure
and establish an empirical upper bound. Compare it with public compiler output,
then extract only reusable semantics-preserving ideas (for example cursor-aware
indexing, logical offsets, or delayed permutations). Never dispatch production
lowering by problem identity, exact source text, or expected output.

### [PERMANENT] Raw BF step counts and optimized-interpreter wall time are different metrics

The Python reference interpreter is intentionally literal and excellent for deterministic complexity/regression checks. Tritium performs substantial static optimization and JIT/optimized execution. Billions of literal BF operations can therefore correspond to sub-second execution for this structured program. Keep both metrics: do not replace correctness-oriented raw-step gates with noisy wall-clock CI, and do not use raw steps alone to reject a practically fast Tritium program.

### [PERMANENT] Do not build list iteration from repeated indexed heap lookup

On the current one-element-per-block heap, a 70-element two-pass generic list iteration exceeded 1,000,000,000 raw steps even though direct length/index/alias tests were correct. Sequential operations need a carried physical/chunk cursor. Repeated `get(index)` from head plus ordinal handle lookup from heap origin is an architectural anti-pattern for scalable BF containers.

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

CI is sharded into `arithmetic`, `runtime`, `frontend`, and `contest`, with xdist inside each job. Runtime raw-step telemetry is part of the main test workflow. Exact Tritium wall-time benchmarking is manual-only because it clones/builds an external project and is inherently host-sensitive.

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

1. Merge PR #6 with the proven 363,109-byte bounded-prefix specialization, runtime/object foundation, and manual exact-Tritium benchmark; do not expose the deferred generic dynamic-list prototype.
2. Start the next PR at the container representation level: add a chunked/contiguous int64 list layout and a carried sequential walker that does not resolve each element through heap-origin ordinal lookup.
3. Re-run the >64-element generic tests from commits `e79a12c` / `5417e27`, especially repeated passes, under the unchanged raw-step guard.
4. Once the container is scalable, reconnect `list(map(int, input().split()))`, `len`, indexing, aliasing, `append`, and iteration to ordinary AST lowering.
5. Then extend copy/slice, nested/reference behavior, deepcopy and stable sort on top of the stable list ABI.
6. Return to partition reader/candidate micro-optimization only if a real Tritium benchmark exposes it as a practical bottleneck.

---

## Maintenance rule

At each meaningful design change:

1. update the relevant retained decision instead of appending a duplicate;
2. for a failed experiment, record the reason and the general prevention rule;
3. update a numeric baseline only when it changes an acceptance or architectural conclusion;
4. delete obsolete MILESTONE/PR-LIFETIME details when their retention condition expires.
