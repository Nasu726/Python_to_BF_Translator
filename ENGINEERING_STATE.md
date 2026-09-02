# Engineering State / Decision Log

This file is a compact memory aid for long-running compiler work. It is **not** a chronological diary. Keep only information that can change future implementation decisions or prevent a repeated failure.

## Retention policy

Each item has a retention class.

- **PERMANENT** — keep while the project exists unless the design decision is explicitly reversed.
- **MILESTONE** — keep until the named milestone is completed, then replace with the resulting invariant/decision and delete obsolete detail.
- **PR-LIFETIME** — keep until the named PR is merged/closed and the relevant regression test exists on `main`.
- **DIAGNOSTIC** — keep only until the associated failure is fixed and protected by a regression test; then reduce to one sentence or delete.

Do not keep successful CI run IDs or routine green-test history. Record success only when it establishes a new invariant or measured baseline.

---

## Current target

### [PERMANENT] Product target

- Public entry point remains ordinary Python source -> standalone standard Brainfuck (`><+-.,[]`).
- Python is compile-time only; generated BF must not require Python/runtime services.
- Primary practical target is competitive programming, currently **AtCoder Beginner Contest C without libraries**.
- Source-size is a correctness-adjacent requirement because AtCoder limits submissions to 512 KiB.
- ABC-C scale means designs must expect common constraints around **N <= 200,000**. A fixed 64-element list cannot be the final storage model.

### [MILESTONE: runtime-sized sequence]

A large runtime sequence must have **BF source size essentially independent of N**. Runtime steps and tape usage may grow with N; emitted source must not be statically unrolled N times.

Acceptance direction:

- `N=64` and `N=200000` should use the same runtime loop body / same generated algorithmic source shape.
- Sequential processing should use cursor/sentinel walks rather than repeated general random-index machinery when data-flow proves this is safe.
- If an input sequence is only consumed once, compiler streaming/fusion may avoid materializing it.
- If it is consumed more than once (e.g. total sum then prefix pass), runtime storage is required.

---

## Stable semantic decisions

### [PERMANENT] Python list/reference semantics

- `b = a` for lists is alias/reference assignment, not shallow copy and not deep copy.
- `a.copy()` / `a[:]` are shallow copies.
- `deepcopy(a)` will be an explicit compiler intrinsic; eventual goal is memoized object-graph copy.
- Python list repetition semantics are preserved, including shared nested references in `[[0] * m] * n`.

### [PERMANENT] Runtime / BF ABI

- 8-bit wrapping tape cells.
- Tape is zero-initialized before execution; compiler may rely on this to omit redundant clear operations.
- Sufficient rightward tape is required by the abstract ABI.
- Standard byte I/O via `,` / `.`.

### [MILESTONE: AtCoder runtime validation] Tritium tape reality

AtCoder currently runs Brainfuck with **Tritium 1.2.73** using `tritium -b -e Main.bf`.

Upstream Tritium on normal 64-bit Linux enables its huge-tape path when `mmap`/signal support is available. That implementation maps tape with `MAP_NORESERVE`; for byte cells the 64-bit path requests a multi-gigabyte virtual region rather than the fallback 1 Mi-cell allocation. Therefore a runtime sequence using millions of cells is technically plausible on AtCoder and should not be rejected merely because it exceeds 1 Mi cells.

Retention condition: keep until an actual AtCoder submission or exact AtCoder Tritium build inspection confirms the large-tape behavior. Runtime time and resident-memory usage remain practical constraints even if virtual address space is large.

### [PERMANENT] Sorting

- Standard `list.sort()` target is stable bottom-up merge sort.
- Worst-case O(n log n), O(n) auxiliary storage, in-place observable semantics, returns `None`.
- Reverse sort must preserve stability; do not sort ascending then blindly reverse equal-key groups.

---

## Current architecture direction

### [MILESTONE: runtime-sized sequence] Numeric/storage layout

Current public compact scalar work uses `Quad64Ref` (32 two-bit lanes plus markers) while fixed `list[int]` stores packed 8-byte values. Boundary conversions are expensive, so this split representation is not the intended large-array ABI.

`pybf/bfbase4.py` now prototypes a radix-4 int64 representation:

- 32 radix-4 digits;
- one value cell plus one traversal marker per lane;
- 66 cells/word versus 99 for Quad;
- source-compact runtime-lane copy/add validated by arithmetic CI;
- subtraction/unsigned-compare are being validated next.

Do not promote base-4 to the public ABI until add/sub/compare, signed behavior, decimal input/output, and sequence integration show acceptable runtime as well as source size. For a sequential record walker, per-value markers may later be avoidable because the record walker itself provides traversal state.

### [MILESTONE: runtime-sized sequence] Sequential storage primitive

`pybf/bfstreamseq.py::RuntimeByteSequence` is the first proven runtime-sized contiguous sequence primitive.

It demonstrates:

- no compile-time capacity/N parameter;
- one BF loop body grows storage at runtime;
- source size does not grow with input length;
- a second BF pass can replay the stored records;
- a reverse marker walk restores the pointer to a known static anchor.

This byte version is an isolated scalability proof, not public Python `list` semantics. Extend the same structure to numeric records rather than enlarging the old fixed-64 list.

### [MILESTONE: runtime heap placement] Memory ordering / layout planning

Do not place huge runtime arrays between hot scalars/scratch/workspace. In Brainfuck, tape distance directly becomes emitted `<`/`>` characters.

Desired layout:

```text
static scalars / small objects
hot scratch
shared arithmetic workspace
compile-time temporary high-water region
----------------------------------------
runtime heap / large sequences
```

`PeakTempArena` is now wired only into the final public compiler through `compiler_layout.py`; legacy frontends retain `_TempArena`. `LayoutPlan` records the final temporary high-water boundary so runtime-sized storage can begin strictly after compile-time temporaries.

Next step is using that measured boundary for actual runtime sequence/list placement. Do not expand a fixed-capacity list to 200000 static slots.

### [MILESTONE: shared IR]

Potential future frontend split:

```text
Python frontend --\
                  -> typed contest IR -> BF backend
C subset frontend-/
```

A C frontend is attractive if restricted to a contest-oriented subset (`int64_t`, `char`, arrays, if/for/while, functions, simple I/O). Do not attempt full ISO C first: pointers, arbitrary `malloc`, preprocessor, integer promotion, structs/unions, UB modeling, etc. would erase much of the simplicity advantage.

---

## Proven source-size lessons

### [PERMANENT] Avoid static expansion of runtime repetition

Major successful reductions came from replacing Python-side/static BF expansion with a small BF runtime loop:

- string iteration streaming when safe;
- list dynamic-index/append slot walks instead of per-capacity unrolling;
- Quad add/sub/compare/copy lane loops;
- compact decimal printing;
- range induction-variable byte shadows when nonnegativity is proven;
- fused `sum += values[i]` and `min(abs(...))` patterns where semantics permit.

General rule: **if the operation logically repeats at runtime, first ask whether the BF itself can contain one loop body rather than the compiler emitting one body per lane/slot/item.**

### [PERMANENT] Host-language speed is secondary until BF size is compact

Measured compilation improved from tens of seconds to ~1 second mainly by emitting less BF, without moving the compiler from Python. A Rust/C++ rewrite would not solve a 30 MB output-size problem.

If native code is later justified, first candidate is the large-string/optimizer stage, not the Python AST frontend.

---

## Important empirical baselines

### [PR-LIFETIME: PR #6] User-provided ABC-B prefix/partition program

Regression source is in `tests/test_compile_performance.py`.

Latest stable compact-lowering baseline:

- final BF: **825,690 bytes**
- compile: about **0.8-1.0 s** on GitHub runner
- still fails only the 512 KiB gate; arithmetic/runtime/frontend shards are green.

This program was later identified as an **ABC-B** solution, not ABC-C. Keep it as a useful lower-bound/regression fixture, but do not mistake passing it for ABC-C readiness.

Earlier progression (retain only until PR #6 closes): roughly 40 MB / ~78 s -> 30.8 MB -> 16.5 MB -> 9.5 MB -> 4.3 MB -> 3.0 MB -> 2.2 MB -> 1.26 MB -> 0.85 MB. The exact intermediate numbers are not architectural requirements.

### [PERMANENT] String-loop real submission result

PR #5 reduced the reported `for c in s` example from about 10.6 MB to **199,730 bytes**, and the generated Brainfuck received an actual AtCoder AC. This validates runtime/streaming source-size optimization as a practical direction.

---

## Failures / rejected approaches

### [PERMANENT] Do not reintroduce full heap handle scans while handles are ordinal

Early heap lookup compared the requested 4-byte handle against every allocated block and exceeded BF step limits. While allocation is monotonic and handles are 1-based block ordinals, use direct ordinal walking. A future free-list may replace this with an indirection table.

### [PERMANENT] Heap reverse-walk coordinate bug

A result-return lane walker once reused coordinates relative to the original block on the second BF-loop iteration, causing nontermination even at a 1B-command test limit. Runtime repeated-loop builders must explicitly rebase their relative coordinate origin when the BF loop moves to a new block.

### [PERMANENT] Right-sentinel requirement for fixed list walkers

A fixed-list lane walker once treated the following variable/control cell as the next-slot sentinel, causing capacity overflow/corruption. Any walker that peeks into a next slot needs a physically reserved sentinel independent of neighboring allocation contents.

### [PERMANENT] Reject value-proportional decimal multiplication loops

A packed decimal parser experiment implemented byte `*10` with nested loops proportional to the byte value. It saved only ~792 source bytes but exceeded 500M runtime BF steps. Do not trade tiny source savings for value-proportional runtime explosions. Prefer fixed-iteration shift/add or a different numeric lane representation.

### [PERMANENT] Repository editing safety

A partial fetch of `transpiler_v2.py` was accidentally used as full replacement, deleting the rest of the module and causing import failures (`clean_bf` missing). It was restored exactly from the previous blob.

Rule: for large foundational files, never perform `partial fetch -> full contents replacement`. Use a full verified blob/file, a narrow new module/subclass, or Git tree/blob replacement from a known-good object.

### [PR-LIFETIME: PR #6] Diagnostic step limits

Do not "fix" heap/runtime failures by only increasing step limits. A temporary 1B limit proved a genuine nontermination bug; root-cause repair restored fast tests. Keep step budgets meaningful after diagnosis.

---

## Current PR #6 working state

### [PR-LIFETIME: PR #6]

- Branch: `abc-c-runtime-heap-1`
- Experimental heap/object/list foundation exists but is **not yet the public list frontend**.
- Public list semantics still rely on fixed-capacity storage for many paths.
- CI is sharded (`arithmetic`, `runtime`, `frontend`, `contest`) and uses xdist inside jobs.
- Current expected failure is the contest 512 KiB gate; other shards should remain green.
- Final public compilation is routed through `compiler_layout.py` and high-water-aware `PeakTempArena`.
- Runtime-sized byte sequence prototype is green and proves N-independent emitted source for contiguous runtime growth/replay.
- Base-4 compact numeric prototype has green copy/add tests; sub/compare validation is in progress.

Next implementation order:

1. Finish base-4 add/sub/compare validation and reject it if runtime costs are poor.
2. Build a runtime-sized numeric sequence on the contiguous record walker, using a representation that avoids per-element public-Quad conversion.
3. Add scale/source-shape tests representing N up to 200000 without emitting N copies of code.
4. Add decimal-token input directly into the scalable numeric sequence.
5. Add sequential iteration/reduction over that sequence; then handle multi-pass cases such as the ABC-B fixture.
6. Migrate ordinary Python `list(map(int, input().split()))` to scalable storage when safe/final semantics are ready.
7. Only after scalable storage is stable, resume alias/deepcopy/sort work on top of it.

---

## Maintenance rule for this file

At each meaningful design change:

1. Update the relevant retained decision, rather than appending another duplicate paragraph.
2. For a failed experiment, record **why it failed** and the general rule that prevents recurrence.
3. For a successful CI run, normally record nothing. Only update a baseline if it changes an acceptance threshold or architectural conclusion.
4. When a `MILESTONE`, `PR-LIFETIME`, or `DIAGNOSTIC` retention condition expires, delete obsolete detail immediately.
