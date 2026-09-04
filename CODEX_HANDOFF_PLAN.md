# Python-to-Brainfuck Translator — Persistent Handoff / Execution Plan

This document is the persistent source of truth for continuing the current compiler work across Codex, ChatGPT, or another implementation session.

It is intentionally written so that a future session does **not** need the original chat history to recover:

- what has already been implemented;
- what is currently under active development;
- which claims are proven by CI / Tritium measurements;
- which ideas failed and must not be repeated blindly;
- the exact next implementation order;
- the architectural constraints that should not be weakened merely to make tests pass.

If chat context and this file disagree, first inspect the current GitHub PRs / branch heads and CI. Treat repository state + green CI as authoritative, then update this document.

---

# 1. Project objective

Translate ordinary contest-style Python into standalone **standard 8-command Brainfuck** while remaining practically executable under the AtCoder Brainfuck environment (Tritium), rather than merely producing semantically correct but unusably slow or enormous BF.

The main product goal is not task-specific BF generation. Prefer reusable compiler transformations and runtime primitives that allow ordinary Python source to scale.

The compiler intentionally uses a fixed signed-int64 arithmetic ABI rather than Python arbitrary-precision integers unless explicitly stated otherwise.

Do not weaken source-size, correctness, or CI gates to make a feature appear complete.

---

# 2. Working rules / non-negotiable constraints

1. **Never merge a PR unless the user explicitly asks.**
2. Keep PR boundaries reviewable. Do not mix an unfinished next architecture phase into a PR whose current feature is already complete.
3. Standard BF only: emitted programs must contain only `><+-.,[]`.
4. Treat raw Python BF interpreter step counts as regression/complexity proxies, not as AtCoder wall-clock predictions.
5. For practical performance claims, benchmark the AtCoder-published Tritium revision when possible:
   - repository: `rdebath/Brainfuck`
   - commit: `14a729d`
   - Tritium version observed: `1.2.73`
   - execution command: `tritium -b -e Main.bf`
6. GitHub Actions and AtCoder do not have identical host hardware. Tritium source revision / command can match while wall-clock time still differs.
7. Do not replace correctness failures with larger step limits unless the computation is demonstrably correct and the limit itself is the intended tested property. Structural O(n^2) failures are design failures, not CI configuration failures.
8. Prefer actual AtCoder ABC problems as acceptance workloads before inventing synthetic feature APIs.
9. Capacity scalability and runtime practicality are separate gates. Storing 400k bytes is not equivalent to solving ABC199 C at maximum constraints.
10. Keep full test collection guard enabled so stale/unlisted tests cannot silently disappear from explicit matrix shards.

---

# 3. Current PR stack

The current work is stacked. Landing order matters.

## PR #7 — `dynamic-list-linear-1`

Title: **Make single-use int input lists stream linearly**

Base: `main`

Head at last verified state:

```text
d3c2eaebe17e1576fedef6e209f4ba82a8fa3bd2
```

Status at handoff:

- open
- Ready for review (not draft)
- mergeable
- normal four-shard CI green
- **not merged by the assistant**

Main feature:

```python
A = list(map(int, input().split()))
total = 0
for x in A:
    total += x
```

When `A` is dead except for a single consumer loop, the compiler eliminates the temporary list and streams signed integer tokens directly into the loop target.

Real acceptance workloads:

- ABC153 B — Common Raccoon vs Monster
- ABC103 C — Modulo Summation

ABC153 B public-default BF size measured:

```text
297,843 bytes
```

At official max `N=100000`, exact AtCoder Tritium source revision on GitHub Actions produced correct output with median elapsed approximately:

```text
all 1       0.12 s
mixed       0.31 s
all 10000   1.09 s
peak RSS    ~6.3 MB
```

These timings are GitHub-runner timings, not literal AtCoder judge timings.

---

## PR #8 — `string-conversions-1`

Title: **Add zero-copy char-list views and int/string conversions**

Base: `dynamic-list-linear-1` (PR #7)

Head at last fully verified state:

```text
6dcc60c1d38b92be901acc4383a158d254c1ecda
```

Status at handoff:

- open
- Ready for review (not draft)
- mergeable
- latest workflow run #475: four-shard CI green
- **not merged by the assistant**

Implemented feature class:

```python
chars = list(input())
chars[i] = "X"
print("".join(chars))
```

`list(input())` and empty-separator `join` are representation-preserving views over the existing byte-string payload rather than general `list[str]` materialization.

Supported restricted character-list operations include:

- positive / negative / runtime index load and store;
- one-character temporaries and swaps;
- `len(chars)` with cached logical length;
- iteration;
- direct empty-separator join output;
- snapshot semantics when join result is assigned to an immutable string;
- re-reading the same restricted variable from `list(input())`.

Important restrictions:

- element values are currently one non-NUL byte, code point `1..255`;
- multi-character element assignment rejected;
- NUL element rejected because current payload uses NUL termination;
- non-byte Unicode rejected;
- general mutable alias semantics are not claimed;
- append/insert/delete are not part of this representation;
- direct scalar-string / scalar-integer semantic leakage from the physical `StringRef` is rejected;
- out-of-range access temporarily uses empty-load / no-op-store instead of Python `IndexError` until runtime exception propagation exists.

Explicit conversions implemented:

```python
s = str(n)
n = int(s)
```

Contract:

- `str(int64)` including `INT64_MIN` / `INT64_MAX`;
- `int(str)` for valid signed ASCII decimal text with surrounding ASCII whitespace;
- `str(str)` value semantics;
- `int(int)` identity under fixed signed-int64 ABI;
- invalid-text `ValueError` propagation is deferred.

Important correctness fixes already made:

- Quad64 vs legacy Int64 parser ABI mismatch;
- constant/runtime index 256 must not wrap to slot 0;
- character temporary inference changed to must-analysis so later `"XY"` assignment cannot silently truncate;
- NUL/non-byte assignment rejected;
- type-inference AST and liveness AST separated: inference may rewrite expressions, but lifetimes must be calculated from the original user AST;
- logical character-list identity cannot be inferred only from reused physical tape addresses;
- direct `str(int)` formatting snapshots integer source before destination clear because liveness allocation can reuse source/destination storage.

Latest green source telemetry at public capacity 255:

```text
runtime character store + join   430,146 B
"".join(list(input()))            38,200 B
int(str_value)                    341,636 B
str(int_value)                    401,007 B
direct int(input())               194,381 B
```

All remain under the repository's 512 KiB source gate.

Real acceptance target currently included:

- ABC199 C — IPFL official samples

Do **not** claim maximum-constraint ABC199 C support yet. The fixed string backing is still capacity-limited and max-scale random access has not been validated.

---

## PR #9 — `runtime-byte-sequence-1`

Title: **Extend runtime byte sequences toward scalable character storage**

Base: `string-conversions-1` (PR #8)

Status at handoff:

- open
- draft
- runtime primitive only; public Python routing intentionally unchanged

### Last known GitHub-verified baseline

The pre-S1c branch head `277e01034a825cbba65717b70c048ea17f7ca9a3`
was verified by workflow run #478:

- arithmetic: green
- runtime: green
- frontend: green
- contest: green

That baseline includes S1a and the first S1b implementation. The S1b-specific
boundary matrix was added and verified locally in the next implementation
commit described below.

### Current implementation commit / active work

Latest implementation commit in this work session:

```text
d136924c6e8fffca85abd6110ca7aa5d24c142ae
```

It completes the missing S1b verification matrix and adds S1c non-negative
runtime index load/store. Local verification after the implementation:

- focused runtime-byte-sequence tests: **54 passed**;
- complete repository suite before the final three swap cases were added:
  **348 passed**;
- the three final repeated-operation/swap cases also pass;
- S1b read source: **4,298 bytes**;
- S1b read + replay source: **4,819 bytes**;
- dynamic-index load fixture: **21,057 bytes**;
- dynamic-index store fixture: **21,447 bytes**;
- two-load/two-store swap fixture: **74,083 bytes**.

The new commit still needs GitHub four-shard CI. Do not call S1c remotely
verified until that run is green.

Current intended persistent record layout:

```text
[marker][back][count][length-carrier:4][payload:8]
```

Constants at current head:

```text
RECORD_STRIDE = 15
MARKER        = 0
BACK          = 1
COUNT         = 2
LENGTH        = 3       # 4 little-endian bytes
PAYLOAD0      = 7
PAYLOAD_BYTES = 8
```

The S1b implementation uses one runtime loop per chunk and a bounded 0..7 lane selector rather than emitting eight complete input readers.

---

# 4. Immediate resume point — do this first

Do **not** start frontend integration yet.

Resume at **S1d negative-index normalization**, after confirming the current
PR head's four-shard CI.

## Completed S1b verification matrix

For a payload of length `n`, inspect both roundtrip and materialized chunk metadata.

Required boundaries:

```text
n=0
n=1
n=7
n=8
n=9
n=15
n=16
n=17
n=255
n=256
```

Expected chunk counts / per-record `COUNT` values:

```text
0   -> []
1   -> [1]
7   -> [7]
8   -> [8]
9   -> [8, 1]
15  -> [8, 7]
16  -> [8, 8]
17  -> [8, 8, 1]
255 -> 31 full chunks + [7]
256 -> 32 full chunks
```

For every case verify:

1. exact roundtrip output;
2. runtime length equals `n`;
3. every materialized record has correct `COUNT`;
4. first zero marker is immediately after the last materialized chunk;
5. `BACK==0` on record zero and `BACK==1` on later records / relevant sentinel;
6. left sentinel marker remains zero;
7. pointer returns to `seq.base` after reader and replay;
8. payload bytes beyond `COUNT` in a partial chunk remain zero;
9. exact-multiple-of-8 input does not create a fake zero-length materialized chunk;
10. generated source remains independent of runtime length;
11. retain the current `<5000 bytes` source-size gate unless real measurements justify a design change. Do not loosen it preemptively.

Also add a test that the reader stops on LF and does not consume the following logical input line unexpectedly.

These cases now live in `tests/test_bfstreamseq.py`. Preserve them when changing
the record walker. The normal four-shard CI remains required after every new
head.

---

# 5. S1 implementation plan

## S1a — runtime length carrier — DONE / VERIFIED

Goal:

Avoid updating a fixed header at the left side of the tape for every byte, which would create growing pointer travel.

Design:

- packed little-endian u32 length;
- length travels forward with runtime record walker;
- increment occurs locally;
- after LF, final carrier is propagated left once to fixed sentinel metadata.

Verified boundaries include at least 0/1/7/8/9/255/256.

---

## S1b — eight-byte chunking — DONE / LOCALLY VERIFIED

Goal:

Reduce physical tape distance from approximately one record per character to one record per eight characters.

Reason:

Standard Brainfuck pays for pointer travel. ABC199 C requires many runtime indexed swaps; capacity alone is insufficient if the representation spaces characters too far apart.

Design constraints:

- use all eight payload lanes;
- keep runtime source-size independent of input length;
- `COUNT` stores valid bytes in final chunk;
- exact 8-byte boundaries must not create fake data records;
- one bounded lane selector is acceptable;
- avoid emitting an entire decimal/string reader eight times.

The full boundary/metadata matrix above is green. The current source remains
under the original 5,000-byte gate.

---

## S1c — non-negative runtime index load/store — DONE / LOCALLY VERIFIED

Logical decomposition:

```text
record_index = index >> 3
lane         = index & 7
```

The implementation realizes the same decomposition as a bounded eight-lane
countdown per materialized chunk. It never truncates the packed-u32 index to one
byte, and emitted source is independent of sequence capacity/runtime length.

Required primitive operations:

```text
load_byte(dst, sequence, index)
store_byte(sequence, index, src)
```

Temporary out-of-range behavior remains:

```text
load  -> 0 / empty character
store -> no-op
```

until runtime exception propagation is implemented.

Load uses one forward location pass plus a backward result carrier. Store uses a
location/tag pass and a second forward value-carrier pass. Both currently return
to the fixed base after each operation and are linear in record distance; S1e
must measure and improve repeated access.

Verified cases include:

```text
indices 0, 1, 7, 8, 9
last valid index
first invalid positive index
index 255 / 256 boundary
store followed by replay
load must preserve sequence
store must alter exactly one byte
```

The tests also cover index `0xffffffff`, preservation of the packed index and
runtime length, cleaned scratch metadata, standard BF only, and repeated
two-load/two-store swaps. Python negative indexing remains isolated for S1d.

---

## S1d — negative index + range normalization

Use runtime length to normalize:

```text
if i < 0:
    i += len
```

Then verify:

```text
-len      -> first byte
-1        -> final byte
-(len+1)  -> out of range
```

Never reduce a potentially large index to an 8-bit selector before range checking. The project previously had an index-256-to-slot-0 bug; do not reintroduce it.

---

## S1e — cursor + telemetry

Before public Python routing, determine whether the representation is practically indexable.

Add a cursor abstraction that can retain a current record position where semantically safe.

Measure raw BF steps separately for:

1. sequential replay;
2. one index access at increasing distances;
3. repeated adjacent/local accesses;
4. alternating first/last accesses;
5. uniformly pseudo-random accesses;
6. repeated swaps.

The purpose is to distinguish:

- capacity scalability;
- source-size scalability;
- pointer-distance/runtime scalability.

Do not claim maximum-scale ABC199 C support until S1e/S3 measurements support it.

---

# 6. S2 — restricted dynamic character-list frontend

Only start after S1 primitive tests and telemetry are credible.

Target ordinary Python:

```python
s = list(input())
i = int(input())
c = s[i]
s[i] = "X"
print("".join(s))
```

Frontend integration must reuse PR #8 semantic restrictions:

- byte element ABI `1..255`, no NUL;
- negative index semantics;
- range guards;
- one-character temporaries / swap;
- `len`;
- iteration;
- representation-preserving `list(input())` / empty join;
- liveness from original user AST;
- no accidental scalar-string / scalar-int leakage;
- no unsupported alias semantics.

Initially support only one runtime-sized escaping character sequence per safe program/region if necessary. Reject unsupported multiple-object cases rather than faking object identity.

Keep the fixed `StringRef` implementation as fallback for programs not selected for scalable backing.

Do not implement an ABC199-specific query algorithm. The optimization must operate on generic character-list semantics.

---

# 7. S3 — scaling / Tritium acceptance

Create a benchmark harness, likely:

```text
tools/bench_tritium_abc199.py
```

Use the same AtCoder Tritium source revision as previous benchmarks.

Progression:

1. official ABC199 C samples;
2. small synthetic random swaps with Python oracle;
3. medium sequence lengths / query counts;
4. local-index distribution;
5. alternating-end distribution;
6. random distribution;
7. official maximum dimensions only if curves remain credible.

Acceptance for a maximum-scale claim requires all of:

- ordinary Python source goes through public compiler;
- independently verified output;
- standard BF only;
- generated source within submission limit;
- runtime practically credible under Tritium;
- no task-specific answer shortcut.

If linear record-distance traversal is too slow, stop and redesign indexing. Do not hide failure with special casing.

Possible next research directions if random access is too slow:

- cursor-relative indexing;
- chunk-local access caching;
- multiple anchor cursors;
- batched semantics-preserving transformations when source/dataflow allows them;
- stronger indexed storage representation.

Avoid assuming a classic RAM-style tree automatically helps: Brainfuck tape movement to reach tree metadata can dominate.

## Problem-specialized BF laboratory (discovery only)

For each difficult ABC tier, it is useful to write or code-golf a separate
problem-specialized Brainfuck solution before finalizing the general compiler
architecture. Treat that program as an experimental upper bound/oracle, not as
the public lowering.

The purpose is to expose BF-native structure that ordinary RAM-model design may
hide. For ABC199 C, likely examples include logical half offsets, delayed
permutations, cursor placement, and avoiding physical whole-string movement.
Measure source size, tape travel and Tritium time, then extract only reusable
ideas such as generic index remapping, deferred permutation, or cursor-aware
access.

Rules:

1. keep specialized BF/harnesses clearly separated from public compiler code;
2. never dispatch on a problem name, exact source text, or expected answer;
3. compare the specialized baseline with ordinary-Python compiler output;
4. promote an idea only after it is phrased as a semantics-preserving reusable
   transformation/runtime primitive and covered by non-problem-specific tests;
5. record failed golf structures too when they reveal a tape-movement lower
   bound or an architectural dead end.

---

# 8. S4 — generalization after byte primitive is measured

Only after dynamic bytes have a known cost model:

1. support multiple runtime-sized byte objects via object/handle semantics;
2. reuse proven chunk/walker ideas for dynamic `list[int]`;
3. implement alias/reference semantics correctly;
4. replace the previous one-element-per-heap-block list path where it causes structural O(n^2) behavior;
5. add indexed/mutating integer-pass acceptance (ABC136 C class);
6. add stable sort;
7. promote ABC192 C / ABC221 C end-to-end tests;
8. add queue/deque workloads (ABC247 D class).

Do not rebuild the previous linked one-element list approach as the primary scalable backend without addressing its measured complexity problem.

---

# 9. Known failed / rejected approaches — do not repeat without new evidence

## Generic dynamic list experiment after PR #6

A correctness-first heap-backed dynamic list route used:

- one heap block per element;
- ordinal handle lookup from heap start;
- list index traversal from head.

70 elements traversed twice exceeded **1,000,000,000 raw BF steps**.

Conclusion:

The representation had structural O(n^2)-style behavior. The unfinished public route was removed instead of raising the step limit.

---

## Derived signed difference state in partition optimization

Using

```text
D_i = D_{i-1} - 2*a[i]
```

reduced state/source but made runtime much worse because negative fixed-width hex values become dense high-F digits and existing BF arithmetic/transport cost depends partly on digit value.

Lesson:

Fewer logical state words do not necessarily mean faster BF.

---

## Reader / candidate micro-fusions that only win on all-ones

Several variants looked better on all-ones but lost badly on mixed or negative input distributions.

Lesson:

Never promote a BF micro-optimization from one low-value input distribution. Benchmark signed/mixed/high-digit cases.

---

## Dynamic fixed-string selector explosion

Early runtime character store / string conversion implementations emitted MB-scale BF because they statically expanded many candidate absolute destinations.

Observed examples during PR #8 development:

```text
runtime character store initial      ~22 MB
int(str) naive runtime-loop variant  ~12 MB
str(int) dynamic-slot variant        ~5.75 MB
```

Preserving rotations and direct formatting reduced these below 512 KiB.

Lesson:

Avoid capacity-wide static destination selection when a local runtime walker/rotation can express the same operation.

---

# 10. Existing reusable runtime components

Do not assume S1/S4 must be built from scratch.

Important existing modules:

```text
pybf/bfstreamseq.py
    Runtime-sized byte sequence. Current active S1 target.

pybf/bfpackedseq.py
    Runtime-sized contiguous packed int64 sequence used by contest/partition work.

pybf/bfheap.py
    Runtime-walked fixed-stride heap blocks with ordinal handles.

pybf/bfobjects.py
    Object handle primitives.

pybf/bfdynlist.py
    Correctness-first heap-backed list[int] root / append / indexed access.
    Useful semantic reference, but current physical representation is not the desired scalable end state.

pybf/bfpacked.py
    Packed little-endian u32 metadata arithmetic.
```

Before adding a new primitive, inspect these modules for reusable walkers / carrier / packed metadata logic.

---

# 11. ABC compatibility roadmap

Use real ABC source shapes to decide what to build.

Current anchors:

```text
ABC129 B   prefix/partition shape
ABC153 B   runtime int-list -> linear fold
ABC103 C   runtime int-list -> linear fold
ABC199 C   character array, runtime swap/indexing
ABC136 C   indexed/mutating integer pass
ABC088 B   sort/reordered traversal
ABC192 C   int <-> decimal string + digit sorting
ABC221 C   decimal digit rearrangement / int conversion
ABC247 D   queue/deque workload
```

For every promoted tier, ideally retain:

- ordinary Python source;
- official samples;
- scaling/max benchmark where meaningful;
- independent expected output;
- source-size measurement;
- raw BF step measurement where useful;
- Tritium wall-clock measurement for practical claims.

---

# 12. Float plan

Float is deliberately low priority.

Do not spend implementation time on float until int/string/list/container work is substantially stable.

If/when implemented, prefer an explicit software float64 model rather than silently approximating with integers.

Expected eventual scope:

- decimal parse/print;
- literals;
- int <-> float conversion;
- `+ - * /`;
- comparisons;
- explicit NaN / Inf / rounding policy tests.

Until then, unsupported float syntax should fail clearly.

---

# 13. CI / performance discipline

Normal CI currently uses four logical shards:

```text
arithmetic
runtime
frontend
contest
```

Contest shard also performs full pytest collection.

Do not remove the full collection guard.

When adding a new feature:

1. add focused tests first;
2. ensure the focused tests are also included in normal shard CI;
3. run all four shards before declaring a milestone green;
4. keep external Tritium benchmarks manual-only unless there is a strong reason to put external builds in every PR run;
5. add source-size telemetry for important public source shapes rather than merely a Boolean limit test.

---

# 14. How to resume in a fresh Codex / ChatGPT session

A future session should begin with this checklist.

## Step 1 — inspect repository / PR state

Check PRs #7, #8, #9 and current branch heads.

Expected stack at the time this document was written:

```text
main
  -> PR #7 dynamic-list-linear-1
       -> PR #8 string-conversions-1
            -> PR #9 runtime-byte-sequence-1
```

If any have been merged, rebase/retarget descendants as appropriate before new work.

Do not assume the SHA values in this document remain current after subsequent commits.

## Step 2 — inspect latest CI

Do not trust only an older green run if the current head has newer commits.

## Step 3 — determine current milestone

At the latest local handoff:

```text
S1a = DONE + four-shard green
S1b = DONE + local boundary/full-suite verification
S1c = DONE + local primitive/full-suite verification; head CI pending
S1d = NOT STARTED
S1e = NOT STARTED
S2  = NOT STARTED
S3  = NOT STARTED
S4  = NOT STARTED
```

The immediate next action is to confirm current-head CI, then implement S1d.

## Step 4 — update this file after each milestone

After S1b, S1c, etc., update:

- milestone status;
- current head SHA;
- exact green workflow run;
- measured source size / steps / Tritium metrics;
- any rejected experiments and why.

This is important so the project remains recoverable after chat truncation or tool/session changes.

---

# 15. What to report back to the user after a work session

Keep the status concise but concrete:

1. current milestone completed / active;
2. exact correctness status;
3. exact performance/source metrics if measured;
4. any design that was rejected and why;
5. what the next milestone is;
6. whether a PR is draft / ready / merged;
7. never state that something is maximum-scale practical unless the appropriate Tritium benchmark has actually been run.

When handing back from Codex to ChatGPT, point ChatGPT to this file first and provide the latest branch / PR number if known.
