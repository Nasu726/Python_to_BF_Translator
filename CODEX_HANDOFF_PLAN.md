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
- S1 runtime primitive complete;
- restricted S2 public Python routing is active for one statically safe dynamic
  character list.

### Last known GitHub-verified baseline

The last remotely verified implementation head is
`5db4ae750e927a85b3f714f55dce95279cbf6c55`. Workflow run #481 is green in all
four normal shards:

- arithmetic: green;
- runtime: green;
- frontend: green;
- contest: green.

The earlier S1d documentation head
`d165c200df91865da772b5419813555623155a90` was likewise verified by run #480.
The S1b/S1c baseline `5bdfe17377c0620c1b7ed99125b3f7fcc5cdddfc`
was verified by run #479.

### Current implementation commit / active work

The current branch head is
`fa36fd83a20d92f48c8601d787d939a211f3231c`. It adds the first restricted S2
frontend route. CI for this new head is pending; local full-suite verification
is green.

The preceding commit `5db4ae750e927a85b3f714f55dce95279cbf6c55`
completes the S1e primitive and telemetry slice. It adds a scoped cursor API
that can keep the physical BF head at the runtime-selected record during a
relative-access session. Ordinary fixed-address emitter operations are
deliberately unavailable while that scope is open; finishing the scope walks
back to the static base.

S1 verification:

- focused runtime-byte-sequence tests: **111 passed**;
- complete repository suite: **408 passed**;
- exact line round trips include lengths **1,024** and **4,097**;
- length 4,097 round trip: **81,585,175 raw steps**;
- read source: **4,465 bytes**;
- read + replay source: **4,990 bytes** (the existing `<5,000` gate remains);
- rooted dynamic-index load/store: **21,790 / 22,104 bytes**;
- rooted two-load/two-store swap: **76,350 bytes**;
- signed-index load/store: **31,152 / 31,466 bytes**;
- cursor load loop: **11,545 bytes**;
- cursor swap loop: **24,062 bytes**.

The emitted source uses only the standard eight Brainfuck commands and remains
independent of runtime sequence length and query count. The cursor ABI currently
accepts prevalidated relative record deltas plus a lane; runtime Python-index to
cursor-coordinate conversion remains S2 work.

S2 local verification at `fa36fd83a20d92f48c8601d787d939a211f3231c`:

- full repository suite: **429 passed in 405.02s**;
- new restricted-dynamic suite: **21 passed**;
- runtime lengths: **0, 1, 8, 9, 256, 300, 1,024**;
- signed indexes: **255, 256, -1, -300, -301** on a 300-byte line;
- `len`, direct empty join, load/store, and generic character iteration work
  beyond the configured fixed capacity;
- iteration preserves `break`, `continue`, and loop `else` behavior;
- all generated output remains standard eight-command Brainfuck.

Current public-capacity source telemetry:

```text
dynamic read + direct join                 5,966 B
dynamic read + len + direct join         155,700 B
load/store/len/join vertical slice       361,931 B
generic character iteration              411,334 B
ABC199 C ordinary source                1,909,540 B
```

The first four fixtures stay within AtCoder's 512 KiB submission limit. ABC199
C is correct on its existing official-sample tests but is still **1,385,252
bytes over the limit**, before maximum-scale runtime is considered. Do not call
S2 complete or hide this with a larger source gate.

Current persistent record layout:

```text
[marker][back][count][length-carrier:4][payload:8][cursor-value]
```

Constants at current head:

```text
RECORD_STRIDE = 16
MARKER        = 0
BACK          = 1
COUNT         = 2
LENGTH        = 3       # 4 little-endian bytes
PAYLOAD0      = 7
PAYLOAD_BYTES = 8
CURSOR_VALUE  = 15
```

The S1b implementation uses one runtime loop per chunk and a bounded 0..7 lane selector rather than emitting eight complete input readers.

---

# 4. Immediate resume point — do this first

S1 is complete and remotely verified. S2a now routes a statically safe single
escaping character list to runtime storage. Resume by confirming the S2a head's
four-shard CI, then reduce the general scalar/query and repeated-access source
cost exposed by ABC199 C. Preserve the fixed `StringRef` fallback for rejected
ownership shapes. Do not claim S2 or maximum-scale ABC199 C support while the
ordinary source remains above 512 KiB.

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

## S1b — eight-byte chunking — DONE / VERIFIED

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

## S1c — non-negative runtime index load/store — DONE / VERIFIED

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

The rooted load uses one forward location pass plus a backward result carrier.
The rooted store uses a location/tag pass and a second forward value-carrier
pass. These compatibility operations return to the fixed base after each call.
S1e adds a separate scoped relative cursor for repeated accesses.

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

## S1d — negative index + range normalization — DONE / VERIFIED

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

The implemented API accepts a preserved packed signed-int64 index. Negative
values in range are normalized against the fixed packed-u32 runtime length;
values outside the representable/range boundary map to `0xffffffff`, which is
guaranteed out of range even for the maximum u32 length. Tests cover `-len`,
`-1`, `-(len+1)`, 255/256 and 65535/65536 borrow boundaries, ±2^32, and both
signed-int64 endpoints. Normalization work is bounded by byte width/value, not
the represented signed magnitude.

---

## S1e — cursor + telemetry — DONE / VERIFIED

The scoped `RuntimeByteCursor` retains the physical BF head at the current
runtime record while consuming relative moves. It supports forward/backward
seek, lane load/store/exchange, a mobile input loop, output/clear, and a final
return to the static base. A single carrier cell can hold the loaded byte and
drive the query loop, making a generic two-index swap source-compact.

Measure raw BF steps separately for:

1. sequential replay;
2. one index access at increasing distances;
3. repeated adjacent/local accesses;
4. alternating first/last accesses;
5. uniformly pseudo-random accesses;
6. repeated swaps.

The profiler `tools/profile_runtime_byte_sequence.py` distinguishes:

- capacity scalability;
- source-size scalability;
- pointer-distance/runtime scalability.

At length 256 with eight swaps, measured incremental raw steps per query were:

| distribution | rooted | cursor | speedup |
|---|---:|---:|---:|
| head-adjacent | 95,215.1 | 14,655.5 | 6.50x |
| middle-adjacent | 6,373,397.6 | 24,271.9 | 262.6x |
| tail-adjacent | 22,607,740.5 | 38,880.4 | 581.5x |
| alternating ends | 11,605,051.8 | 394,652.5 | 29.4x |
| deterministic pseudo-random | 8,300,418.8 | 117,558.6 | 70.6x |

The benchmark fixture intentionally pre-encodes absolute byte indexes as
`(forward record delta, backward record delta, lane)`. This proves the storage
and cursor session but not yet frontend/runtime coordinate conversion. Do not
claim maximum-scale ABC199 C support until S2/S3 end-to-end measurements support
it.

---

# 6. S2 — restricted dynamic character-list frontend — IN PROGRESS

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

## S2a implemented slice

`compiler_dynamic_charlist.py` selects exactly one direct
`name = list(input())` construction when every use is one of:

- signed runtime subscript load or simple-assignment store;
- `len(name)`;
- `for char in name` with a distinct simple target;
- direct `print("".join(name), ...)`.

Aliasing, rebinding, multiple constructions, materialized join results, and
other object uses remain on the fixed compatibility route. The public layout
compiler performs a probe pass, then converges the runtime base so the complete
16-cell left sentinel begins exactly at the temporary high-water boundary.

Quad64 indexes are snapshotted with one runtime lane walker and destructively
packed in adjacent temporary cells. This reduced the ordinary ABC199 C source
from **4,399,170 B** for the first naive S2 connection to **1,909,540 B**. The
remaining excess is spread across general tuple integer input/control-flow and
multiple separately lowered list accesses; it is not evidence for a task-name
special case.

## S2b next gate

Reduce generic source cost far enough that the ordinary ABC199 C program is
below 512 KiB, without weakening any existing gate. Candidate reusable work:

1. keep packed query integers packed longer instead of expanding and repacking
   every scalar around `map(int, input().split())`;
2. fuse the general three-statement character swap idiom or add a two-index
   sequence primitive with ordinary alias/evaluation semantics;
3. feed normalized runtime coordinates into the proven mobile cursor session;
4. measure each change against unrelated dynamic-character programs as well as
   ABC199 C.

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
S1b = DONE + four-shard green (run #479)
S1c = DONE + four-shard green (run #479)
S1d = DONE + four-shard green (run #480)
S1e = DONE + four-shard green (run #481)
S2a = IMPLEMENTED + local 429-test green; head CI pending
S2b = NOT STARTED (ABC199 C source reduction / cursor coordinates)
S3  = NOT STARTED
S4  = NOT STARTED
```

The immediate next action is to verify the S2a branch head in normal CI, then
start the generic source-reduction work listed under S2b.

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
