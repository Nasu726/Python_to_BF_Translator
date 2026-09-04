# Implementation Plan Addendum — contest syntax and conversions

This file supplements `IMPLEMENTATION_PLAN.md` with the near-term priorities
identified while validating the compiler against real AtCoder ABC programs.
The main plan remains authoritative for the long-term object/container model.

## Priority A — representation-preserving character-list syntax

Support the ordinary contest idiom:

```python
chars = list(input())
chars[i] = "X"
print("".join(chars))
```

The compiler should not allocate/copy a second runtime container merely because
Python syntax changes between `str` and `list[str]` here. For the restricted
character-list case, a NUL-terminated byte string and a mutable list of
one-character strings have the same useful payload representation. Therefore:

1. `list(input())` becomes a **mutable character-list view** over the input byte
   buffer rather than a generic heap `list[str]` materialization.
2. `"".join(character_list)` becomes the corresponding **string view** over the
   same buffer. The conversion itself performs no runtime data reshaping.
3. Index load/store, negative indexing, `len`, and iteration operate directly on
   that byte buffer.
4. Assignment to an element is initially restricted to a statically known
   one-character string. Supporting arbitrary string elements belongs to the
   general dynamic `list[str]` backend.
5. Do not silently fake Python list semantics. Alias assignment, direct list
   repr printing, insertion/deletion/append, or non-empty-separator joins must
   fall back to / wait for the general mutable-list implementation unless their
   semantics are proven separately.
6. Runtime indices must be range-checked before conversion to the current
   byte-sized physical selector. Values such as 256 must never wrap to slot 0.

Acceptance examples:

```python
s = list(input())
s[0] = "A"
s[-1] = "Z"
print("".join(s))
```

```python
s = list(input())
i = int(input())
s[i] = "#"
for c in s:
    print(c, end="")
```

The target is that `list(input())` and `"".join(s)` themselves add effectively
zero BF runtime work beyond the input, mutation, or output operations actually
required by the program.

### Real target: ABC199 C — IPFL

ABC199 C is the first direct real-problem target for this syntax. It requires
runtime-index character swaps and delayed half-string swapping. Official
constraints include `N <= 2*10^5`, so `|S| <= 4*10^5`, and `Q <= 3*10^5`.

Two separate gates are required:

1. **source-shape gate now** — compile/run the official samples using
   `list(input())`, runtime character indexing, a one-character temporary, and
   `"".join(s)`;
2. **maximum-scale gate later** — replace the current <=255-byte fixed
   `StringRef` backing with a scalable byte-sequence representation, then run
   the same ordinary Python shape at the official maximum constraints.

Passing samples alone must not be described as full ABC199 C support while the
fixed string-capacity ceiling remains.

## Priority B — explicit `int` / `str` conversion

Implement these as normal expressions, not only as I/O special cases:

```python
s = str(n)
n = int(s)
```

Required early semantics:

- `str(int64)` including `0`, negative values, `INT64_MAX`, and `INT64_MIN`;
- `str(str_value)` identity;
- `int(str_value)` for ordinary signed ASCII decimal strings, including optional
  leading `+` / `-`;
- `int(int_value)` identity under the project's fixed signed-int64 ABI;
- conversions compose inside larger expressions and assignments;
- CPython differential tests cover round trips and boundary values.

The project intentionally uses fixed signed int64 rather than Python arbitrary
precision for contest arithmetic. Conversion behavior must therefore be stated
and tested against that ABI rather than pretending arbitrary-precision Python
semantics are already implemented.

Invalid-text `ValueError` behavior is part of the later runtime-error phase. In
the correctness-first contest slice, valid decimal input is the immediate
acceptance target; invalid syntax must never be documented as supported until a
runtime error state exists.

### Real targets after sort support

Two ABC tasks make the conversion priority concrete:

- **ABC192 C — Kaprekar Number**: repeatedly converts an integer to decimal
  digits, reorders those digits, converts the resulting strings back to
  integers, and repeats up to `K <= 10^5` times.
- **ABC221 C — Select Mul**: rearranges the decimal digits of `N <= 10^9`, splits
  them into two positive integers, and maximizes their product.

These become end-to-end compatibility targets once stable character sorting /
reordered traversal is available. Until then, conversion-specific fixtures
should isolate `str(int64)`, `int(str)`, and round trips without inventing a
problem-specific lowering.

## Next implementation phase — scalable byte sequence

The next architecture step is not "make `StringRef` larger". Capacity
scalability and random-index runtime performance are separate engineering
problems, and both must pass before maximum-constraint ABC199 C support can be
claimed.

### Stage S1 — runtime-sized flat byte-sequence primitive

Build a standalone Brainfuck runtime primitive whose generated source size does
not depend on the runtime line length.

Initial scope:

- storage begins strictly after `LayoutPlan.runtime_base()`;
- one runtime-sized flat byte sequence is supported by the first specialization;
- input appends bytes until physical line end without a compile-time 255-byte
  ceiling;
- runtime length is tracked explicitly;
- sequential iteration and output are source-compact and preserve the sequence;
- index load/store and negative-index normalization are correct even if the
  first implementation uses linear head movement;
- no Python runtime or nonstandard BF instruction is introduced.

Planned implementation units:

```text
pybf/bfdynbytes.py             # runtime byte-sequence record/layout primitives
pybf/compiler_dynchars.py      # safe source-shape specialization/router
 tests/test_bfdynbytes.py       # primitive correctness + source-independence
 tests/test_dynamic_charlist.py # ordinary-Python integration
```

Exact filenames may change if an existing runtime-sequence module is a cleaner
home, but the primitive/compiler/test separation should remain.

S1 acceptance gates:

- lengths 0, 1, 32, 255, 256, 1024 and a multi-thousand-byte line;
- generated BF source size independent of those runtime lengths;
- exact input/output preservation;
- positive/negative index load and replacement;
- out-of-range access follows the project's temporary empty-load/no-op-store
  contract until runtime exceptions exist;
- standard BF eight-command output only;
- no overlap with compile-time temporary high-water storage.

### Stage S2 — restricted dynamic character-list frontend

Integrate S1 with ordinary Python only when escape/lifetime analysis proves the
specialized backing is safe.

Target shape begins with:

```python
s = list(input())
# len / iteration / runtime index load/store / one-char swap
print("".join(s))
```

Rules:

- retain the existing fixed `StringRef` path as a fallback for unsupported
  programs;
- initially allow only one escaping runtime-sized character sequence per safe
  region/program, rather than pretending a general heap already exists;
- reject aliasing, rebinding or multiple dynamic objects when object ownership
  is not yet representable;
- keep `list(input())` / empty join representation-preserving;
- reuse the PR #8 byte-element, liveness and semantic-boundary checks.

The official ABC199 C source used by current sample tests becomes the first
integration fixture, but the lowering must be phrased in terms of the general
restricted character-list operations, not task-specific query semantics.

### Stage S3 — indexing complexity and Tritium scaling

A runtime-sized sequence that is merely correct is not automatically practical.
Standard Brainfuck pays for tape-head movement, so random accesses to distant
positions need explicit measurement.

Measure separately:

1. sequential read / iterate / print cost as sequence length grows;
2. one indexed load/store at increasing distances;
3. repeated swaps under local, alternating-end and random index distributions;
4. ABC199 C sample, medium-scale synthetic cases, then official maximum-scale
   dimensions only if earlier curves are credible.

Prefer cursor-relative movement so consecutive nearby indices cost their delta
rather than repeatedly returning to origin. Record both raw BF steps and real
Tritium wall-clock time.

A planned benchmark entry point is:

```text
tools/bench_tritium_abc199.py
```

Maximum-constraint support is accepted only if:

- correctness is independently checked;
- generated source remains within the submission limit;
- the same ordinary Python source is compiled, with no hand-written BF or
  task-specific answer routine;
- Tritium execution is practically within the intended contest environment.

If linear-distance random access is too slow, stop the maximum-scale claim at
that gate. The next research task is then a stronger indexed representation or a
semantics-preserving general batched/cursor optimization. Do not hide the
problem with an ABC199-specific compiler shortcut.

### Stage S4 — generalize only after the primitive is measured

Once dynamic bytes are correct and their cost model is understood:

1. support more than one runtime-sized byte object through the object/handle
   model from `IMPLEMENTATION_PLAN.md`;
2. reuse the proven allocation/traversal machinery for scalable `list[int]`;
3. add alias/reference semantics instead of copying values accidentally;
4. only then move to sort, nested containers and queue/deque workloads.

This order prevents the full heap model from being built around an unmeasured
random-access primitive.

## Priority C — real ABC compatibility corpus

Continue selecting actual ABC tasks before extending the backend so feature
work is driven by real source shapes rather than synthetic APIs. Near-term tiers:

1. streaming integer input/list folds — ABC153 B and ABC103 C;
2. character-array editing — ABC199 C;
3. indexed/mutating integer passes — ABC136 C class;
4. string/int conversion plus digit reorder — ABC192 C / ABC221 C after sort;
5. sort/reordered list traversal — ABC088 B class;
6. queue/deque workloads — ABC247 D class.

For each tier, retain:

- ordinary Python source as the fixture;
- official samples where applicable;
- at least one maximum-constraint or scaling benchmark;
- standard-BF-only output;
- source-size and runtime measurements as separate gates.

## Priority D — float is deliberately deferred

`float` / float64 remains desirable but is **not a near-term blocker**.

Implement it only after the int/string/list path is stable enough that float work
will not displace higher-value contest compatibility. When undertaken, prefer a
software IEEE-754 binary64 runtime with:

- decimal parse / print;
- `int -> float` and `float -> int` conversions;
- literals;
- `+ - * /`;
- comparisons;
- explicit tests for NaN/Inf/rounding policy before claiming Python-compatible
  behavior.

Until then, unsupported float syntax should fail clearly rather than being
approximated with integer arithmetic.

## Immediate implementation order

1. Land PR #7 (single-use int input-list streaming foundation) after final
   review; its current head already passes the normal four-shard CI.
2. Land PR #8 (restricted character-list views + explicit `int`/`str`
   conversions) after the stacked base is handled; keep its fixed-capacity scope
   explicit.
3. Implement Stage S1 runtime-sized flat byte sequence.
4. Integrate Stage S2 restricted dynamic character-list lowering.
5. Run Stage S3 scaling/Tritium benchmarks and decide the ABC199 C maximum-scale
   claim from measurements, not from capacity alone.
6. Generalize through Stage S4 into scalable indexed/mutable integer lists.
7. Proceed to stable sort; then promote ABC192 C / ABC221 C into the end-to-end
   corpus.
8. Proceed to queue/deque semantics.
9. Revisit float64 only after those higher-priority paths are stable.
