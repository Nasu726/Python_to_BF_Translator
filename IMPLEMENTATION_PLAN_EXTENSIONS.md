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

1. Finish zero-copy `list(input())` + `"".join(character_list)` syntax, mutable
   character indexing, swaps, and ABC199 C sample compatibility.
2. Finish `str(int64)` and `int(str)` with boundary/differential tests.
3. Replace fixed-capacity string/character backing with a scalable byte sequence
   and revisit ABC199 C at maximum constraints.
4. Proceed to scalable indexed/mutable integer lists.
5. Proceed to stable sort; then promote ABC192 C / ABC221 C into the end-to-end
   corpus.
6. Proceed to queue/deque semantics.
7. Revisit float64 only after those higher-priority paths are stable.
