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

## Priority C — real ABC compatibility corpus

Continue selecting actual ABC tasks before extending the backend so feature
work is driven by real source shapes rather than synthetic APIs. Near-term tiers:

1. streaming integer input/list folds — already represented by ABC153 B and
   ABC103 C;
2. character-array editing with `list(input())` / `"".join(...)`;
3. indexed/mutating integer passes (ABC136 C class);
4. sort/reordered traversal (ABC088 B class);
5. queue/deque workloads (ABC247 D class).

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

1. Finish zero-copy `list(input())` + `"".join(character_list)` syntax and
   mutable character indexing.
2. Finish `str(int64)` and `int(str)` with boundary/differential tests.
3. Locate real ABC problems using character-array mutation and promote selected
   tasks into the compatibility corpus.
4. Proceed to scalable indexed/mutable integer lists.
5. Proceed to stable sort and queue/deque semantics.
6. Revisit float64 only after those higher-priority paths are stable.
