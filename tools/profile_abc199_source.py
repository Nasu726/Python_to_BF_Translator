"""Profile generated-source attribution for the ordinary ABC199 C shape.

This is a compiler diagnostics tool, not a problem-specific lowering.  It uses
exactly the public-layout compiler stack and reports which ordinary Python
statements/nested statements account for emitted Brainfuck source.  Keep this
around while S2 reduces general dynamic-character-list/query overhead.
"""

from __future__ import annotations

import ast
from textwrap import dedent

from bfopt import optimize_bf
from bfstreamseq import RECORD_STRIDE
from compiler_dynamic_charlist import select_dynamic_char_list
from compiler_layout import PythonToBFLayout, compile_source


ABC199_SOURCE = '''
n = int(input())
s = list(input())
q = int(input())
flipped = 0
for k in range(q):
    t, a, b = map(int, input().split())
    if t == 1:
        a -= 1
        b -= 1
        if flipped:
            if a < n:
                a += n
            else:
                a -= n
            if b < n:
                b += n
            else:
                b -= n
        tmp = s[a]
        s[a] = s[b]
        s[b] = tmp
    else:
        flipped = 1 - flipped
if flipped:
    for i in range(n):
        tmp = s[i]
        s[i] = s[i + n]
        s[i + n] = tmp
print("".join(s))
'''


MICRO_CASES = {
    "base": '''
        s = list(input())
        print("".join(s))
    ''',
    "one_load": '''
        s = list(input())
        i = int(input())
        tmp = s[i]
        print(tmp)
        print("".join(s))
    ''',
    "subscript_copy": '''
        s = list(input())
        i = int(input())
        j = int(input())
        s[i] = s[j]
        print("".join(s))
    ''',
    "three_statement_swap": '''
        s = list(input())
        i = int(input())
        j = int(input())
        tmp = s[i]
        s[i] = s[j]
        s[j] = tmp
        print("".join(s))
    ''',
}


def lower_profile(source: str):
    tree = ast.parse(source)
    if select_dynamic_char_list(tree) is None:
        raise SystemExit("ABC199 profile source no longer selects dynamic char-list lowering")

    # Mirror compiler_layout.lower_with_layout while retaining the final
    # compiler object so its built-in source attribution can be inspected.
    probe = PythonToBFLayout(tree, string_capacity=255, list_capacity=64)
    probe.compile_module(tree)
    runtime_base = probe.layout_plan.runtime_base(guard_cells=RECORD_STRIDE)

    compiler = None
    raw = ""
    for _attempt in range(3):
        compiler = PythonToBFLayout(
            tree,
            string_capacity=255,
            list_capacity=64,
            runtime_charlist_base=runtime_base,
        )
        raw = compiler.compile_module(tree)
        plan = compiler.layout_plan
        exact_base = plan.runtime_base(guard_cells=RECORD_STRIDE)
        if exact_base != runtime_base:
            runtime_base = exact_base
            continue
        if plan.temp_peak <= runtime_base - RECORD_STRIDE:
            break
        runtime_base = exact_base
    else:
        raise SystemExit("runtime character-list layout did not converge")

    assert compiler is not None
    return raw, optimize_bf(raw), compiler


def _show(title: str, rows: list[tuple[int, str, int]], *, limit: int | None = None):
    ordered = sorted(rows, key=lambda row: row[2], reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    print(title)
    for line, kind, size in ordered:
        print(f"  L{line:02d} {kind:<12} {size:>10,d} B")


def _show_micro_cases() -> None:
    print("dynamic-char micro cases:")
    previous = None
    for name, source in MICRO_CASES.items():
        code = compile_source(dedent(source))
        size = len(code.encode("ascii"))
        delta = ""
        if previous is not None:
            delta = f" delta_from_previous={size - previous:+,}"
        print(f"  {name:<22} {size:>10,d} B{delta}")
        previous = size


def main() -> None:
    raw, final, compiler = lower_profile(ABC199_SOURCE)
    limit = 512 * 1024
    print(f"abc199_raw_bytes={len(raw.encode('ascii')):,}")
    print(f"abc199_final_bytes={len(final.encode('ascii')):,}")
    print(f"abc199_headroom_to_512k={limit - len(final.encode('ascii')):,}")
    print(f"dynamic_charlist_base={compiler.runtime_charlist_base}")
    print(f"temp_peak={compiler.layout_plan.temp_peak}")
    _show("top-level attribution:", compiler.statement_sizes)
    _show("largest nested attribution:", compiler.detail_sizes, limit=30)
    _show_micro_cases()


if __name__ == "__main__":
    main()
