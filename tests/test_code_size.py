from bf_runtime import run_bf
from pybf import compile_source


BF_COMMANDS = set("><+-.,[]")
ATCODER_SOURCE_LIMIT = 512 * 1024


USER_STRING_LOOP = '''
s = input()
for c in s:
    if c == "A":
        print("A", end="")
    else:
        print(".", end="")
'''


def test_user_string_loop_fits_512k_submission_limit():
    code = compile_source(USER_STRING_LOOP)
    assert set(code) <= BF_COMMANDS
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"generated BF is {len(code):,} bytes; limit is {ATCODER_SOURCE_LIMIT:,}"
    )


def test_size_optimized_user_string_loop_still_executes():
    code = compile_source(USER_STRING_LOOP)
    result = run_bf(code, "ABCA\n", step_limit=300_000_000)
    assert result.output == "A..A"


def test_stream_fusion_break_still_consumes_original_input_line():
    source = '''
s = input()
for c in s:
    if c == "B":
        break
x = input()
print(c, x)
'''
    code = compile_source(source)
    result = run_bf(code, "ABCD\nNEXT\n", step_limit=300_000_000)
    assert result.output == "B NEXT\n"


def test_stream_fusion_preserves_continue_and_for_else():
    source = '''
s = input()
for c in s:
    if c == "B":
        continue
    print(c, end="")
else:
    print("!", end="")
'''
    code = compile_source(source)
    result = run_bf(code, "ABC\n", step_limit=300_000_000)
    assert result.output == "AC!"



def test_p0_runtime_intlist_repr_has_attributed_source_budget():
    # Keep the complete ordinary-Python P0 prefix below AtCoder's source limit,
    # including the new mobile repr printer, rather than testing it only via
    # manual Tritium scripts. This also verifies that the independent
    # layout-attribution diagnostic stays aligned with the public compiler.
    from tools.bench_tritium_intlist_repr import SOURCE
    from tools.profile_p0_intlist_source import profile_p0_source

    profile = profile_p0_source(SOURCE)
    public = compile_source(SOURCE)
    assert set(public) <= BF_COMMANDS
    assert len(public) == profile.final_bytes
    assert 0 < profile.final_bytes <= ATCODER_SOURCE_LIMIT
    assert profile.headroom == ATCODER_SOURCE_LIMIT - len(public)
    assert profile.raw_bytes > 0
    assert profile.runtime_base > profile.temp_peak
    assert profile.statement_sizes
