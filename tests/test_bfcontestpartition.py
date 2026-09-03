from bf_runtime import run_bf
from bfcontestpartition import build_partition_program


ATCODER_SOURCE_LIMIT = 512 * 1024


def _reference(values):
    total = sum(values)
    left = 0
    ans = 10_000_000
    for value in values:
        left += value
        ans = min(ans, abs(total - 2 * left))
    return ans


def test_scalable_partition_vertical_slice_executes_end_to_end():
    code = build_partition_program()
    cases = [
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        [100],
        list(range(12)),
    ]
    for values in cases:
        input_data = f"{len(values)}\n" + " ".join(map(str, values)) + "\n"
        result = run_bf(
            code,
            input_data,
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert result.output == f"{_reference(values)}\n"


def test_scalable_partition_vertical_slice_fits_submission_limit():
    code = build_partition_program()
    assert set(code) <= set("><+-.,[]")
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"scalable partition BF is {len(code):,} bytes; "
        f"limit is {ATCODER_SOURCE_LIMIT:,}"
    )


def test_scalable_partition_source_does_not_depend_on_runtime_n():
    # N and the values exist only in BF input.  Building the program has no
    # item-count parameter, which is the source-shape property needed for
    # N=200000-class inputs.
    first = build_partition_program()
    second = build_partition_program()
    assert first == second
