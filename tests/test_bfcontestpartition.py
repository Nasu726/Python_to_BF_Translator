from bf_runtime import run_bf
from bfcontestpartition import build_partition_program, partition_program_size_breakdown
from bfcore import BFEmitter
from bfhexio import print_record_hex_s64_compact
from bfhexseq import ANS, RuntimeHexIntSequence


ATCODER_SOURCE_LIMIT = 512 * 1024
MASK64 = (1 << 64) - 1


def _reference(values):
    total = sum(values)
    left = 0
    ans = 10_000_000
    for value in values:
        left += value
        ans = min(ans, abs(total - 2 * left))
    return ans


def _compact_print_program(value):
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=320)
    bits = value & MASK64
    for i in range(16):
        bf.set_const(seq.field(0, ANS) + i, (bits >> (4 * i)) & 0xF)
    print_record_hex_s64_compact(bf, seq, field_base=ANS)
    return bf.code()


def test_compact_hex_decimal_printer_signed_int64_boundaries():
    values = [
        0,
        1,
        10,
        9_999_999,
        -1,
        -123_456_789,
        -(1 << 63),
        (1 << 63) - 1,
    ]
    for value in values:
        result = run_bf(
            _compact_print_program(value),
            "",
            memory_size=10_000,
            step_limit=1_000_000_000,
        )
        assert result.output == f"{value}\n"


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


def test_scalable_partition_execution_steps_grow_linearly_with_runtime_n():
    code = build_partition_program()
    measured = []
    for n in (8, 16, 32):
        values = [1] * n
        result = run_bf(
            code,
            f"{n}\n" + " ".join(map(str, values)) + "\n",
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert result.output == f"{_reference(values)}\n"
        measured.append(result.steps)

    first_increment = measured[1] - measured[0]
    second_increment = measured[2] - measured[1]
    assert first_increment > 0
    assert second_increment > 0
    # Doubling the number of added records doubles the added work.  Leave a
    # small constant-factor margin for decimal-output and input-length effects,
    # while still detecting accidental quadratic record scans.
    assert second_increment < first_increment * 2.25, (
        f"unexpected runtime-step growth: steps={measured}, "
        f"increments={[first_increment, second_increment]}"
    )


def test_scalable_partition_vertical_slice_fits_submission_limit():
    code = build_partition_program()
    sizes = partition_program_size_breakdown()
    checkpoints = list(sizes.items())
    deltas = {}
    previous = 0
    for name, cumulative in checkpoints:
        if name == "optimized":
            continue
        deltas[name] = cumulative - previous
        previous = cumulative
    diagnostics = f"cumulative={sizes} deltas={deltas}"

    assert set(code) <= set("><+-.,[]")
    assert len(code.encode("ascii")) <= ATCODER_SOURCE_LIMIT, (
        f"scalable partition BF is {len(code):,} bytes; "
        f"limit is {ATCODER_SOURCE_LIMIT:,}; {diagnostics}"
    )


def test_scalable_partition_source_does_not_depend_on_runtime_n():
    # N and the values exist only in BF input.  Building the program has no
    # item-count parameter, which is the source-shape property needed for
    # N=200000-class inputs.
    first = build_partition_program()
    second = build_partition_program()
    assert first == second
