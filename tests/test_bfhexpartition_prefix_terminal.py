from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexcounted_prefix import read_counted_two_line_s64s_and_sum
from bfhexpartition_prefix_terminal import run_partition_min_pass_and_print_terminal
from bfhexseq import RuntimeHexIntSequence


MASK64 = (1 << 64) - 1


def _s64(value):
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def _abs_s64(value):
    value = _s64(value)
    return _s64(-value) if value < 0 else value


def _reference(values, initial_ans=10_000_000):
    total = 0
    for value in values:
        total = _s64(total + value)
    left = 0
    ans = _s64(initial_ans)
    for value in values:
        left = _s64(left + value)
        candidate = _abs_s64(_s64(total - _s64(2 * left)))
        if candidate < ans:
            ans = candidate
    return ans


def _program(initial_ans=10_000_000):
    bf = BFEmitter()
    seq = RuntimeHexIntSequence(base=320)
    read_counted_two_line_s64s_and_sum(bf, seq)
    seq.propagate_total_back_to_first(bf)
    run_partition_min_pass_and_print_terminal(bf, seq, initial_ans=initial_ans)
    return bf.code()


def test_terminal_sentinel_output_matches_partition_reference():
    code = _program()
    cases = (
        [],
        [100],
        [1, 2, 3, 4],
        [5, -2, 10, 1234, -77],
        list(range(12)),
        [0xFFFF, 1, 0xFFFF, 1],
        [2**62, 2**62],
        [-(1 << 63), 0, 1, -1],
        [(1 << 63) - 1, 1, 0],
    )
    for values in cases:
        result = run_bf(
            code,
            f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert result.output == f"{_reference(values)}\n"


def test_terminal_sentinel_output_handles_zero_answer_and_int64_min_abs():
    for values in ([1] * 32, [-(1 << 63)], [0, 1, (1 << 63) - 1]):
        code = _program(initial_ans=0)
        result = run_bf(
            code,
            f"{len(values)}\n" + " ".join(map(str, values)) + "\n",
            memory_size=30_000,
            step_limit=1_000_000_000,
        )
        assert result.output == f"{_reference(values, 0)}\n"


def test_terminal_sentinel_output_is_standard_brainfuck():
    assert set(_program()) <= set("><+-.,[]")
