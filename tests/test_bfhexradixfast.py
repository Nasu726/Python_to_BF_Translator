from bf_runtime import run_bf
from bfhexradixfast import map_total_base16_threshold
from bfhexseq import _RelativeBuilder


def _program(value):
    total = 2
    out = 3
    carry = 4
    r = _RelativeBuilder()
    r.set_const(total, value)
    map_total_base16_threshold(r, total, out, carry)
    r.move(0)
    return r.code(), total, out, carry


def test_threshold_radix_mapper_exhaustive_0_to_31():
    for value in range(32):
        code, total, out, carry = _program(value)
        result = run_bf(code, "", memory_size=64, step_limit=1_000_000)
        assert result.memory[total] == 0
        assert result.memory[out] == value % 16
        assert result.memory[carry] == value // 16
        assert result.pointer == 0
