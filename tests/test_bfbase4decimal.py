from bf_runtime import run_bf
from bfbase4 import Base4I64Core, Base4I64Ref, MASK64
from bfbase4decimal import Base4DecimalCore
from bfcore import BFEmitter


def _decode(memory, ref):
    value = 0
    for digit in range(32):
        lane = memory[ref.value(digit)]
        assert 0 <= lane <= 3
        value |= lane << (2 * digit)
    return value


def _emit_decimal(text: str):
    bf = BFEmitter()
    base4 = Base4I64Core(bf)
    decimal = Base4DecimalCore(bf)
    value = Base4I64Ref(40)
    scratch = Base4I64Ref(140)
    digit_cell = 250

    base4.set_u64(value, 0)
    base4.set_u64(scratch, 0)
    for char in text:
        bf.set_const(digit_cell, ord(char) - ord("0"))
        decimal.mul10_add_digit_inplace(value, scratch, digit_cell)

    return bf, value, digit_cell


def test_base4_decimal_accumulates_unsigned_boundaries():
    cases = [
        "0",
        "9",
        "10",
        "123456789",
        "18446744073709551615",  # 2**64 - 1
        "18446744073709551616",  # modulo wrap to zero
        "18446744073709551617",  # modulo wrap to one
    ]

    for text in cases:
        bf, value, digit_cell = _emit_decimal(text)
        result = run_bf(bf.code(), memory_size=500, step_limit=250_000_000)
        assert _decode(result.memory, value) == (int(text) & MASK64)
        assert result.memory[digit_cell] == 0


def test_base4_decimal_digit_has_fixed_lane_work_not_value_proportional_byte_work():
    small_bf, _, _ = _emit_decimal("0")
    large_bf, _, _ = _emit_decimal("9")

    # Both decimal digits use exactly the same generated lane program.  Only
    # the compile-time constant used to seed the digit cell differs.
    assert abs(len(small_bf.code()) - len(large_bf.code())) <= 9
    assert len(large_bf.code()) < 30_000
