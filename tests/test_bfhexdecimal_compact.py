from bf_runtime import run_bf
from bfcore import BFEmitter
from bfhexdecimal_compact import decimal_digit_kernel, negate_data_kernel
from bfhexseq import CH, DATA


MASK64 = (1 << 64) - 1
COUNT = 400


def _decode_u64(memory, base=DATA):
    value = 0
    for i in range(16):
        nibble = memory[base + i]
        assert 0 <= nibble <= 15
        value |= nibble << (4 * i)
    return value


def _program(digit_count, *, negate=False):
    bf = BFEmitter()
    bf.set_const(COUNT, digit_count)
    bf.move(COUNT)
    bf.emit("[")
    bf.add_const(COUNT, -1)
    bf.move(CH)
    bf.emit(",")
    bf.add_const(CH, -ord("0"))
    bf.move(0)
    bf.emit(decimal_digit_kernel())
    bf.ptr = 0
    bf.move(COUNT)
    bf.emit("]")
    if negate:
        bf.move(0)
        bf.emit(negate_data_kernel())
        bf.ptr = 0
    bf.move(0)
    return bf.code()


def test_compact_direct_hex_decimal_matches_u64_modulo():
    texts = [
        "0",
        "1",
        "10",
        "15",
        "16",
        "255",
        "256",
        "123456789",
        str((1 << 63) - 1),
        str((1 << 64) - 1),
        str(1 << 64),
        "99999999999999999999",
    ]
    for text in texts:
        result = run_bf(
            _program(len(text)),
            text,
            memory_size=2_000,
            step_limit=1_000_000_000,
        )
        assert _decode_u64(result.memory) == int(text) & MASK64


def test_compact_direct_hex_decimal_negation_handles_signed_boundary():
    texts = ["1", "123456789", str(1 << 63), str((1 << 64) - 1)]
    for text in texts:
        result = run_bf(
            _program(len(text), negate=True),
            text,
            memory_size=2_000,
            step_limit=1_000_000_000,
        )
        assert _decode_u64(result.memory) == (-int(text)) & MASK64
