from bfcore import BFEmitter, Int64Ref
from bf_runtime import run_bf
from bftokens import BinaryTokenIO


def signed(mem, base):
    raw = sum((mem[base + i] & 1) << i for i in range(64))
    return raw - (1 << 64) if raw & (1 << 63) else raw


def test_repeated_signed_tokens_and_whitespace():
    bf = BFEmitter()
    backend = BinaryTokenIO(bf, scratch_base=400)
    a, b, c = Int64Ref(0), Int64Ref(64), Int64Ref(128)
    backend.read_s64_token(a, 200)
    backend.read_s64_token(b, 200)
    backend.read_s64_token(c, 200)
    result = run_bf(bf.code(), '  -12\t34   5\n', memory_size=1024)
    assert signed(result.memory, a.base) == -12
    assert signed(result.memory, b.base) == 34
    assert signed(result.memory, c.base) == 5
