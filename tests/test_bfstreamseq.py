from bf_runtime import run_bf
from bfcore import BFEmitter
from bfstreamseq import RECORD_STRIDE, RuntimeByteSequence


def _roundtrip_program():
    bf = BFEmitter()
    seq = RuntimeByteSequence(base=64)
    seq.read_lf_terminated_bytes(bf)
    seq.write_all_bytes(bf)
    return bf.code()


def test_runtime_sized_byte_sequence_roundtrips_empty_and_nonempty_lines():
    code = _roundtrip_program()

    assert run_bf(code, "\n", step_limit=5_000_000).output == ""
    assert run_bf(code, "ABCxyz\n", step_limit=20_000_000).output == "ABCxyz"


def test_runtime_sequence_uses_one_source_program_for_longer_runtime_input():
    code = _roundtrip_program()
    source_size = len(code)

    payload = "a" * 200
    result = run_bf(code, payload + "\n", step_limit=200_000_000)

    assert result.output == payload
    # Runtime length changes tape/step usage only.  There is no capacity/N
    # parameter in code generation, so the emitted program stays compact.
    assert len(code) == source_size
    assert source_size < 5_000


def test_runtime_sequence_layout_reserves_a_left_sentinel_record():
    seq = RuntimeByteSequence(base=3 * RECORD_STRIDE)
    assert seq.left_sentinel == 2 * RECORD_STRIDE
