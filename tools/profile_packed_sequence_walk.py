"""Reproduce runtime-repeat and physical-walk source/step measurements.

Run: PYTHONPATH=pybf python tools/profile_packed_sequence_walk.py
These are primitive measurements, not public Python-list acceptance results.
"""

from bfcore import BFEmitter
from bfpacked import PackedU32Ref
from bfpackedseq import RECORD_STRIDE, RuntimePackedIntSequence
from bf_runtime import run_bf


def main():
    bf = BFEmitter()
    for byte in range(4):
        bf.move(byte)
        bf.emit(",")
    seq = RuntimePackedIntSequence(64)
    seq.repeat_constant(bf, PackedU32Ref(0), 0)
    creation = bf.code()
    seq.walk_records(bf, lambda record: record.write_value())
    seq.walk_records(bf, lambda record: record.write_value(), reverse=True)
    walked = bf.code()
    print(f"repeat_source_bytes={len(creation)}")
    print(f"repeat_and_two_passes_source_bytes={len(walked)}")
    print(f"two_passes_extra_source_bytes={len(walked) - len(creation)}")
    for count in (0, 64, 128, 256, 1024):
        data = "".join(chr((count >> (8 * i)) & 255) for i in range(4))
        kwargs = dict(memory_size=64 + (count + 2) * RECORD_STRIDE,
                      step_limit=100_000_000)
        baseline = run_bf(creation, data, **kwargs)
        result = run_bf(walked, data, **kwargs)
        assert result.memory == baseline.memory
        assert result.pointer == baseline.pointer == seq.base
        assert result.input_consumed == baseline.input_consumed == 4
        assert result.output == "\0" * (16 * count)
        traversal = result.steps - baseline.steps
        assert traversal == 98 * count + 8
        print(f"n={count} creation_steps={baseline.steps} "
              f"total_steps={result.steps} traversal_steps={traversal}")


if __name__ == "__main__":
    main()
