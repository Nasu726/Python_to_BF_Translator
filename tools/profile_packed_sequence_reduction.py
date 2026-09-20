"""Measure mobile sum/length separately from runtime sequence construction.

PYTHONPATH=pybf python tools/profile_packed_sequence_reduction.py
PYTHONPATH=pybf python tools/profile_packed_sequence_reduction.py --value=-1 --counts 1 8 16

This is a runtime primitive, not public dynamic-list / AtCoder acceptance.
"""

import argparse

from bfcore import BFEmitter
from bfpacked import PackedU32Ref
from bfpacked64 import PackedI64Ref
from bfpackedseq import RECORD_STRIDE, REDUCTION_WORKSPACE_CELLS, RuntimePackedIntSequence
from bf_runtime import run_bf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value", type=int, default=0)
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 256, 512, 1024, 2048])
    args = parser.parse_args()
    if any(n < 0 or n > 100_000 for n in args.counts):
        parser.error("counts must be between 0 and 100000")
    bf = BFEmitter()
    for byte in range(4):
        bf.move(byte)
        bf.emit(",")
    seq = RuntimePackedIntSequence(64)
    seq.repeat_constant(bf, PackedU32Ref(0), args.value)
    creation = bf.code()
    seq.sum_and_length(bf, PackedI64Ref(8), PackedU32Ref(16))
    reduced = bf.code()
    print(f"reduction_extra_source_bytes={len(reduced) - len(creation)}")
    print(f"reduction_workspace_cells={REDUCTION_WORKSPACE_CELLS}")
    print(f"combined_source_bytes={len(reduced)}")
    for count in args.counts:
        data = count.to_bytes(4, "little").decode("latin1")
        kwargs = dict(memory_size=64 + (count + 2) * RECORD_STRIDE,
                      step_limit=500_000_000)
        before = run_bf(creation, data, **kwargs)
        result = run_bf(reduced, data, **kwargs)
        expected = before.memory[:]
        expected[8:16] = ((count * args.value) % (1 << 64)).to_bytes(8, "little")
        expected[16:20] = count.to_bytes(4, "little")
        assert result.memory == expected
        assert result.pointer == before.pointer == seq.base
        assert result.input_consumed == before.input_consumed == 4
        assert result.output == before.output == ""
        print(f"n={count} value={args.value} creation_steps={before.steps} "
              f"reduction_steps={result.steps - before.steps}")


if __name__ == "__main__":
    main()
