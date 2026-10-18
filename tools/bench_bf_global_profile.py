"""Benchmark global BF profiling at large source sizes.

Run:
    PYTHONPATH=pybf python tools/bench_bf_global_profile.py
Synthetic workloads are balanced BF text with many repeated loop fragments.
The benchmark measures profiler complexity; it makes no execution-equivalence
or compression-ratio claim. ru_maxrss is process high-water RSS (not a per-run
increment).
"""

from __future__ import annotations

import argparse
import json
import resource
import time

from bfglobalprofile import profile_bf


MOTIF = "[->+<]>>>++<<<[-]"


def make_source(byte_count: int) -> str:
    if byte_count < 1:
        raise ValueError("byte_count must be positive")
    repetitions, remainder = divmod(byte_count, len(MOTIF))
    return MOTIF * repetitions + "+" * remainder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[64 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024],
    )
    args = parser.parse_args()
    print("bytes,profile_seconds,peak_rss_kb,loops,distinct_loop_fingerprints")
    for size in args.sizes:
        source = make_source(size)
        start = time.perf_counter()
        result = profile_bf(source)
        duration = time.perf_counter() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            f"{size},{duration:.6f},{peak},{result['loops']['instances']},"
            f"{result['loops']['distinct_fingerprints']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
