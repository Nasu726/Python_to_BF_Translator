"""Command-line source profiler for global Brainfuck optimization research.

Run with:
    PYTHONPATH=pybf python tools/profile_bf_global.py --bf Main.bf
    PYTHONPATH=pybf python tools/profile_bf_global.py --abc199 --json
    PYTHONPATH=pybf python tools/profile_bf_global.py --python Main.py

The Python options compile ordinary source through the public frontend. Reported
duplicate text is not an achievable compression ratio: BF has no macro calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bfglobalprofile import profile_bf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bf", type=Path, help="existing standard BF program")
    source.add_argument("--python", type=Path, help="Python program to compile")
    source.add_argument(
        "--abc199", action="store_true",
        help="compile the ordinary ABC199 C workload in profile_abc199_source",
    )
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if args.bf is not None:
        code = args.bf.read_text(encoding="utf-8")
        label = str(args.bf)
    else:
        from compiler_layout import compile_source

        if args.python is not None:
            source_code = args.python.read_text(encoding="utf-8")
            label = str(args.python)
        else:
            from profile_abc199_source import ABC199_SOURCE

            source_code = ABC199_SOURCE
            label = "ABC199 C public compiler"
        code = compile_source(source_code)

    report = profile_bf(code, top=args.top)
    if args.json:
        print(json.dumps({"workload": label, **report}, indent=2))
        return

    print(f"workload={label}")
    print(f"source_bytes={report['source_bytes']:,}")
    print(f"ignored_chars={report['ignored_non_command_chars']:,}")
    for command, count in report["commands"].items():
        print(f"command_{command}={count:,}")
    for name, count in report["runs"].items():
        print(f"{name}={count:,}")
    for name, count in report["loops"].items():
        print(f"loops_{name}={count:,}")
    print("top repeated loop-text fingerprints (NOT realizable savings):")
    for item in report["top_loop_motifs"]:
        print(
            f"  length={item['length']:,} instances={item['instances']:,} "
            f"duplicate_text_bytes={item['duplicate_text_bytes']:,} "
            f"first_offset={item['first_offset']:,} "
            f"fingerprint={item['fingerprint']}"
        )


if __name__ == "__main__":
    main()
