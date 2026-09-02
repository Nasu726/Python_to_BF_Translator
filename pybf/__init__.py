"""Public API for the Python -> Brainfuck compiler.

Everything under ``pybf`` is implementation detail except ``compile_source``
and the ABI constants exported here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Internal modules intentionally use sibling imports.  Add the package
# directory once so the user-facing root stays clean and only ``main.py`` is an
# executable entrypoint.
_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

from abi import INT_BITS, LIST_CAPACITY, STRING_CAPACITY  # noqa: E402
from compiler_strings import compile_source as _compile_source  # noqa: E402
from transpiler_v2 import CompileError  # noqa: E402


def compile_source(source: str, filename: str = "<string>") -> str:
    """Compile source using the fixed runtime ABI."""
    return _compile_source(
        source,
        filename,
        string_capacity=STRING_CAPACITY,
        list_capacity=LIST_CAPACITY,
    )


__all__ = [
    "CompileError",
    "INT_BITS",
    "STRING_CAPACITY",
    "LIST_CAPACITY",
    "compile_source",
]
