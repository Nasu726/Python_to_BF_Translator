"""Final streaming compiler router with narrow scalable whole-program lowering.

The established generic implementation lives in ``compiler_stream_generic``.
``compiler_stream_intfusion`` adds dead-list producer/consumer fusion,
``compiler_charconv`` adds zero-copy character-list views plus explicit
int/string conversions, and ``compiler_decimalconv`` makes dynamic decimal
string parsing source-compact by emitting one runtime digit-update loop.  This
module keeps the separately proven whole-program specializations in front of
those generic layers.
"""

from __future__ import annotations

import ast

from bfopt import optimize_bf
from compiler_decimalconv import CompileError
from compiler_decimalconv import PythonToBFStream as _GenericPythonToBFStream
from compiler_partition import lower_partition_program_if_supported


class PythonToBFStream(_GenericPythonToBFStream):
    """Generic stream compiler plus narrowly proven scalable specializations."""

    def compile_module(self, tree: ast.AST) -> str:
        if not isinstance(tree, ast.Module):
            raise CompileError("expected ast.Module")

        specialized = lower_partition_program_if_supported(tree)
        if specialized is not None:
            # Preserve the diagnostics contract used by compile-performance
            # tests even though the complete program is emitted as one runtime
            # kernel instead of statement-by-statement generic lowering.
            self.statement_sizes = [
                (1, "ScalablePartitionProgram", len(specialized))
            ]
            self.detail_sizes = [
                (1, "ScalablePartitionProgram", len(specialized))
            ]
            return specialized

        return super().compile_module(tree)


def compile_source(
    source: str,
    filename: str = "<string>",
    *,
    string_capacity: int = 255,
    list_capacity: int = 64,
) -> str:
    tree = ast.parse(source, filename=filename)
    code = PythonToBFStream(
        tree,
        string_capacity=string_capacity,
        list_capacity=list_capacity,
    ).compile_module(tree)
    return optimize_bf(code)


__all__ = ["CompileError", "PythonToBFStream", "compile_source"]
