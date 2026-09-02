"""Current recommended Python -> Brainfuck compiler entrypoint.

This layers common competitive-programming syntax on the typed v3 frontend.
Older versioned frontends remain available as regression/reference layers.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from bftokens import BinaryTokenIO
from transpiler_v2 import CompileError, clean_bf
from transpiler_v3 import PythonToBFV3


def _is_map_int_input_split(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "map"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "int"
    ):
        return False
    split = node.args[1]
    return (
        isinstance(split, ast.Call)
        and isinstance(split.func, ast.Attribute)
        and split.func.attr == "split"
        and not split.args
        and not split.keywords
        and isinstance(split.func.value, ast.Call)
        and isinstance(split.func.value.func, ast.Name)
        and split.func.value.func.id == "input"
        and not split.func.value.args
        and not split.func.value.keywords
    )


class PythonToBF(PythonToBFV3):
    def __init__(self, tree: ast.AST, *, string_capacity: int = 128) -> None:
        super().__init__(tree, string_capacity=string_capacity)
        # No code is emitted during frontend construction, so swapping the
        # richer subclass in here preserves all v3 addresses and state.
        self.backend = BinaryTokenIO(self.bf, scratch_base=self.scratch_base)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and _is_map_int_input_split(node.value)
        ):
            targets = node.targets[0].elts
            if not targets or not all(isinstance(t, ast.Name) for t in targets):
                raise self._error(node, "map(int, input().split()) unpacking requires simple names")
            for target in targets:
                assert isinstance(target, ast.Name)
                if target.id in self.strings:
                    raise self._error(target, "integer token cannot be assigned to a string variable")
                self.backend.read_s64_token(self._var(target), self.workspace_base)
            return
        return super()._compile_stmt_inner(node)


def compile_source(source: str, filename: str = "<string>", *, string_capacity: int = 128) -> str:
    tree = ast.parse(source, filename=filename)
    return PythonToBF(tree, string_capacity=string_capacity).compile_module(tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the supported Python subset to Brainfuck")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--string-capacity", type=int, default=128)
    args = parser.parse_args()
    source = args.input.read_text(encoding="utf-8")
    try:
        code = compile_source(source, str(args.input), string_capacity=args.string_capacity)
    except (CompileError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output or args.input.with_suffix(".bf")
    output.write_text(clean_bf(code), encoding="utf-8")
    print(f"wrote {output} ({len(code):,} BF commands)")


if __name__ == "__main__":
    main()
