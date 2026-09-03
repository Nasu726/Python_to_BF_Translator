"""Structural recognition for the scalable partition contest lowering.

The generic compiler intentionally remains the fallback for all programs that
do not match this complete data-flow shape.  This module recognizes the common
contest pattern::

    n = int(input())
    a = list(map(int, input().split()))
    total = 0
    for i in range(n):
        total += a[i]
    ans = C
    left = 0
    for i in range(n):
        left += a[i]
        ans = min(ans, abs(total - 2 * left))
    print(ans)

Variable names are irrelevant; relationships between them are not.  The
runtime-sized vertical slice currently relies on the standard contest input
contract that the second line contains exactly n integers.  Until the carried
sequence itself consumes an explicit n counter, this recognizer must remain
narrow and its precondition documented.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from bfcontestpartition import build_partition_program


@dataclass(frozen=True)
class PartitionProgramMatch:
    n_name: str
    list_name: str
    total_name: str
    ans_name: str
    left_name: str
    initial_ans: int


def _single_name_assignment(stmt: ast.stmt) -> tuple[str, ast.AST] | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return None
    return stmt.targets[0].id, stmt.value


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _is_int_input(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "input"
        and not node.args[0].args
        and not node.args[0].keywords
    )


def _is_split_input(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "split"
        and not node.args
        and not node.keywords
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "input"
        and not node.func.value.args
        and not node.func.value.keywords
    )


def _is_int_list_input(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and not node.keywords
    ):
        return False
    mapped = node.args[0]
    return (
        isinstance(mapped, ast.Call)
        and isinstance(mapped.func, ast.Name)
        and mapped.func.id == "map"
        and len(mapped.args) == 2
        and not mapped.keywords
        and isinstance(mapped.args[0], ast.Name)
        and mapped.args[0].id == "int"
        and _is_split_input(mapped.args[1])
    )


def _is_range_name(loop: ast.For, n_name: str) -> bool:
    return (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "range"
        and len(loop.iter.args) == 1
        and not loop.iter.keywords
        and isinstance(loop.iter.args[0], ast.Name)
        and loop.iter.args[0].id == n_name
        and not loop.orelse
    )


def _is_index(node: ast.AST, list_name: str, index_name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == list_name
        and isinstance(node.slice, ast.Name)
        and node.slice.id == index_name
    )


def _matches_sum_loop(
    stmt: ast.stmt,
    *,
    n_name: str,
    list_name: str,
    total_name: str,
    forbidden_targets: set[str],
) -> bool:
    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id not in forbidden_targets
        and _is_range_name(stmt, n_name)
        and len(stmt.body) == 1
    ):
        return False
    index_name = stmt.target.id
    body = stmt.body[0]
    return (
        isinstance(body, ast.AugAssign)
        and isinstance(body.target, ast.Name)
        and body.target.id == total_name
        and isinstance(body.op, ast.Add)
        and _is_index(body.value, list_name, index_name)
    )


def _is_twice_name(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    left_literal = _literal_int(node.left)
    right_literal = _literal_int(node.right)
    return (
        left_literal == 2
        and isinstance(node.right, ast.Name)
        and node.right.id == name
    ) or (
        right_literal == 2
        and isinstance(node.left, ast.Name)
        and node.left.id == name
    )


def _matches_min_abs_update(
    stmt: ast.stmt,
    *,
    ans_name: str,
    total_name: str,
    left_name: str,
) -> bool:
    assigned = _single_name_assignment(stmt)
    if assigned is None or assigned[0] != ans_name:
        return False
    value = assigned[1]
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "min"
        and len(value.args) == 2
        and not value.keywords
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == ans_name
    ):
        return False
    magnitude = value.args[1]
    if not (
        isinstance(magnitude, ast.Call)
        and isinstance(magnitude.func, ast.Name)
        and magnitude.func.id == "abs"
        and len(magnitude.args) == 1
        and not magnitude.keywords
        and isinstance(magnitude.args[0], ast.BinOp)
        and isinstance(magnitude.args[0].op, ast.Sub)
        and isinstance(magnitude.args[0].left, ast.Name)
        and magnitude.args[0].left.id == total_name
    ):
        return False
    return _is_twice_name(magnitude.args[0].right, left_name)


def _matches_partition_loop(
    stmt: ast.stmt,
    *,
    n_name: str,
    list_name: str,
    total_name: str,
    ans_name: str,
    left_name: str,
    forbidden_targets: set[str],
) -> bool:
    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id not in forbidden_targets
        and _is_range_name(stmt, n_name)
        and len(stmt.body) == 2
    ):
        return False
    index_name = stmt.target.id
    first, second = stmt.body
    return (
        isinstance(first, ast.AugAssign)
        and isinstance(first.target, ast.Name)
        and first.target.id == left_name
        and isinstance(first.op, ast.Add)
        and _is_index(first.value, list_name, index_name)
        and _matches_min_abs_update(
            second,
            ans_name=ans_name,
            total_name=total_name,
            left_name=left_name,
        )
    )


def _is_print_name(stmt: ast.stmt, name: str) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == "print"
        and len(stmt.value.args) == 1
        and not stmt.value.keywords
        and isinstance(stmt.value.args[0], ast.Name)
        and stmt.value.args[0].id == name
    )


def match_partition_program(tree: ast.AST) -> PartitionProgramMatch | None:
    """Return the complete structural match, or None for generic fallback."""
    if not isinstance(tree, ast.Module) or len(tree.body) != 8:
        return None

    n_assign = _single_name_assignment(tree.body[0])
    list_assign = _single_name_assignment(tree.body[1])
    total_assign = _single_name_assignment(tree.body[2])
    ans_assign = _single_name_assignment(tree.body[4])
    left_assign = _single_name_assignment(tree.body[5])
    if None in (n_assign, list_assign, total_assign, ans_assign, left_assign):
        return None
    assert n_assign is not None
    assert list_assign is not None
    assert total_assign is not None
    assert ans_assign is not None
    assert left_assign is not None

    n_name, n_value = n_assign
    list_name, list_value = list_assign
    total_name, total_value = total_assign
    ans_name, ans_value = ans_assign
    left_name, left_value = left_assign

    initial_ans = _literal_int(ans_value)
    if not (
        _is_int_input(n_value)
        and _is_int_list_input(list_value)
        and _literal_int(total_value) == 0
        and _literal_int(left_value) == 0
        and initial_ans is not None
    ):
        return None

    critical = {n_name, list_name, total_name, ans_name, left_name}
    if len(critical) != 5:
        return None

    if not _matches_sum_loop(
        tree.body[3],
        n_name=n_name,
        list_name=list_name,
        total_name=total_name,
        forbidden_targets=critical,
    ):
        return None
    if not _matches_partition_loop(
        tree.body[6],
        n_name=n_name,
        list_name=list_name,
        total_name=total_name,
        ans_name=ans_name,
        left_name=left_name,
        forbidden_targets=critical,
    ):
        return None
    if not _is_print_name(tree.body[7], ans_name):
        return None

    return PartitionProgramMatch(
        n_name=n_name,
        list_name=list_name,
        total_name=total_name,
        ans_name=ans_name,
        left_name=left_name,
        initial_ans=initial_ans,
    )


def lower_partition_program_if_supported(tree: ast.AST) -> str | None:
    """Lower the recognized whole-program shape to the scalable BF runtime."""
    match = match_partition_program(tree)
    if match is None:
        return None
    return build_partition_program(initial_ans=match.initial_ans)


__all__ = [
    "PartitionProgramMatch",
    "lower_partition_program_if_supported",
    "match_partition_program",
]
