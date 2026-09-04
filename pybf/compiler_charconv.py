"""Character-list views and explicit int/string conversions.

This layer keeps two common contest idioms cheap::

    chars = list(input())
    ...
    text = "".join(chars)

A character list is represented by the same contiguous NUL-terminated byte
buffer as a scalar string.  The conversions themselves therefore need no
runtime data reshaping.  The compiler only adds mutable one-character indexing
semantics for names created by ``list(input())``.

The same layer also lowers explicit ``str(int64)`` and ``int(str)`` conversions.
The project intentionally retains its signed-int64 ABI; dynamic string parsing
targets ordinary ASCII signed-decimal contest input rather than CPython
arbitrary-precision integers.
"""

from __future__ import annotations

import ast
import copy

from bfquad import WORD_CELLS
from compiler_stream_intfusion import CompileError
from compiler_stream_intfusion import PythonToBFStream as _BasePythonToBFStream


_INT_TEXT_SCAN_CELLS = 21  # sign + 19/20 decimal digits + NUL headroom


def _is_input_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
        and not node.args
        and not node.keywords
    )


def _is_list_input(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
        and not node.keywords
        and _is_input_call(node.args[0])
    )


def _empty_join_arg(node: ast.AST) -> ast.AST | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and node.func.value.value == ""
        and len(node.args) == 1
        and not node.keywords
    ):
        return None
    return node.args[0]


def _direct_char_list_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not _is_list_input(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _char_value_names(tree: ast.AST, char_lists: set[str]) -> set[str]:
    """Infer names that are statically one-character strings.

    This is intentionally narrower than general string inference.  It exists so
    code such as ``tmp = chars[i]; chars[j] = tmp`` can preserve the fact that
    ``tmp`` contains exactly one character even when its physical StringRef has
    the normal string capacity.
    """
    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                value = node.value
                one_char = (
                    isinstance(value, ast.Subscript)
                    and isinstance(value.value, ast.Name)
                    and value.value.id in char_lists
                ) or (isinstance(value, ast.Name) and value.id in result)
                if one_char and target.id not in result:
                    result.add(target.id)
                    changed = True
            elif (
                isinstance(node, ast.For)
                and isinstance(node.iter, ast.Name)
                and node.iter.id in char_lists
                and isinstance(node.target, ast.Name)
                and node.target.id not in result
            ):
                result.add(node.target.id)
                changed = True
    return result


class _InferenceRewrite(ast.NodeTransformer):
    """Teach the established static allocator about representation aliases.

    The rewritten AST is used only for type/layout inference.  Actual lowering
    still receives the user's original AST.
    """

    def __init__(self, char_lists: set[str]) -> None:
        self.char_lists = char_lists

    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        assert isinstance(node, ast.Subscript)
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.char_lists
        ):
            return ast.copy_location(ast.Constant(value="x"), node)
        return node

    def visit_Call(self, node: ast.Call):
        join_arg = _empty_join_arg(node)
        if isinstance(join_arg, ast.Name) and join_arg.id in self.char_lists:
            return ast.copy_location(ast.Constant(value=""), node)
        if join_arg is not None and _is_list_input(join_arg):
            return ast.copy_location(ast.Constant(value=""), node)

        if _is_list_input(node):
            return ast.copy_location(copy.deepcopy(node.args[0]), node)

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        ):
            return ast.copy_location(ast.Constant(value=""), node)

        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        return node


class PythonToBFStream(_BasePythonToBFStream):
    """Streaming compiler plus string-backed mutable character lists."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
    ) -> None:
        self.char_list_names = _direct_char_list_names(tree)
        self.char_value_names = _char_value_names(tree, self.char_list_names)
        inference_tree = copy.deepcopy(tree)
        inference_tree = _InferenceRewrite(self.char_list_names).visit(inference_tree)
        assert isinstance(inference_tree, ast.Module)
        ast.fix_missing_locations(inference_tree)
        super().__init__(
            inference_tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
        )

    def _is_char_list_subscript(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.char_list_names
        )

    def _is_empty_char_join(self, node: ast.AST) -> bool:
        arg = _empty_join_arg(node)
        if isinstance(arg, ast.Name):
            return arg.id in self.char_list_names
        return arg is not None and _is_list_input(arg)

    @staticmethod
    def _is_str_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "str"
            and len(node.args) == 1
            and not node.keywords
        )

    def _expr_is_string(self, node: ast.AST) -> bool:
        if self._is_char_list_subscript(node):
            return True
        if self._is_empty_char_join(node) or self._is_str_call(node):
            return True
        return super()._expr_is_string(node)

    @staticmethod
    def _constant_int(node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) is int
        ):
            return -node.operand.value
        return None

    def _index_word_to_byte(self, index) -> int:
        out = self.temps.cell()
        gate = self.temps.cell()
        self.bf.clear(out)
        for bit, weight in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
            self.backend.copy_cell(index.bit(bit), gate, self.backend.s0)
            self.bf.begin_while(gate)
            self.bf.add_const(gate, -1)
            self.bf.add_const(out, weight)
            self.bf.end_while(gate)
        self.backend._clear_scratch()
        return out

    def _char_list_runtime_index_byte(self, node: ast.AST, ref) -> int:
        raw = self.compile_expr(node)
        index = self._new_word()
        self.backend.copy64(index, raw)

        length = self._new_word()
        zero = self._new_word(0)
        self.backend.string_length(length, ref, self.temps.cell())

        # Python negative indexing: normalize once by the current logical
        # length.  Values still negative after that are out of range.
        negative = self.temps.cell()
        self.backend.slt64(negative, index, zero)
        self.bf.begin_while(negative)
        self.bf.add_const(negative, -1)
        summed = self._new_word()
        self.backend.add64(summed, index, length)
        self.backend.copy64(index, summed)
        self.bf.end_while(negative)

        # Physical selection is byte-sized because StringRef capacity <=255.
        # Prove the normalized index is in [0, len) before dropping high bits;
        # otherwise map it to 255, which is outside every valid StringRef slot.
        valid = self.temps.cell()
        still_negative = self.temps.cell()
        self.backend.slt64(valid, index, length)
        self.backend.slt64(still_negative, index, zero)
        self.bf.begin_while(still_negative)
        self.bf.add_const(still_negative, -1)
        self.bf.clear(valid)
        self.bf.end_while(still_negative)

        out = self._index_word_to_byte(index)
        invalid = self.temps.cell()
        invalid_tmp = self.temps.cell()
        self._flag_not(invalid, valid, invalid_tmp)
        self.bf.begin_while(invalid)
        self.bf.add_const(invalid, -1)
        self.bf.set_const(out, 255)
        self.bf.end_while(invalid)
        return out

    def _load_char_list_subscript(self, node: ast.Subscript):
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]
        result = self._new_string()
        self.backend.clear_string(result)

        constant = self._constant_int(node.slice)
        if constant is not None and constant >= 0:
            if constant >= ref.capacity:
                raise self._error(node, "constant character-list index exceeds capacity")
            self.backend.copy_cell(ref.char(constant), result.char(0), self.backend.s0)
            self.backend._clear_scratch()
            return result

        index_byte = self._char_list_runtime_index_byte(node.slice, ref)
        self._load_string_char_at(result, ref, index_byte)
        return result

    def _char_value(self, node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) != 1:
                raise self._error(
                    node,
                    "string-backed character lists currently require one-character assignments",
                )
            return self._eval_string(node)
        if isinstance(node, ast.Name) and node.id in self.char_value_names:
            return self._eval_string(node)
        if isinstance(node, ast.Name) and node.id in self.strings:
            if self.strings[node.id].capacity == 1:
                return self._eval_string(node)
        if self._is_char_list_subscript(node):
            return self._load_char_list_subscript(node)
        raise self._error(
            node,
            "character-list assignment requires a statically one-character string",
        )

    def _store_char_list_subscript(self, node: ast.Subscript, value) -> None:
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]
        constant = self._constant_int(node.slice)
        if constant is not None and constant >= 0:
            if constant >= ref.capacity:
                raise self._error(node, "constant character-list index exceeds capacity")
            self.backend.copy_cell(value.char(0), ref.char(constant), self.backend.s0)
            self.backend._clear_scratch()
            return

        index_byte = self._char_list_runtime_index_byte(node.slice, ref)
        match = self.temps.cell()
        for i in range(ref.capacity):
            self.backend._eq_byte_const(match, index_byte, i)
            self.bf.begin_while(match)
            self.bf.add_const(match, -1)
            self.backend.copy_cell(value.char(0), ref.char(i), self.backend.s0)
            self.bf.end_while(match)
        self.backend._clear_scratch()

    @staticmethod
    def _signed64(value: int) -> int:
        raw = value & ((1 << 64) - 1)
        return raw if raw < (1 << 63) else raw - (1 << 64)

    def _parse_decimal_string(self, src):
        result = self._new_word(0)
        twice = self._new_word()
        eight = self._new_word()
        summed = self._new_word()

        active = self.temps.cell()
        sign = self.temps.cell()
        slot_gate = self.temps.cell()
        process = self.temps.cell()
        ch = self.temps.cell()
        is_zero = self.temps.cell()
        is_minus = self.temps.cell()
        is_plus = self.temps.cell()

        for cell in (
            active,
            sign,
            slot_gate,
            process,
            ch,
            is_zero,
            is_minus,
            is_plus,
        ):
            self.bf.clear(cell)
        self.bf.set_const(active, 1)

        scan = min(src.capacity, _INT_TEXT_SCAN_CELLS)
        for i in range(scan):
            self.backend.copy_cell(active, slot_gate, self.backend.s0)
            self.bf.begin_while(slot_gate)
            self.bf.add_const(slot_gate, -1)

            self.backend.copy_cell(src.char(i), ch, self.backend.s0)
            self.bf.set_const(process, 1)
            self.backend._eq_byte_const(is_zero, ch, 0)
            self.bf.begin_while(is_zero)
            self.bf.add_const(is_zero, -1)
            self.bf.clear(active)
            self.bf.clear(process)
            self.bf.end_while(is_zero)

            if i == 0:
                self.backend._eq_byte_const(is_minus, ch, ord("-"))
                self.bf.begin_while(is_minus)
                self.bf.add_const(is_minus, -1)
                self.bf.set_const(sign, 1)
                self.bf.clear(process)
                self.bf.end_while(is_minus)

                self.backend._eq_byte_const(is_plus, ch, ord("+"))
                self.bf.begin_while(is_plus)
                self.bf.add_const(is_plus, -1)
                self.bf.clear(process)
                self.bf.end_while(is_plus)

            self.bf.begin_while(process)
            self.bf.add_const(process, -1)
            self.backend.copy64(twice, result)
            self.backend.copy64(eight, result)
            self.backend.shl1_inplace(twice)
            self.backend.shl1_inplace(eight)
            self.backend.shl1_inplace(eight)
            self.backend.shl1_inplace(eight)
            self.backend.add64(summed, twice, eight)
            self.backend.copy64(result, summed)

            self.bf.add_const(ch, -ord("0"))
            self.bf.begin_while(ch)
            self.bf.add_const(ch, -1)
            self.backend._inc64_inplace(result)
            self.bf.end_while(ch)
            self.bf.end_while(process)
            self.bf.end_while(slot_gate)

        self.bf.begin_while(sign)
        self.bf.add_const(sign, -1)
        self.backend._neg64_inplace(result)
        self.bf.end_while(sign)
        self.backend._clear_scratch()
        return result

    def _store_formatted_byte(self, dst, index: int, value: int) -> None:
        match = self.temps.cell()
        for i in range(min(dst.capacity, _INT_TEXT_SCAN_CELLS)):
            self.backend._eq_byte_const(match, index, i)
            self.bf.begin_while(match)
            self.bf.add_const(match, -1)
            self.backend.copy_cell(value, dst.char(i), self.backend.s0)
            self.bf.end_while(match)
        self.backend._clear_scratch()

    def _format_int_string(self, src):
        dst = self._new_string()
        self.backend.clear_string(dst)
        magnitude = self._new_word()
        self.backend.copy64(magnitude, src)

        out_index = self.temps.cell()
        sign = self.temps.cell()
        ascii_cell = self.temps.cell()
        self.bf.clear(out_index)
        self.backend.copy_cell(src.bit(63), sign, self.backend.s0)
        self.bf.begin_while(sign)
        self.bf.add_const(sign, -1)
        self.bf.set_const(ascii_cell, ord("-"))
        if dst.capacity:
            self.backend.copy_cell(ascii_cell, dst.char(0), self.backend.s0)
        self.bf.set_const(out_index, 1)
        self.backend._neg64_inplace(magnitude)
        self.bf.end_while(sign)
        self.backend._clear_scratch()

        decimal_base = self.workspace_base + WORD_CELLS
        if not hasattr(self.backend, "_quad_to_decimal_bytes"):
            raise CompileError("int->str requires the final Quad scalar backend")
        self.backend._quad_to_decimal_bytes(magnitude, decimal_base)

        started = self.temps.cell()
        control = self.temps.cell()
        tmp = self.temps.cell()
        helper = self.temps.cell()
        for cell in (started, control, tmp, helper):
            self.bf.clear(cell)

        for digit_index in range(19, -1, -1):
            digit = decimal_base + digit_index * 7 + 1
            if digit_index == 0:
                self.bf.set_const(started, 1)
            else:
                self.backend.copy_cell(digit, tmp, helper)
                self.bf.begin_while(tmp)
                self.bf.clear(tmp)
                self.bf.set_const(started, 1)
                self.bf.end_while(tmp)

            self.backend.copy_cell(started, control, helper)
            self.bf.begin_while(control)
            self.bf.add_const(control, -1)
            self.bf.set_const(ascii_cell, ord("0"))
            self.bf.begin_while(digit)
            self.bf.add_const(digit, -1)
            self.bf.add_const(ascii_cell, 1)
            self.bf.end_while(digit)
            self._store_formatted_byte(dst, out_index, ascii_cell)
            self.bf.add_const(out_index, 1)
            self.bf.end_while(control)

        return dst

    def compile_expr(self, node: ast.AST):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and not node.keywords
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id in self.char_list_names:
                raise self._error(arg, "int(list) is not a valid integer conversion")
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                try:
                    return self._new_word(self._signed64(int(arg.value)))
                except ValueError as exc:
                    raise self._error(arg, "invalid decimal integer string") from exc
            if self._expr_is_string(arg):
                return self._parse_decimal_string(self._eval_string(arg))
            return self.compile_expr(arg)
        return super().compile_expr(node)

    def _eval_string(self, node: ast.AST):
        if self._is_char_list_subscript(node):
            return self._load_char_list_subscript(node)

        if self._is_empty_char_join(node):
            arg = _empty_join_arg(node)
            assert arg is not None
            if isinstance(arg, ast.Name):
                return self.strings[arg.id]
            assert _is_list_input(arg)
            dst = self._new_string()
            self.backend.read_line(dst, self.workspace_base)
            return dst

        if self._is_str_call(node):
            assert isinstance(node, ast.Call)
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id in self.char_list_names:
                raise self._error(arg, "str(character_list) list repr is not lowered yet")
            if self._expr_is_string(arg):
                return self._eval_string(arg)
            if isinstance(arg, ast.Constant) and type(arg.value) is int:
                value = self._signed64(arg.value)
                dst = self._new_string()
                self.backend.set_string_literal(dst, str(value))
                return dst
            return self._format_int_string(self.compile_expr(arg))

        return super()._eval_string(node)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self.char_list_names
                and _is_list_input(node.value)
            ):
                self.backend.read_line(self.strings[node.targets[0].id], self.workspace_base)
                return

            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and self._is_char_list_subscript(node.targets[0])
            ):
                value = self._char_value(node.value)
                self._store_char_list_subscript(node.targets[0], value)
                return

            if isinstance(node.value, ast.Name) and node.value.id in self.char_list_names:
                if any(
                    isinstance(target, ast.Name) and target.id != node.value.id
                    for target in node.targets
                ):
                    raise self._error(
                        node,
                        "character-list alias assignment awaits the general mutable-object backend",
                    )

        return super()._compile_stmt_inner(node)

    def _compile_print(self, call: ast.Call) -> None:
        for arg in call.args:
            if isinstance(arg, ast.Name) and arg.id in self.char_list_names:
                raise self._error(
                    arg,
                    "direct character-list repr printing is not lowered; use ''.join(list)",
                )
        return super()._compile_print(call)


__all__ = ["CompileError", "PythonToBFStream"]
