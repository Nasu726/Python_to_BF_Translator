"""Source-compact signed-int64 -> string formatting.

The decimal digit generator in the Quad backend is already source-compact and
is shared with ``print(int)``. The original ``str(int)`` lowering became huge
because every emitted decimal character selected one of up to 21 destination
string slots through a statically expanded dynamic-index store.

This layer keeps the existing decimal generator but builds the destination
string by preserving rotations. Each accepted output byte rotates the payload
left once and is moved into the now-zero tail. A final ``remaining`` rotation
loop completes exactly one full cycle, leaving the characters in canonical
prefix order. No runtime destination selector is needed.

For the common ``s = str(n)`` statement, formatting is performed directly into
``s`` rather than formatting a 255-byte temporary and then snapshot-copying it.
The integer value itself is snapshotted first because the liveness allocator is
allowed to reuse ``n`` storage for ``s`` once the assignment consumes ``n``.
"""

from __future__ import annotations

import ast

from bfquad import WORD_CELLS
from compiler_decimalconv import CompileError
from compiler_decimalconv import PythonToBFStream as _BasePythonToBFStream


_DECIMAL_DIGITS = 20
_DECIMAL_STRIDE = 7


class PythonToBFStream(_BasePythonToBFStream):
    """Decimal conversion compiler with compact int->string construction."""

    def _eval_string(self, node: ast.AST):
        # A restricted character-list deliberately reuses StringRef storage, but
        # that physical representation must not leak scalar-string semantics.
        # Dedicated list operations (subscript, len, iteration and empty join)
        # are lowered before reaching this generic string-evaluation boundary.
        if isinstance(node, ast.Name) and node.id in self.char_list_names:
            raise self._error(
                node,
                "restricted character-list values cannot be consumed as scalar strings; use ''.join(list)",
            )
        return super()._eval_string(node)

    def _append_ascii_if_room(
        self,
        dst,
        remaining: int,
        ascii_cell: int,
        room: int,
    ) -> None:
        """Append ``ascii_cell`` to the rotation-built string if capacity remains.

        ``ascii_cell`` is consumed when appended. ``remaining`` is a byte
        counter because fixed StringRef capacity is at most 255.
        """
        self.bf.clear(room)
        self.backend.copy_cell(remaining, room, self.backend.s0)
        self.bf.begin_while(room)
        self.bf.clear(room)
        self._rotate_payload_left_once(dst)
        tail = dst.char(dst.capacity - 1)
        self.bf.clear(tail)
        self.bf.begin_while(ascii_cell)
        self.bf.add_const(ascii_cell, -1)
        self.bf.add_const(tail, 1)
        self.bf.end_while(ascii_cell)
        self.bf.add_const(remaining, -1)
        self.bf.end_while(room)

    def _format_int_string_into(self, dst, src) -> None:
        """Write signed ``src`` into existing ``dst`` in canonical string form.

        ``dst`` may overlap ``src`` at runtime because compile-time liveness
        allocation can reuse a dead scalar block for the string target. Copy the
        complete integer and sign before clearing destination storage.
        """
        magnitude = self._new_word()
        self.backend.copy64(magnitude, src)

        remaining = self.temps.cell()
        sign = self.temps.cell()
        ascii_cell = self.temps.cell()
        room = self.temps.cell()
        for cell in (remaining, sign, ascii_cell, room):
            self.bf.clear(cell)
        self.backend.copy_cell(magnitude.bit(63), sign, self.backend.s0)

        # Only now is it safe to initialize destination storage.
        self.backend.clear_string(dst)
        self.bf.set_const(remaining, dst.capacity)

        # Emit '-' first when needed. Negation is modulo 2**64, so INT64_MIN
        # becomes the unsigned magnitude 2**63 as required by decimal output.
        self.bf.begin_while(sign)
        self.bf.add_const(sign, -1)
        self.bf.set_const(ascii_cell, ord("-"))
        self._append_ascii_if_room(dst, remaining, ascii_cell, room)
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

        for digit_index in range(_DECIMAL_DIGITS - 1, -1, -1):
            digit = decimal_base + digit_index * _DECIMAL_STRIDE + 1
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
            self._append_ascii_if_room(dst, remaining, ascii_cell, room)
            self.bf.end_while(control)

        self.bf.begin_while(remaining)
        self.bf.add_const(remaining, -1)
        self._rotate_payload_left_once(dst)
        self.bf.end_while(remaining)
        self.bf.clear(dst.terminator)
        self.backend._clear_scratch()

    def _format_int_string(self, src):
        dst = self._new_string()
        self._format_int_string_into(dst, src)
        return dst

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.strings
            and node.targets[0].id not in self.char_list_names
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "str"
            and len(node.value.args) == 1
            and not node.value.keywords
        ):
            arg = node.value.args[0]
            # Keep constant int -> string compile-time folding in the older
            # layer; direct runtime formatting is only needed for dynamic values.
            if not self._expr_is_string(arg) and self._constant_int(arg) is None:
                value = self.compile_expr(arg)
                self._format_int_string_into(self.strings[node.targets[0].id], value)
                return

        return super()._compile_stmt_inner(node)


__all__ = ["CompileError", "PythonToBFStream"]
