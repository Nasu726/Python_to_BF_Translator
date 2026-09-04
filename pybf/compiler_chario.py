"""Source-compact I/O for string-backed character-list views.

The generic fixed ``StringRef`` backend is correctness-first: reading and
printing a 255-byte string statically expands one absolute-address operation per
slot.  That is acceptable for small scalar strings but made the ordinary
``list(input())`` / ``''.join(chars)`` path multi-megabyte.

A character-list view has stronger invariants: it is a fixed-capacity sequence
of one-byte elements, element replacement does not change its logical length,
and its cached length is refreshed on every ``list(input())`` assignment.  We
use those invariants to emit one runtime loop body instead of 255 copies.

Input construction keeps newly read characters at the physical tail by rotating
the payload left once per stored byte.  After newline/EOF, the remaining
rotations complete one full cycle and normalize the logical string back to the
prefix.  Direct ``''.join(chars)`` output similarly rotates exactly one full
cycle while printing slot zero, restoring the payload exactly.
"""

from __future__ import annotations

import ast

from compiler_charconv import _empty_join_arg, _is_list_input
from compiler_charindex import CompileError
from compiler_charindex import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Character-view compiler with compact line input and direct join output."""

    def _read_char_list_line_compact(self, name: str) -> None:
        ref = self.strings[name]
        length = self.char_list_lengths[name]

        self.backend.clear_string(ref)
        self.backend.set_u64(length, 0)

        remaining = self.temps.cell()
        active = self.temps.cell()
        ch = self.temps.cell()
        is_end = self.temps.cell()
        end_tmp = self.temps.cell()
        end_gate = self.temps.cell()
        data_gate = self.temps.cell()
        data_tmp = self.temps.cell()
        has_room = self.temps.cell()

        for cell in (
            remaining,
            active,
            ch,
            is_end,
            end_tmp,
            end_gate,
            data_gate,
            data_tmp,
            has_room,
        ):
            self.bf.clear(cell)
        self.bf.set_const(remaining, ref.capacity)
        self.bf.set_const(active, 1)

        # One emitted reader body handles the whole physical line, including
        # draining excess bytes after the fixed character capacity is full.
        self.bf.begin_while(active)
        self.bf.move(ch)
        self.bf.emit(",")
        self.backend._set_line_end(is_end, ch, end_tmp)

        self.backend.copy_cell(is_end, end_gate, self.backend.s0)
        self.bf.begin_while(end_gate)
        self.bf.add_const(end_gate, -1)
        self.bf.clear(active)
        self.bf.end_while(end_gate)

        self._flag_not(data_gate, is_end, data_tmp)
        self.bf.begin_while(data_gate)
        self.bf.add_const(data_gate, -1)

        # has_room is a one-shot boolean, not a copy of the byte counter.
        self.bf.clear(has_room)
        self.backend.copy_cell(remaining, has_room, self.backend.s0)
        self.bf.begin_while(has_room)
        self.bf.clear(has_room)

        self._rotate_payload_left_once(ref)
        tail = ref.char(ref.capacity - 1)
        self.bf.clear(tail)
        # ch is dead after the line-end classification, so move it directly.
        self.bf.begin_while(ch)
        self.bf.add_const(ch, -1)
        self.bf.add_const(tail, 1)
        self.bf.end_while(ch)
        self.bf.add_const(remaining, -1)
        self.backend._inc64_inplace(length)

        self.bf.end_while(has_room)
        self.bf.end_while(data_gate)
        self.bf.end_while(active)

        # k stored bytes performed k left rotations and now occupy the last k
        # payload slots in order.  Completing capacity-k more rotations places
        # them in the canonical prefix and restores a zero suffix.
        self.bf.begin_while(remaining)
        self.bf.add_const(remaining, -1)
        self._rotate_payload_left_once(ref)
        self.bf.end_while(remaining)
        self.bf.clear(ref.terminator)
        self.backend._clear_scratch()

    def _print_char_list_join_compact(self, name: str) -> None:
        ref = self.strings[name]
        turns = self.temps.cell()
        gate = self.temps.cell()

        self.bf.clear(turns)
        self.bf.clear(gate)
        self.bf.set_const(turns, ref.capacity)

        # Well-formed char-list views contain the logical bytes in a prefix and
        # zero in every suffix slot.  Printing every nonzero slot over one full
        # rotation therefore emits exactly the joined string, then restores it.
        self.bf.begin_while(turns)
        self.bf.add_const(turns, -1)
        self.backend.copy_cell(ref.char(0), gate, self.backend.s0)
        self.bf.begin_while(gate)
        self.bf.clear(gate)
        self.bf.move(ref.char(0))
        self.bf.emit(".")
        self.bf.end_while(gate)
        self._rotate_payload_left_once(ref)
        self.bf.end_while(turns)
        self.bf.clear(ref.terminator)
        self.backend._clear_scratch()

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.char_list_names
            and _is_list_input(node.value)
        ):
            self._read_char_list_line_compact(node.targets[0].id)
            return
        return super()._compile_stmt_inner(node)

    def _compile_print(self, call: ast.Call) -> None:
        sep = " "
        end = "\n"
        for kw in call.keywords:
            if (
                kw.arg not in ("sep", "end")
                or not isinstance(kw.value, ast.Constant)
                or not isinstance(kw.value.value, str)
            ):
                raise self._error(
                    call, "print only supports constant-string sep= and end="
                )
            if kw.arg == "sep":
                sep = kw.value.value
            else:
                end = kw.value.value

        for arg_index, arg in enumerate(call.args):
            if arg_index:
                self._emit_string(sep)

            join_arg = _empty_join_arg(arg)
            if isinstance(join_arg, ast.Name) and join_arg.id in self.char_list_names:
                # The join conversion itself is a view/no-op.  Direct output can
                # consume that view without materializing a scalar string copy.
                self._print_char_list_join_compact(join_arg.id)
            elif isinstance(arg, ast.Name) and arg.id in self.char_list_names:
                raise self._error(
                    arg,
                    "direct character-list repr printing is not lowered; use ''.join(list)",
                )
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self._emit_string(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in self.string_lists:
                self._print_string_list(self.string_lists[arg.id])
            elif self._expr_is_string(arg):
                value = self._eval_string(arg)
                self.backend.print_string(value, self.temps.cell())
            elif isinstance(arg, ast.Name) and arg.id in self.lists:
                self._print_list(self.lists[arg.id])
            else:
                value = self.compile_expr(arg)
                self.backend.print_s64(value, self.workspace_base)
        self._emit_string(end)


__all__ = ["CompileError", "PythonToBFStream"]
