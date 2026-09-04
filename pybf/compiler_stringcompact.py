"""Source-compact scalar-string operations built on preserving rotations.

Character-list work exposed a broader fixed-string problem: generic ``read_line``,
``copy_string`` and ``print_string`` statically repeat distant scratch traffic
for every one of 255 payload slots.  Explicit ``int(str)`` / ``str(int)`` needs
ordinary scalar strings to remain practical too.

This layer reuses the same rotation invariant for scalar strings:

* line input builds bytes at the physical tail and finishes the remaining
  rotations to normalize them into the prefix;
* copying rotates the immutable source through slot zero while building the
  destination, then both values end in canonical form;
* printing rotates one full cycle while outputting nonzero slot-zero bytes.

All three preserve the NUL-terminated byte-string ABI.  Runtime work grows with
the fixed capacity, but emitted source contains only one body for each runtime
loop instead of one absolute-address body per slot.
"""

from __future__ import annotations

import ast

from compiler_charconv import _empty_join_arg, _is_input_call
from compiler_chario import CompileError
from compiler_chario import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Character/conversion compiler with compact scalar-string primitives."""

    def _read_string_ref_compact(self, ref, *, length=None) -> None:
        self.backend.clear_string(ref)
        if length is not None:
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

        self.bf.clear(has_room)
        self.backend.copy_cell(remaining, has_room, self.backend.s0)
        self.bf.begin_while(has_room)
        self.bf.clear(has_room)

        self._rotate_payload_left_once(ref)
        tail = ref.char(ref.capacity - 1)
        self.bf.clear(tail)
        self.bf.begin_while(ch)
        self.bf.add_const(ch, -1)
        self.bf.add_const(tail, 1)
        self.bf.end_while(ch)
        self.bf.add_const(remaining, -1)
        if length is not None:
            self.backend._inc64_inplace(length)

        self.bf.end_while(has_room)
        self.bf.end_while(data_gate)
        self.bf.end_while(active)

        self.bf.begin_while(remaining)
        self.bf.add_const(remaining, -1)
        self._rotate_payload_left_once(ref)
        self.bf.end_while(remaining)
        self.bf.clear(ref.terminator)
        self.backend._clear_scratch()

    def _copy_string_ref_compact(self, dst, src) -> None:
        if dst.base == src.base:
            return
        if dst.capacity < src.capacity:
            raise ValueError("destination string capacity is smaller than source")

        self.backend.clear_string(dst)
        turns = self.temps.cell()
        remaining = self.temps.cell()
        ch = self.temps.cell()
        data_gate = self.temps.cell()

        for cell in (turns, remaining, ch, data_gate):
            self.bf.clear(cell)
        self.bf.set_const(turns, src.capacity)
        self.bf.set_const(remaining, dst.capacity)

        # Canonical strings have a nonzero prefix followed by zeros, and NUL is
        # not a legal payload byte in the current ABI. Reading each physical
        # source slot exactly once therefore copies precisely the logical value.
        self.bf.begin_while(turns)
        self.bf.add_const(turns, -1)
        self.backend.copy_cell(src.char(0), ch, self.backend.s0)
        self.backend.copy_cell(ch, data_gate, self.backend.s0)
        self.bf.begin_while(data_gate)
        self.bf.clear(data_gate)

        self._rotate_payload_left_once(dst)
        tail = dst.char(dst.capacity - 1)
        self.bf.clear(tail)
        self.bf.begin_while(ch)
        self.bf.add_const(ch, -1)
        self.bf.add_const(tail, 1)
        self.bf.end_while(ch)
        self.bf.add_const(remaining, -1)

        self.bf.end_while(data_gate)
        self._rotate_payload_left_once(src)
        self.bf.end_while(turns)

        self.bf.begin_while(remaining)
        self.bf.add_const(remaining, -1)
        self._rotate_payload_left_once(dst)
        self.bf.end_while(remaining)
        self.bf.clear(src.terminator)
        self.bf.clear(dst.terminator)
        self.backend._clear_scratch()

    def _print_string_ref_compact(self, ref) -> None:
        turns = self.temps.cell()
        gate = self.temps.cell()
        self.bf.clear(turns)
        self.bf.clear(gate)
        self.bf.set_const(turns, ref.capacity)

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

    def _eval_string(self, node: ast.AST):
        if isinstance(node, ast.Name) and node.id in self.strings:
            # Python strings are immutable.  Consumers may inspect the stable
            # source directly; assignment creates the required snapshot below.
            return self.strings[node.id]
        if _is_input_call(node):
            result = self._new_string()
            self._read_string_ref_compact(result)
            return result
        return super()._eval_string(node)

    def _assign_string_to(self, target: ast.Name, value) -> None:
        if target.id not in self.strings:
            raise self._error(
                target, f"cannot assign string to integer variable {target.id!r}"
            )
        self._copy_string_ref_compact(self.strings[target.id], value)

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        # Avoid input -> temporary string -> destination copy for the common
        # single-target assignment.  Multiple-target assignment still falls
        # back to evaluate-once + compact snapshot semantics.
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.strings
            and node.targets[0].id not in self.char_list_names
            and _is_input_call(node.value)
        ):
            self._read_string_ref_compact(self.strings[node.targets[0].id])
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
                self._print_string_ref_compact(self.strings[join_arg.id])
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
                self._print_string_ref_compact(value)
            elif isinstance(arg, ast.Name) and arg.id in self.lists:
                self._print_list(self.lists[arg.id])
            else:
                value = self.compile_expr(arg)
                self.backend.print_s64(value, self.workspace_base)
        self._emit_string(end)


__all__ = ["CompileError", "PythonToBFStream"]
