"""Source-compact runtime decimal-string parsing for the final compiler.

``compiler_charconv`` establishes the type/view semantics and
``compiler_charindex`` makes character-list indexing logically range-safe.

The first runtime-loop parser still used generic dynamic string indexing.  At
public capacity 255 that expanded to 12 MB because each character fetch emitted
255 absolute-address candidates.  This implementation instead keeps the parser
at ``src[0]`` and rotates the fixed string buffer left exactly ``capacity``
times.  After a complete cycle the source bytes are restored exactly, while the
64-bit ``value = value*10 + digit`` body appears only once in generated BF.

Runtime work is O(capacity**2) byte moves in the worst case, but capacity is at
most 255 and valid signed-int64 decimal text is tiny.  The trade is therefore
appropriate for explicit ``int(str)`` and dramatically reduces emitted source.

For valid Python decimal strings under the project's int64 ABI, the parser
accepts ASCII digits, an optional leading sign, and ASCII whitespace around the
number. Runtime ValueError propagation for invalid text remains a later error-
state feature; unsupported invalid spellings are not claimed to match CPython.
"""

from __future__ import annotations

import ast

from compiler_charconv import _is_input_call
from compiler_charindex import CompileError
from compiler_charindex import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Character/conversion compiler with preserving rotation decimal parse."""

    def _set_ascii_digit_flag(self, result: int, ch: int, tmp: int) -> None:
        """result = 1 iff ch is ASCII '0'..'9', preserving ch."""
        self.bf.clear(result)
        for value in range(ord("0"), ord("9") + 1):
            self.backend._eq_byte_const(tmp, ch, value)
            self.bf.begin_while(tmp)
            self.bf.add_const(tmp, -1)
            self.bf.set_const(result, 1)
            self.bf.end_while(tmp)

    def _apply_sign_if_first(
        self,
        *,
        char_matches: int,
        started: int,
        negative_sign: int | None,
    ) -> None:
        gate = self.temps.cell()
        not_started = self.temps.cell()
        not_started_tmp = self.temps.cell()
        self.backend.copy_cell(char_matches, gate, self.backend.s0)
        self.bf.begin_while(gate)
        self.bf.add_const(gate, -1)
        self._flag_not(not_started, started, not_started_tmp)
        self.bf.begin_while(not_started)
        self.bf.add_const(not_started, -1)
        self.bf.set_const(started, 1)
        if negative_sign is not None:
            self.bf.set_const(negative_sign, 1)
        self.bf.end_while(not_started)
        self.bf.end_while(gate)

    def _rotate_string_left_one(self, src, saved: int) -> None:
        """Rotate all payload cells left once, consuming ``saved`` into tail.

        ``saved`` must contain a copy of the pre-rotation first byte.  The
        transfer is deliberately local: adjacent source cells are consumed into
        their left neighbor without routing through the distant shared scratch
        area, keeping pointer source compact.
        """
        for i in range(src.capacity - 1):
            dst = src.char(i)
            nxt = src.char(i + 1)
            self.bf.clear(dst)
            self.bf.begin_while(nxt)
            self.bf.add_const(nxt, -1)
            self.bf.add_const(dst, 1)
            self.bf.end_while(nxt)

        tail = src.char(src.capacity - 1)
        self.bf.clear(tail)
        self.bf.begin_while(saved)
        self.bf.add_const(saved, -1)
        self.bf.add_const(tail, 1)
        self.bf.end_while(saved)

    def _parse_decimal_string(self, src):
        result = self._new_word(0)
        twice = self._new_word()
        eight = self._new_word()
        summed = self._new_word()

        turns = self.temps.cell()
        active = self.temps.cell()
        started = self.temps.cell()
        negative_sign = self.temps.cell()
        ch = self.temps.cell()
        saved = self.temps.cell()
        active_gate = self.temps.cell()
        parse_gate = self.temps.cell()
        is_zero = self.temps.cell()
        is_digit = self.temps.cell()
        is_minus = self.temps.cell()
        is_plus = self.temps.cell()
        class_tmp = self.temps.cell()
        digit_gate = self.temps.cell()

        for cell in (
            turns,
            active,
            started,
            negative_sign,
            ch,
            saved,
            active_gate,
            parse_gate,
            is_zero,
            is_digit,
            is_minus,
            is_plus,
            class_tmp,
            digit_gate,
        ):
            self.bf.clear(cell)
        self.bf.set_const(turns, src.capacity)
        self.bf.set_const(active, 1)

        # Exactly capacity rotations restore the original payload byte-for-byte.
        # Once the first NUL terminator is encountered, parsing is disabled so
        # wrapped-around original characters near the end are not parsed twice.
        self.bf.begin_while(turns)
        self.bf.add_const(turns, -1)
        self.backend.copy_cell(src.char(0), ch, self.backend.s0)
        self.backend.copy_cell(src.char(0), saved, self.backend.s0)

        self.backend.copy_cell(active, active_gate, self.backend.s0)
        self.bf.begin_while(active_gate)
        self.bf.add_const(active_gate, -1)

        self.backend._eq_byte_const(is_zero, ch, 0)
        self.bf.begin_while(is_zero)
        self.bf.add_const(is_zero, -1)
        self.bf.clear(active)
        self.bf.end_while(is_zero)

        self.backend.copy_cell(active, parse_gate, self.backend.s0)
        self.bf.begin_while(parse_gate)
        self.bf.add_const(parse_gate, -1)

        # Signs are meaningful only before the first sign/digit. Whitespace
        # leaves ``started`` unchanged, so ordinary leading whitespace works.
        self.backend._eq_byte_const(is_minus, ch, ord("-"))
        self._apply_sign_if_first(
            char_matches=is_minus,
            started=started,
            negative_sign=negative_sign,
        )
        self.backend._eq_byte_const(is_plus, ch, ord("+"))
        self._apply_sign_if_first(
            char_matches=is_plus,
            started=started,
            negative_sign=None,
        )

        self._set_ascii_digit_flag(is_digit, ch, class_tmp)
        self.backend.copy_cell(is_digit, digit_gate, self.backend.s0)
        self.bf.begin_while(digit_gate)
        self.bf.add_const(digit_gate, -1)
        self.bf.set_const(started, 1)

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
        self.bf.end_while(digit_gate)
        self.bf.end_while(parse_gate)
        self.bf.end_while(active_gate)

        self._rotate_string_left_one(src, saved)
        self.bf.end_while(turns)

        self.bf.begin_while(negative_sign)
        self.bf.add_const(negative_sign, -1)
        self.backend._neg64_inplace(result)
        self.bf.end_while(negative_sign)
        self.backend._clear_scratch()
        return result

    def compile_expr(self, node: ast.AST):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and not node.keywords
        ):
            arg = node.args[0]

            # Do not route the long-established int(input()) fast path through
            # string materialization.  PR #8 initially did so accidentally and
            # regressed ordinary integer input by many megabytes of BF source.
            if _is_input_call(arg):
                result = self._new_word()
                self._read_single_int_line(result)
                return result

            # Named strings already have stable storage.  The rotation parser
            # restores them exactly, so avoid the generic _eval_string copy.
            if isinstance(arg, ast.Name) and arg.id in self.strings:
                if arg.id in self.char_list_names:
                    raise self._error(arg, "int(list) is not a valid integer conversion")
                return self._parse_decimal_string(self.strings[arg.id])

        return super().compile_expr(node)


__all__ = ["CompileError", "PythonToBFStream"]
