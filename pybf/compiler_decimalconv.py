"""Source-compact runtime decimal-string parsing for the final compiler.

``compiler_charconv`` establishes the type/view semantics.  This layer replaces
its correctness-first statically unrolled ``int(str)`` parser with one runtime
character loop.  The expensive 64-bit ``value = value*10 + digit`` body is
therefore emitted once regardless of string capacity.

For valid Python decimal strings under the project's int64 ABI, the parser
accepts ASCII digits, an optional leading sign, and ASCII whitespace around the
number.  Runtime ValueError propagation for invalid text remains a later error-
state feature; unsupported invalid spellings are not claimed to match CPython.
"""

from __future__ import annotations

from bfstrings import StringRef
from compiler_charconv import CompileError
from compiler_charconv import PythonToBFStream as _BasePythonToBFStream


class PythonToBFStream(_BasePythonToBFStream):
    """Character/conversion compiler with runtime-loop decimal parsing."""

    def _new_char_buffer(self) -> StringRef:
        ref = StringRef(self.temps.top, 1)
        self.temps.top += ref.capacity + 1
        return ref

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

    def _parse_decimal_string(self, src):
        result = self._new_word(0)
        twice = self._new_word()
        eight = self._new_word()
        summed = self._new_word()

        char_ref = self._new_char_buffer()
        index = self.temps.cell()
        control = self.temps.cell()
        started = self.temps.cell()
        negative_sign = self.temps.cell()
        ch = self.temps.cell()
        is_digit = self.temps.cell()
        is_minus = self.temps.cell()
        is_plus = self.temps.cell()
        class_tmp = self.temps.cell()
        digit_gate = self.temps.cell()

        for cell in (
            index,
            control,
            started,
            negative_sign,
            ch,
            is_digit,
            is_minus,
            is_plus,
            class_tmp,
            digit_gate,
        ):
            self.bf.clear(cell)
        self.backend.clear_string(char_ref)

        self._load_string_char_at(char_ref, src, index)
        self._set_char_loop_control(control, char_ref.char(0))

        self.bf.begin_while(control)
        self.bf.add_const(control, -1)
        self.backend.copy_cell(char_ref.char(0), ch, self.backend.s0)

        # Signs are meaningful only before the first sign/digit.  Whitespace
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

        # Only an actual ASCII digit reaches the numeric update.  This avoids
        # the old ``ch -= '0'; while ch`` behavior exploding on whitespace or
        # unsupported invalid characters.
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

        self.bf.add_const(index, 1)
        self._load_string_char_at(char_ref, src, index)
        self._set_char_loop_control(control, char_ref.char(0))
        self.bf.end_while(control)

        self.bf.begin_while(negative_sign)
        self.bf.add_const(negative_sign, -1)
        self.backend._neg64_inplace(result)
        self.bf.end_while(negative_sign)
        self.backend._clear_scratch()
        return result


__all__ = ["CompileError", "PythonToBFStream"]
