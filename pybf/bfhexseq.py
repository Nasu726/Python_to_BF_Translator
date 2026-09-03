"""Runtime-sized hexadecimal-lane integer sequence with carried loop state.

This experiment targets repeated competitive-programming passes over large
integer arrays.  A record keeps the source value plus three 64-bit state words,
all as sixteen little-endian hexadecimal digits (one nibble per tape cell)::

    marker, back,
    data[16], total[16], left[16], ans[16]

The 66-cell stride is larger than packed storage, but every arithmetic operand
for a sequential pass is local to the current record.  Loop-carried state moves
only one record at a time, so emitted Brainfuck does not contain pointer travel
proportional to runtime N.

During input, ``total`` is updated and transferred into the following record.
The final zero-marker sentinel therefore holds the complete sum.  A compact
backward pass can move that total back to record zero before a second forward
pass.  This is the core transport pattern needed by prefix/partition programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from bfbase4 import Base4I64Core, Base4I64Ref, WORD_CELLS as BASE4_WORD_CELLS
from bfbase4decimal import Base4DecimalCore
from bfcore import BFEmitter


HEX_DIGITS = 16
MARKER = 0
BACK = 1
DATA = 2
TOTAL = DATA + HEX_DIGITS
LEFT = TOTAL + HEX_DIGITS
ANS = LEFT + HEX_DIGITS
RECORD_STRIDE = ANS + HEX_DIGITS  # 66

# Rolling parse control begins at the next record and may overwrite future
# records because they are not materialized yet. It is scrubbed before any next
# record marker/state is exposed.
CH = RECORD_STRIDE
SIGN = CH + 1
IS_MINUS = CH + 2
SKIP = CH + 3
TMP = CH + 4
ACTIVE = CH + 5
DELIMITER = CH + 6
END_LINE = CH + 7
EQ = CH + 8
RESTORE = CH + 9
CONT = CH + 10
HAS_TOKEN = CH + 11
GATE = CH + 12
LINE_TMP = CH + 13

ACC_BASE = 96
DECIMAL_SCRATCH_BASE = ACC_BASE + BASE4_WORD_CELLS
NEG_RESULT_BASE = DECIMAL_SCRATCH_BASE + BASE4_WORD_CELLS
WORKSPACE_END = NEG_RESULT_BASE + BASE4_WORD_CELLS


class _RelativeBuilder:
    def __init__(self, initial_pos: int = 0) -> None:
        self.pos = initial_pos
        self.parts: list[str] = []

    def move(self, target: int) -> None:
        delta = target - self.pos
        if delta > 0:
            self.parts.append(">" * delta)
        elif delta < 0:
            self.parts.append("<" * -delta)
        self.pos = target

    def emit(self, code: str) -> None:
        self.parts.append(code)

    def add(self, target: int, amount: int) -> None:
        self.move(target)
        amount %= 256
        if amount <= 128:
            self.parts.append("+" * amount)
        else:
            self.parts.append("-" * (256 - amount))

    def clear(self, target: int) -> None:
        self.move(target)
        self.parts.append("[-]")

    def set_const(self, target: int, value: int) -> None:
        self.clear(target)
        self.add(target, value)

    def transfer(self, src: int, dst: int) -> None:
        self.move(src)
        self.emit("[")
        self.add(src, -1)
        self.add(dst, 1)
        self.move(src)
        self.emit("]")

    def copy_preserved(self, src: int, dst: int, tmp: int) -> None:
        self.clear(dst)
        self.clear(tmp)
        self.move(src)
        self.emit("[")
        self.add(src, -1)
        self.add(dst, 1)
        self.add(tmp, 1)
        self.move(src)
        self.emit("]")
        self.move(tmp)
        self.emit("[")
        self.add(tmp, -1)
        self.add(src, 1)
        self.move(tmp)
        self.emit("]")

    def code(self) -> str:
        return "".join(self.parts)


def _eq_const(r, result, src, value, tmp=TMP, restore=RESTORE):
    r.set_const(result, 1)
    r.copy_preserved(src, tmp, restore)
    r.add(tmp, -value)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(result)
    r.move(tmp)
    r.emit("]")


def _is_hspace(r, result, src):
    r.clear(result)
    for value in (ord(" "), ord("\t"), ord("\r")):
        _eq_const(r, EQ, src, value)
        r.move(EQ)
        r.emit("[")
        r.add(EQ, -1)
        r.set_const(result, 1)
        r.move(EQ)
        r.emit("]")


def _is_line_end(r, result, src):
    r.clear(result)
    for value in (ord("\n"), 0):
        _eq_const(r, EQ, src, value)
        r.move(EQ)
        r.emit("[")
        r.add(EQ, -1)
        r.set_const(result, 1)
        r.move(EQ)
        r.emit("]")


def _flag_not(r, dst, src):
    scratch = LINE_TMP if dst == GATE else GATE
    r.set_const(dst, 1)
    r.copy_preserved(src, scratch, RESTORE)
    r.move(scratch)
    r.emit("[")
    r.add(scratch, -1)
    r.clear(dst)
    r.move(scratch)
    r.emit("]")


@lru_cache(maxsize=1)
def _decimal_digit_kernel() -> str:
    bf = BFEmitter()
    decimal = Base4DecimalCore(bf)
    decimal.mul10_add_digit_inplace(
        Base4I64Ref(ACC_BASE),
        Base4I64Ref(DECIMAL_SCRATCH_BASE),
        CH,
    )
    bf.move(0)
    return bf.code()


@lru_cache(maxsize=1)
def _negate_kernel() -> str:
    bf = BFEmitter()
    core = Base4I64Core(bf)
    acc = Base4I64Ref(ACC_BASE)
    zero = Base4I64Ref(DECIMAL_SCRATCH_BASE)
    result = Base4I64Ref(NEG_RESULT_BASE)
    core.set_u64(zero, 0)
    core.sub64(result, zero, acc)
    core.copy64(acc, result)
    bf.move(0)
    return bf.code()


@lru_cache(maxsize=1)
def _pack_hex_kernel() -> str:
    """Consume base-4 accumulator into sixteen little-endian nibble cells."""
    bf = BFEmitter()
    acc = Base4I64Ref(ACC_BASE)
    for nibble in range(HEX_DIGITS):
        out = DATA + nibble
        bf.clear(out)
        low = acc.value(2 * nibble)
        high = acc.value(2 * nibble + 1)
        bf.begin_while(low)
        bf.add_const(low, -1)
        bf.add_const(out, 1)
        bf.end_while(low)
        bf.begin_while(high)
        bf.add_const(high, -1)
        bf.add_const(out, 4)
        bf.end_while(high)
    bf.move(0)
    return bf.code()


def _map_total_base16(r: _RelativeBuilder, total: int, out: int, carry: int):
    """Consume total 0..31 into one hex digit and carry 0/1."""
    r.clear(out)
    r.clear(carry)
    for step in range(1, 32):
        r.move(total)
        r.emit("[")
        r.add(total, -1)
        if step == 16:
            r.clear(out)
            r.add(carry, 1)
        else:
            r.add(out, 1)
    for _ in range(31):
        r.move(total)
        r.emit("]")


def _add_preserved_small(r, src, total, tmp):
    r.move(src)
    r.emit("[")
    r.add(src, -1)
    r.add(total, 1)
    r.add(tmp, 1)
    r.move(src)
    r.emit("]")
    r.move(tmp)
    r.emit("[")
    r.add(tmp, -1)
    r.add(src, 1)
    r.move(tmp)
    r.emit("]")


@lru_cache(maxsize=1)
def _add_data_to_total_kernel() -> str:
    """TOTAL += DATA modulo 2**64, preserving DATA; start/end at marker 0."""
    r = _RelativeBuilder()
    total_tmp = LEFT
    copy_tmp = LEFT + 1
    carry = LEFT + 2
    next_carry = LEFT + 3
    r.clear(carry)

    for i in range(HEX_DIGITS):
        a = TOTAL + i
        b = DATA + i
        r.clear(total_tmp)
        r.clear(next_carry)
        r.transfer(carry, total_tmp)
        r.transfer(a, total_tmp)
        _add_preserved_small(r, b, total_tmp, copy_tmp)
        _map_total_base16(r, total_tmp, a, next_carry)
        r.transfer(next_carry, carry)

    for cell in (total_tmp, copy_tmp, carry, next_carry):
        r.clear(cell)
    r.move(0)
    return r.code()


def _transfer_word(r: _RelativeBuilder, src: int, dst: int):
    for i in range(HEX_DIGITS):
        r.transfer(src + i, dst + i)


@lru_cache(maxsize=1)
def _read_record_body() -> str:
    r = _RelativeBuilder()
    r.move(CH)
    r.emit(",")
    _is_hspace(r, SKIP, CH)
    r.move(SKIP)
    r.emit("[")
    r.add(SKIP, -1)
    r.move(CH)
    r.emit(",")
    _is_hspace(r, SKIP, CH)
    r.move(SKIP)
    r.emit("]")

    _is_line_end(r, END_LINE, CH)
    _flag_not(r, HAS_TOKEN, END_LINE)
    r.copy_preserved(HAS_TOKEN, GATE, RESTORE)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)

    _eq_const(r, IS_MINUS, CH, ord("-"))
    r.move(IS_MINUS)
    r.emit("[")
    r.add(IS_MINUS, -1)
    r.set_const(SIGN, 1)
    r.move(CH)
    r.emit(",")
    r.move(IS_MINUS)
    r.emit("]")

    _is_line_end(r, END_LINE, CH)
    _is_hspace(r, DELIMITER, CH)
    r.copy_preserved(END_LINE, LINE_TMP, RESTORE)
    r.move(LINE_TMP)
    r.emit("[")
    r.add(LINE_TMP, -1)
    r.set_const(DELIMITER, 1)
    r.move(LINE_TMP)
    r.emit("]")
    _flag_not(r, ACTIVE, DELIMITER)

    r.move(ACTIVE)
    r.emit("[")
    r.add(ACTIVE, -1)
    r.add(CH, -ord("0"))
    r.move(0)
    r.emit(_decimal_digit_kernel())
    r.pos = 0
    r.move(CH)
    r.emit(",")
    _is_line_end(r, END_LINE, CH)
    _is_hspace(r, DELIMITER, CH)
    r.copy_preserved(END_LINE, LINE_TMP, RESTORE)
    r.move(LINE_TMP)
    r.emit("[")
    r.add(LINE_TMP, -1)
    r.set_const(DELIMITER, 1)
    r.move(LINE_TMP)
    r.emit("]")
    _flag_not(r, ACTIVE, DELIMITER)
    r.move(ACTIVE)
    r.emit("]")

    r.move(SIGN)
    r.emit("[")
    r.add(SIGN, -1)
    r.move(0)
    r.emit(_negate_kernel())
    r.pos = 0
    r.move(SIGN)
    r.emit("]")

    r.move(0)
    r.emit(_pack_hex_kernel())
    r.pos = 0
    r.emit(_add_data_to_total_kernel())
    r.pos = 0
    _flag_not(r, CONT, END_LINE)
    r.move(GATE)
    r.emit("]")

    _flag_not(r, GATE, HAS_TOKEN)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)
    r.clear(MARKER)
    r.move(GATE)
    r.emit("]")

    # Scrub future records used as parsing workspace. Current DATA/TOTAL remain.
    for cell in range(CH, WORKSPACE_END):
        if cell not in (CONT, HAS_TOKEN):
            r.clear(cell)

    # A valid current record transfers its running TOTAL into the following
    # record even when that following record is the zero end sentinel.
    r.move(HAS_TOKEN)
    r.emit("[")
    r.add(HAS_TOKEN, -1)
    _transfer_word(r, TOTAL, RECORD_STRIDE + TOTAL)
    r.set_const(RECORD_STRIDE + BACK, 1)
    r.move(HAS_TOKEN)
    r.emit("]")

    r.move(CONT)
    r.emit("[")
    r.add(CONT, -1)
    r.set_const(RECORD_STRIDE + MARKER, 1)
    r.move(CONT)
    r.emit("]")
    r.move(RECORD_STRIDE + MARKER)
    return r.code()


@dataclass(frozen=True)
class RuntimeHexIntSequence:
    base: int

    def marker(self, index: int) -> int:
        return self.base + index * RECORD_STRIDE + MARKER

    def back(self, index: int) -> int:
        return self.base + index * RECORD_STRIDE + BACK

    def field(self, index: int, field_base: int) -> int:
        return self.base + index * RECORD_STRIDE + field_base

    def read_lf_terminated_s64s_and_sum(self, bf: BFEmitter) -> None:
        """Read one integer line; the end sentinel receives TOTAL sum."""
        if self.base < 0:
            raise ValueError("sequence base must be non-negative")
        bf.move(self.base)
        bf.set_const(self.base + MARKER, 1)
        bf.clear(self.base + BACK)
        bf.move(self.base + MARKER)
        bf.emit("[" + _read_record_body() + "]")

        # Construction exits one active iteration to the right. Return to base
        # using the BACK chain without knowing runtime item count.
        bf.emit("<" * (RECORD_STRIDE - BACK))
        bf.emit("[" + "<" * RECORD_STRIDE + "]")
        bf.emit("<" * BACK)
        bf.ptr = self.base

    def propagate_total_back_to_first(self, bf: BFEmitter) -> None:
        """Move sentinel TOTAL backward until record zero owns it."""
        # Walk forward to the zero-marker sentinel.
        bf.move(self.base + MARKER)
        bf.emit("[" + ">" * RECORD_STRIDE + "]")
        # Sentinel BACK is one iff the sequence is non-empty.
        bf.emit(">" * BACK)

        r = _RelativeBuilder(initial_pos=BACK)
        # Coordinates stay relative to the current record marker even though the
        # BF pointer starts at BACK. Transfer all sixteen TOTAL nibbles; using
        # TOTAL-BACK here would silently omit the most-significant nibble.
        _transfer_word(r, TOTAL, TOTAL - RECORD_STRIDE)
        r.move(BACK - RECORD_STRIDE)
        bf.emit("[" + r.code() + "]")

        # BACK loop exits at record-zero BACK for non-empty data, or starts there
        # immediately for an empty sequence.
        bf.emit("<" * BACK)
        bf.ptr = self.base


__all__ = [
    "HEX_DIGITS",
    "MARKER",
    "BACK",
    "DATA",
    "TOTAL",
    "LEFT",
    "ANS",
    "RECORD_STRIDE",
    "RuntimeHexIntSequence",
]
