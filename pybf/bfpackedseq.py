"""Runtime-sized contiguous ``int64`` sequence storage.

This module is the scalable integer-storage vertical slice.  Persistent records
are deliberately small and source generation never receives a capacity/N::

    [marker][back][packed int64:8 bytes]

``marker == 1`` denotes a materialized item.  The first zero marker is the end
sentinel. ``back == 1`` on every record after record zero lets a runtime walker
return to the fixed base without knowing the sequence length.

Decimal parsing temporarily borrows zero-initialized *future* tape cells.  A
radix-4 accumulator gives fixed-work ``x = x*10 + digit``; once a token ends it
is packed into the current eight-byte record and the borrowed window is scrubbed
before the next record is armed.  Therefore persistent tape usage is only ten
cells per integer even though parsing uses a larger rolling scratch window.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from bfbase4 import Base4I64Core, Base4I64Ref, WORD_CELLS as BASE4_WORD_CELLS
from bfbase4decimal import Base4DecimalCore
from bfcore import BFEmitter
from bfpacked64 import PackedI64Ref


RECORD_STRIDE = 10
MARKER = 0
BACK = 1
PAYLOAD = 2
PAYLOAD_BYTES = 8

# Control scratch begins exactly at the next, still-unmaterialized record.
# NEXT_MARKER/NEXT_BACK are intentionally reused as CH/SIGN until the very end
# of an iteration; all rolling scratch is cleared before those cells are armed.
CH = RECORD_STRIDE
SIGN = RECORD_STRIDE + 1
IS_MINUS = RECORD_STRIDE + 2
SKIP = RECORD_STRIDE + 3
TMP = RECORD_STRIDE + 4
ACTIVE = RECORD_STRIDE + 5
DELIMITER = RECORD_STRIDE + 6
END_LINE = RECORD_STRIDE + 7
EQ = RECORD_STRIDE + 8
RESTORE = RECORD_STRIDE + 9
CONT = RECORD_STRIDE + 10
HAS_TOKEN = RECORD_STRIDE + 11
GATE = RECORD_STRIDE + 12
LINE_TMP = RECORD_STRIDE + 13

# The base-4 words are temporary rolling parse state.  They overlap records
# that do not exist yet and are zeroed before the next record is materialized.
ACC_BASE = 32
DECIMAL_SCRATCH_BASE = ACC_BASE + BASE4_WORD_CELLS
NEG_RESULT_BASE = DECIMAL_SCRATCH_BASE + BASE4_WORD_CELLS
WORKSPACE_END = NEG_RESULT_BASE + BASE4_WORD_CELLS


class _RelativeBuilder:
    def __init__(self) -> None:
        self.pos = 0
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


def _eq_const(
    r: _RelativeBuilder,
    result: int,
    src: int,
    value: int,
    tmp: int,
    restore: int,
) -> None:
    """result = (src == value), preserving src."""
    r.set_const(result, 1)
    r.copy_preserved(src, tmp, restore)
    r.add(tmp, -value)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(result)
    r.move(tmp)
    r.emit("]")


def _is_hspace(r: _RelativeBuilder, result: int, src: int) -> None:
    r.clear(result)
    for value in (ord(" "), ord("\t"), ord("\r")):
        _eq_const(r, EQ, src, value, TMP, RESTORE)
        r.move(EQ)
        r.emit("[")
        r.add(EQ, -1)
        r.set_const(result, 1)
        r.move(EQ)
        r.emit("]")


def _is_line_end(r: _RelativeBuilder, result: int, src: int) -> None:
    r.clear(result)
    for value in (ord("\n"), 0):
        _eq_const(r, EQ, src, value, TMP, RESTORE)
        r.move(EQ)
        r.emit("[")
        r.add(EQ, -1)
        r.set_const(result, 1)
        r.move(EQ)
        r.emit("]")


def _flag_not(
    r: _RelativeBuilder,
    dst: int,
    src: int,
) -> None:
    """dst = not src for a preserved Boolean src, including dst==GATE."""
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
    """Start/end at relative cell zero; CH contains numeric digit 0..9."""
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
    """Two's-complement negate ACC using scratch/result words; start/end at 0."""
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
def _pack_kernel() -> str:
    """Destructively pack ACC's 32 radix-4 digits into eight payload bytes."""
    bf = BFEmitter()
    acc = Base4I64Ref(ACC_BASE)
    for byte_index in range(PAYLOAD_BYTES):
        out = PAYLOAD + byte_index
        bf.clear(out)
        for within, scale in enumerate((1, 4, 16, 64)):
            digit = acc.value(byte_index * 4 + within)
            bf.begin_while(digit)
            bf.add_const(digit, -1)
            bf.add_const(out, scale)
            bf.end_while(digit)
    bf.move(0)
    return bf.code()


@lru_cache(maxsize=1)
def _read_record_body() -> str:
    """One capacity-independent token-record iteration.

    Entry is the current record marker. Exit is the next record marker, which
    is one iff another token may follow on the same logical line.
    """
    r = _RelativeBuilder()

    # Read first byte and skip horizontal whitespace without crossing LF/EOF.
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

    # Dynamically gate all token work. An empty/whitespace-only line clears the
    # pre-armed current marker, making record zero itself the end sentinel.
    r.copy_preserved(HAS_TOKEN, GATE, RESTORE)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)

    _eq_const(r, IS_MINUS, CH, ord("-"), TMP, RESTORE)
    r.move(IS_MINUS)
    r.emit("[")
    r.add(IS_MINUS, -1)
    r.set_const(SIGN, 1)
    r.move(CH)
    r.emit(",")
    r.move(IS_MINUS)
    r.emit("]")

    # Recompute end/delimiter after an optional minus sign.
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

    # One source loop handles every decimal digit. The expensive arithmetic is
    # a fixed 32-lane radix-4 kernel, never a loop proportional to byte value.
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

    # Signed tokens use exact two's complement before persistent packing.
    r.move(SIGN)
    r.emit("[")
    r.add(SIGN, -1)
    r.move(0)
    r.emit(_negate_kernel())
    r.pos = 0
    r.move(SIGN)
    r.emit("]")

    r.move(0)
    r.emit(_pack_kernel())
    r.pos = 0
    _flag_not(r, CONT, END_LINE)

    r.move(GATE)
    r.emit("]")

    # If no token existed, current marker becomes the zero end sentinel.
    _flag_not(r, GATE, HAS_TOKEN)
    r.move(GATE)
    r.emit("[")
    r.add(GATE, -1)
    r.clear(MARKER)
    r.move(GATE)
    r.emit("]")

    # Scrub the complete rolling window before any future record is exposed.
    # CONT/HAS_TOKEN survive just long enough to arm next marker/back.
    for cell in range(CH, WORKSPACE_END):
        if cell not in (CONT, HAS_TOKEN):
            r.clear(cell)

    # Every valid record creates a back-link on the following record, including
    # the zero sentinel. Only a non-line-ending token arms the next marker.
    r.move(HAS_TOKEN)
    r.emit("[")
    r.add(HAS_TOKEN, -1)
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
class RuntimePackedIntSequence:
    """Contiguous runtime-grown signed-int64 records beginning at ``base``."""

    base: int

    def _check_layout(self) -> None:
        if self.base < 0:
            raise ValueError("sequence base must be non-negative")

    def marker(self, index: int) -> int:
        if index < 0:
            raise IndexError(index)
        return self.base + index * RECORD_STRIDE + MARKER

    def back(self, index: int) -> int:
        if index < 0:
            raise IndexError(index)
        return self.base + index * RECORD_STRIDE + BACK

    def item(self, index: int) -> PackedI64Ref:
        if index < 0:
            raise IndexError(index)
        return PackedI64Ref(self.base + index * RECORD_STRIDE + PAYLOAD)

    def read_lf_terminated_s64s(self, bf: BFEmitter) -> None:
        """Read one whitespace-separated signed-int line with no capacity limit."""
        self._check_layout()
        bf.move(self.base)
        bf.set_const(self.base + MARKER, 1)
        bf.clear(self.base + BACK)
        bf.move(self.base + MARKER)
        bf.emit("[" + _read_record_body() + "]")

        # Body exits on the marker one record to the right of the last active
        # iteration. Move to the previous record's BACK and follow BACK==1 to
        # record zero. This also works for the empty-line case.
        bf.emit("<" * (RECORD_STRIDE - BACK))
        bf.emit("[" + "<" * RECORD_STRIDE + "]")
        bf.emit("<" * BACK)
        bf.ptr = self.base


__all__ = [
    "RECORD_STRIDE",
    "MARKER",
    "BACK",
    "PAYLOAD",
    "PAYLOAD_BYTES",
    "RuntimePackedIntSequence",
]
