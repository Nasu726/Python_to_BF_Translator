"""Runtime-sized sequential tape records with source-size-independent growth.

This is the first scalable-storage vertical slice.  It deliberately starts with
raw bytes rather than Python ``list[int]`` semantics so the hard Brainfuck
property can be tested in isolation:

* runtime record count is not a compiler parameter;
* one BF loop body advances across fixed-stride records;
* storage grows only in tape cells/runtime work, not emitted source size;
* a later pass can traverse the stored records again.

Persistent record layout is intentionally tiny::

    [marker][payload byte][7 reserved payload bytes]

``marker == 1`` means a materialized record.  The first zero marker is the end
sentinel.  Scratch lives *ahead* of the current record and is cleared before the
walker advances, so during construction that scratch window overlaps future,
still-unmaterialized records instead of increasing persistent stride.

The current input helper treats LF (``\n``) as the line terminator.  It is an
internal scalability primitive, not yet the public ``input()`` implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter


RECORD_STRIDE = 9
MARKER = 0
PAYLOAD0 = 1
PAYLOAD_BYTES = 8

# Rolling construction scratch.  These cells overlap future payload cells and
# therefore must finish zero before the next record becomes current.
TMP = RECORD_STRIDE + 1
RESTORE = RECORD_STRIDE + 2
IS_LF = RECORD_STRIDE + 3
DATA_GATE = RECORD_STRIDE + 4


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


@dataclass(frozen=True)
class RuntimeByteSequence:
    """A runtime-grown contiguous sequence beginning at ``base``.

    ``base - RECORD_STRIDE`` is a permanent zero left sentinel.  Callers should
    place ``base`` at least one full record after every static/temp cell.
    """

    base: int

    @property
    def left_sentinel(self) -> int:
        return self.base - RECORD_STRIDE

    def _check_layout(self) -> None:
        if self.left_sentinel < 0:
            raise ValueError("runtime sequence requires one record of left guard space")

    @staticmethod
    def _read_line_body() -> str:
        r = _RelativeBuilder()

        # Read the byte directly into the current persistent payload.
        r.move(PAYLOAD0)
        r.emit(",")

        # IS_LF = (payload == 10), preserving payload.
        r.copy_preserved(PAYLOAD0, TMP, RESTORE)
        r.add(TMP, -10)
        r.set_const(IS_LF, 1)
        r.move(TMP)
        r.emit("[")
        r.clear(TMP)
        r.clear(IS_LF)
        r.move(TMP)
        r.emit("]")

        # DATA_GATE = not IS_LF.  LF turns the current pre-armed marker into the
        # zero end sentinel; data arms the next record marker instead.
        r.set_const(DATA_GATE, 1)
        r.move(IS_LF)
        r.emit("[")
        r.add(IS_LF, -1)
        r.clear(DATA_GATE)
        r.clear(MARKER)
        r.clear(PAYLOAD0)
        r.move(IS_LF)
        r.emit("]")

        r.move(DATA_GATE)
        r.emit("[")
        r.add(DATA_GATE, -1)
        r.set_const(RECORD_STRIDE + MARKER, 1)
        r.move(DATA_GATE)
        r.emit("]")

        # Future-record cells used as rolling scratch must be clean before that
        # record becomes current.
        for cell in (TMP, RESTORE, IS_LF, DATA_GATE):
            r.clear(cell)

        # BF loop continuation is the next record's marker.
        r.move(RECORD_STRIDE + MARKER)
        return r.code()

    @staticmethod
    def _reverse_to_left_sentinel_code() -> str:
        # Entry is the zero end sentinel.  Step once to the preceding record;
        # then the BF loop walks left while markers are one.  Empty sequences
        # immediately land on the permanent left sentinel.
        step = "<" * RECORD_STRIDE
        return step + "[" + step + "]"

    def read_lf_terminated_bytes(self, bf: BFEmitter) -> None:
        """Materialize one LF-terminated input line into runtime records."""
        self._check_layout()

        # The left guard and future tape are zero by ABI.  Arm the first record;
        # every successful data iteration arms exactly one next record.
        bf.move(self.base)
        bf.set_const(self.base + MARKER, 1)
        bf.move(self.base + MARKER)
        bf.emit("[" + self._read_line_body() + "]")

        # The walking loop exits one record to the right of the actual zero end
        # sentinel.  Move to that sentinel, then walk back to the permanent left
        # sentinel and finally restore the known compile-time base anchor.
        bf.emit("<" * RECORD_STRIDE)
        bf.emit(self._reverse_to_left_sentinel_code())
        bf.emit(">" * RECORD_STRIDE)
        bf.ptr = self.base

    def write_all_bytes(self, bf: BFEmitter) -> None:
        """Replay every stored byte to stdout with one forward BF walker."""
        self._check_layout()

        bf.move(self.base + MARKER)
        # Each iteration starts on a materialized marker and ends on the next
        # record marker.  The zero sentinel terminates the same static loop body.
        bf.emit("[")
        bf.emit(">.")
        bf.emit(">" * (RECORD_STRIDE - 1))
        bf.emit("]")

        # Entry is now the zero end sentinel.  Return to the static base anchor
        # so subsequent compiler/runtime code can again use absolute addresses.
        bf.emit(self._reverse_to_left_sentinel_code())
        bf.emit(">" * RECORD_STRIDE)
        bf.ptr = self.base


__all__ = [
    "RECORD_STRIDE",
    "MARKER",
    "PAYLOAD0",
    "PAYLOAD_BYTES",
    "RuntimeByteSequence",
]
