"""Runtime-sized sequential tape records with source-size-independent growth.

This is the first scalable-storage vertical slice. It deliberately starts with
raw bytes rather than Python ``list[int]`` semantics so the hard Brainfuck
property can be tested in isolation:

* runtime record count is not a compiler parameter;
* one BF loop body advances across fixed-stride records;
* storage grows only in tape cells/runtime work, not emitted source size;
* a later pass can traverse the stored records again.

The record layout now reserves the metadata needed by the next chunked-storage
stage while S1a still stores one input byte per materialized record::

    [marker][back][count][length-carrier:4][payload:8]

``marker == 1`` means a materialized record. ``back`` is zero on record zero and
one on later armed records. ``count`` is currently one for a data record and
zero for the sentinel; S1b will use it for partial eight-byte chunks.

Runtime length is deliberately not updated at a distant fixed header on every
input byte. A four-byte little-endian length carrier moves forward with the
runtime walker, is incremented next to the current record, then is carried back
once after LF is reached. The final length lives in the permanent left-sentinel
record, whose marker remains zero and is therefore still a valid traversal
anchor.

Scratch lives ahead of the current record and is cleared before the walker
advances. The current input helper treats LF (``\n``) as the line terminator. It
is an internal scalability primitive, not yet the public ``input()`` lowering.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter
from bfpacked import PackedU32Ref


RECORD_STRIDE = 15
MARKER = 0
BACK = 1
COUNT = 2
LENGTH = 3
LENGTH_BYTES = 4
PAYLOAD0 = 7
PAYLOAD_BYTES = 8

# Rolling construction scratch uses unused lanes in the next, still-unmaterialized
# payload. These cells must finish zero before that record becomes current.
TMP = RECORD_STRIDE + PAYLOAD0 + 1
RESTORE = RECORD_STRIDE + PAYLOAD0 + 2
IS_LF = RECORD_STRIDE + PAYLOAD0 + 3
DATA_GATE = RECORD_STRIDE + PAYLOAD0 + 4

# Once the current length carrier has moved forward, its four cells are dead and
# can serve as local increment scratch without increasing persistent stride.
INC_CARRY = LENGTH
INC_GATE = LENGTH + 1
INC_TMP = LENGTH + 2
INC_RESTORE = LENGTH + 3


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
        """Destructively move one byte from ``src`` into a zero ``dst``."""
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


def _set_zero_flag(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    """``result = (src == 0)`` while preserving ``src``."""
    r.set_const(result, 1)
    r.copy_preserved(src, tmp, restore)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(result)
    r.move(tmp)
    r.emit("]")


def _increment_u32(r: _RelativeBuilder, base: int) -> None:
    """Increment a little-endian four-byte integer modulo 2**32.

    The current record's old carrier cells are zero before this helper runs, so
    they are reused as carry/gate/copy scratch. Runtime work is fixed at four
    byte lanes and does not depend on sequence length.
    """
    r.set_const(INC_CARRY, 1)
    for byte_index in range(LENGTH_BYTES):
        r.clear(INC_GATE)
        r.transfer(INC_CARRY, INC_GATE)
        r.move(INC_GATE)
        r.emit("[")
        r.add(INC_GATE, -1)
        cell = base + byte_index
        r.add(cell, 1)
        _set_zero_flag(r, INC_CARRY, cell, INC_TMP, INC_RESTORE)
        r.move(INC_GATE)
        r.emit("]")
    # Overflow beyond 32 bits is outside the intended contest-size range, but
    # keep the carrier scratch canonical even when wraparound occurs.
    for cell in (INC_CARRY, INC_GATE, INC_TMP, INC_RESTORE):
        r.clear(cell)


@dataclass(frozen=True)
class RuntimeByteSequence:
    """A one-shot runtime-grown contiguous sequence beginning at ``base``.

    ``base - RECORD_STRIDE`` is a permanent zero-marker left sentinel. Its four
    LENGTH cells hold the final runtime length after construction. Callers place
    ``base`` at least one full record after every compile-time static/temp cell.
    """

    base: int

    @property
    def left_sentinel(self) -> int:
        return self.base - RECORD_STRIDE

    @property
    def length_ref(self) -> PackedU32Ref:
        return PackedU32Ref(self.left_sentinel + LENGTH)

    def _check_layout(self) -> None:
        if self.left_sentinel < 0:
            raise ValueError("runtime sequence requires one record of left guard space")

    @staticmethod
    def _read_line_body() -> str:
        r = _RelativeBuilder()

        # Move the prefix length into the next record before current payload
        # construction. The current carrier becomes zero and is then scratch.
        for byte_index in range(LENGTH_BYTES):
            r.transfer(
                LENGTH + byte_index,
                RECORD_STRIDE + LENGTH + byte_index,
            )

        # S1a materializes one data byte per record. S1b will fill all eight
        # payload lanes while preserving the same carrier/header contract.
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

        # DATA_GATE = not IS_LF. LF turns the current pre-armed marker into the
        # zero end sentinel; data records their count and arm the next record.
        r.set_const(DATA_GATE, 1)
        r.move(IS_LF)
        r.emit("[")
        r.add(IS_LF, -1)
        r.clear(DATA_GATE)
        r.clear(MARKER)
        r.clear(COUNT)
        r.clear(PAYLOAD0)
        r.move(IS_LF)
        r.emit("]")

        r.move(DATA_GATE)
        r.emit("[")
        r.add(DATA_GATE, -1)
        r.set_const(COUNT, 1)
        _increment_u32(r, RECORD_STRIDE + LENGTH)
        r.set_const(RECORD_STRIDE + BACK, 1)
        r.set_const(RECORD_STRIDE + MARKER, 1)
        r.move(DATA_GATE)
        r.emit("]")

        # Current carrier/scratch must not survive as false persistent metadata.
        for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
            r.clear(cell)

        # Future-record cells used as rolling classification scratch must be
        # clean before that record becomes current.
        for cell in (TMP, RESTORE, IS_LF, DATA_GATE):
            r.clear(cell)

        r.move(RECORD_STRIDE + MARKER)
        return r.code()

    @staticmethod
    def _move_final_length_to_end_sentinel_code() -> str:
        """Entry: record one step right of the actual zero end sentinel."""
        r = _RelativeBuilder()
        for byte_index in range(LENGTH_BYTES):
            r.transfer(
                LENGTH + byte_index,
                -RECORD_STRIDE + LENGTH + byte_index,
            )
        r.move(-RECORD_STRIDE + MARKER)
        return r.code()

    @staticmethod
    def _return_length_to_left_sentinel_code() -> str:
        """Carry final length left once per record and stop on left marker zero.

        Entry is the actual zero end sentinel. The first transfer handles the
        empty sequence as well: record-zero sentinel length moves directly into
        the permanent left-sentinel metadata.
        """
        r = _RelativeBuilder()
        for byte_index in range(LENGTH_BYTES):
            r.transfer(
                LENGTH + byte_index,
                -RECORD_STRIDE + LENGTH + byte_index,
            )
        r.move(-RECORD_STRIDE + MARKER)
        r.emit("[")

        # Runtime loop iterations are relative to the newly reached record.
        r.pos = MARKER
        for byte_index in range(LENGTH_BYTES):
            r.transfer(
                LENGTH + byte_index,
                -RECORD_STRIDE + LENGTH + byte_index,
            )
        r.move(-RECORD_STRIDE + MARKER)
        r.emit("]")
        return r.code()

    @staticmethod
    def _reverse_to_left_sentinel_code() -> str:
        # Entry is a zero end sentinel. Step once to the preceding record; then
        # markers walk left until the permanent zero-marker sentinel is reached.
        step = "<" * RECORD_STRIDE
        return step + "[" + step + "]"

    def read_lf_terminated_bytes(self, bf: BFEmitter) -> None:
        """Materialize one LF-terminated input line into runtime records."""
        self._check_layout()

        # A sequence object is currently one-shot. The permanent metadata is
        # nevertheless explicitly initialized so the contract does not depend on
        # callers remembering which sentinel payload cells are meaningful.
        for byte_index in range(LENGTH_BYTES):
            bf.clear(self.length_ref.base + byte_index)

        bf.move(self.base)
        bf.clear(self.base + BACK)
        bf.clear(self.base + COUNT)
        bf.set_const(self.base + MARKER, 1)
        bf.move(self.base + MARKER)
        bf.emit("[" + self._read_line_body() + "]")

        # The one-byte S1a walker exits one record right of the zero sentinel.
        # First bring the final moving length into that sentinel, then carry it
        # left along the same record chain exactly once.
        bf.emit(self._move_final_length_to_end_sentinel_code())
        bf.emit(self._return_length_to_left_sentinel_code())
        bf.emit(">" * RECORD_STRIDE)
        bf.ptr = self.base

    def write_all_bytes(self, bf: BFEmitter) -> None:
        """Replay every stored byte to stdout with one forward BF walker."""
        self._check_layout()

        bf.move(self.base + MARKER)
        bf.emit("[")
        bf.emit(">" * PAYLOAD0)
        bf.emit(".")
        bf.emit(">" * (RECORD_STRIDE - PAYLOAD0))
        bf.emit("]")

        # Entry is the zero end sentinel. Return to the static base anchor so
        # subsequent compiler/runtime code can again use absolute addresses.
        bf.emit(self._reverse_to_left_sentinel_code())
        bf.emit(">" * RECORD_STRIDE)
        bf.ptr = self.base


__all__ = [
    "RECORD_STRIDE",
    "MARKER",
    "BACK",
    "COUNT",
    "LENGTH",
    "LENGTH_BYTES",
    "PAYLOAD0",
    "PAYLOAD_BYTES",
    "RuntimeByteSequence",
]
