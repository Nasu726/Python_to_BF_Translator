"""Runtime-sized sequential tape records with source-size-independent growth.

This is the scalable raw-byte storage vertical slice used before public Python
container routing. Runtime record count is not a compiler parameter: one fixed
Brainfuck record loop grows and revisits tape storage at runtime.

Records are now eight-byte chunks::

    [marker][back][count][length-carrier:4][payload:8]

``marker == 1`` means a materialized chunk, ``back`` is zero on record zero and
one on later records/sentinels, and ``count`` is the number of valid payload
bytes (1..8). The first zero marker is the end sentinel.

The four-byte little-endian runtime length is never updated through a distant
fixed address for every character. A carrier moves forward beside the runtime
walker, is incremented locally for each accepted byte, and is carried back once
after LF terminates the line. The final length lives in the permanent
left-sentinel record, whose marker remains zero.

One runtime loop fills up to eight payload lanes per chunk. The only statically
expanded selector has eight cases for the lane inside the current chunk; source
size therefore remains independent of runtime line length while persistent tape
distance drops from one record per character to one record per eight characters.

The current helper treats LF (``\n``) as line terminator. It is an internal
scalability primitive, not yet the public ``input()`` lowering.
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

# Rolling construction scratch occupies the payload of the next,
# still-unmaterialized record. All eight cells are scrubbed before that record
# can become current.
CH = RECORD_STRIDE + PAYLOAD0
TMP = CH + 1
RESTORE = CH + 2
ACTIVE = CH + 3
REMAINING = CH + 4
LANE = CH + 5
FLAG = CH + 6
GATE = CH + 7

# After the current length carrier moves forward, its four cells are zero and
# can serve as bounded local scratch for lane selection / packed increment.
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


def _set_equal_const(
    r: _RelativeBuilder,
    result: int,
    src: int,
    value: int,
    tmp: int,
    restore: int,
) -> None:
    """``result = (src == value)`` while preserving ``src``."""
    r.set_const(result, 1)
    r.copy_preserved(src, tmp, restore)
    r.add(tmp, -value)
    r.move(tmp)
    r.emit("[")
    r.clear(tmp)
    r.clear(result)
    r.move(tmp)
    r.emit("]")


def _set_zero_flag(
    r: _RelativeBuilder,
    result: int,
    src: int,
    tmp: int,
    restore: int,
) -> None:
    _set_equal_const(r, result, src, 0, tmp, restore)


def _increment_u32(r: _RelativeBuilder, base: int) -> None:
    """Increment a little-endian four-byte integer modulo 2**32."""
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
    for cell in (INC_CARRY, INC_GATE, INC_TMP, INC_RESTORE):
        r.clear(cell)


def _store_ch_in_selected_lane(r: _RelativeBuilder) -> None:
    """Move CH into payload[LANE], where LANE is statically bounded to 0..7."""
    for lane_index in range(PAYLOAD_BYTES):
        _set_equal_const(
            r,
            INC_CARRY,
            LANE,
            lane_index,
            INC_TMP,
            INC_RESTORE,
        )
        r.move(INC_CARRY)
        r.emit("[")
        r.add(INC_CARRY, -1)
        r.transfer(CH, PAYLOAD0 + lane_index)
        r.move(INC_CARRY)
        r.emit("]")


@dataclass(frozen=True)
class RuntimeByteSequence:
    """A one-shot runtime-grown contiguous byte sequence beginning at ``base``.

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

        # Carry the prefix length one record ahead. Current LENGTH becomes local
        # scratch; next LENGTH stays close to every per-byte increment.
        for byte_index in range(LENGTH_BYTES):
            r.transfer(
                LENGTH + byte_index,
                RECORD_STRIDE + LENGTH + byte_index,
            )

        r.clear(COUNT)
        r.set_const(ACTIVE, 1)
        r.set_const(REMAINING, PAYLOAD_BYTES)
        r.clear(LANE)

        # One emitted loop fills up to eight bytes. ACTIVE is re-armed only when
        # data was read and at least one lane remains; LF consumes no payload.
        r.move(ACTIVE)
        r.emit("[")
        r.add(ACTIVE, -1)
        r.move(CH)
        r.emit(",")

        _set_equal_const(r, FLAG, CH, ord("\n"), TMP, RESTORE)
        r.set_const(GATE, 1)
        r.move(FLAG)
        r.emit("[")
        r.add(FLAG, -1)
        r.clear(GATE)
        r.clear(CH)
        r.move(FLAG)
        r.emit("]")

        r.move(GATE)
        r.emit("[")
        r.add(GATE, -1)
        _store_ch_in_selected_lane(r)
        r.add(COUNT, 1)
        _increment_u32(r, RECORD_STRIDE + LENGTH)
        r.add(LANE, 1)
        r.add(REMAINING, -1)

        # ACTIVE = (REMAINING != 0).
        _set_zero_flag(r, FLAG, REMAINING, TMP, RESTORE)
        r.set_const(ACTIVE, 1)
        r.move(FLAG)
        r.emit("[")
        r.add(FLAG, -1)
        r.clear(ACTIVE)
        r.move(FLAG)
        r.emit("]")
        r.move(GATE)
        r.emit("]")
        r.move(ACTIVE)
        r.emit("]")

        # count == 0 occurs on an empty line or on the sentinel iteration after
        # an exactly full chunk. Turn that pre-armed current marker into zero.
        _set_zero_flag(r, FLAG, COUNT, TMP, RESTORE)
        r.move(FLAG)
        r.emit("[")
        r.add(FLAG, -1)
        r.clear(MARKER)
        r.move(FLAG)
        r.emit("]")

        # Any materialized current chunk gives the next record a back-link,
        # including a zero sentinel after a partial final chunk.
        _set_zero_flag(r, FLAG, COUNT, TMP, RESTORE)
        r.set_const(GATE, 1)
        r.move(FLAG)
        r.emit("[")
        r.add(FLAG, -1)
        r.clear(GATE)
        r.move(FLAG)
        r.emit("]")
        r.move(GATE)
        r.emit("[")
        r.add(GATE, -1)
        r.set_const(RECORD_STRIDE + BACK, 1)
        r.move(GATE)
        r.emit("]")

        # REMAINING == 0 means eight data bytes filled this chunk without LF;
        # only then is the next record armed for another runtime iteration.
        _set_zero_flag(r, FLAG, REMAINING, TMP, RESTORE)
        r.move(FLAG)
        r.emit("[")
        r.add(FLAG, -1)
        r.set_const(RECORD_STRIDE + MARKER, 1)
        r.move(FLAG)
        r.emit("]")

        for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
            r.clear(cell)
        for cell in range(CH, GATE + 1):
            r.clear(cell)

        r.move(RECORD_STRIDE + MARKER)
        return r.code()

    @staticmethod
    def _move_final_length_one_record_left_code() -> str:
        """Move the final carrier one record left and follow it to that marker."""
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
        """Carry final length left through materialized records to fixed metadata."""
        r = _RelativeBuilder()
        for byte_index in range(LENGTH_BYTES):
            r.transfer(
                LENGTH + byte_index,
                -RECORD_STRIDE + LENGTH + byte_index,
            )
        r.move(-RECORD_STRIDE + MARKER)
        r.emit("[")
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
        step = "<" * RECORD_STRIDE
        return step + "[" + step + "]"

    @staticmethod
    def _write_record_body() -> str:
        """Output COUNT payload bytes and advance to the next record marker."""
        r = _RelativeBuilder()
        remaining = LENGTH
        lane_gate = LENGTH + 1
        helper = LENGTH + 2

        r.copy_preserved(COUNT, remaining, helper)
        for lane_index in range(PAYLOAD_BYTES):
            r.copy_preserved(remaining, lane_gate, helper)
            r.move(lane_gate)
            r.emit("[")
            r.clear(lane_gate)
            r.move(PAYLOAD0 + lane_index)
            r.emit(".")
            r.add(remaining, -1)
            r.move(lane_gate)
            r.emit("]")

        for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
            r.clear(cell)
        r.move(RECORD_STRIDE + MARKER)
        return r.code()

    def read_lf_terminated_bytes(self, bf: BFEmitter) -> None:
        """Materialize one LF-terminated input line into runtime-sized chunks."""
        self._check_layout()

        for byte_index in range(LENGTH_BYTES):
            bf.clear(self.length_ref.base + byte_index)

        bf.move(self.base)
        bf.clear(self.base + BACK)
        bf.clear(self.base + COUNT)
        bf.set_const(self.base + MARKER, 1)
        bf.move(self.base + MARKER)
        bf.emit("[" + self._read_line_body() + "]")

        # For a partial final chunk the loop exits on its following zero sentinel;
        # for an exact multiple of eight it exits one record to the right of the
        # pre-armed sentinel iteration. An unconditional one-record carrier move
        # is valid in both cases, after which marker-guided propagation reaches
        # the permanent left-sentinel metadata.
        bf.emit(self._move_final_length_one_record_left_code())
        bf.emit(self._return_length_to_left_sentinel_code())
        bf.emit(">" * RECORD_STRIDE)
        bf.ptr = self.base

    def write_all_bytes(self, bf: BFEmitter) -> None:
        """Replay every stored byte to stdout with one forward chunk walker."""
        self._check_layout()

        bf.move(self.base + MARKER)
        bf.emit("[" + self._write_record_body() + "]")

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
