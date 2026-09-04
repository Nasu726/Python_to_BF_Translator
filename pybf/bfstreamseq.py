"""Runtime-sized sequential tape records with source-size-independent growth.

This is the scalable raw-byte storage vertical slice used before public Python
container routing. Runtime record count is not a compiler parameter: one fixed
Brainfuck record loop grows and revisits tape storage at runtime.

Records are now eight-byte chunks with a dormant cursor carrier::

    [marker][back][count][length-carrier:4][payload:8]
    [cursor-value]

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
from functools import lru_cache

from bfcore import BFEmitter
from bfpacked import PackedU32Core, PackedU32Ref
from bfpacked64 import PackedI64Ref


RECORD_STRIDE = 16
MARKER = 0
BACK = 1
COUNT = 2
LENGTH = 3
LENGTH_BYTES = 4
PAYLOAD0 = 7
PAYLOAD_BYTES = 8
CURSOR_VALUE = 15
CURSOR_POSITION = LENGTH + LENGTH_BYTES - 1

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


def _is_zero_u32(
    r: _RelativeBuilder,
    base: int,
    result: int,
    tmp: int,
    helper: int,
) -> None:
    """``result = (packed_u32(base) == 0)`` while preserving the value."""
    r.set_const(result, 1)
    for byte_index in range(LENGTH_BYTES):
        r.copy_preserved(base + byte_index, tmp, helper)
        r.move(tmp)
        r.emit("[")
        r.clear(tmp)
        r.clear(result)
        r.move(tmp)
        r.emit("]")


def _decrement_u32(
    r: _RelativeBuilder,
    base: int,
    borrow: int,
    gate: int,
    tmp: int,
    helper: int,
) -> None:
    """Decrement a nonzero packed u32 using caller-selected zero scratch."""
    r.set_const(borrow, 1)
    for byte_index in range(LENGTH_BYTES):
        r.clear(gate)
        r.transfer(borrow, gate)
        r.move(gate)
        r.emit("[")
        r.add(gate, -1)
        cell = base + byte_index
        _set_zero_flag(r, borrow, cell, tmp, helper)
        r.add(cell, -1)
        r.move(gate)
        r.emit("]")
    for cell in (borrow, gate, tmp, helper):
        r.clear(cell)


def _copy_cell_preserved(bf: BFEmitter, src: int, dst: int, tmp: int) -> None:
    """Copy one fixed-address byte while preserving ``src`` and zeroing tmp."""
    bf.clear(dst)
    bf.clear(tmp)
    bf.begin_while(src)
    bf.add_const(src, -1)
    bf.add_const(dst, 1)
    bf.add_const(tmp, 1)
    bf.end_while(src)
    bf.begin_while(tmp)
    bf.add_const(tmp, -1)
    bf.add_const(src, 1)
    bf.end_while(tmp)


def _set_zero_flag_fixed(
    bf: BFEmitter,
    result: int,
    src: int,
    tmp: int,
    helper: int,
) -> None:
    """Fixed-address counterpart of ``_set_zero_flag``."""
    bf.set_const(result, 1)
    _copy_cell_preserved(bf, src, tmp, helper)
    bf.begin_while(tmp)
    bf.clear(tmp)
    bf.clear(result)
    bf.end_while(tmp)


def _extract_packed_sign(
    bf: BFEmitter,
    src: int,
    byte: int,
    quotient: int,
    parity: int,
    gate: int,
) -> None:
    """Extract bit 7 of ``src`` into ``byte``, preserving ``src``."""
    _copy_cell_preserved(bf, src, byte, quotient)

    # Seven bounded divisions by two leave floor(src / 128), which is exactly
    # the sign bit of the packed int64's most-significant byte.
    for _ in range(7):
        bf.clear(quotient)
        bf.clear(parity)
        bf.begin_while(byte)
        bf.add_const(byte, -1)
        bf.set_const(gate, 1)
        bf.begin_while(parity)
        bf.add_const(parity, -1)
        bf.clear(gate)
        bf.add_const(quotient, 1)
        bf.end_while(parity)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.add_const(parity, 1)
        bf.end_while(gate)
        bf.end_while(byte)
        bf.begin_while(quotient)
        bf.add_const(quotient, -1)
        bf.add_const(byte, 1)
        bf.end_while(quotient)
        bf.clear(parity)
        bf.clear(gate)


def _subtract_u32_consuming(
    bf: BFEmitter,
    dst: PackedU32Ref,
    subtrahend: PackedU32Ref,
    *,
    scratch_base: int,
) -> None:
    """``dst -= subtrahend``; map underflow to ``0xffffffff``.

    The subtrahend is scratch and is consumed. Work is bounded by four byte
    values (at most 1,020 decrement iterations), never by the represented u32.
    """
    borrow = scratch_base
    gate = scratch_base + 1
    tmp = scratch_base + 2
    helper = scratch_base + 3
    bf.clear(borrow)

    for byte_index in range(4):
        dst_byte = dst.byte(byte_index)
        sub_byte = subtrahend.byte(byte_index)

        # Apply the incoming borrow once and replace it with this byte's
        # outgoing borrow.
        bf.clear(gate)
        bf.begin_while(borrow)
        bf.add_const(borrow, -1)
        bf.add_const(gate, 1)
        bf.end_while(borrow)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        _set_zero_flag_fixed(bf, borrow, dst_byte, tmp, helper)
        bf.add_const(dst_byte, -1)
        bf.end_while(gate)

        # A byte-sized amount can cross zero at most once. Preserve that fact
        # in ``borrow`` while consuming the magnitude byte.
        bf.begin_while(sub_byte)
        bf.add_const(sub_byte, -1)
        _set_zero_flag_fixed(bf, gate, dst_byte, tmp, helper)
        bf.add_const(dst_byte, -1)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.set_const(borrow, 1)
        bf.end_while(gate)
        bf.end_while(sub_byte)

    bf.begin_while(borrow)
    bf.add_const(borrow, -1)
    for byte_index in range(4):
        bf.set_const(dst.byte(byte_index), 0xFF)
    bf.end_while(borrow)
    for cell in range(scratch_base, scratch_base + 4):
        bf.clear(cell)


@lru_cache(maxsize=2)
def _locate_index_body(*, load: bool) -> str:
    """Find a non-negative byte index with one runtime record walker.

    Entry is a materialized record marker. ``LENGTH`` carries the remaining
    byte index. On a hit, the following record's marker/back-link are cleared
    temporarily so the outer walker exits there. A load leaves the selected
    byte in the target record's first length cell; a store-location pass leaves
    ``lane + 1`` there for the later value-carrying pass.
    """
    r = _RelativeBuilder()
    next_remaining = RECORD_STRIDE + LENGTH
    next_valid_gate = next_remaining + 1
    next_work = next_remaining + 2
    next_helper = next_remaining + 3
    next_back = RECORD_STRIDE + BACK
    next_marker = RECORD_STRIDE + MARKER

    # BACK==2 marks record zero only while the locator starts. Restore its
    # persistent BACK==0 before any possible return walk; all later records use
    # the normal BACK==1 invariant.
    _set_equal_const(
        r,
        next_remaining,
        BACK,
        2,
        next_valid_gate,
        next_work,
    )
    r.move(next_remaining)
    r.emit("[")
    r.add(next_remaining, -1)
    r.clear(BACK)
    r.move(next_remaining)
    r.emit("]")

    r.copy_preserved(COUNT, next_remaining, next_helper)

    # COUNT is at most eight, so statically expanding only the lane selector
    # keeps source size independent of both capacity and runtime input length.
    for lane_index in range(PAYLOAD_BYTES):
        # Execute this lane once iff it is part of the current (possibly
        # partial) chunk. Clearing the copied numeric gate makes it one-shot.
        r.copy_preserved(next_remaining, next_valid_gate, next_helper)
        r.move(next_valid_gate)
        r.emit("[")
        r.clear(next_valid_gate)
        r.add(next_remaining, -1)

        # MARKER remains one until an earlier lane finds the target. This extra
        # gate prevents any later lane from observing the repurposed counter.
        r.copy_preserved(MARKER, next_work, next_helper)
        r.move(next_work)
        r.emit("[")
        r.clear(next_work)

        _is_zero_u32(
            r,
            LENGTH,
            next_work,
            next_valid_gate,
            next_helper,
        )

        # next_valid_gate = not next_work. next_back is a known one after every
        # materialized record, so it may be borrowed as the copy helper and
        # immediately restored.
        r.set_const(next_valid_gate, 1)
        r.copy_preserved(next_work, next_helper, next_back)
        r.set_const(next_back, 1)
        r.move(next_helper)
        r.emit("[")
        r.clear(next_helper)
        r.clear(next_valid_gate)
        r.move(next_helper)
        r.emit("]")

        # A nonzero remaining index consumes exactly one valid byte. The four
        # scratch cells are all local and next_back is restored afterwards.
        r.move(next_valid_gate)
        r.emit("[")
        r.clear(next_valid_gate)
        r.clear(next_back)
        _decrement_u32(
            r,
            LENGTH,
            next_valid_gate,
            next_helper,
            next_work,
            next_back,
        )
        r.set_const(next_back, 1)
        r.move(next_valid_gate)
        r.emit("]")

        # Zero means this lane is the requested byte. The current marker is a
        # temporary "still searching" flag; it is restored before the body
        # exits. Clearing the following marker/back stops the outer walker.
        r.move(next_work)
        r.emit("[")
        r.clear(next_work)
        if load:
            r.copy_preserved(PAYLOAD0 + lane_index, LENGTH, next_valid_gate)
        else:
            r.set_const(LENGTH, lane_index + 1)
        r.clear(MARKER)
        r.clear(next_remaining)
        r.clear(next_back)
        r.clear(next_marker)
        r.move(next_work)
        r.emit("]")

        r.move(next_work)
        r.emit("]")
        r.move(next_valid_gate)
        r.emit("]")

    # No lane matched: clean the borrowed next-record cells and carry the
    # remaining index forward. On a hit MARKER is zero, so the result/lane tag
    # remains in this record instead.
    r.move(MARKER)
    r.emit("[")
    r.add(MARKER, -1)
    for cell in range(next_remaining, next_helper + 1):
        r.clear(cell)
    for byte_index in range(LENGTH_BYTES):
        r.transfer(LENGTH + byte_index, next_remaining + byte_index)
    r.move(MARKER)
    r.emit("]")
    r.set_const(MARKER, 1)

    r.move(next_marker)
    return r.code()


@lru_cache(maxsize=2)
def _finish_location_code(*, carry_loaded_byte: bool) -> str:
    """Restore scan metadata and return from the runtime record to base."""
    r = _RelativeBuilder()

    # A hit uniquely clears BACK on the record where the outer scan stopped.
    # Empty input instead retains the temporary BACK==2 marker and a natural
    # nonempty end sentinel has BACK==1.
    r.clear(LENGTH + 3)
    _set_zero_flag(r, LENGTH, BACK, LENGTH + 1, LENGTH + 2)
    r.move(LENGTH)
    r.emit("[")
    r.add(LENGTH, -1)
    r.set_const(BACK, 1)
    if carry_loaded_byte:
        r.transfer(-RECORD_STRIDE + LENGTH, LENGTH + 3)
    r.move(LENGTH)
    r.emit("]")

    # A temporarily cleared next marker is restored iff this is another data
    # record. COUNT==0 identifies the real end sentinel.
    _set_zero_flag(r, LENGTH, COUNT, LENGTH + 1, LENGTH + 2)
    r.set_const(MARKER, 1)
    r.move(LENGTH)
    r.emit("[")
    r.add(LENGTH, -1)
    r.clear(MARKER)
    r.move(LENGTH)
    r.emit("]")

    # Empty input never entered the locator body, so normalize its temporary
    # base BACK marker before the common reverse walk.
    _set_equal_const(r, LENGTH, BACK, 2, LENGTH + 1, LENGTH + 2)
    r.move(LENGTH)
    r.emit("[")
    r.add(LENGTH, -1)
    r.clear(BACK)
    r.move(LENGTH)
    r.emit("]")

    # Every processed record left its LENGTH scratch zero. Carrying via the
    # fourth lane therefore returns a loaded byte without touching persistent
    # payload or the fixed length stored in the separate left sentinel.
    r.move(BACK)
    r.emit("[")
    if carry_loaded_byte:
        r.transfer(LENGTH + 3, -RECORD_STRIDE + LENGTH + 3)
    r.move(-RECORD_STRIDE + BACK)
    r.emit("]")
    r.emit("<")
    return r.code()


@lru_cache(maxsize=1)
def _store_located_body() -> str:
    """Carry a source byte forward and replace the tagged payload lane."""
    r = _RelativeBuilder()
    next_marker = RECORD_STRIDE + MARKER

    for lane_index in range(PAYLOAD_BYTES):
        _set_equal_const(
            r,
            LENGTH + 2,
            LENGTH,
            lane_index + 1,
            LENGTH + 3,
            MARKER,
        )
        r.set_const(MARKER, 1)
        r.move(LENGTH + 2)
        r.emit("[")
        r.clear(LENGTH + 2)
        r.clear(PAYLOAD0 + lane_index)
        r.transfer(LENGTH + 1, PAYLOAD0 + lane_index)
        r.move(LENGTH + 2)
        r.emit("]")

    # A nonzero lane tag is unique. Consume it and stop at the following
    # record. A missing/out-of-range location carries the source to the natural
    # end sentinel without modifying any payload.
    r.copy_preserved(LENGTH, LENGTH + 2, LENGTH + 3)
    r.move(LENGTH + 2)
    r.emit("[")
    r.clear(LENGTH + 2)
    r.clear(LENGTH)
    r.clear(next_marker)
    r.move(LENGTH + 2)
    r.emit("]")

    r.transfer(LENGTH + 1, RECORD_STRIDE + LENGTH + 1)
    r.move(next_marker)
    return r.code()


@lru_cache(maxsize=1)
def _finish_store_code() -> str:
    """Restore the second store pass and return to the fixed sequence base."""
    r = _RelativeBuilder()
    _set_zero_flag(r, LENGTH + 2, COUNT, LENGTH, LENGTH + 3)
    r.set_const(MARKER, 1)
    r.move(LENGTH + 2)
    r.emit("[")
    r.add(LENGTH + 2, -1)
    r.clear(MARKER)
    r.move(LENGTH + 2)
    r.emit("]")
    for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
        r.clear(cell)

    r.move(BACK)
    r.emit("[" + "<" * RECORD_STRIDE + "]")
    r.emit("<")
    return r.code()


@lru_cache(maxsize=2)
def _cursor_seek_code(direction: int) -> str:
    """Consume a local packed record delta while retaining a mobile cursor.

    The code starts and finishes at ``CURSOR_POSITION`` in whichever runtime
    record is current. Each loop iteration moves the value carrier and the
    remaining delta by exactly one record. The literal BF loop is therefore
    source-size independent of both the sequence length and represented delta.

    ``direction`` is +1 or -1. The caller must have range-normalized the
    relative record coordinate; walking beyond either sequence boundary is not
    part of this deliberately low-level cursor ABI.
    """
    if direction not in (-1, 1):
        raise ValueError("cursor direction must be -1 or 1")

    r = _RelativeBuilder(CURSOR_POSITION)
    neighbor = direction * RECORD_STRIDE

    # A valid byte index occupies at most 29 record-index bits after division
    # by eight. Bit 31 of the high counter byte is therefore free and doubles
    # as the mobile loop flag, avoiding another per-record control cell.
    # The following record is materialized or the zero sentinel and has clean
    # LENGTH scratch, so it can host the initial zero-test in either direction.
    probe = RECORD_STRIDE + LENGTH
    _is_zero_u32(r, LENGTH, probe, probe + 1, probe + 2)
    r.set_const(probe + 3, 1)
    r.move(probe)
    r.emit("[")
    r.add(probe, -1)
    r.clear(probe + 3)
    r.move(probe)
    r.emit("]")
    r.move(probe + 3)
    r.emit("[")
    r.add(probe + 3, -1)
    r.add(CURSOR_POSITION, 128)
    r.move(probe + 3)
    r.emit("]")

    r.move(CURSOR_POSITION)
    r.emit("[")
    r.add(CURSOR_POSITION, -128)

    # The destination LENGTH cells are known-zero cursor scratch. Borrow them
    # for a packed decrement, then use them as the next mobile counter.
    neighbor_length = neighbor + LENGTH
    _decrement_u32(
        r,
        LENGTH,
        neighbor_length,
        neighbor_length + 1,
        neighbor_length + 2,
        neighbor_length + 3,
    )
    for byte_index in range(LENGTH_BYTES):
        r.transfer(LENGTH + byte_index, neighbor_length + byte_index)

    # A value loaded earlier in the session follows the physical cursor.
    r.transfer(CURSOR_VALUE, neighbor + CURSOR_VALUE)

    # Re-arm at the destination iff another record step remains. The source
    # record's LENGTH cells are zero after the transfer and form local scratch.
    _is_zero_u32(r, neighbor_length, LENGTH, LENGTH + 1, LENGTH + 2)
    r.set_const(LENGTH + 3, 1)
    r.move(LENGTH)
    r.emit("[")
    r.add(LENGTH, -1)
    r.clear(LENGTH + 3)
    r.move(LENGTH)
    r.emit("]")
    r.move(LENGTH + 3)
    r.emit("[")
    r.add(LENGTH + 3, -1)
    r.add(neighbor + CURSOR_POSITION, 128)
    r.move(LENGTH + 3)
    r.emit("]")
    r.move(neighbor + CURSOR_POSITION)
    r.emit("]")

    # The loop body is translation-invariant: on each back-edge the physical
    # pointer is the control field of the new current record. Reset only the
    # builder's compile-time coordinate for code emitted after the loop.
    r.pos = CURSOR_POSITION
    return r.code()


@lru_cache(maxsize=1)
def _cursor_read_u32_code() -> str:
    r = _RelativeBuilder(CURSOR_POSITION)
    for byte_index in range(LENGTH_BYTES):
        r.clear(LENGTH + byte_index)
        r.move(LENGTH + byte_index)
        r.emit(",")
    r.move(CURSOR_POSITION)
    return r.code()


@lru_cache(maxsize=1)
def _cursor_load_lane_code() -> str:
    """Read a lane byte and load that payload into the mobile value."""
    r = _RelativeBuilder(CURSOR_POSITION)
    r.clear(LENGTH)
    r.move(LENGTH)
    r.emit(",")
    r.clear(CURSOR_VALUE)

    for lane_index in range(PAYLOAD_BYTES):
        _set_equal_const(
            r,
            CURSOR_POSITION,
            LENGTH,
            lane_index,
            LENGTH + 1,
            LENGTH + 2,
        )
        r.move(CURSOR_POSITION)
        r.emit("[")
        r.add(CURSOR_POSITION, -1)
        r.copy_preserved(PAYLOAD0 + lane_index, CURSOR_VALUE, LENGTH + 1)
        r.move(CURSOR_POSITION)
        r.emit("]")

    for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
        r.clear(cell)
    r.move(CURSOR_POSITION)
    return r.code()


@lru_cache(maxsize=1)
def _cursor_store_lane_code() -> str:
    """Read a lane byte and store the preserved mobile value into that lane."""
    r = _RelativeBuilder(CURSOR_POSITION)
    r.clear(LENGTH)
    r.move(LENGTH)
    r.emit(",")

    for lane_index in range(PAYLOAD_BYTES):
        _set_equal_const(
            r,
            CURSOR_POSITION,
            LENGTH,
            lane_index,
            LENGTH + 1,
            LENGTH + 2,
        )
        r.move(CURSOR_POSITION)
        r.emit("[")
        r.add(CURSOR_POSITION, -1)
        r.copy_preserved(CURSOR_VALUE, PAYLOAD0 + lane_index, LENGTH + 1)
        r.move(CURSOR_POSITION)
        r.emit("]")

    for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
        r.clear(cell)
    r.move(CURSOR_POSITION)
    return r.code()


@lru_cache(maxsize=1)
def _cursor_exchange_lane_code() -> str:
    """Swap the mobile value with an input-selected payload lane."""
    r = _RelativeBuilder(CURSOR_POSITION)
    r.clear(LENGTH)
    r.move(LENGTH)
    r.emit(",")

    for lane_index in range(PAYLOAD_BYTES):
        _set_equal_const(
            r,
            CURSOR_POSITION,
            LENGTH,
            lane_index,
            LENGTH + 1,
            LENGTH + 2,
        )
        r.move(CURSOR_POSITION)
        r.emit("[")
        r.add(CURSOR_POSITION, -1)
        r.clear(LENGTH + 1)
        r.transfer(PAYLOAD0 + lane_index, LENGTH + 1)
        r.transfer(CURSOR_VALUE, PAYLOAD0 + lane_index)
        r.transfer(LENGTH + 1, CURSOR_VALUE)
        r.move(CURSOR_POSITION)
        r.emit("]")

    for cell in range(LENGTH, LENGTH + LENGTH_BYTES):
        r.clear(cell)
    r.move(CURSOR_POSITION)
    return r.code()


@lru_cache(maxsize=1)
def _cursor_return_to_base_code() -> str:
    """Carry the mobile value left and finish at the fixed base marker."""
    r = _RelativeBuilder(CURSOR_POSITION)
    r.move(BACK)
    r.emit("[")
    r.transfer(CURSOR_VALUE, -RECORD_STRIDE + CURSOR_VALUE)
    r.move(-RECORD_STRIDE + BACK)
    r.emit("]")
    r.pos = BACK
    r.move(MARKER)
    return r.code()


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

    @property
    def _fixed_access_tmp(self) -> int:
        # COUNT is unused in the permanent left sentinel. Keeping the copy
        # helper here leaves its eight payload cells available for signed-index
        # normalization state.
        return self.left_sentinel + COUNT

    @property
    def _normalized_index(self) -> PackedU32Ref:
        return PackedU32Ref(self.left_sentinel + PAYLOAD0)

    @property
    def _normalization_scratch_base(self) -> int:
        return self.left_sentinel + PAYLOAD0 + 4

    def _clear_normalization_workspace(self, bf: BFEmitter) -> None:
        for cell in range(
            self.left_sentinel + PAYLOAD0,
            self.left_sentinel + RECORD_STRIDE,
        ):
            bf.clear(cell)

    def _normalize_signed_index(
        self,
        bf: BFEmitter,
        index: PackedI64Ref,
    ) -> PackedU32Ref:
        """Normalize a packed signed-int64 index into the internal u32 ABI.

        Negative values use ``len + index`` exactly once. Values still outside
        ``[0, len)`` are represented as ``0xffffffff``; the S1c locator already
        maps that value to empty-load/no-op-store for every possible u32 length.
        The signed source is preserved.
        """
        normalized = self._normalized_index
        magnitude = PackedU32Ref(self.base + LENGTH)
        scratch = self._normalization_scratch_base
        s0, s1, s2, s3 = range(scratch, scratch + 4)
        packed = PackedU32Core(bf, scratch)

        self._clear_normalization_workspace(bf)
        for byte_index in range(4):
            bf.clear(magnitude.byte(byte_index))

        _extract_packed_sign(bf, index.byte(7), s0, s1, s2, s3)
        bf.set_const(s1, 1)  # non-negative gate

        bf.begin_while(s0)
        bf.add_const(s0, -1)
        bf.clear(s1)

        # Default negative result is the guaranteed out-of-range sentinel.
        for byte_index in range(4):
            bf.set_const(normalized.byte(byte_index), 0xFF)

        # Only -1..-(2**32-1) can possibly normalize into a u32-length
        # sequence. Their upper four bytes are all 0xff and their low word is
        # nonzero. This also rejects -2**32 before its low-word negation wraps.
        bf.set_const(s0, 1)
        for byte_index in range(4, 8):
            _copy_cell_preserved(bf, index.byte(byte_index), s2, s3)
            bf.add_const(s2, 1)
            bf.begin_while(s2)
            bf.clear(s2)
            bf.clear(s0)
            bf.end_while(s2)

        bf.set_const(s1, 1)  # low word is zero until a nonzero byte is seen
        for byte_index in range(4):
            _copy_cell_preserved(bf, index.byte(byte_index), s2, s3)
            bf.begin_while(s2)
            bf.clear(s2)
            bf.clear(s1)
            bf.end_while(s2)
        bf.begin_while(s1)
        bf.add_const(s1, -1)
        bf.clear(s0)
        bf.end_while(s1)

        bf.begin_while(s0)
        bf.add_const(s0, -1)

        # magnitude = two's-complement negate(low_u32(index)).
        for byte_index in range(4):
            out = magnitude.byte(byte_index)
            bf.set_const(out, 0xFF)
            _copy_cell_preserved(bf, index.byte(byte_index), s1, s2)
            bf.begin_while(s1)
            bf.add_const(s1, -1)
            bf.add_const(out, -1)
            bf.end_while(s1)
        packed.increment(magnitude)

        packed.copy(normalized, self.length_ref)
        _subtract_u32_consuming(
            bf,
            normalized,
            magnitude,
            scratch_base=scratch,
        )
        bf.end_while(s0)
        bf.end_while(s0)

        # Non-negative values are valid u32 candidates only when their upper
        # four bytes are zero. Range against length remains the locator's job.
        bf.begin_while(s1)
        bf.add_const(s1, -1)
        for byte_index in range(4):
            bf.set_const(normalized.byte(byte_index), 0xFF)
        bf.set_const(s0, 1)
        for byte_index in range(4, 8):
            _copy_cell_preserved(bf, index.byte(byte_index), s2, s3)
            bf.begin_while(s2)
            bf.clear(s2)
            bf.clear(s0)
            bf.end_while(s2)
        bf.begin_while(s0)
        bf.add_const(s0, -1)
        for byte_index in range(4):
            _copy_cell_preserved(
                bf,
                index.byte(byte_index),
                normalized.byte(byte_index),
                s2,
            )
        bf.end_while(s0)
        bf.end_while(s1)

        for cell in range(scratch, scratch + 4):
            bf.clear(cell)
        bf.move(self.base)
        return normalized

    def _copy_index_to_base(self, bf: BFEmitter, index: PackedU32Ref) -> None:
        for byte_index in range(LENGTH_BYTES):
            _copy_cell_preserved(
                bf,
                index.byte(byte_index),
                self.base + LENGTH + byte_index,
                self._fixed_access_tmp,
            )

    def _locate_index(self, bf: BFEmitter, index: PackedU32Ref, *, load: bool) -> None:
        self._copy_index_to_base(bf, index)

        # BACK==2 distinguishes an empty base sentinel from the temporary
        # BACK==0 stop marker used for a successful lookup. The first data-body
        # iteration restores the normal base BACK==0 invariant.
        bf.set_const(self.base + BACK, 2)
        bf.move(self.base + MARKER)
        bf.emit("[" + _locate_index_body(load=load) + "]")
        bf.emit(_finish_location_code(carry_loaded_byte=load))
        bf.ptr = self.base

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

    def load_byte(self, bf: BFEmitter, dst: int, index: PackedU32Ref) -> None:
        """Load a non-negative runtime index, preserving the sequence/index.

        ``dst`` receives zero when ``index >= len(sequence)``. The index and
        destination must be fixed cells outside this runtime sequence's record
        area. Runtime work is linear in the traversed chunk distance while
        emitted source size is independent of sequence length and index value.
        """
        self._check_layout()
        self._locate_index(bf, index, load=True)

        bf.clear(dst)
        result = self.base + LENGTH + 3
        bf.begin_while(result)
        bf.add_const(result, -1)
        bf.add_const(dst, 1)
        bf.end_while(result)
        bf.move(self.base)

    def store_byte(self, bf: BFEmitter, index: PackedU32Ref, src: int) -> None:
        """Replace a non-negative runtime index while preserving ``src``.

        Out-of-range stores are no-ops. The first pass locates and tags one
        payload lane; the second carries ``src`` forward without combining a
        fifth carrier byte with the packed-u32 index. Both passes are runtime
        walkers whose emitted source is independent of sequence length.
        """
        self._check_layout()
        self._locate_index(bf, index, load=False)

        _copy_cell_preserved(
            bf,
            src,
            self.base + LENGTH + 1,
            self._fixed_access_tmp,
        )
        bf.move(self.base + MARKER)
        bf.emit("[" + _store_located_body() + "]")
        bf.emit(_finish_store_code())
        bf.ptr = self.base

    def open_cursor(self, bf: BFEmitter) -> "RuntimeByteCursor":
        """Open a scoped physical-record cursor at the sequence base.

        The returned cursor emits only translation-invariant, record-relative
        code until ``finish`` returns to the fixed base. Callers must not emit
        ordinary fixed-address ``BFEmitter`` operations while the cursor is
        open. Relative record deltas must already be range-normalized; this
        internal ABI is intended for sequential or safely batched lowering,
        while ``load_byte`` / ``store_byte`` remain the range-safe public
        random-access primitives.
        """
        return RuntimeByteCursor(self, bf)

    def load_byte_signed(self, bf: BFEmitter, dst: int, index: PackedI64Ref) -> None:
        """Load using Python-style signed negative-index normalization."""
        self._check_layout()
        normalized = self._normalize_signed_index(bf, index)
        self.load_byte(bf, dst, normalized)
        self._clear_normalization_workspace(bf)
        bf.move(self.base)

    def store_byte_signed(self, bf: BFEmitter, index: PackedI64Ref, src: int) -> None:
        """Store using Python-style signed negative-index normalization."""
        self._check_layout()
        normalized = self._normalize_signed_index(bf, index)
        self.store_byte(bf, normalized, src)
        self._clear_normalization_workspace(bf)
        bf.move(self.base)


class RuntimeByteCursor:
    """Scoped mobile frame for cursor-relative byte-sequence operations.

    The physical BF pointer may be at a runtime-dependent record while this
    object is active. Its methods are consequently the only supported emitter
    operations inside the scope. ``seek_from_input`` consumes two packed-u32
    record deltas (forward, then backward); canonical callers encode at most
    one as nonzero. A valid delta is at most ``0x1fffffff`` because a u32 byte
    index addresses eight-byte records. Lane operations consume a following
    raw lane byte in ``0..7``.

    ``begin_input_loop`` / ``end_input_loop`` implement a mobile loop whose
    one-byte continuation flag is read both before the first iteration and
    after every iteration. The single cursor carrier doubles as this loop flag
    between iterations and as the loaded byte inside an iteration. This keeps
    the record stride small while retaining the physical cursor between
    queries.
    """

    def __init__(self, sequence: RuntimeByteSequence, bf: BFEmitter) -> None:
        sequence._check_layout()
        self.sequence = sequence
        self.bf = bf
        self._active = True
        self._input_loop_open = False

        bf.clear(sequence.base + CURSOR_VALUE)
        for byte_index in range(LENGTH_BYTES):
            bf.clear(sequence.base + LENGTH + byte_index)
        bf.move(sequence.base + CURSOR_POSITION)

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("runtime byte cursor is already finished")

    def _emit(self, code: str) -> None:
        self._require_active()
        self.bf.emit(code)
        # This is a logical coordinate only. At runtime the pointer is the same
        # field in an unknown record until finish() walks back to the base.
        self.bf.ptr = self.sequence.base + CURSOR_POSITION

    def begin_input_loop(self) -> None:
        """Read the first 0/1 continuation flag and begin a mobile loop."""
        self._require_active()
        if self._input_loop_open:
            raise RuntimeError("runtime byte cursor input loop is already open")
        r = _RelativeBuilder(CURSOR_POSITION)
        r.clear(CURSOR_VALUE)
        r.move(CURSOR_VALUE)
        r.emit(",[")
        r.clear(CURSOR_VALUE)
        r.move(CURSOR_POSITION)
        self._emit(r.code())
        self._input_loop_open = True

    def end_input_loop(self) -> None:
        """Read the next continuation flag and close the mobile loop."""
        self._require_active()
        if not self._input_loop_open:
            raise RuntimeError("runtime byte cursor input loop is not open")
        r = _RelativeBuilder(CURSOR_POSITION)
        # A query body may leave a loaded/exchanged byte in the carrier. It is
        # dead at the iteration boundary where the next continuation flag is
        # read, so clear it before reusing the same cell as the mobile loop bit.
        r.clear(CURSOR_VALUE)
        r.move(CURSOR_VALUE)
        r.emit(",]")
        r.move(CURSOR_POSITION)
        self._emit(r.code())
        self._input_loop_open = False

    def seek_forward_from_input(self) -> None:
        """Read and consume a packed-u32 forward record delta."""
        self._emit(_cursor_read_u32_code())
        self._emit(_cursor_seek_code(1))

    def seek_backward_from_input(self) -> None:
        """Read and consume a packed-u32 backward record delta."""
        self._emit(_cursor_read_u32_code())
        self._emit(_cursor_seek_code(-1))

    def seek_from_input(self) -> None:
        """Read forward/backward deltas; at most one is canonically nonzero."""
        self.seek_forward_from_input()
        self.seek_backward_from_input()

    def load_lane_from_input(self) -> None:
        """Read a lane byte and load it into the mobile value."""
        self._emit(_cursor_load_lane_code())

    def store_lane_from_input(self) -> None:
        """Read a lane byte and store the preserved mobile value."""
        self._emit(_cursor_store_lane_code())

    def exchange_lane_from_input(self) -> None:
        """Exchange the mobile value with an input-selected payload lane."""
        self._emit(_cursor_exchange_lane_code())

    def output_value(self) -> None:
        """Write the mobile value byte without consuming it."""
        r = _RelativeBuilder(CURSOR_POSITION)
        r.move(CURSOR_VALUE)
        r.emit(".")
        r.move(CURSOR_POSITION)
        self._emit(r.code())

    def clear_value(self) -> None:
        r = _RelativeBuilder(CURSOR_POSITION)
        r.clear(CURSOR_VALUE)
        r.move(CURSOR_POSITION)
        self._emit(r.code())

    def finish(self, *, dst: int | None = None) -> None:
        """Return to the fixed base, optionally copying out the mobile value."""
        self._require_active()
        if self._input_loop_open:
            raise RuntimeError("finish() called with an open cursor input loop")

        self._emit(_cursor_return_to_base_code())
        self.bf.ptr = self.sequence.base

        field = self.sequence.base + CURSOR_VALUE
        if dst is not None:
            if dst == field:
                raise ValueError("cursor destination must be outside its frame")
            _copy_cell_preserved(
                self.bf,
                field,
                dst,
                self.sequence._fixed_access_tmp,
            )
        self.bf.clear(field)
        self.bf.move(self.sequence.base)
        self._active = False


__all__ = [
    "RECORD_STRIDE",
    "MARKER",
    "BACK",
    "COUNT",
    "LENGTH",
    "LENGTH_BYTES",
    "PAYLOAD0",
    "PAYLOAD_BYTES",
    "CURSOR_VALUE",
    "CURSOR_POSITION",
    "RuntimeByteSequence",
    "RuntimeByteCursor",
]
