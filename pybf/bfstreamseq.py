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
from functools import lru_cache

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
        # The left sentinel payload is not sequence data. Keep one fixed cell
        # for preserving copies made before/after a runtime-relative walk.
        return self.left_sentinel + PAYLOAD0

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
