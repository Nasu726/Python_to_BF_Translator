"""Fixed-capacity int64 list primitives for the Brainfuck runtime.

List payloads stay packed as eight little-endian bytes, but every slot also
contains a tiny amount of traversal metadata.  That metadata lets runtime
indices walk from slot to slot with *one emitted Brainfuck loop body* instead
of Python statically emitting one candidate branch per list-capacity slot.

Layout::

    [length]
    [walk back target payload:8 result:8] * capacity

The extra cells trade a small amount of tape for a major source-size win:
``get_dynamic``/``set_dynamic``/``append`` are independent of list capacity
apart from marker initialization and the numeric bounds check.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import Int64Ref, WORD_BITS
from bfpacked64 import I64_BYTES, PackedI64Core, PackedI64Ref
from bfpackedtokens import PackedBinaryTokenIO


SLOT_WALK = 0
SLOT_BACK = 1
SLOT_TARGET = 2
SLOT_PAYLOAD = 3
SLOT_RESULT = SLOT_PAYLOAD + I64_BYTES
SLOT_STRIDE = SLOT_RESULT + I64_BYTES


@dataclass(frozen=True)
class IntListRef:
    base: int
    capacity: int

    @property
    def length_cell(self) -> int:
        return self.base

    def slot(self, index: int) -> int:
        if not 0 <= index < self.capacity:
            raise IndexError(index)
        return self.base + 1 + index * SLOT_STRIDE

    def walk_cell(self, index: int) -> int:
        return self.slot(index) + SLOT_WALK

    def back_cell(self, index: int) -> int:
        return self.slot(index) + SLOT_BACK

    def target_cell(self, index: int) -> int:
        return self.slot(index) + SLOT_TARGET

    def item(self, index: int) -> PackedI64Ref:
        return PackedI64Ref(self.slot(index) + SLOT_PAYLOAD)

    def result(self, index: int) -> PackedI64Ref:
        return PackedI64Ref(self.slot(index) + SLOT_RESULT)

    @property
    def cells(self) -> int:
        return 1 + self.capacity * SLOT_STRIDE


class _RelativeBuilder:
    """Build a lane-walking BF loop with coordinates relative to one slot."""

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

    def transfer(self, src: int, dst: int) -> None:
        self.move(src)
        self.parts.append("[")
        self.parts.append("-")
        self.move(dst)
        self.parts.append("+")
        self.move(src)
        self.parts.append("]")

    def copy_preserved(self, src: int, dst: int, tmp: int) -> None:
        self.clear(dst)
        self.clear(tmp)
        self.move(src)
        self.parts.append("[")
        self.parts.append("-")
        self.add(dst, 1)
        self.add(tmp, 1)
        self.move(src)
        self.parts.append("]")
        self.move(tmp)
        self.parts.append("[")
        self.parts.append("-")
        self.add(src, 1)
        self.move(tmp)
        self.parts.append("]")

    def code(self) -> str:
        return "".join(self.parts)


class BinaryListIO(PackedBinaryTokenIO):
    """Integer-list operations over packed values and runtime slot walkers."""

    def __init__(self, bf, scratch_base: int) -> None:
        super().__init__(bf, scratch_base=scratch_base)
        self.packed64 = PackedI64Core(bf, scratch_base)

    def copy64(self, dst, src) -> None:
        """Compatibility bridge between arithmetic words and packed list slots."""
        if isinstance(dst, PackedI64Ref):
            if isinstance(src, PackedI64Ref):
                self.packed64.copy(dst, src)
                return
            if isinstance(src, Int64Ref):
                self.packed64.from_int64(dst, src)
                return
        if isinstance(dst, Int64Ref) and isinstance(src, PackedI64Ref):
            self.packed64.to_int64(dst, src)
            return
        super().copy64(dst, src)

    # ------------------------------------------------------------------
    # list layout / invariants
    # ------------------------------------------------------------------
    def clear_list(self, ref: IntListRef) -> None:
        """Reset length and establish persistent reverse-walk markers."""
        bf = self.bf
        bf.clear(ref.length_cell)
        for i in range(ref.capacity):
            bf.clear(ref.walk_cell(i))
            bf.clear(ref.target_cell(i))
            bf.set_const(ref.back_cell(i), 0 if i == 0 else 1)
            self.packed64.clear(ref.result(i))

    def set_list_literal(self, ref: IntListRef, values: list[int]) -> None:
        if len(values) > ref.capacity:
            raise ValueError("list literal exceeds allocated list capacity")
        self.clear_list(ref)
        self.bf.set_const(ref.length_cell, len(values))
        for i, value in enumerate(values):
            self.packed64.set_u64(ref.item(i), value)

    def copy_list(self, dst: IntListRef, src: IntListRef) -> None:
        if dst.capacity < src.capacity:
            raise ValueError("destination list capacity is smaller than source")
        self.clear_list(dst)
        self.copy_cell(src.length_cell, dst.length_cell, self.s0)
        for i in range(src.capacity):
            self.packed64.copy(dst.item(i), src.item(i))
        self._clear_scratch()

    def list_length(self, dst: Int64Ref, ref: IntListRef, tmp: int) -> None:
        """Convert the compact byte length to a normal uint64 word."""
        bf = self.bf
        self._clear_word(dst)
        self.copy_cell(ref.length_cell, tmp, self.s0)
        bf.begin_while(tmp)
        bf.add_const(tmp, -1)
        self._inc64_inplace(dst)
        bf.end_while(tmp)
        self._clear_scratch()

    def get_const(self, dst: Int64Ref, ref: IntListRef, index: int) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.packed64.to_int64(dst, ref.item(index))

    def set_const(self, ref: IntListRef, index: int, value: Int64Ref) -> None:
        if not 0 <= index < ref.capacity:
            raise IndexError(index)
        self.packed64.from_int64(ref.item(index), value)

    # ------------------------------------------------------------------
    # source-compact runtime slot walk
    # ------------------------------------------------------------------
    def _int64_low_byte(self, dst: int, src: Int64Ref) -> None:
        """Pack src bits 0..7 into one byte while preserving src."""
        bf = self.bf
        bf.clear(dst)
        for bit_index in range(8):
            self.copy_cell(src.bit(bit_index), self.s0, self.s1)
            bf.begin_while(self.s0)
            bf.add_const(self.s0, -1)
            bf.add_const(dst, 1 << bit_index)
            bf.end_while(self.s0)
        self._clear_scratch()

    def _arm_walk_from_byte(self, ref: IntListRef, index_byte: int) -> None:
        """Set slot-0 walk counter to index+1 (valid indices are <=254)."""
        self.copy_cell(index_byte, ref.walk_cell(0), self.s0)
        self.bf.add_const(ref.walk_cell(0), 1)

    def _walk_body(self, *, write: bool) -> str:
        """One runtime lane body; starts at current WALK and ends at next WALK."""
        r = _RelativeBuilder(initial_pos=SLOT_WALK)

        # The counter stores remaining_index+1.  After the first decrement,
        # zero means this is the selected slot.
        r.add(SLOT_WALK, -1)
        r.clear(SLOT_TARGET)
        r.add(SLOT_TARGET, 1)

        # Non-target path.  Consume the remaining counter into the next slot,
        # carry the packed value/result corridor forward once, and clear the
        # target flag.  Repeating the inner loop for counter>1 only moves zeros
        # after the first transfer, which keeps the emitted source tiny.
        r.move(SLOT_WALK)
        r.parts.append("[")
        r.add(SLOT_WALK, -1)
        r.add(SLOT_STRIDE + SLOT_WALK, 1)
        r.clear(SLOT_TARGET)
        for i in range(I64_BYTES):
            r.transfer(SLOT_RESULT + i, SLOT_STRIDE + SLOT_RESULT + i)
        r.move(SLOT_WALK)
        r.parts.append("]")

        # Selected-slot operation. WALK is zero here and is safe as the
        # temporary cell required by preserved byte copies.
        r.move(SLOT_TARGET)
        r.parts.append("[")
        r.add(SLOT_TARGET, -1)
        for i in range(I64_BYTES):
            if write:
                r.copy_preserved(SLOT_RESULT + i, SLOT_PAYLOAD + i, SLOT_WALK)
                r.clear(SLOT_RESULT + i)
            else:
                r.copy_preserved(SLOT_PAYLOAD + i, SLOT_RESULT + i, SLOT_WALK)
        r.move(SLOT_TARGET)
        r.parts.append("]")

        r.move(SLOT_STRIDE + SLOT_WALK)
        return r.code()

    def _reverse_result_body(self) -> str:
        """Move selected RESULT one slot left; starts/ends on BACK markers."""
        r = _RelativeBuilder(initial_pos=0)
        current_result = SLOT_RESULT - SLOT_BACK
        previous_result = current_result - SLOT_STRIDE
        for i in range(I64_BYTES):
            r.transfer(current_result + i, previous_result + i)
        r.move(-SLOT_STRIDE)
        return r.code()

    def _return_to_first(self, ref: IntListRef, *, carry_result: bool) -> None:
        """Return from the dynamic target to slot 0 using persistent BACK bits.

        The forward walker exits on the WALK cell immediately after the target.
        Moving back by ``SLOT_STRIDE-SLOT_BACK`` lands on the target BACK bit.
        BACK is zero for slot 0 and one for every later slot, so a loop whose
        body ends on the previous BACK bit naturally walks to a fixed address.
        """
        bf = self.bf
        bf.emit("<" * (SLOT_STRIDE - SLOT_BACK))
        bf.emit("[")
        if carry_result:
            bf.emit(self._reverse_result_body())
        else:
            bf.emit("<" * SLOT_STRIDE)
        bf.emit("]")
        bf.ptr = ref.back_cell(0)

    def _run_walk(self, ref: IntListRef, *, write: bool) -> None:
        bf = self.bf
        bf.move(ref.walk_cell(0))
        bf.emit("[")
        bf.emit(self._walk_body(write=write))
        bf.emit("]")
        # The runtime pointer is now at the WALK cell after the dynamic target;
        # BFEmitter cannot infer that address, so return with a relative lane
        # walk before restoring its compile-time pointer model.
        self._return_to_first(ref, carry_result=not write)

    def get_dynamic(
        self,
        dst: Int64Ref,
        ref: IntListRef,
        index: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Load a runtime index with one emitted slot-walking body."""
        bf = self.bf
        index_byte = workspace_word.base

        self.set_u64(workspace_word, ref.capacity)
        self.slt64(match, index, workspace_word)
        self._int64_low_byte(index_byte, index)

        bf.begin_while(match)
        bf.add_const(match, -1)
        self._arm_walk_from_byte(ref, index_byte)
        self._run_walk(ref, write=False)
        bf.end_while(match)

        self.packed64.to_int64(dst, ref.result(0))
        self.packed64.clear(ref.result(0))
        bf.clear(index_byte)
        bf.clear(match)
        self._clear_scratch()

    def set_dynamic(
        self,
        ref: IntListRef,
        index: Int64Ref,
        value: Int64Ref,
        workspace_word: Int64Ref,
        match: int,
    ) -> None:
        """Store through a runtime index with one emitted slot-walking body."""
        bf = self.bf
        index_byte = workspace_word.base

        self.set_u64(workspace_word, ref.capacity)
        self.slt64(match, index, workspace_word)
        self._int64_low_byte(index_byte, index)

        bf.begin_while(match)
        bf.add_const(match, -1)
        self.packed64.from_int64(ref.result(0), value)
        self._arm_walk_from_byte(ref, index_byte)
        self._run_walk(ref, write=True)
        bf.end_while(match)

        self.packed64.clear(ref.result(0))
        bf.clear(index_byte)
        bf.clear(match)
        self._clear_scratch()

    # ------------------------------------------------------------------
    # append / input
    # ------------------------------------------------------------------
    def append_packed(
        self,
        ref: IntListRef,
        value: PackedI64Ref,
        length_copy: int,
        match: int,
    ) -> None:
        """Append packed data through the same capacity-independent walker."""
        bf = self.bf

        # active = length != capacity.  A full fixed-capacity list remains
        # unchanged; callers such as input parsing continue consuming tokens.
        self._eq_byte_const(match, ref.length_cell, ref.capacity)
        self._toggle_bit(match, self.s0)
        self._clear_scratch()

        bf.begin_while(match)
        bf.add_const(match, -1)
        self.packed64.copy(ref.result(0), value)
        self.copy_cell(ref.length_cell, ref.walk_cell(0), self.s0)
        bf.add_const(ref.walk_cell(0), 1)
        self._run_walk(ref, write=True)
        bf.add_const(ref.length_cell, 1)
        bf.end_while(match)

        bf.clear(length_copy)
        bf.clear(match)
        self._clear_scratch()

    def append(
        self,
        ref: IntListRef,
        value: Int64Ref,
        length_copy: int,
        match: int,
        packed_tmp: PackedI64Ref | None = None,
    ) -> None:
        """Pack once, then append with the runtime lane walker."""
        if packed_tmp is None:
            packed_tmp = PackedI64Ref(match + 1)
        self.packed64.from_int64(packed_tmp, value)
        self.append_packed(ref, packed_tmp, length_copy, match)
        self.packed64.clear(packed_tmp)
        self._clear_scratch()

    def read_int_list_line(
        self,
        ref: IntListRef,
        workspace_base: int,
        active: int,
        gate: int,
        has_token: int,
        end_line: int,
    ) -> None:
        """Implement ``list(map(int, input().split()))`` with compact runtime loops.

        Decimal tokens are parsed directly into packed int64 bytes.  Appending
        a token no longer statically emits one branch per list-capacity slot;
        the same runtime lane walker handles every destination index.  Excess
        tokens are still consumed through the line terminator.
        """
        bf = self.bf
        self.clear_list(ref)
        for c in (active, gate, has_token, end_line):
            bf.clear(c)

        # Packed parser controls occupy workspace_base+128..+151.  Keep the
        # token and append scratch beyond that private region.
        token = PackedI64Ref(workspace_base + WORD_BITS * 2 + 32)
        length_copy = token.base + I64_BYTES
        append_match = length_copy + 1

        self.packed64.clear(token)
        bf.clear(length_copy)
        bf.clear(append_match)
        bf.set_const(active, 1)
        bf.begin_while(active)

        self.read_packed_s64_line_token(
            token,
            has_token,
            end_line,
            workspace_base,
        )

        self.copy_cell(has_token, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        self.append_packed(ref, token, length_copy, append_match)
        bf.end_while(gate)

        self.copy_cell(end_line, gate, self.s0)
        bf.begin_while(gate)
        bf.add_const(gate, -1)
        bf.clear(active)
        bf.end_while(gate)

        bf.end_while(active)
        self.packed64.clear(token)
        self._clear_scratch()


__all__ = ["IntListRef", "BinaryListIO", "SLOT_STRIDE"]
