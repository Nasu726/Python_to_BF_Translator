"""Experimental base-4/bit-pair backend.

Each 64-bit word uses 32 lanes plus a sentinel::

    [marker, bit0, bit1] * 32 + [sentinel, 0, 0]

The value is still binary at the bit level; grouping two bits lets one marker
cell service two bits.  The key Brainfuck optimization is that corresponding
lanes in A/B/D have constant offsets.  A single loop body therefore walks all
32 base-4 digits at runtime instead of Python emitting 32 copies of the full
adder.  Marker cells are disposable control state; value bits are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfcore import BFEmitter


DIGITS = 32
STRIDE = 3
WORD_CELLS = (DIGITS + 1) * STRIDE
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Quad64Ref:
    base: int

    def marker(self, digit: int) -> int:
        return self.base + STRIDE * digit

    def bit0(self, digit: int) -> int:
        return self.base + STRIDE * digit + 1

    def bit1(self, digit: int) -> int:
        return self.base + STRIDE * digit + 2

    def bit(self, bit_index: int) -> int:
        digit, within = divmod(bit_index, 2)
        return self.bit0(digit) if within == 0 else self.bit1(digit)


class _RelativeBuilder:
    """Build one loop body using offsets relative to the current A marker."""

    def __init__(self) -> None:
        self.pos = 0
        self.parts: list[str] = []

    def move(self, target: int) -> None:
        delta = target - self.pos
        if delta > 0:
            self.parts.append('>' * delta)
        elif delta < 0:
            self.parts.append('<' * -delta)
        self.pos = target

    def emit(self, code: str) -> None:
        self.parts.append(code)

    def clear(self, cell: int) -> None:
        self.move(cell)
        self.emit('[-]')

    def add(self, cell: int, amount: int) -> None:
        self.move(cell)
        self.emit('+' * amount if amount >= 0 else '-' * -amount)

    def transfer(self, src: int, dst: int) -> None:
        """Destructively add a 0/1 source into dst."""
        self.move(src)
        self.emit('[-')
        self.move(dst)
        self.emit('+')
        self.move(src)
        self.emit(']')

    def code(self) -> str:
        return ''.join(self.parts)


def _map_total(builder: _RelativeBuilder, total: int, out: int, carry: int) -> None:
    """Map total in 0..3 to parity and floor(total/2), consuming total."""
    r = builder
    r.move(total)
    r.emit('[')
    r.emit('-')
    r.add(out, 1)
    r.move(total)
    r.emit('[')
    r.emit('-')
    r.add(out, -1)
    r.add(carry, 1)
    r.move(total)
    r.emit('[')
    r.emit('-')
    r.add(out, 1)
    r.move(total)
    r.emit(']')
    r.move(total)
    r.emit(']')
    r.move(total)
    r.emit(']')


def _add_preserved(builder: _RelativeBuilder, src: int, total: int, tmp: int) -> None:
    """total += src for a Boolean src, restoring src and leaving tmp zero."""
    r = builder
    r.move(src)
    r.emit('[')
    r.emit('-')
    r.add(total, 1)
    r.add(tmp, 1)
    r.move(src)
    r.emit(']')
    r.move(tmp)
    r.emit('[')
    r.emit('-')
    r.add(src, 1)
    r.move(tmp)
    r.emit(']')


def _add_not_preserved(
    builder: _RelativeBuilder,
    src: int,
    total: int,
    tmp: int,
    helper: int,
) -> None:
    """total += (1-src), preserving Boolean src; tmp/helper finish zero."""
    r = builder

    # Copy src into tmp, restoring src through helper.
    r.move(src)
    r.emit('[')
    r.emit('-')
    r.add(tmp, 1)
    r.add(helper, 1)
    r.move(src)
    r.emit(']')
    r.move(helper)
    r.emit('[')
    r.emit('-')
    r.add(src, 1)
    r.move(helper)
    r.emit(']')

    # tmp ^= 1, using helper as the one-shot flag.
    r.add(helper, 1)
    r.move(tmp)
    r.emit('[')
    r.emit('-')
    r.clear(helper)
    r.move(tmp)
    r.emit(']')
    r.move(helper)
    r.emit('[')
    r.emit('-')
    r.add(tmp, 1)
    r.move(helper)
    r.emit(']')

    r.transfer(tmp, total)


class Quad64Core:
    def __init__(self, bf: BFEmitter) -> None:
        self.bf = bf

    def set_u64(self, dst: Quad64Ref, value: int) -> None:
        value &= MASK64
        for digit in range(DIGITS):
            self.bf.clear(dst.marker(digit))
            self.bf.set_const(dst.bit0(digit), (value >> (2 * digit)) & 1)
            self.bf.set_const(dst.bit1(digit), (value >> (2 * digit + 1)) & 1)
        # Sentinel lane is always zero.
        self.bf.clear(dst.marker(DIGITS))
        self.bf.clear(dst.base + STRIDE * DIGITS + 1)
        self.bf.clear(dst.base + STRIDE * DIGITS + 2)

    def _prepare_binary_op(self, dst: Quad64Ref, a: Quad64Ref, b: Quad64Ref) -> None:
        for digit in range(DIGITS):
            # A markers drive the walking loop.  B markers are local temporary
            # cells.  D markers carry state between adjacent digits.
            self.bf.set_const(a.marker(digit), 1)
            self.bf.clear(b.marker(digit))
            self.bf.clear(dst.marker(digit))
            self.bf.clear(dst.bit0(digit))
            self.bf.clear(dst.bit1(digit))
        self.bf.clear(a.marker(DIGITS))
        self.bf.clear(b.marker(DIGITS))
        self.bf.clear(dst.marker(DIGITS))

    def _emit_add_body(self, b_delta: int, d_delta: int) -> str:
        # Relative lane layout:
        # A: 0 marker/temp, 1 a0, 2 a1
        # B: b_delta marker/tmp, +1 b0, +2 b1
        # D: d_delta carry-in, +1 d0, +2 d1
        r = _RelativeBuilder()
        total = 0
        a0, a1 = 1, 2
        tmp = b_delta
        b0, b1 = b_delta + 1, b_delta + 2
        carry_in = d_delta
        d0, d1 = d_delta + 1, d_delta + 2
        carry_out = d_delta + STRIDE

        r.clear(total)  # consume current A marker
        r.transfer(carry_in, total)
        _add_preserved(r, a0, total, tmp)
        _add_preserved(r, b0, total, tmp)
        _map_total(r, total, d0, tmp)  # B marker becomes low-bit carry

        r.transfer(tmp, total)
        _add_preserved(r, a1, total, tmp)
        _add_preserved(r, b1, total, tmp)
        _map_total(r, total, d1, carry_out)

        r.move(STRIDE)  # matching ] tests the next A marker
        return r.code()

    def _emit_sub_body(self, b_delta: int, d_delta: int) -> str:
        r = _RelativeBuilder()
        total = 0
        a0, a1 = 1, 2
        tmp = b_delta
        b0, b1 = b_delta + 1, b_delta + 2
        carry_in = d_delta
        d0, d1 = d_delta + 1, d_delta + 2
        carry_out = d_delta + STRIDE

        r.clear(total)
        r.transfer(carry_in, total)
        _add_preserved(r, a0, total, tmp)
        _add_not_preserved(r, b0, total, tmp, d0)
        _map_total(r, total, d0, tmp)

        r.transfer(tmp, total)
        _add_preserved(r, a1, total, tmp)
        _add_not_preserved(r, b1, total, tmp, d1)
        _map_total(r, total, d1, carry_out)

        r.move(STRIDE)
        return r.code()

    def add64(self, dst: Quad64Ref, a: Quad64Ref, b: Quad64Ref) -> None:
        """dst = a+b modulo 2**64, preserving all value bits of a and b."""
        self._prepare_binary_op(dst, a, b)
        self.bf.move(a.marker(0))
        body = self._emit_add_body(b.base - a.base, dst.base - a.base)
        self.bf.emit('[' + body + ']')
        # The body advances one lane each runtime iteration; after 32 passes
        # the actual pointer is at A's sentinel, not merely one stride ahead.
        self.bf.ptr = a.marker(DIGITS)
        self.bf.clear(dst.marker(DIGITS))  # discard unsigned overflow

    def sub64(self, dst: Quad64Ref, a: Quad64Ref, b: Quad64Ref) -> None:
        """dst = a-b modulo 2**64 via a + ~b + 1."""
        self._prepare_binary_op(dst, a, b)
        self.bf.set_const(dst.marker(0), 1)  # initial +1 carry
        self.bf.move(a.marker(0))
        body = self._emit_sub_body(b.base - a.base, dst.base - a.base)
        self.bf.emit('[' + body + ']')
        self.bf.ptr = a.marker(DIGITS)
        self.bf.clear(dst.marker(DIGITS))
