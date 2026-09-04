"""Range-safe compact indexing for string-backed character-list views.

``compiler_charconv`` establishes the source-level view semantics. This layer
makes indexing logically range-safe and source-compact.

The correctness-first implementation selected a runtime index by emitting one
absolute-address comparison for every possible string slot. At capacity 255 a
single store produced tens of megabytes of Brainfuck because every candidate
repeated long pointer travel to shared scratch. The current implementation uses
preserving rotations instead:

    rotate payload left index times -> operate on slot 0 -> rotate right back

Each rotation body contains only adjacent destructive transfers and is emitted
once; runtime index affects execution count, not generated source size. The
payload and explicit terminator are restored exactly except for the requested
one-character replacement.

The currently supported character-list view permits element replacement but no
append/delete/insert. Its logical length is therefore immutable between
``list(input())`` assignments. We cache that length once in a persistent Quad64
word and reuse it for every later index check and ``len(character_list)`` call
instead of rescanning up to 255 string cells on every access.
"""

from __future__ import annotations

import ast

from bfstrings import StringRef
from compiler_charconv import CompileError, _is_list_input
from compiler_charconv import PythonToBFStream as _BasePythonToBFStream


def _must_char_value_names(tree: ast.AST, char_lists: set[str]) -> set[str]:
    """Return names whose every write is provably a one-character string.

    The earlier compatibility layer used an existential rule: once a name had
    ever received ``chars[i]`` it remained classified as one-character forever.
    That is unsound after a later assignment such as ``tmp = "XY"``.  Character
    stores would then silently use only ``tmp[0]``.

    This conservative must-analysis records every simple assignment / for-loop
    write that the current compiler understands and compares that count with all
    ``Store`` occurrences for the name.  A name qualifies only when every write
    is accounted for and every producer is itself known to be one character.
    """
    stores: dict[str, int] = {}
    writes: dict[str, list[tuple[str, ast.AST]]] = {}
    accounted: dict[str, int] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1

        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            writes.setdefault(name, []).append(("assign", node.value))
            accounted[name] = accounted.get(name, 0) + 1
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            name = node.target.id
            writes.setdefault(name, []).append(("assign", node.value))
            accounted[name] = accounted.get(name, 0) + 1
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            writes.setdefault(name, []).append(("for", node.iter))
            accounted[name] = accounted.get(name, 0) + 1

    candidates = {
        name
        for name, producers in writes.items()
        if name not in char_lists
        and producers
        and accounted.get(name, 0) == stores.get(name, 0)
    }

    result: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name in candidates - result:
            one_char = True
            for kind, producer in writes[name]:
                if kind == "for":
                    ok = (
                        isinstance(producer, ast.Name)
                        and producer.id in char_lists
                    )
                else:
                    ok = (
                        isinstance(producer, ast.Constant)
                        and isinstance(producer.value, str)
                        and len(producer.value) == 1
                    ) or (
                        isinstance(producer, ast.Subscript)
                        and isinstance(producer.value, ast.Name)
                        and producer.value.id in char_lists
                    ) or (
                        isinstance(producer, ast.Name)
                        and producer.id in result
                    )
                if not ok:
                    one_char = False
                    break
            if one_char:
                result.add(name)
                changed = True

    return result


class PythonToBFStream(_BasePythonToBFStream):
    """Character-view compiler with safe rotation-based indexing."""

    def __init__(
        self,
        tree: ast.Module,
        *,
        string_capacity: int = 255,
        list_capacity: int = 64,
    ) -> None:
        super().__init__(
            tree,
            string_capacity=string_capacity,
            list_capacity=list_capacity,
        )

        # Tighten the compatibility layer's provisional character-name
        # inference before statement lowering starts.  Type/layout inference has
        # already happened, but this set only controls whether a later mutable
        # character-list store is semantically permitted.
        self.char_value_names = _must_char_value_names(tree, self.char_list_names)

        # These words are allocated before statement lowering begins. Every
        # compile_stmt mark therefore starts above them, so statement-local temp
        # rewinds cannot reclaim the cached metadata. compiler_layout later
        # places its PeakTempArena above this persistent prefix as well.
        self.char_list_lengths = {
            name: self._new_word(0) for name in sorted(self.char_list_names)
        }
        self._char_list_name_by_base = {
            self.strings[name].base: name for name in self.char_list_names
        }

    def _new_char_buffer(self) -> StringRef:
        ref = StringRef(self.temps.top, 1)
        self.temps.top += ref.cells
        return ref

    def _cached_char_list_length(self, ref):
        name = self._char_list_name_by_base.get(ref.base)
        if name is None:
            return None
        return self.char_list_lengths[name]

    def compile_expr(self, node: ast.AST):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.char_list_names
        ):
            return self._copy_new(self.char_list_lengths[node.args[0].id])
        return super().compile_expr(node)

    def _char_list_runtime_index_byte(self, node: ast.AST, ref) -> int:
        length = self._cached_char_list_length(ref)
        if length is None:
            return super()._char_list_runtime_index_byte(node, ref)

        raw = self.compile_expr(node)
        index = self._new_word()
        self.backend.copy64(index, raw)
        zero = self._new_word(0)

        # Python negative indexing: normalize once by the cached logical length.
        negative = self.temps.cell()
        self.backend.slt64(negative, index, zero)
        self.bf.begin_while(negative)
        self.bf.add_const(negative, -1)
        summed = self._new_word()
        self.backend.add64(summed, index, length)
        self.backend.copy64(index, summed)
        self.bf.end_while(negative)

        valid = self.temps.cell()
        still_negative = self.temps.cell()
        self.backend.slt64(valid, index, length)
        self.backend.slt64(still_negative, index, zero)
        self.bf.begin_while(still_negative)
        self.bf.add_const(still_negative, -1)
        self.bf.clear(valid)
        self.bf.end_while(still_negative)

        out = self._index_word_to_byte(index)
        invalid = self.temps.cell()
        invalid_tmp = self.temps.cell()
        self._flag_not(invalid, valid, invalid_tmp)
        self.bf.begin_while(invalid)
        self.bf.add_const(invalid, -1)
        # 255 is outside every valid StringRef payload slot (0..254).
        self.bf.set_const(out, 255)
        self.bf.end_while(invalid)
        return out

    def _rotate_payload_left_once(self, ref) -> None:
        """Rotate all payload bytes left by one, preserving terminator zero."""
        saved = ref.terminator
        self.bf.clear(saved)

        # Move old slot 0 into the adjacent-after-payload saved cell. The long
        # pointer distance is emitted once in this reusable runtime body.
        first = ref.char(0)
        self.bf.begin_while(first)
        self.bf.add_const(first, -1)
        self.bf.add_const(saved, 1)
        self.bf.end_while(first)

        # Every destination is zero because its previous contents were consumed
        # by the preceding transfer, so no per-slot scratch or clear is needed.
        for i in range(ref.capacity - 1):
            dst = ref.char(i)
            src = ref.char(i + 1)
            self.bf.begin_while(src)
            self.bf.add_const(src, -1)
            self.bf.add_const(dst, 1)
            self.bf.end_while(src)

        tail = ref.char(ref.capacity - 1)
        self.bf.begin_while(saved)
        self.bf.add_const(saved, -1)
        self.bf.add_const(tail, 1)
        self.bf.end_while(saved)

    def _rotate_payload_right_once(self, ref) -> None:
        """Rotate all payload bytes right by one, preserving terminator zero."""
        saved = ref.terminator
        self.bf.clear(saved)

        tail = ref.char(ref.capacity - 1)
        self.bf.begin_while(tail)
        self.bf.add_const(tail, -1)
        self.bf.add_const(saved, 1)
        self.bf.end_while(tail)

        for i in range(ref.capacity - 1, 0, -1):
            dst = ref.char(i)
            src = ref.char(i - 1)
            self.bf.begin_while(src)
            self.bf.add_const(src, -1)
            self.bf.add_const(dst, 1)
            self.bf.end_while(src)

        first = ref.char(0)
        self.bf.begin_while(saved)
        self.bf.add_const(saved, -1)
        self.bf.add_const(first, 1)
        self.bf.end_while(saved)

    def _rotation_controls(self, index_byte: int) -> tuple[int, int, int]:
        """Return left-count, right-count, and one-shot valid access gate."""
        left_turns = self.temps.cell()
        right_turns = self.temps.cell()
        invalid = self.temps.cell()
        valid = self.temps.cell()
        valid_tmp = self.temps.cell()

        self.backend.copy_cell(index_byte, left_turns, self.backend.s0)
        self.backend.copy_cell(index_byte, right_turns, self.backend.s0)
        self.backend._eq_byte_const(invalid, index_byte, 255)
        self._flag_not(valid, invalid, valid_tmp)
        return left_turns, right_turns, valid

    def _rotate_left_n(self, ref, turns: int) -> None:
        self.bf.begin_while(turns)
        self.bf.add_const(turns, -1)
        self._rotate_payload_left_once(ref)
        self.bf.end_while(turns)

    def _rotate_right_n(self, ref, turns: int) -> None:
        self.bf.begin_while(turns)
        self.bf.add_const(turns, -1)
        self._rotate_payload_right_once(ref)
        self.bf.end_while(turns)

    def _load_char_list_subscript(self, node: ast.Subscript):
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]

        # Constants intentionally share the same logical range-check path as
        # runtime indices. In particular, 256 must become the invalid sentinel
        # rather than wrapping to physical slot 0 when narrowed to one byte.
        index_byte = self._char_list_runtime_index_byte(node.slice, ref)
        left_turns, right_turns, valid = self._rotation_controls(index_byte)
        result = self._new_char_buffer()
        self.backend.clear_string(result)

        self.bf.begin_while(valid)
        self.bf.add_const(valid, -1)
        self._rotate_left_n(ref, left_turns)
        self.backend.copy_cell(ref.char(0), result.char(0), self.backend.s0)
        self._rotate_right_n(ref, right_turns)
        self.bf.end_while(valid)
        self.backend._clear_scratch()
        return result

    def _store_char_list_subscript(self, node: ast.Subscript, value) -> None:
        assert isinstance(node.value, ast.Name)
        ref = self.strings[node.value.id]

        index_byte = self._char_list_runtime_index_byte(node.slice, ref)
        left_turns, right_turns, valid = self._rotation_controls(index_byte)

        self.bf.begin_while(valid)
        self.bf.add_const(valid, -1)
        self._rotate_left_n(ref, left_turns)
        self.backend.copy_cell(value.char(0), ref.char(0), self.backend.s0)
        self._rotate_right_n(ref, right_turns)
        self.bf.end_while(valid)
        self.backend._clear_scratch()

    def _compile_stmt_inner(self, node: ast.stmt) -> None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.char_list_names
            and _is_list_input(node.value)
        ):
            name = node.targets[0].id
            ref = self.strings[name]
            self.backend.read_line(ref, self.workspace_base)
            self.backend.string_length(
                self.char_list_lengths[name], ref, self.temps.cell()
            )
            return

        return super()._compile_stmt_inner(node)


__all__ = ["CompileError", "PythonToBFStream"]
