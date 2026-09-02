from bfcore import BFEmitter, Int64Ref
from bf_runtime import run_bf
from bfstrings import BinaryStringIO, StringRef


def test_literal_copy_print_and_length():
    bf = BFEmitter()
    backend = BinaryStringIO(bf, scratch_base=300)
    a = StringRef(0, 12)
    b = StringRef(20, 12)
    length = Int64Ref(40)
    backend.set_string_literal(a, "hello")
    backend.copy_string(b, a)
    backend.print_string(b, control=110)
    backend.string_length(length, b, control=111)
    result = run_bf(bf.code(), memory_size=512)
    assert result.output == "hello"
    assert sum((result.memory[length.base + i] & 1) << i for i in range(64)) == 5


def test_read_line_truncates_and_drains():
    bf = BFEmitter()
    backend = BinaryStringIO(bf, scratch_base=300)
    first = StringRef(0, 4)
    second = StringRef(10, 4)
    backend.read_line(first, workspace_base=100)
    backend.read_line(second, workspace_base=120)
    backend.print_string(first, control=140)
    backend.print_char(ord('|'), 141)
    backend.print_string(second, control=142)
    result = run_bf(bf.code(), "abcdef\nxy\n", memory_size=512)
    assert result.output == "abcd|xy"


def test_string_equality():
    bf = BFEmitter()
    backend = BinaryStringIO(bf, scratch_base=300)
    a, b, c = StringRef(0, 8), StringRef(10, 8), StringRef(20, 8)
    backend.set_string_literal(a, "nasu")
    backend.set_string_literal(b, "nasu")
    backend.set_string_literal(c, "nasa")
    backend.eq_string(40, a, b)
    backend.eq_string(41, a, c)
    result = run_bf(bf.code(), memory_size=512)
    assert result.memory[40] == 1
    assert result.memory[41] == 0
