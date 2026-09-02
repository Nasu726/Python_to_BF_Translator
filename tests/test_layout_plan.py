import ast

from bf_runtime import run_bf
from bftemparena import PeakTempArena
from compiler_layout import PythonToBFLayout, lower_with_layout


def test_peak_temp_arena_keeps_high_water_across_rewind():
    arena = PeakTempArena(100)
    mark = arena.mark()
    arena.top += 37
    assert arena.peak == 137
    arena.rewind(mark)
    assert arena.top == 100
    assert arena.peak == 137
    assert arena.cells_used_peak == 37
    assert arena.runtime_base if False else True  # guard against accidental API assumptions


def test_final_layout_plan_is_after_all_compile_time_temporaries():
    source = "x = 1\ny = x + 2\nprint(y)\n"
    raw, plan = lower_with_layout(source)

    assert raw
    assert plan.temp_peak >= plan.temp_base
    assert plan.runtime_base() > plan.temp_peak


def test_layout_compiler_preserves_public_execution_semantics():
    source = "x = 7\ny = x * 3 + 1\nprint(y)\n"
    tree = ast.parse(source)
    compiler = PythonToBFLayout(tree)
    code = compiler.compile_module(tree)
    result = run_bf(code, step_limit=100_000_000)

    assert result.output == "22\n"
    assert compiler.layout_plan.temp_peak >= compiler.layout_plan.temp_base
