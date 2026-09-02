import ast

from bfmemory import allocate_live_blocks


def test_non_overlapping_scalars_reuse_storage():
    tree = ast.parse('a = 1\nprint(a)\nb = 2\nprint(b)\n')
    blocks, top = allocate_live_blocks(tree, {'a': 64, 'b': 64})
    assert blocks['a'].base == blocks['b'].base
    assert top == 64


def test_loop_variables_are_pinned_across_back_edge():
    tree = ast.parse('a = 0\nfor i in range(3):\n    a += 1\n    b = 5\nprint(a)\n')
    blocks, _ = allocate_live_blocks(tree, {'a': 64, 'b': 64, 'i': 64})
    assert blocks['a'].base != blocks['b'].base
    assert blocks['a'].base != blocks['i'].base
    assert blocks['b'].base != blocks['i'].base


def test_mixed_size_best_fit_reuse():
    tree = ast.parse('s = 1\nprint(s)\nx = 2\nprint(x)\n')
    blocks, top = allocate_live_blocks(tree, {'s': 129, 'x': 64})
    assert blocks['s'].base == blocks['x'].base
    assert top == 129
