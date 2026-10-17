"""Analysis-only: no unsafe loop deduplication is performed."""
import pytest
from bfglobalprofile import analyze


def test_counts_and_nested_fingerprint():
    report = analyze("x[->+<][->+<]", top=3)
    assert report["bf_bytes"] == len("[->+<][->+<]")
    assert report["ignored_bytes"] == 1
    assert report["loop_count"] == 2
    assert report["distinct_loop_fingerprints"] == 1
    assert report["top_repeated_loops"][0]["occurrences"] == 2
    assert report["top_repeated_loops"][0]["bytes_per_loop"] == 6


def test_nested_loops_are_overlapping_motifs_not_compressible_macros():
    report = analyze("[[+]][[+]]")
    assert report["loop_count"] == 4
    assert report["max_loop_depth"] == 2
    assert report["repeated_loop_groups"] == 2
    assert report["repeat_motif_bytes_upper_bound_with_overlap"] > 0


def test_run_group_statistics_and_invalid_brackets():
    report = analyze(">><<++++---.,", top=0)
    assert report["pointer_run_count"] == 1
    assert report["pointer_max_run"] == 4
    assert report["arithmetic_max_run"] == 7
    assert report["top_repeated_loops"] == []
    with pytest.raises(ValueError, match="unmatched"):
        analyze("[[+]")
    with pytest.raises(ValueError, match="unmatched"):
        analyze("+]")


def test_large_repeated_loops_have_no_quadratic_substring_materialization():
    report = analyze("[->+<]" * 100000, top=1)
    assert report["bf_bytes"] == 600000
    assert report["top_repeated_loops"][0]["occurrences"] == 100000
    assert report["distinct_loop_fingerprints"] == 1
