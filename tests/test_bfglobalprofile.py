"""Correctness and scaling checks for the read-only global BF profiler."""

import pytest

from bfglobalprofile import profile_bf


def test_counts_commands_and_ignores_non_bf_text():
    report = profile_bf("abc[-]xyz>[-]")
    assert report["source_bytes"] == 7
    assert report["ignored_non_command_chars"] == 6
    assert report["commands"]["["] == 2
    assert report["commands"]["]"] == 2
    assert report["commands"][">"] == 1
    assert report["loops"]["instances"] == 2
    assert report["loops"]["max_depth"] == 1
    assert report["top_loop_motifs"][0]["instances"] == 2
    assert report["top_loop_motifs"][0]["length"] == 3
    assert report["top_loop_motifs"][0]["duplicate_text_bytes"] == 3


def test_records_cancellable_runs_without_rewriting_anything():
    source = "<>><++++---."
    report = profile_bf(source)
    assert source == "<>><++++---."
    assert report["runs"]["locally_cancellable_pointer_bytes"] == 4
    assert report["runs"]["locally_reducible_arithmetic_bytes"] == 6
    assert report["runs"]["pointer_runs"] == 1
    assert report["runs"]["arithmetic_runs"] == 1


def test_nested_loop_fingerprints_have_correct_spans():
    report = profile_bf("[[-]][[-]]", top=3)
    assert report["loops"]["instances"] == 4
    assert report["loops"]["max_depth"] == 2
    assert report["loops"]["distinct_fingerprints"] == 2
    assert report["loops"]["repeated_fingerprints"] == 2
    assert sorted(row["length"] for row in report["top_loop_motifs"]) == [3, 5]
    assert report["loops"]["duplicate_text_bytes_not_savings"] == 8


@pytest.mark.parametrize("source", ("[+", "abc]", "[[]", "[]]"))
def test_rejects_unmatched_brackets(source):
    with pytest.raises(ValueError, match="unmatched"):
        profile_bf(source)


def test_top_limit_and_validation():
    source = "[-][-][+][+][->+<][->+<]"
    assert len(profile_bf(source, top=1)["top_loop_motifs"]) == 1
    with pytest.raises(ValueError, match="positive"):
        profile_bf(source, top=0)


def test_half_megabyte_profile_has_no_pairwise_loop_comparisons():
    # ~0.5 MiB and ~75k loop instances. The profiler only hashes/aggregates
    # each byte once and sorts at most the final distinct fingerprints.
    piece = "[->+<]>+<"
    code = piece * 60000
    report = profile_bf(code)
    assert report["source_bytes"] == len(code)
    assert report["loops"]["instances"] == 60000
    assert report["loops"]["distinct_fingerprints"] == 1
    assert report["top_loop_motifs"][0]["instances"] == 60000
    assert report["runs"]["pointer_runs"] > 0
