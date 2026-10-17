"""Unit tests for pairwise feature engineering, rarity weighting, and missing-value handling."""

from __future__ import annotations

from alt_detector.features import (
    char_ngram_cosine,
    compute_core_lcs_ratio,
    extract_pairwise_features,
)
from alt_detector.graph_analysis import CandidateGraphAnalyzer
from tests.fixtures.sample_accounts import (
    make_lexical_only_candidate,
    make_popular_group_only_candidate,
    make_strong_alt_candidate,
    make_target_account,
)


def test_lexical_similarity_primitives() -> None:
    assert compute_core_lcs_ratio("witrecx64", "witrecx") == 1.0
    assert compute_core_lcs_ratio("witrecx64", "witrecx_416") > 0.80
    assert compute_core_lcs_ratio("witrecx64", "RandomPlayer90210") == 0.0
    assert char_ngram_cosine("witrecx64", "witrecx") > 0.65


def test_popular_versus_rare_shared_groups() -> None:
    target = make_target_account()
    strong_cand = make_strong_alt_candidate()
    popular_cand = make_popular_group_only_candidate()

    graph_analyzer = CandidateGraphAnalyzer()
    g_metrics, _ = graph_analyzer.build_and_analyze(target, [strong_cand, popular_cand])

    f_strong = extract_pairwise_features(target, strong_cand, g_metrics[strong_cand.user_id])
    f_popular = extract_pairwise_features(target, popular_cand, g_metrics[popular_cand.user_id])

    # Strong candidate shares 3 small/medium groups (79, 128, 445 members) with elevated roles
    assert (f_strong.values["rarity_weighted_group_sim"] or 0.0) > 0.70
    assert (f_strong.values["small_group_co_membership_score"] or 0.0) > 0.70
    assert (f_strong.values["shared_elevated_role_score"] or 0.0) > 0.70

    # Popular-group candidate only shares a 920,100-member group at rank 1
    assert (f_popular.values["rarity_weighted_group_sim"] or 0.0) < 0.25
    assert f_popular.values["small_group_co_membership_score"] == 0.0
    assert f_popular.values["shared_elevated_role_score"] == 0.0


def test_missing_features_not_treated_as_zero_dissimilarity() -> None:
    target = make_target_account()
    sparse = make_lexical_only_candidate()

    pf = extract_pairwise_features(target, sparse)
    # Candidate has empty description, no created games, no favorite games, private inventory
    assert pf.values["description_similarity"] is None
    assert pf.values["created_games_similarity"] is None
    assert pf.values["inventory_asset_overlap"] is None
    assert "description_similarity" in pf.missing_features
    assert "username_lcs_ratio" in pf.available_features
