"""Unit tests for TransparentBaselineScorer, single-family guardrails, and explainable assessments."""

from __future__ import annotations

from alt_detector.features import extract_pairwise_features
from alt_detector.graph_analysis import CandidateGraphAnalyzer
from alt_detector.schemas import CandidateDiscoveryReason, CandidateRecord
from alt_detector.scoring import TransparentBaselineScorer
from tests.fixtures.sample_accounts import (
    make_lexical_only_candidate,
    make_popular_group_only_candidate,
    make_strong_alt_candidate,
    make_target_account,
)


def test_deterministic_scoring_and_confidence_bands() -> None:
    target = make_target_account()
    strong_cand = make_strong_alt_candidate()
    lexical_cand = make_lexical_only_candidate()
    popular_cand = make_popular_group_only_candidate()

    graph_analyzer = CandidateGraphAnalyzer()
    g_metrics, _ = graph_analyzer.build_and_analyze(
        target, [strong_cand, lexical_cand, popular_cand]
    )
    scorer = TransparentBaselineScorer()

    # 1. Strong candidate (lexical + rare group/role + direct friend/mutuals + game cross-fav)
    f_strong = extract_pairwise_features(target, strong_cand, g_metrics[strong_cand.user_id])
    rec_strong = CandidateRecord(
        user_id=strong_cand.user_id,
        username=strong_cand.username,
        display_name=strong_cand.display_name,
        reasons=[
            CandidateDiscoveryReason(
                strategy="lexical_search",
                detail="Matched stem 'witrecx'",
                source_endpoint="https://users.roblox.com/v1/users/search",
                prior_weight=0.72,
            )
        ],
    )
    a_strong_1 = scorer.build_assessment(target, strong_cand, rec_strong, f_strong)
    a_strong_2 = scorer.build_assessment(target, strong_cand, rec_strong, f_strong)
    assert a_strong_1.combined_score == a_strong_2.combined_score
    assert a_strong_1.confidence_band == "Strong evidence"
    assert a_strong_1.combined_score >= 0.68
    assert len(a_strong_1.supporting_evidence) >= 3
    assert len(a_strong_1.alternative_explanations) >= 1

    # 2. Lexical-only candidate (witrecx_416: similar handle, 0 friends, 0 groups)
    # Single-family guardrail must prevent Strong or Moderate evidence!
    f_lex = extract_pairwise_features(target, lexical_cand, g_metrics[lexical_cand.user_id])
    rec_lex = CandidateRecord(
        user_id=lexical_cand.user_id,
        username=lexical_cand.username,
        display_name=lexical_cand.display_name,
    )
    a_lex = scorer.build_assessment(target, lexical_cand, rec_lex, f_lex)
    assert a_lex.confidence_band in ("Weak evidence", "Insufficient evidence")
    assert a_lex.combined_score < scorer.thresholds["moderate"]
    assert any(ev.direction == "against" for ev in a_lex.conflicting_evidence)

    # 3. Popular-group-only control candidate
    f_pop = extract_pairwise_features(target, popular_cand, g_metrics[popular_cand.user_id])
    rec_pop = CandidateRecord(
        user_id=popular_cand.user_id,
        username=popular_cand.username,
        display_name=popular_cand.display_name,
    )
    a_pop = scorer.build_assessment(target, popular_cand, rec_pop, f_pop)
    assert a_pop.confidence_band == "Insufficient evidence"
    assert a_pop.combined_score < scorer.thresholds["weak"]
    assert a_strong_1.combined_score > a_lex.combined_score > a_pop.combined_score
