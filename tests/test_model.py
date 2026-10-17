"""Unit tests for SupervisedPairwiseModel, grouped cross-validation, serialization, and Layer C."""

from __future__ import annotations

from pathlib import Path
import joblib
import pytest

from alt_detector.exceptions import ModelError
from alt_detector.features import extract_pairwise_features
from alt_detector.model import (
    SupervisedPairwiseModel,
    UnsupervisedAccountAnalyzer,
    deduplicate_labeled_pairs,
)
from tests.fixtures.sample_accounts import (
    make_labeled_dataset_fixture,
    make_lexical_only_candidate,
    make_popular_group_only_candidate,
    make_strong_alt_candidate,
    make_target_account,
)


def test_grouped_training_evaluation_and_serialization(tmp_path: Path) -> None:
    pairs = make_labeled_dataset_fixture()
    # Add an intentional duplicate pair with lower confidence to test deduplication
    dup_pair = pairs[0].model_copy(update={"label_confidence": 0.40})
    pairs_with_dup = pairs + [dup_pair]

    deduped, removed = deduplicate_labeled_pairs(pairs_with_dup)
    assert removed == 1
    assert len(deduped) == len(pairs)

    model = SupervisedPairwiseModel(random_seed=42)
    report = model.train_and_evaluate(pairs_with_dup, n_splits=4)

    assert model.is_trained
    assert report.duplicates_removed == 1
    assert report.uncertain_excluded_pairs == 1
    assert report.positive_pairs == 8
    assert report.negative_pairs == 14
    assert report.unique_operator_clusters == 8
    assert report.selected_model_name in (
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    )
    assert report.baseline_metrics.pr_auc > 0.70

    # Verify prediction on strong vs unrelated pair
    target = make_target_account()
    strong = make_strong_alt_candidate()
    popular = make_popular_group_only_candidate()
    pf_strong = extract_pairwise_features(target, strong)
    pf_pop = extract_pairwise_features(target, popular)

    score_strong = model.predict_score(pf_strong)
    score_pop = model.predict_score(pf_pop)
    assert score_strong is not None and score_pop is not None
    assert score_strong > score_pop

    # Serialization & reloading
    artifact_path = tmp_path / "models" / "pairwise_model.joblib"
    model.save(artifact_path)
    loaded = SupervisedPairwiseModel.load(artifact_path)
    assert loaded.is_trained
    assert loaded.selected_model_name == model.selected_model_name
    assert loaded.predict_score(pf_strong) == score_strong

    # Corrupted schema version raises ModelError
    bad_bundle = joblib.load(artifact_path)
    bad_bundle["feature_schema_version"] = "0.0.1-incompatible"
    bad_path = tmp_path / "models" / "bad_model.joblib"
    joblib.dump(bad_bundle, bad_path)
    with pytest.raises(ModelError, match="Feature schema version mismatch"):
        SupervisedPairwiseModel.load(bad_path)


def test_unsupervised_account_analyzer_cpu_only() -> None:
    target = make_target_account()
    cands = [
        make_strong_alt_candidate(),
        make_lexical_only_candidate(),
        make_popular_group_only_candidate(),
    ]
    pfs = {c.user_id: extract_pairwise_features(target, c) for c in cands}
    analyzer = UnsupervisedAccountAnalyzer(random_seed=42)
    res = analyzer.analyze_candidates(target, cands, pfs)

    assert len(res) == 3
    # witrecx_416 has 0 groups, 0 friends, empty description -> sparse_minimal_profile
    assert res[8814991503]["profile_archetype"] == "sparse_minimal_profile"
    assert 0.0 <= float(res[2349691539]["isolation_forest_anomaly_score"]) <= 1.0
