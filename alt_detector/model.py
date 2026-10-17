"""Supervised pairwise models, grouped evaluation pipeline, and unsupervised analysis (Layers B & C)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alt_detector import FEATURE_SCHEMA_VERSION, MODEL_VERSION
from alt_detector.exceptions import ModelError
from alt_detector.features import FEATURE_NAMES
from alt_detector.schemas import AccountObservation, LabeledAccountPair, PairwiseFeatures
from alt_detector.scoring import TransparentBaselineScorer


def features_to_vector(feature_map: dict[str, float | None]) -> np.ndarray:
    """Convert a pairwise feature dict into a deterministic float64 vector with `np.nan` for missing."""
    vec = np.empty(len(FEATURE_NAMES), dtype=np.float64)
    for idx, name in enumerate(FEATURE_NAMES):
        val = feature_map.get(name)
        vec[idx] = np.nan if val is None else float(val)
    return vec


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if np.any(mask):
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return round(float(ece), 4)


@dataclass
class ModelEvaluationMetrics:
    """Comprehensive evaluation report for a model or baseline on grouped splits."""

    model_name: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    brier_score: float
    expected_calibration_error: float
    confusion_matrix: list[list[int]]
    fold_f1_mean: float
    fold_f1_ci95: float
    fold_pr_auc_mean: float
    fold_pr_auc_ci95: float
    performance_by_confidence_band: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkComparisonReport:
    """Grouped evaluation benchmark comparing ML candidates against TransparentBaselineScorer."""

    total_pairs_submitted: int
    evaluated_pairs: int
    positive_pairs: int
    negative_pairs: int
    uncertain_excluded_pairs: int
    duplicates_removed: int
    unique_operator_clusters: int
    n_splits: int
    selected_model_name: str
    selection_rationale: str
    baseline_metrics: ModelEvaluationMetrics
    candidate_model_metrics: dict[str, ModelEvaluationMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pairs_submitted": self.total_pairs_submitted,
            "evaluated_pairs": self.evaluated_pairs,
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "uncertain_excluded_pairs": self.uncertain_excluded_pairs,
            "duplicates_removed": self.duplicates_removed,
            "unique_operator_clusters": self.unique_operator_clusters,
            "n_splits": self.n_splits,
            "selected_model_name": self.selected_model_name,
            "selection_rationale": self.selection_rationale,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "candidate_model_metrics": {
                k: v.to_dict() for k, v in self.candidate_model_metrics.items()
            },
        }


def deduplicate_labeled_pairs(
    pairs: list[LabeledAccountPair],
) -> tuple[list[LabeledAccountPair], int]:
    """Deduplicate symmetric `(user_id_a, user_id_b)` pairs, keeping highest confidence."""
    by_key: dict[tuple[int, int], LabeledAccountPair] = {}
    duplicates = 0
    for p in pairs:
        key = (min(p.user_id_a, p.user_id_b), max(p.user_id_a, p.user_id_b))
        if key in by_key:
            duplicates += 1
            if p.label_confidence > by_key[key].label_confidence:
                by_key[key] = p
        else:
            by_key[key] = p
    ordered = [by_key[k] for k in sorted(by_key.keys())]
    return ordered, duplicates


def _build_candidate_classifiers(random_seed: int = 42) -> dict[str, Any]:
    """Build lightweight CPU-efficient tabular models with explicit missing-value handling."""
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=-1.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=-1.0,
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=60,
                        max_depth=3,
                        min_samples_leaf=2,
                        learning_rate=0.08,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }


def _evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_Scores: np.ndarray,
    fold_f1s: list[float],
    fold_prs: list[float],
    baseline_scorer: TransparentBaselineScorer,
    threshold: float = 0.50,
) -> ModelEvaluationMetrics:
    y_pred = (y_Scores >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr_auc = (
            float(average_precision_score(y_true, y_Scores))
            if len(np.unique(y_true)) > 1
            else 0.0
        )
        roc_auc = (
            float(roc_auc_score(y_true, y_Scores))
            if len(np.unique(y_true)) > 1
            else 0.0
        )
    brier = float(brier_score_loss(y_true, np.clip(y_Scores, 0.0, 1.0)))
    ece = compute_expected_calibration_error(y_true, np.clip(y_Scores, 0.0, 1.0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    k = max(len(fold_f1s), 1)
    f1_mean = float(np.mean(fold_f1s)) if fold_f1s else f1
    f1_ci = float(1.96 * np.std(fold_f1s) / np.sqrt(k)) if len(fold_f1s) > 1 else 0.0
    pr_mean = float(np.mean(fold_prs)) if fold_prs else pr_auc
    pr_ci = float(1.96 * np.std(fold_prs) / np.sqrt(k)) if len(fold_prs) > 1 else 0.0

    # Performance by confidence band
    bands: dict[str, dict[str, Any]] = {
        "Strong evidence": {"count": 0, "positives": 0, "empirical_precision": None},
        "Moderate evidence": {"count": 0, "positives": 0, "empirical_precision": None},
        "Weak evidence": {"count": 0, "positives": 0, "empirical_precision": None},
        "Insufficient evidence": {"count": 0, "positives": 0, "empirical_precision": None},
    }
    for label_val, score_val in zip(y_true, y_Scores):
        band_name, _ = baseline_scorer.classify_band(float(score_val))
        bands[band_name]["count"] += 1
        bands[band_name]["positives"] += int(label_val == 1)

    for b_info in bands.values():
        if b_info["count"] > 0:
            b_info["empirical_precision"] = round(
                b_info["positives"] / b_info["count"], 4
            )

    return ModelEvaluationMetrics(
        model_name=model_name,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        brier_score=round(brier, 4),
        expected_calibration_error=ece,
        confusion_matrix=cm,
        fold_f1_mean=round(f1_mean, 4),
        fold_f1_ci95=round(f1_ci, 4),
        fold_pr_auc_mean=round(pr_mean, 4),
        fold_pr_auc_ci95=round(pr_ci, 4),
        performance_by_confidence_band=bands,
    )


class SupervisedPairwiseModel:
    """Layer B supervised pairwise classifier with grouped cross-validation and versioning."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.model_version = MODEL_VERSION
        self.feature_schema_version = FEATURE_SCHEMA_VERSION
        self.feature_names = list(FEATURE_NAMES)
        self.selected_model_name: str = "untrained"
        self.scoring_mode: str = "heuristic_baseline_primary"
        self._estimator: Any | None = None
        self.benchmark_report: BenchmarkComparisonReport | None = None

    @property
    def is_trained(self) -> bool:
        return self._estimator is not None

    def train_and_evaluate(
        self,
        pairs: list[LabeledAccountPair],
        baseline_scorer: TransparentBaselineScorer | None = None,
        n_splits: int = 4,
    ) -> BenchmarkComparisonReport:
        """Evaluate candidate ML models vs TransparentBaselineScorer using grouped splits."""
        scorer = baseline_scorer or TransparentBaselineScorer()
        deduped, dup_count = deduplicate_labeled_pairs(pairs)
        uncertain_count = sum(1 for p in deduped if p.label == -1)
        valid_pairs = [p for p in deduped if p.label in (0, 1)]

        if len(valid_pairs) < 6:
            raise ModelError(
                f"At least 6 labeled pairs (with both classes) are required for grouped evaluation; got {len(valid_pairs)}"
            )

        X = np.vstack([features_to_vector(p.features) for p in valid_pairs])
        y = np.array([int(p.label) for p in valid_pairs], dtype=int)
        groups = np.array([p.operator_cluster_id for p in valid_pairs])
        sample_weights = np.array([float(p.label_confidence) for p in valid_pairs])

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        if pos_count == 0 or neg_count == 0:
            raise ModelError("Both positive (1) and negative (0) pairs are required for evaluation")

        unique_clusters = len(set(groups))
        actual_splits = min(n_splits, unique_clusters, pos_count, neg_count)
        if actual_splits < 2:
            raise ModelError(
                f"At least 2 distinct operator clusters per class are required to prevent group leakage (got {unique_clusters})"
            )

        # Evaluate TransparentBaselineScorer across folds
        try:
            splitter: Any = StratifiedGroupKFold(n_splits=actual_splits)
            split_indices = list(splitter.split(X, y, groups))
            # Verify every test fold has at least 1 sample
            if any(len(test_idx) == 0 for _, test_idx in split_indices):
                raise ValueError("Empty fold")
        except Exception:
            splitter = GroupKFold(n_splits=actual_splits)
            split_indices = list(splitter.split(X, y, groups))

        baseline_oof = np.zeros(len(valid_pairs), dtype=np.float64)
        for idx, pair in enumerate(valid_pairs):
            pf = PairwiseFeatures(
                target_user_id=pair.user_id_a,
                candidate_user_id=pair.user_id_b,
                values=pair.features,
                available_features=[k for k, v in pair.features.items() if v is not None],
                missing_features=[k for k in FEATURE_NAMES if pair.features.get(k) is None],
            )
            b_score, _, _ = scorer.compute_score_and_breakdown(pf)
            baseline_oof[idx] = b_score

        base_fold_f1s: list[float] = []
        base_fold_prs: list[float] = []
        for _, test_idx in split_indices:
            yt = y[test_idx]
            yp = baseline_oof[test_idx]
            base_fold_f1s.append(
                float(f1_score(yt, (yp >= scorer.thresholds["moderate"]).astype(int), zero_division=0))
            )
            if len(np.unique(yt)) > 1:
                base_fold_prs.append(float(average_precision_score(yt, yp)))

        baseline_metrics = _evaluate_predictions(
            "transparent_baseline",
            y,
            baseline_oof,
            base_fold_f1s,
            base_fold_prs,
            scorer,
            threshold=scorer.thresholds["moderate"],
        )

        # Evaluate candidate supervised models out-of-fold with strict group isolation
        candidate_defs = _build_candidate_classifiers(self.random_seed)
        candidate_metrics: dict[str, ModelEvaluationMetrics] = {}

        for name, _ in candidate_defs.items():
            oof_probs = np.zeros(len(valid_pairs), dtype=np.float64)
            fold_f1s: list[float] = []
            fold_prs: list[float] = []

            for train_idx, test_idx in split_indices:
                fold_est = _build_candidate_classifiers(self.random_seed)[name]
                X_tr, y_tr, w_tr = X[train_idx], y[train_idx], sample_weights[train_idx]
                X_te, y_te = X[test_idx], y[test_idx]

                if len(np.unique(y_tr)) < 2:
                    oof_probs[test_idx] = baseline_oof[test_idx]
                    continue

                if isinstance(fold_est, Pipeline):
                    fold_est.fit(X_tr, y_tr, clf__sample_weight=w_tr)
                else:
                    fold_est.fit(X_tr, y_tr, sample_weight=w_tr)

                probs = fold_est.predict_proba(X_te)[:, 1]
                oof_probs[test_idx] = probs
                fold_f1s.append(float(f1_score(y_te, (probs >= 0.50).astype(int), zero_division=0)))
                if len(np.unique(y_te)) > 1:
                    fold_prs.append(float(average_precision_score(y_te, probs)))

            candidate_metrics[name] = _evaluate_predictions(
                name,
                y,
                oof_probs,
                fold_f1s,
                fold_prs,
                scorer,
                threshold=0.50,
            )

        # Select best model by measured validation performance (PR-AUC + F1)
        best_ml_name = max(
            candidate_metrics.keys(),
            key=lambda m: (candidate_metrics[m].pr_auc, candidate_metrics[m].f1, -candidate_metrics[m].brier_score),
        )
        best_ml = candidate_metrics[best_ml_name]

        # Fit the selected estimator on all valid pairs
        final_est = _build_candidate_classifiers(self.random_seed)[best_ml_name]
        if isinstance(final_est, Pipeline):
            final_est.fit(X, y, clf__sample_weight=sample_weights)
        else:
            final_est.fit(X, y, sample_weight=sample_weights)

        self._estimator = final_est
        self.selected_model_name = best_ml_name

        verified_clusters = {
            p.operator_cluster_id
            for p in valid_pairs
            if p.adjudication_status == "verified"
        }
        if len(verified_clusters) >= 25 and best_ml.pr_auc >= baseline_metrics.pr_auc:
            self.scoring_mode = "calibrated_supervised"
        else:
            self.scoring_mode = "research_weak_supervision"

        rationale = (
            f"Selected '{best_ml_name}' (PR-AUC={best_ml.pr_auc:.3f}, F1={best_ml.f1:.3f}) "
            f"compared to transparent_baseline (PR-AUC={baseline_metrics.pr_auc:.3f}, F1={baseline_metrics.f1:.3f}) "
            f"across {actual_splits} group-isolated folds ({unique_clusters} operator clusters). "
            f"Operating mode set to '{self.scoring_mode}'."
        )

        self.benchmark_report = BenchmarkComparisonReport(
            total_pairs_submitted=len(pairs),
            evaluated_pairs=len(valid_pairs),
            positive_pairs=pos_count,
            negative_pairs=neg_count,
            uncertain_excluded_pairs=uncertain_count,
            duplicates_removed=dup_count,
            unique_operator_clusters=unique_clusters,
            n_splits=actual_splits,
            selected_model_name=best_ml_name,
            selection_rationale=rationale,
            baseline_metrics=baseline_metrics,
            candidate_model_metrics=candidate_metrics,
        )
        return self.benchmark_report

    def predict_score(self, features: PairwiseFeatures | dict[str, float | None]) -> float | None:
        """Predict pairwise association score in [0, 1] using the fitted local model."""
        if self._estimator is None:
            return None
        fmap = features.values if isinstance(features, PairwiseFeatures) else features
        vec = features_to_vector(fmap).reshape(1, -1)
        prob = float(self._estimator.predict_proba(vec)[0, 1])
        return round(min(1.0, max(0.0, prob)), 4)

    def save(self, path: Path | str) -> Path:
        """Serialize the trained model, schema metadata, and benchmark report locally."""
        if self._estimator is None:
            raise ModelError("Cannot save an untrained SupervisedPairwiseModel")
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": self.feature_names,
            "selected_model_name": self.selected_model_name,
            "scoring_mode": self.scoring_mode,
            "random_seed": self.random_seed,
            "estimator": self._estimator,
            "benchmark_report": (
                self.benchmark_report.to_dict() if self.benchmark_report else None
            ),
        }
        joblib.dump(bundle, target_path)
        return target_path

    @classmethod
    def load(cls, path: Path | str) -> SupervisedPairwiseModel:
        """Load and validate a serialized local model artifact."""
        target_path = Path(path)
        if not target_path.exists():
            raise ModelError(f"Model file not found at {target_path}")
        try:
            bundle = joblib.load(target_path)
        except Exception as exc:
            raise ModelError(f"Failed to deserialize model from {target_path}: {exc}") from exc

        if not isinstance(bundle, dict) or "estimator" not in bundle:
            raise ModelError(f"Invalid model artifact structure in {target_path}")
        if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ModelError(
                f"Feature schema version mismatch: artifact={bundle.get('feature_schema_version')}, "
                f"expected={FEATURE_SCHEMA_VERSION}"
            )
        if bundle.get("feature_names") != list(FEATURE_NAMES):
            raise ModelError("Feature name ordering in artifact does not match current schema")

        inst = cls(random_seed=int(bundle.get("random_seed", 42)))
        inst.model_version = str(bundle.get("model_version", MODEL_VERSION))
        inst.selected_model_name = str(bundle.get("selected_model_name", "loaded"))
        inst.scoring_mode = str(bundle.get("scoring_mode", "research_weak_supervision"))
        inst._estimator = bundle["estimator"]
        return inst


class UnsupervisedAccountAnalyzer:
    """Layer C unsupervised anomaly and profile-sparsity analyzer.

    Evaluates account-level structural properties (e.g., disposable sparse profiles vs
    established accounts) using IsolationForest and LocalOutlierFactor. Kept strictly
    separate from pairwise linkage scoring because an anomalous or sparse account is
    not automatically an alternative account of the target.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    @staticmethod
    def _account_profile_vector(obs: AccountObservation) -> np.ndarray:
        """Extract an account-intrinsic profile vector for anomaly/sparsity analysis."""
        group_cnt = float(len(obs.groups)) if obs.groups is not None else 0.0
        friend_cnt = float(obs.friends_count if obs.friends_count is not None else len(obs.friends or []))
        desc_len = float(len((obs.description or "").strip()))
        avatar_assets = float(len(obs.avatar.asset_ids)) if obs.avatar else 0.0
        fav_games = float(len(obs.favorite_games)) if obs.favorite_games is not None else 0.0
        created_games = float(len(obs.created_games)) if obs.created_games is not None else 0.0
        age_days = 0.0
        if obs.created_datetime is not None:
            age_days = max(0.0, (2026 - obs.created_datetime.year) * 365.0)
        return np.array(
            [
                np.log1p(group_cnt),
                np.log1p(friend_cnt),
                np.log1p(desc_len),
                np.log1p(avatar_assets),
                np.log1p(fav_games + created_games),
                np.log1p(age_days),
            ],
            dtype=np.float64,
        )

    def analyze_candidates(
        self,
        target: AccountObservation,
        candidates: list[AccountObservation],
        pairwise_features_by_uid: dict[int, PairwiseFeatures],
    ) -> dict[int, dict[str, float | str]]:
        """Return separate unsupervised profile-anomaly and feature-cluster diagnostics."""
        if not candidates:
            return {}

        all_obs = [target] + candidates
        prof_matrix = np.vstack([self._account_profile_vector(o) for o in all_obs])

        # Compute IsolationForest outlier scores if enough accounts exist
        iso_scores = np.zeros(len(all_obs), dtype=np.float64)
        lof_flags = np.zeros(len(all_obs), dtype=int)

        if len(all_obs) >= 4:
            iso = IsolationForest(
                n_estimators=50,
                contamination="auto",
                random_state=self.random_seed,
            )
            iso.fit(prof_matrix)
            # decision_function < 0 indicates outlier; map to [0, 1] anomaly index
            raw_dec = iso.decision_function(prof_matrix)
            iso_scores = np.clip(0.5 - raw_dec, 0.0, 1.0)

        if len(all_obs) >= 6:
            n_neighbors = min(5, len(all_obs) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
            preds = lof.fit_predict(prof_matrix)
            lof_flags = (preds == -1).astype(int)

        results: dict[int, dict[str, float | str]] = {}
        for idx, cand in enumerate(candidates, start=1):
            g_cnt = len(cand.groups or [])
            f_cnt = cand.friends_count if cand.friends_count is not None else len(cand.friends or [])
            is_sparse = g_cnt <= 1 and f_cnt <= 2 and not (cand.description or "").strip()

            pf = pairwise_features_by_uid.get(cand.user_id)
            vec = features_to_vector(pf.values) if pf else np.zeros(len(FEATURE_NAMES))
            valid_vals = vec[~np.isnan(vec)]
            mean_signal_density = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

            profile_archetype = (
                "sparse_minimal_profile"
                if is_sparse
                else ("outlier_activity_profile" if lof_flags[idx] == 1 else "standard_active_profile")
            )
            results[cand.user_id] = {
                "isolation_forest_anomaly_score": round(float(iso_scores[idx]), 4),
                "lof_outlier_flag": int(lof_flags[idx]),
                "pairwise_signal_density": round(mean_signal_density, 4),
                "profile_archetype": profile_archetype,
                "note": "Unsupervised profile archetype is separate from linkage score; sparse/outlier profiles are not automatically alts.",
            }
        return results
