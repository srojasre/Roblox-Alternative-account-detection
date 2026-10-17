"""Explainable baseline scoring and evidence synthesis (Layer A & Section 9)."""

from __future__ import annotations

from typing import Any

from alt_detector import FEATURE_SCHEMA_VERSION, MODEL_VERSION
from alt_detector.config import DEFAULT_FAMILY_WEIGHTS, DEFAULT_SCORE_THRESHOLDS, EngineConfig
from alt_detector.schemas import (
    AccountObservation,
    CandidateAssessment,
    CandidateRecord,
    ConfidenceBand,
    EvidenceItem,
    PairwiseFeatures,
)

# Within-family subweights (normalized across available features in each family)
WITHIN_FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "lexical_naming": {
        "username_char_ngram_sim": 0.30,
        "username_lcs_ratio": 0.42,
        "display_name_similarity": 0.16,
        "shared_distinctive_affix_score": 0.12,
    },
    "group_structure": {
        "group_jaccard": 0.10,
        "rarity_weighted_group_sim": 0.30,
        "small_group_co_membership_score": 0.30,
        "shared_elevated_role_score": 0.30,
    },
    "social_graph": {
        "is_direct_friend": 0.42,
        "mutual_friends_jaccard": 0.23,
        "mutual_friends_containment": 0.35,
    },
    "profile_avatar": {
        "description_similarity": 0.35,
        "avatar_asset_overlap": 0.35,
        "avatar_style_similarity": 0.15,
        "creation_burst_proximity": 0.15,
    },
    "activity_games": {
        "favorite_games_overlap": 0.45,
        "created_games_similarity": 0.25,
        "inventory_asset_overlap": 0.20,
        "activity_timestamp_overlap": 0.10,
    },
    "graph_topology": {
        "graph_adamic_adar": 0.35,
        "graph_resource_allocation": 0.30,
        "graph_personalized_pagerank": 0.20,
        "graph_same_community": 0.15,
    },
}


class TransparentBaselineScorer:
    """Layer A interpretable baseline scorer with family-level collinearity caps.

    Combines normalized evidence across orthogonal signal families, distinguishes
    unavailable data from confirmed absence, and prevents single-family signals
    (such as a similar username alone or a single shared group alone) from
    producing uncorroborated high-confidence attributions.
    """

    def __init__(
        self,
        family_weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        threshold_status: str = "provisional",
    ) -> None:
        raw_w = dict(family_weights or DEFAULT_FAMILY_WEIGHTS)
        total_w = sum(raw_w.values()) or 1.0
        self.family_weights = {k: v / total_w for k, v in raw_w.items()}
        self.thresholds = dict(thresholds or DEFAULT_SCORE_THRESHOLDS)
        self.threshold_status = (
            "validated" if threshold_status == "validated" else "provisional"
        )

    def compute_score_and_breakdown(
        self,
        features: PairwiseFeatures,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        """Return `(baseline_score, family_scores, feature_contributions)`."""
        vals = features.values
        family_scores: dict[str, float] = {}
        raw_feature_contribs: dict[str, float] = {}

        for family, subweights in WITHIN_FAMILY_WEIGHTS.items():
            avail_sub = {
                feat: w
                for feat, w in subweights.items()
                if vals.get(feat) is not None
            }
            if not avail_sub:
                family_scores[family] = 0.0
                continue

            sub_sum = sum(avail_sub.values()) or 1.0
            fam_val = 0.0
            for feat, w in avail_sub.items():
                v = float(vals[feat] or 0.0)
                norm_sub_w = w / sub_sum
                fam_val += norm_sub_w * v
                raw_feature_contribs[feat] = round(
                    self.family_weights.get(family, 0.0) * norm_sub_w * v, 4
                )
            family_scores[family] = round(min(1.0, max(0.0, fam_val)), 4)

        # Weighted linear combination across families
        weighted_sum = sum(
            self.family_weights.get(fam, 0.0) * score
            for fam, score in family_scores.items()
        )

        # Count independent corroborating core families (lexical, group, social, profile/activity)
        core_families = [
            "lexical_naming",
            "group_structure",
            "social_graph",
            "profile_avatar",
            "activity_games",
        ]
        active_core = [
            fam for fam in core_families if family_scores.get(fam, 0.0) >= 0.25
        ]
        strong_core = [
            fam for fam in core_families if family_scores.get(fam, 0.0) >= 0.50
        ]

        adjusted = weighted_sum

        # Multi-family corroboration synergy vs single-family guardrail
        if len(active_core) >= 3 and len(strong_core) >= 2:
            # Independent corroboration across 3+ orthogonal families (e.g., lexical + rare group role + mutual friends)
            adjusted = min(1.0, adjusted * 1.22 + 0.06)
        elif (
            family_scores.get("lexical_naming", 0.0) >= 0.65
            and family_scores.get("group_structure", 0.0) >= 0.55
            and family_scores.get("social_graph", 0.0) >= 0.45
        ):
            adjusted = min(1.0, adjusted * 1.25 + 0.08)
        elif len(active_core) <= 1:
            # Guardrail: a single signal family alone (e.g. username similarity with 0 shared groups/friends,
            # or a single shared group with 0 lexical/social overlap) cannot exceed Weak/low-Moderate evidence
            adjusted = min(adjusted * 0.82, self.thresholds["moderate"] - 0.03)

        # Sparse observation discount if most core features are missing
        observed_ratio = len(features.available_features) / max(len(vals), 1)
        if observed_ratio < 0.45:
            adjusted *= 0.75 + 0.25 * (observed_ratio / 0.45)

        final_baseline = round(min(1.0, max(0.0, adjusted)), 4)
        return final_baseline, family_scores, raw_feature_contribs

    def classify_band(self, score: float) -> tuple[ConfidenceBand, str]:
        """Map a numeric score in [0, 1] to a qualitative evidence category."""
        if score >= self.thresholds["strong"]:
            return (
                "Strong evidence",
                "Multiple independent public signal families (lexical, rare group/role structure, "
                "and social graph) corroborate a possible shared operator or close operational link.",
            )
        if score >= self.thresholds["moderate"]:
            return (
                "Moderate evidence",
                "Noticeable overlap across two or more public signal categories, though also "
                "consistent with close collaborators, clan peers, or unconfirmed name reuse.",
            )
        if score >= self.thresholds["weak"]:
            return (
                "Weak evidence",
                "Limited or single-category overlap (e.g. lexical similarity without social/group "
                "links, or co-membership in a small group without identity overlap).",
            )
        return (
            "Insufficient evidence",
            "Publicly observable signals do not provide meaningful evidence of shared account operation.",
        )

    def build_assessment(
        self,
        target: AccountObservation,
        candidate: AccountObservation,
        candidate_record: CandidateRecord,
        features: PairwiseFeatures,
        *,
        ml_score: float | None = None,
        scoring_mode: str = "heuristic_baseline_primary",
        unsupervised_signals: dict[str, float | str] | None = None,
        config: EngineConfig | None = None,
    ) -> CandidateAssessment:
        """Synthesize a complete, human-explainable CandidateAssessment."""
        baseline_score, family_scores, contribs = self.compute_score_and_breakdown(features)

        if ml_score is not None and scoring_mode == "calibrated_supervised":
            combined = round(0.45 * baseline_score + 0.55 * ml_score, 4)
        elif ml_score is not None:
            # In heuristic/weakly-supervised mode, baseline leads and ML provides a bounded adjustment
            combined = round(0.75 * baseline_score + 0.25 * ml_score, 4)
            # Respect single-family guardrail even when ML score is blended
            core_active = sum(
                1
                for f in ("lexical_naming", "group_structure", "social_graph", "profile_avatar", "activity_games")
                if family_scores.get(f, 0.0) >= 0.25
            )
            if core_active <= 1:
                combined = min(combined, self.thresholds["moderate"] - 0.02)
        else:
            combined = baseline_score

        band, band_interp = self.classify_band(combined)
        supporting, conflicting, missing = self._extract_evidence_items(
            target, candidate, features, contribs
        )
        uncertainties, alternatives = self._build_uncertainty_and_alternatives(
            target, candidate, features, family_scores, band
        )

        source_set = sorted(
            {s.source_name for s in target.sources}
            | {s.source_name for s in candidate.sources}
        )
        graph_metrics = {
            k: float(features.values[k])
            for k in (
                "graph_adamic_adar",
                "graph_resource_allocation",
                "graph_personalized_pagerank",
                "graph_same_community",
            )
            if features.values.get(k) is not None
        }

        return CandidateAssessment(
            candidate_user_id=candidate.user_id,
            candidate_username=candidate.username,
            candidate_display_name=candidate.display_name,
            discovery_reasons=candidate_record.reasons,
            baseline_score=baseline_score,
            ml_score=round(ml_score, 4) if ml_score is not None else None,
            combined_score=combined,
            confidence_band=band,
            confidence_interpretation=band_interp,
            threshold_status=(
                config.threshold_status if config else self.threshold_status  # type: ignore[arg-type]
            ),
            scoring_mode=scoring_mode,
            supporting_evidence=supporting,
            conflicting_evidence=conflicting,
            missing_evidence=missing,
            family_scores=family_scores,
            feature_contributions=contribs,
            raw_features=features.values,
            graph_metrics=graph_metrics,
            unsupervised_signals=unsupervised_signals or {},
            uncertainty_notes=uncertainties,
            alternative_explanations=alternatives,
            data_sources=source_set,
            model_version=MODEL_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )

    def _extract_evidence_items(
        self,
        target: AccountObservation,
        candidate: AccountObservation,
        features: PairwiseFeatures,
        contribs: dict[str, float],
    ) -> tuple[list[EvidenceItem], list[EvidenceItem], list[EvidenceItem]]:
        supporting: list[EvidenceItem] = []
        conflicting: list[EvidenceItem] = []
        missing: list[EvidenceItem] = []

        vals = features.values
        diag = features.diagnostics

        # 1. Lexical handle evidence
        lcs = vals.get("username_lcs_ratio") or 0.0
        ngram = vals.get("username_char_ngram_sim") or 0.0
        disp_sim = vals.get("display_name_similarity") or 0.0
        if lcs >= 0.70 or ngram >= 0.55:
            pair = diag.get("matched_username_pair", (target.username, candidate.username))
            supporting.append(
                EvidenceItem(
                    signal_name="username_lexical_similarity",
                    family="lexical_naming",
                    direction="supporting",
                    strength="high" if lcs >= 0.85 else "medium",
                    value_summary=f"core_lcs={lcs:.2f}, char_3gram={ngram:.2f}",
                    explanation=(
                        f"Handles '{pair[0]}' and '{pair[1]}' share a strong lexical stem "
                        f"(LCS ratio {lcs:.2f}, 3-gram cosine {ngram:.2f})."
                    ),
                    contribution=round(
                        contribs.get("username_lcs_ratio", 0.0)
                        + contribs.get("username_char_ngram_sim", 0.0),
                        4,
                    ),
                )
            )
        elif lcs < 0.15 and ngram < 0.15 and disp_sim < 0.20:
            conflicting.append(
                EvidenceItem(
                    signal_name="username_lexical_dissimilarity",
                    family="lexical_naming",
                    direction="against",
                    strength="low",
                    value_summary=f"core_lcs={lcs:.2f}, display_sim={disp_sim:.2f}",
                    explanation=(
                        f"No observable lexical stem overlap between '{target.username}' "
                        f"('{target.display_name}') and '{candidate.username}' ('{candidate.display_name}')."
                    ),
                    contribution=0.0,
                )
            )

        affix = vals.get("shared_distinctive_affix_score") or 0.0
        if affix >= 0.30:
            nums = diag.get("shared_numeric_affixes", [])
            words = diag.get("shared_word_tokens", [])
            supporting.append(
                EvidenceItem(
                    signal_name="shared_distinctive_affix",
                    family="lexical_naming",
                    direction="supporting",
                    strength="medium" if affix >= 0.60 else "low",
                    value_summary=f"affix_score={affix:.2f} (nums={nums}, words={words})",
                    explanation=(
                        f"Shared distinctive naming/group tokens: numbers={nums or 'none'}, "
                        f"lexical tokens={words or 'none'}."
                    ),
                    contribution=contribs.get("shared_distinctive_affix_score", 0.0),
                )
            )

        # 2. Group & Role evidence
        shared_groups: list[dict[str, Any]] = diag.get("shared_groups", [])
        rw_grp = vals.get("rarity_weighted_group_sim")
        elev_role = vals.get("shared_elevated_role_score") or 0.0
        if shared_groups:
            small_shared = [
                g for g in shared_groups if (g.get("member_count") or 999_999) <= 2_500
            ]
            if small_shared or elev_role >= 0.35:
                summary_parts = [
                    f"{g['group_name']} (N={g['member_count']}, roles: {g['target_role']}/{g['candidate_role']})"
                    for g in sorted(small_shared or shared_groups, key=lambda x: x.get("member_count") or 999_999)[:3]
                ]
                supporting.append(
                    EvidenceItem(
                        signal_name="rare_group_and_role_overlap",
                        family="group_structure",
                        direction="supporting",
                        strength="high" if elev_role >= 0.60 or (rw_grp or 0.0) >= 0.65 else "medium",
                        value_summary=f"rarity_sim={rw_grp or 0.0:.2f}, elevated_role={elev_role:.2f}",
                        explanation=(
                            f"Co-membership and/or shared high rank in {len(shared_groups)} group(s), "
                            f"including: {'; '.join(summary_parts)}."
                        ),
                        contribution=round(
                            contribs.get("rarity_weighted_group_sim", 0.0)
                            + contribs.get("small_group_co_membership_score", 0.0)
                            + contribs.get("shared_elevated_role_score", 0.0),
                            4,
                        ),
                    )
                )
            else:
                conflicting.append(
                    EvidenceItem(
                        signal_name="only_popular_groups_shared",
                        family="group_structure",
                        direction="against",
                        strength="medium",
                        value_summary=f"shared_groups={len(shared_groups)} (all >2,500 members)",
                        explanation=(
                            "Shared groups consist solely of large/popular communities with low "
                            "discriminative value for identity linkage."
                        ),
                        contribution=0.0,
                    )
                )
        elif target.groups and candidate.groups:
            conflicting.append(
                EvidenceItem(
                    signal_name="no_shared_groups",
                    family="group_structure",
                    direction="against",
                    strength="medium",
                    value_summary=f"target_groups={len(target.groups)}, candidate_groups={len(candidate.groups)}, overlap=0",
                    explanation=(
                        f"Both accounts have public group memberships ({len(target.groups)} vs "
                        f"{len(candidate.groups)}), but share zero groups in common."
                    ),
                    contribution=0.0,
                )
            )
        elif candidate.groups is not None and len(candidate.groups) == 0:
            conflicting.append(
                EvidenceItem(
                    signal_name="candidate_has_zero_groups",
                    family="group_structure",
                    direction="against",
                    strength="low",
                    value_summary="candidate_groups=0",
                    explanation="Candidate account belongs to 0 public groups, providing no organizational corroboration.",
                    contribution=0.0,
                )
            )

        # 3. Social graph & mutual friends evidence
        is_friend = vals.get("is_direct_friend")
        mut_cnt = int(diag.get("mutual_friends_count") or 0)
        mut_cont = vals.get("mutual_friends_containment") or 0.0
        if is_friend == 1.0 or mut_cnt > 0:
            supporting.append(
                EvidenceItem(
                    signal_name="social_graph_overlap",
                    family="social_graph",
                    direction="supporting",
                    strength="high" if (is_friend == 1.0 and mut_cnt >= 5) else ("medium" if is_friend == 1.0 or mut_cnt >= 3 else "low"),
                    value_summary=f"direct_friend={bool(is_friend)}, mutual_friends={mut_cnt} (containment={mut_cont:.2f})",
                    explanation=(
                        f"{'Direct public friends' if is_friend == 1.0 else 'Not direct friends'}, "
                        f"sharing {mut_cnt} mutual third-party friend(s) (containment {mut_cont:.2f})."
                    ),
                    contribution=round(
                        contribs.get("is_direct_friend", 0.0)
                        + contribs.get("mutual_friends_jaccard", 0.0)
                        + contribs.get("mutual_friends_containment", 0.0),
                        4,
                    ),
                )
            )
        elif is_friend == 0.0 and target.friends and candidate.friends is not None:
            conflicting.append(
                EvidenceItem(
                    signal_name="no_social_graph_connection",
                    family="social_graph",
                    direction="against",
                    strength="medium",
                    value_summary=f"direct_friend=False, mutual_friends=0 (candidate_friends={len(candidate.friends)})",
                    explanation="No direct friendship and zero mutual friends observed between the two accounts.",
                    contribution=0.0,
                )
            )

        # 4. Avatar / Description / Games evidence
        av_overlap = vals.get("avatar_asset_overlap")
        if av_overlap is not None and av_overlap >= 0.25:
            shared_assets = diag.get("shared_custom_avatar_assets", [])
            supporting.append(
                EvidenceItem(
                    signal_name="custom_avatar_asset_overlap",
                    family="profile_avatar",
                    direction="supporting",
                    strength="medium" if av_overlap >= 0.50 else "low",
                    value_summary=f"asset_jaccard={av_overlap:.2f} ({len(shared_assets)} non-default assets)",
                    explanation=f"Accounts wear {len(shared_assets)} identical non-default catalog asset(s): {shared_assets[:6]}.",
                    contribution=contribs.get("avatar_asset_overlap", 0.0),
                )
            )

        fav_overlap = vals.get("favorite_games_overlap")
        if fav_overlap is not None and fav_overlap >= 0.20:
            cross_fav = diag.get("cross_favorited_created_game", False)
            supporting.append(
                EvidenceItem(
                    signal_name="favorite_experiences_overlap",
                    family="activity_games",
                    direction="supporting",
                    strength="medium" if cross_fav or fav_overlap >= 0.50 else "low",
                    value_summary=f"fav_overlap={fav_overlap:.2f}, cross_creator_fav={cross_fav}",
                    explanation=(
                        "Shared favorited experiences or direct favoriting of an experience created by the counterpart."
                    ),
                    contribution=contribs.get("favorite_games_overlap", 0.0),
                )
            )

        # 5. Missing / inaccessible evidence tracking
        for feat_name in features.missing_features:
            missing.append(
                EvidenceItem(
                    signal_name=feat_name,
                    family=WITHIN_FAMILY_WEIGHTS_LOOKUP.get(feat_name, "other"),
                    direction="missing",
                    strength="info",
                    value_summary="unavailable / empty",
                    explanation=(
                        f"Feature '{feat_name}' could not be evaluated because one or both accounts "
                        "had no public data or restricted visibility for this signal."
                    ),
                    contribution=0.0,
                )
            )

        for src_key, reason in sorted(candidate.inaccessible_sources.items()):
            missing.append(
                EvidenceItem(
                    signal_name=f"source:{src_key}",
                    family="data_availability",
                    direction="missing",
                    strength="info",
                    value_summary="restricted / unqueried",
                    explanation=reason,
                    contribution=0.0,
                )
            )

        supporting.sort(key=lambda x: -x.contribution)
        return supporting, conflicting, missing

    def _build_uncertainty_and_alternatives(
        self,
        target: AccountObservation,
        candidate: AccountObservation,
        features: PairwiseFeatures,
        family_scores: dict[str, float],
        band: ConfidenceBand,
    ) -> tuple[list[str], list[str]]:
        uncertainties: list[str] = [
            "Public Roblox telemetry cannot observe IP addresses, hardware/device fingerprints, "
            "or authentication cookies; scores reflect behavioral/structural correlation, not proof of identity."
        ]
        alternatives: list[str] = []

        if family_scores.get("group_structure", 0.0) >= 0.45:
            alternatives.append(
                "Co-officers, co-developers, or close friends within the same small Roblox clan/group "
                "(e.g., roleplay division or development studio) frequently share identical ranks and mutual friends."
            )
        if (
            family_scores.get("lexical_naming", 0.0) >= 0.55
            and family_scores.get("group_structure", 0.0) < 0.20
            and family_scores.get("social_graph", 0.0) < 0.20
        ):
            alternatives.append(
                "An unrelated user or impersonator may have chosen a handle containing a similar "
                "lexical stem or number suffix without any social or organizational tie to the target."
            )
            uncertainties.append(
                "Only lexical naming similarity was observed; absence of shared groups or mutual friends "
                "prevents elevating confidence above Weak evidence."
            )
        if features.values.get("shared_distinctive_affix_score", 0.0) and (
            features.values.get("username_lcs_ratio") or 0.0
        ) < 0.40:
            alternatives.append(
                "Shared numeric suffixes (such as group/division numbers) or shared display-name prefixes "
                "are often adopted as uniform clan tags by multiple distinct group members."
            )
        if not alternatives:
            alternatives.append(
                "Accounts may belong to independent players who happen to intersect in one or more "
                "public communities or friend circles."
            )
        if len(features.missing_features) >= 5:
            uncertainties.append(
                f"{len(features.missing_features)} of {len(features.values)} pairwise features were unavailable "
                "(e.g., empty profile descriptions, private inventories, or unpopulated game favorites)."
            )
        return uncertainties, alternatives


WITHIN_FAMILY_WEIGHTS_LOOKUP: dict[str, str] = {
    feat: fam
    for fam, sub in WITHIN_FAMILY_WEIGHTS.items()
    for feat in sub
}
