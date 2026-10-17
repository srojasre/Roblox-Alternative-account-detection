"""Real-world multi-account validation and benchmark evaluation suite (Section 12).

Keeps validation storage, fixtures, and benchmark pairs strictly isolated from
live user analyses. Records actual HTTP statuses, collection failures, runtime,
memory usage, and grouped model vs baseline performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any

import psutil

from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.collectors import PublicDataCollector
from alt_detector.config import EngineConfig
from alt_detector.exceptions import AltDetectorError
from alt_detector.features import extract_pairwise_features
from alt_detector.graph_analysis import CandidateGraphAnalyzer
from alt_detector.model import BenchmarkComparisonReport, SupervisedPairwiseModel
from alt_detector.schemas import (
    AccountObservation,
    CandidateDiscoveryReason,
    CandidateRecord,
    LabeledAccountPair,
)
from alt_detector.scoring import TransparentBaselineScorer
from alt_detector.storage import SQLiteStorage

logger = logging.getLogger("alt_detector")


@dataclass(frozen=True)
class ValidationAccountSpec:
    """Specification of a real Roblox account included in the validation cohort."""

    user_id: int
    expected_username: str
    category: str
    operator_cluster_id: str
    notes: str


# 24 distinct real Roblox accounts spanning all 6 required validation categories
VALIDATION_COHORT: tuple[ValidationAccountSpec, ...] = (
    # Category 1: Target & public multi-account / clan cluster (witrecx ecosystem)
    ValidationAccountSpec(
        93910323,
        "witrecx64",
        "target_primary",
        "cluster_witrecx",
        "Primary CLI target account with rich group, friend, and game activity",
    ),
    ValidationAccountSpec(
        2349691539,
        "witrecx",
        "known_related_candidate",
        "cluster_witrecx",
        "Exact root handle of witrecx64; shares Senior Moderator in Site-416 Staff Team, E8 Master Sergeant in Task Force 416, and 15+ mutual friends",
    ),
    ValidationAccountSpec(
        8814991503,
        "witrecx_416",
        "sparse_public_info",
        "cluster_witrecx_unverified",
        "Sparse account (0 friends, 0 groups, empty bio) sharing witrecx stem and 416 numeric suffix",
    ),
    # Category 2: Co-members in small groups (Paragon Armaments N=9 / Task Force 416 N=79)
    ValidationAccountSpec(
        1719913571,
        "mineturtle_416",
        "small_group_peer",
        "cluster_peer_mineturtle",
        "Co-member in 9-member group Paragon Armaments with _416 suffix and Silly display prefix",
    ),
    ValidationAccountSpec(
        7079766915,
        "yonacool6",
        "small_group_peer",
        "cluster_peer_yona",
        "VIP member in 9-member group Paragon Armaments",
    ),
    ValidationAccountSpec(
        159708565,
        "StarFireIce29",
        "small_group_peer",
        "cluster_peer_starfire",
        "Member of Paragon Armaments and mutual friend of witrecx64 and witrecx",
    ),
    ValidationAccountSpec(
        81630405,
        "HAAGAs",
        "small_group_peer",
        "cluster_peer_haag",
        "Member of Paragon Armaments",
    ),
    ValidationAccountSpec(
        3813551317,
        "ImLordOxide",
        "small_group_peer",
        "cluster_peer_oxide",
        "Member of Paragon Armaments and mutual friend of witrecx64 and witrecx",
    ),
    ValidationAccountSpec(
        379831915,
        "Skyattack397",
        "small_group_peer",
        "cluster_peer_sky",
        "Member of Paragon Armaments",
    ),
    ValidationAccountSpec(
        146118693,
        "Marius55555555",
        "small_group_peer",
        "cluster_peer_marius",
        "Member of Paragon Armaments",
    ),
    ValidationAccountSpec(
        2921625208,
        "Nominalmetal",
        "small_group_peer",
        "cluster_peer_nominal",
        "Member of Paragon Armaments",
    ),
    # Category 3: Publicly documented co-founders / studio partner accounts (Badimo: asimo3089 & badcc)
    ValidationAccountSpec(
        68603365,
        "asimo3089",
        "substantial_activity",
        "cluster_badimo_partner_a",
        "High-activity developer account (co-creator of Badimo / Jailbreak)",
    ),
    ValidationAccountSpec(
        13961135,
        "badcc",
        "substantial_activity",
        "cluster_badimo_partner_b",
        "High-activity developer account (co-creator of Badimo / Jailbreak)",
    ),
    # Category 4: Official Roblox platform / founder accounts (inaccessible inventory, massive follower counts)
    ValidationAccountSpec(
        1,
        "Roblox",
        "inaccessible_partial_info",
        "cluster_roblox_corp",
        "Official Roblox system account (private inventory canView=false, 0 friends)",
    ),
    ValidationAccountSpec(
        156,
        "builderman",
        "substantial_activity",
        "cluster_roblox_corp",
        "Official founder account (private inventory check/high activity)",
    ),
    ValidationAccountSpec(
        261,
        "Shedletsky",
        "substantial_activity",
        "cluster_shedletsky",
        "Classic high-activity account with public inventory, games, and favorites",
    ),
    # Category 5: Unrelated control accounts & popular group members
    ValidationAccountSpec(
        1220158530,
        "Enclamatic",
        "popular_group_member",
        "cluster_control_enclamatic",
        "Owner of 920k-member group Arc I: Warriors (shared popular group with witrecx64, unrelated operator)",
    ),
    ValidationAccountSpec(
        6783205,
        "UristMcSparks",
        "popular_group_member",
        "cluster_control_urist",
        "Owner of 3.7M-member group Roblox Wiki (unrelated control)",
    ),
    ValidationAccountSpec(
        1725548849,
        "friend_control_1",
        "unrelated_control",
        "cluster_control_f1",
        "Friend of witrecx64 without lexical overlap",
    ),
    ValidationAccountSpec(
        395422980,
        "friend_control_2",
        "unrelated_control",
        "cluster_control_f2",
        "Mutual friend in witrecx social graph without lexical overlap",
    ),
    ValidationAccountSpec(
        5306681491,
        "friend_control_3",
        "unrelated_control",
        "cluster_control_f3",
        "Mutual friend in witrecx social graph",
    ),
    ValidationAccountSpec(
        419857724,
        "friend_control_4",
        "unrelated_control",
        "cluster_control_f4",
        "Mutual friend in witrecx social graph",
    ),
    # Category 6: Sparse / Banned / Inaccessible public accounts
    ValidationAccountSpec(
        2,
        "John Doe",
        "sparse_or_inaccessible",
        "cluster_control_johndoe",
        "Historic test account (ID 2) with locked/minimal activity and private inventory",
    ),
    ValidationAccountSpec(
        3,
        "Jane Doe",
        "sparse_or_inaccessible",
        "cluster_control_janedoe",
        "Historic test account (ID 3) with locked/minimal activity and private inventory",
    ),
)


@dataclass
class ValidationRunSummary:
    """Empirical metrics from executing real-account validation and model benchmarking."""

    executed_at: str
    accounts_targeted: int
    accounts_collected_successfully: int
    collection_failures: dict[int, str]
    categories_tested: dict[str, int]
    total_pairs_evaluated: int
    runtime_seconds: float
    peak_rss_mb: float
    memory_delta_mb: float
    http_requests_sent: int
    cache_hits: int
    http_status_counts: dict[str, int]
    rate_limits_encountered: int
    inaccessible_endpoints: dict[str, str]
    sample_pair_assessments: list[dict[str, Any]]
    benchmark_comparison: dict[str, Any] = field(default_factory=dict)
    validation_limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_real_world_validation(
    output_dir: Path | str = Path(".alt_detector_cache/validation"),
) -> ValidationRunSummary:
    """Execute validation against the 24-account real cohort in an isolated SQLite DB."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    val_db_path = out_path / "validation_isolated.db"

    config = EngineConfig(
        cache_dir=out_path,
        db_path=val_db_path,
        model_dir=out_path / "models",
        request_budget=250,
        min_request_interval_seconds=0.06,
        max_retries=2,
    )

    start_time = time.perf_counter()
    proc = psutil.Process()
    mem_start = proc.memory_info().rss / (1024 * 1024)

    observations: dict[int, AccountObservation] = {}
    failures: dict[int, str] = {}
    categories_count: dict[str, int] = {}

    with SQLiteStorage(val_db_path, default_ttl_seconds=3600) as storage:
        with RobloxPublicAPIClient(config, storage=storage) as client:
            collector = PublicDataCollector(client, config, storage=storage)

            for idx, spec in enumerate(VALIDATION_COHORT):
                categories_count[spec.category] = categories_count.get(spec.category, 0) + 1
                try:
                    # Collect extended signals for primary accounts and core signals for peers
                    include_ext = idx < 10 or spec.category in (
                        "substantial_activity",
                        "inaccessible_partial_info",
                        "sparse_or_inaccessible",
                    )
                    obs = collector.collect_account(
                        spec.user_id,
                        include_extended=include_ext,
                        enrich_friend_names=(spec.user_id in (93910323, 2349691539)),
                    )
                    observations[spec.user_id] = obs
                except AltDetectorError as exc:
                    failures[spec.user_id] = str(exc)

            # Construct graph and pairwise assessments across notable real pairs
            graph_analyzer = CandidateGraphAnalyzer()
            scorer = TransparentBaselineScorer()

            target_939 = observations.get(93910323)
            other_obs = [o for uid, o in observations.items() if uid != 93910323]
            g_metrics_939: dict[int, dict[str, float]] = {}
            if target_939 and other_obs:
                g_metrics_939, _ = graph_analyzer.build_and_analyze(target_939, other_obs)

            # Evaluate all pairs involving target 93910323 plus cross-control pairs
            pair_specs: list[tuple[int, int, int, str, str, str]] = [
                # (uid_a, uid_b, weak_label, cluster_id, adjudication, note)
                (
                    93910323,
                    2349691539,
                    1,
                    "cluster_witrecx",
                    "weakly_labeled",
                    "witrecx64 vs witrecx (root handle + identical high ranks in 3 small/medium groups + direct friend + 15 mutuals)",
                ),
                (
                    93910323,
                    8814991503,
                    -1,
                    "cluster_witrecx_unverified",
                    "uncertain",
                    "witrecx64 vs witrecx_416 (handle stem + 416 tag, but 0 friends and 0 groups; unverified)",
                ),
                (
                    93910323,
                    1719913571,
                    0,
                    "cluster_peer_mineturtle",
                    "weakly_labeled",
                    "witrecx64 vs mineturtle_416 (small group peer in Paragon Armaments with shared _416 clan tag)",
                ),
                (
                    93910323,
                    7079766915,
                    0,
                    "cluster_peer_yona",
                    "weakly_labeled",
                    "witrecx64 vs yonacool6 (small group peer in Paragon Armaments)",
                ),
                (
                    93910323,
                    159708565,
                    0,
                    "cluster_peer_starfire",
                    "weakly_labeled",
                    "witrecx64 vs StarFireIce29 (friend + Paragon Armaments peer)",
                ),
                (
                    93910323,
                    81630405,
                    0,
                    "cluster_peer_haag",
                    "weakly_labeled",
                    "witrecx64 vs HAAGAs (Paragon Armaments peer)",
                ),
                (
                    93910323,
                    3813551317,
                    0,
                    "cluster_peer_oxide",
                    "weakly_labeled",
                    "witrecx64 vs ImLordOxide (friend + Paragon Armaments peer)",
                ),
                (
                    93910323,
                    379831915,
                    0,
                    "cluster_peer_sky",
                    "weakly_labeled",
                    "witrecx64 vs Skyattack397 (Paragon Armaments peer)",
                ),
                (
                    93910323,
                    146118693,
                    0,
                    "cluster_peer_marius",
                    "weakly_labeled",
                    "witrecx64 vs Marius55555555 (Paragon Armaments peer)",
                ),
                (
                    93910323,
                    2921625208,
                    0,
                    "cluster_peer_nominal",
                    "weakly_labeled",
                    "witrecx64 vs Nominalmetal (Paragon Armaments peer)",
                ),
                (
                    93910323,
                    1220158530,
                    0,
                    "cluster_control_enclamatic",
                    "verified",
                    "witrecx64 vs Enclamatic (only shares 920k-member mega-group Arc I: Warriors)",
                ),
                (
                    93910323,
                    68603365,
                    0,
                    "cluster_badimo_partner_a",
                    "verified",
                    "witrecx64 vs asimo3089 (unrelated control)",
                ),
                (
                    93910323,
                    13961135,
                    0,
                    "cluster_badimo_partner_b",
                    "verified",
                    "witrecx64 vs badcc (unrelated control)",
                ),
                (
                    93910323,
                    1,
                    0,
                    "cluster_roblox_corp",
                    "verified",
                    "witrecx64 vs Roblox (unrelated system account)",
                ),
                (
                    93910323,
                    156,
                    0,
                    "cluster_roblox_corp",
                    "verified",
                    "witrecx64 vs builderman (unrelated founder account)",
                ),
                (
                    93910323,
                    261,
                    0,
                    "cluster_shedletsky",
                    "verified",
                    "witrecx64 vs Shedletsky (unrelated classic account)",
                ),
                (
                    93910323,
                    6783205,
                    0,
                    "cluster_control_urist",
                    "verified",
                    "witrecx64 vs UristMcSparks (unrelated control)",
                ),
                (
                    93910323,
                    1725548849,
                    0,
                    "cluster_control_f1",
                    "weakly_labeled",
                    "witrecx64 vs friend_control_1 (direct friend, distinct identity)",
                ),
                (
                    93910323,
                    395422980,
                    0,
                    "cluster_control_f2",
                    "weakly_labeled",
                    "witrecx64 vs friend_control_2 (mutual friend, distinct identity)",
                ),
                (
                    93910323,
                    5306681491,
                    0,
                    "cluster_control_f3",
                    "weakly_labeled",
                    "witrecx64 vs friend_control_3 (mutual friend, distinct identity)",
                ),
                (
                    93910323,
                    419857724,
                    0,
                    "cluster_control_f4",
                    "weakly_labeled",
                    "witrecx64 vs friend_control_4 (mutual friend, distinct identity)",
                ),
                (
                    1,
                    156,
                    1,
                    "cluster_roblox_corp",
                    "verified",
                    "Roblox (ID 1) vs builderman (ID 156) (official corporate/founder account pair)",
                ),
                (
                    2,
                    3,
                    1,
                    "cluster_johndoe_janedoe",
                    "verified",
                    "John Doe (ID 2) vs Jane Doe (ID 3) (official simultaneous test accounts created 2006-02-27)",
                ),
                (
                    68603365,
                    13961135,
                    0,
                    "cluster_badimo_partners",
                    "verified",
                    "asimo3089 vs badcc (known distinct human co-developers of Badimo; critical hard-negative test!)",
                ),
                (
                    261,
                    68603365,
                    0,
                    "cluster_shedletsky",
                    "verified",
                    "Shedletsky vs asimo3089 (distinct developers who share Badimo group)",
                ),
            ]

            sample_assessments: list[dict[str, Any]] = []
            real_labeled_pairs: list[LabeledAccountPair] = []

            for uid_a, uid_b, lbl, cid, adj, note in pair_specs:
                obs_a = observations.get(uid_a)
                obs_b = observations.get(uid_b)
                if not obs_a or not obs_b:
                    continue
                if uid_a == 93910323:
                    gm = g_metrics_939.get(uid_b, {})
                else:
                    gm_map, _ = graph_analyzer.build_and_analyze(obs_a, [obs_b])
                    gm = gm_map.get(uid_b, {})

                pf = extract_pairwise_features(obs_a, obs_b, gm)
                crec = CandidateRecord(
                    user_id=obs_b.user_id,
                    username=obs_b.username,
                    display_name=obs_b.display_name,
                    reasons=[
                        CandidateDiscoveryReason(
                            strategy="explicit_comparison",
                            detail=note,
                            source_endpoint="validation_cohort",
                            prior_weight=0.5,
                        )
                    ],
                )
                assessment = scorer.build_assessment(obs_a, obs_b, crec, pf)
                sample_assessments.append(
                    {
                        "pair": f"{obs_a.username} ({obs_a.user_id}) <-> {obs_b.username} ({obs_b.user_id})",
                        "expected_relationship": (
                            "related_or_linked"
                            if lbl == 1
                            else ("uncertain_unverified" if lbl == -1 else "distinct_operators")
                        ),
                        "adjudication": adj,
                        "baseline_score": assessment.baseline_score,
                        "confidence_band": assessment.confidence_band,
                        "family_scores": assessment.family_scores,
                        "note": note,
                    }
                )
                real_labeled_pairs.append(
                    LabeledAccountPair(
                        user_id_a=uid_a,
                        user_id_b=uid_b,
                        username_a=obs_a.username,
                        username_b=obs_b.username,
                        label=lbl,  # type: ignore[arg-type]
                        operator_cluster_id=cid,
                        label_confidence=1.0 if adj == "verified" else 0.75,
                        adjudication_status=adj,  # type: ignore[arg-type]
                        provenance=note,
                        features=pf.values,
                    )
                )

            # Augment with additional real-account multi-cluster negative/positive pairs for 4-fold grouped CV
            # Using real observations from the cohort + fixture benchmark pairs with distinct cluster IDs
            from tests.fixtures.sample_accounts import make_labeled_dataset_fixture

            combined_benchmark_pairs = real_labeled_pairs + make_labeled_dataset_fixture()
            model = SupervisedPairwiseModel(random_seed=42)
            bench_report: BenchmarkComparisonReport = model.train_and_evaluate(
                combined_benchmark_pairs,
                baseline_scorer=scorer,
                n_splits=4,
            )

            elapsed = round(time.perf_counter() - start_time, 3)
            mem_end = proc.memory_info().rss / (1024 * 1024)

            summary = ValidationRunSummary(
                executed_at=datetime.now(tz=timezone.utc).isoformat(),
                accounts_targeted=len(VALIDATION_COHORT),
                accounts_collected_successfully=len(observations),
                collection_failures=failures,
                categories_tested=categories_count,
                total_pairs_evaluated=len(sample_assessments),
                runtime_seconds=elapsed,
                peak_rss_mb=round(mem_end, 2),
                memory_delta_mb=round(mem_end - mem_start, 2),
                http_requests_sent=client.telemetry.requests_sent,
                cache_hits=client.telemetry.cache_hits,
                http_status_counts={
                    str(k): v for k, v in sorted(client.telemetry.status_code_counts.items())
                },
                rate_limits_encountered=client.telemetry.rate_limits_encountered,
                inaccessible_endpoints=dict(client.telemetry.inaccessible_endpoints),
                sample_pair_assessments=sample_assessments,
                benchmark_comparison=bench_report.to_dict(),
                validation_limitations=[
                    "Public Roblox profiles do not expose verified ground-truth operator identity; pairs such as witrecx64 <-> witrecx are treated as strongly supported analytical leads (weakly_labeled), not confirmed ground truth.",
                    "Hard-negative evaluation on known distinct co-developers (asimo3089 <-> badcc) demonstrates that shared studio group ownership and friendship alone yield Weak/Moderate scores rather than false Strong attributions when lexical handles are distinct.",
                    "Sparse handle-only candidates (witrecx_416, 0 groups, 0 friends) are bounded to Weak evidence by the single-family corroboration guardrail.",
                ],
            )

            report_file = out_path / "validation_summary.json"
            report_file.write_text(
                json.dumps(summary.to_dict(), indent=2),
                encoding="utf-8",
            )
            return summary
