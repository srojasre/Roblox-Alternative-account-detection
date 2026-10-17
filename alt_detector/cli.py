"""Command-line interface and orchestration pipeline for alt_detector."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Sequence

import psutil

from alt_detector import FEATURE_SCHEMA_VERSION, MODEL_VERSION, SCHEMA_VERSION, __version__
from alt_detector.api_client import RobloxPublicAPIClient
from alt_detector.candidate_discovery import CandidateDiscoveryEngine
from alt_detector.collectors import KNOWN_AUTH_RESTRICTED_SOURCES, PublicDataCollector
from alt_detector.config import EngineConfig, detect_local_resources
from alt_detector.exceptions import (
    AltDetectorError,
    InvalidInputError,
    ProfileInaccessibleError,
    RequestBudgetExceededError,
    UserNotFoundError,
)
from alt_detector.features import extract_pairwise_features
from alt_detector.graph_analysis import CandidateGraphAnalyzer
from alt_detector.model import SupervisedPairwiseModel, UnsupervisedAccountAnalyzer
from alt_detector.reporting import (
    export_json_report,
    format_terminal_report,
    safe_write_stdout,
)
from alt_detector.schemas import (
    AccountObservation,
    AnalysisReport,
    CandidateAssessment,
    CandidateDiscoveryReason,
    CandidateRecord,
    PairwiseFeatures,
)
from alt_detector.scoring import TransparentBaselineScorer
from alt_detector.storage import SQLiteStorage

logger = logging.getLogger("alt_detector")


def configure_cli_logging(verbose: bool = False, stream: object | None = None) -> None:
    """Configure real-time terminal logging matching `[HH:MM:SS] LEVEL    message`."""
    target_stream = stream or sys.stderr
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(target_stream)  # type: ignore[arg-type]
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AltDetectionEngine:
    """Local orchestration pipeline connecting collection, discovery, graph, and scoring layers."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        storage: SQLiteStorage | None = None,
        api_client: RobloxPublicAPIClient | None = None,
        supervised_model: SupervisedPairwiseModel | None = None,
    ) -> None:
        self.config = config
        self._owns_storage = storage is None
        self.storage = storage or SQLiteStorage(
            config.db_path or Path(".alt_detector_cache/alt_detector.db"),
            default_ttl_seconds=config.cache_ttl_seconds,
            max_cache_entries=config.max_cache_entries,
        )
        self._owns_client = api_client is None
        self.api_client = api_client or RobloxPublicAPIClient(config, storage=self.storage)
        self.collector = PublicDataCollector(self.api_client, config, storage=self.storage)
        self.discovery = CandidateDiscoveryEngine(self.collector, config)
        self.graph_analyzer = CandidateGraphAnalyzer()
        self.baseline_scorer = TransparentBaselineScorer(
            family_weights=config.family_weights,
            thresholds=config.thresholds,
            threshold_status=config.threshold_status,
        )
        self.unsupervised = UnsupervisedAccountAnalyzer(random_seed=config.random_seed)
        self.supervised_model = supervised_model

    def close(self) -> None:
        if self._owns_client:
            self.api_client.close()
        if self._owns_storage:
            self.storage.close()

    def __enter__(self) -> AltDetectionEngine:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def load_local_model_if_available(self) -> SupervisedPairwiseModel | None:
        """Check local model directory and load serialized model if present and valid."""
        if not self.config.enable_ml:
            logger.info("Checking local model availability: ML disabled via configuration (baseline-only mode)")
            return None
        if self.supervised_model is not None and self.supervised_model.is_trained:
            logger.info(
                "Checking local model availability: active in-memory model (%s)",
                self.supervised_model.selected_model_name,
            )
            return self.supervised_model

        model_path = (self.config.model_dir or Path(".alt_detector_cache/models")) / "pairwise_model.joblib"
        if model_path.exists():
            try:
                loaded = SupervisedPairwiseModel.load(model_path)
                self.supervised_model = loaded
                logger.info(
                    "Checking local model availability: loaded '%s' (%s) from %s",
                    loaded.selected_model_name,
                    loaded.scoring_mode,
                    model_path,
                )
                return loaded
            except Exception as exc:
                logger.warning(
                    "Checking local model availability: could not load %s (%s); falling back to heuristic baseline",
                    model_path,
                    exc,
                )
                return None

        logger.info(
            "Checking local model availability: no trained artifact at %s; operating in transparent heuristic baseline mode",
            model_path,
        )
        return None

    def analyze(
        self,
        identifier: str,
        *,
        explicit_compare: Sequence[str] | None = None,
    ) -> AnalysisReport:
        """Execute end-to-end public data collection, candidate discovery, and explainable scoring."""
        start_time = time.perf_counter()
        proc = psutil.Process()
        mem_start_mb = proc.memory_info().rss / (1024 * 1024)
        hw = detect_local_resources(self.config.max_concurrency)

        logger.info("Initializing local detection engine (v%s, CPU=%d cores)", __version__, hw.logical_cpus)
        self.storage.purge_expired()
        active_model = self.load_local_model_if_available()

        logger.info("Resolving username: %s", identifier)
        uid, canonical_username, display_name = self.collector.resolve_account(identifier)
        logger.info("Resolved user ID: %d (%s)", uid, canonical_username)

        logger.info("Collecting accessible public profile data")
        target_obs = self.collector.collect_account(
            uid,
            include_extended=True,
            enrich_friend_names=True,
            require_accessible=True,
        )
        logger.info(
            "Retrieving publicly accessible group information (%d group memberships, %d public friends)",
            len(target_obs.groups or []),
            len(target_obs.friends or []),
        )

        candidates = self.discovery.discover_candidates(target_obs)
        by_uid = {c.user_id: c for c in candidates}

        # Add any explicitly requested comparison accounts
        if explicit_compare:
            for comp_ident in explicit_compare:
                try:
                    c_uid, c_uname, c_dname = self.collector.resolve_account(comp_ident)
                    if c_uid == target_obs.user_id:
                        continue
                    reason = CandidateDiscoveryReason(
                        strategy="explicit_comparison",
                        detail=f"Explicitly requested comparison account '{comp_ident}'",
                        source_endpoint="cli:--compare",
                        prior_weight=0.90,
                    )
                    if c_uid in by_uid:
                        by_uid[c_uid].reasons.append(reason)
                    else:
                        by_uid[c_uid] = CandidateRecord(
                            user_id=c_uid,
                            username=c_uname,
                            display_name=c_dname,
                            reasons=[reason],
                        )
                except AltDetectorError as exc:
                    logger.warning("Could not resolve explicit comparison account '%s': %s", comp_ident, exc)

        ordered_candidates = sorted(
            by_uid.values(),
            key=lambda c: (-c.priority_score, -len(c.reasons), c.user_id),
        )[: self.config.max_candidates]

        discovered_count = len(by_uid)
        logger.info(
            "Discovered %d candidate account(s); analyzing top %d within request budget",
            discovered_count,
            len(ordered_candidates),
        )

        # Collect observations for candidates using bounded worker pool
        cand_observations: dict[int, AccountObservation] = {}
        top_extended_uids = {
            c.user_id for c in ordered_candidates[: self.config.extended_profile_top_k]
        }

        workers = max(1, min(self.config.max_concurrency, hw.recommended_workers, len(ordered_candidates) or 1))
        if workers == 1 or len(ordered_candidates) <= 2:
            for cand_rec in ordered_candidates:
                try:
                    obs = self.collector.collect_account(
                        cand_rec.user_id,
                        include_extended=(cand_rec.user_id in top_extended_uids),
                        enrich_friend_names=False,
                    )
                    cand_observations[cand_rec.user_id] = obs
                except RequestBudgetExceededError:
                    logger.warning("Request budget reached during candidate collection; proceeding with partial set")
                    break
                except AltDetectorError as exc:
                    logger.warning("Skipping candidate %s (%d): %s", cand_rec.username, cand_rec.user_id, exc)
        else:
            budget_warned = False
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(
                        self.collector.collect_account,
                        cand_rec.user_id,
                        include_extended=(cand_rec.user_id in top_extended_uids),
                        enrich_friend_names=False,
                    ): cand_rec
                    for cand_rec in ordered_candidates
                }
                for fut in as_completed(future_map):
                    cand_rec = future_map[fut]
                    try:
                        obs = fut.result()
                        cand_observations[cand_rec.user_id] = obs
                    except RequestBudgetExceededError:
                        if not budget_warned:
                            logger.warning("Request budget reached; evaluating collected candidates")
                            budget_warned = True
                    except AltDetectorError as exc:
                        logger.warning("Candidate %s (%d) collection failed: %s", cand_rec.username, cand_rec.user_id, exc)

        valid_cand_list = [
            cand_observations[c.user_id]
            for c in ordered_candidates
            if c.user_id in cand_observations
        ]

        logger.info("Constructing candidate account graph (%d account nodes)", 1 + len(valid_cand_list))
        graph_metrics_by_uid, graph_summary = self.graph_analyzer.build_and_analyze(
            target_obs, valid_cand_list
        )

        logger.info("Extracting account relationship features")
        features_by_uid: dict[int, PairwiseFeatures] = {}
        for cand_obs in valid_cand_list:
            gm = graph_metrics_by_uid.get(cand_obs.user_id, {})
            features_by_uid[cand_obs.user_id] = extract_pairwise_features(
                target_obs, cand_obs, graph_metrics=gm
            )

        logger.info("Evaluating candidate relationships")
        unsupervised_by_uid = self.unsupervised.analyze_candidates(
            target_obs, valid_cand_list, features_by_uid
        )

        logger.info("Generating explainable assessment")
        assessments: list[CandidateAssessment] = []
        scoring_mode = (
            active_model.scoring_mode
            if (active_model is not None and active_model.is_trained)
            else "heuristic_baseline_primary"
        )

        for cand_rec in ordered_candidates:
            if cand_rec.user_id not in cand_observations:
                continue
            cand_obs = cand_observations[cand_rec.user_id]
            pf = features_by_uid[cand_rec.user_id]
            ml_prob = (
                active_model.predict_score(pf)
                if (active_model is not None and active_model.is_trained)
                else None
            )
            unsup = unsupervised_by_uid.get(cand_rec.user_id, {})
            assessment = self.baseline_scorer.build_assessment(
                target_obs,
                cand_obs,
                cand_rec,
                pf,
                ml_score=ml_prob,
                scoring_mode=scoring_mode,
                unsupervised_signals=unsup,
                config=self.config,
            )
            assessments.append(assessment)

        # Deterministic ranking by combined_score DESC, baseline_score DESC, candidate_user_id ASC
        assessments.sort(
            key=lambda a: (-a.combined_score, -a.baseline_score, a.candidate_user_id)
        )

        elapsed = round(time.perf_counter() - start_time, 3)
        mem_end_mb = proc.memory_info().rss / (1024 * 1024)
        total_records = 1 + len(valid_cand_list)
        logger.info(
            "Analysis completed in %.2fs (%d account records processed, %d HTTP requests, %d cache hits)",
            elapsed,
            total_records,
            self.api_client.telemetry.requests_sent,
            self.api_client.telemetry.cache_hits,
        )

        status_val = (
            "no_candidates"
            if not assessments
            else (
                "partial"
                if len(valid_cand_list) < len(ordered_candidates)
                else "completed"
            )
        )

        inaccessible_merged = dict(KNOWN_AUTH_RESTRICTED_SOURCES)
        inaccessible_merged.update(self.api_client.telemetry.inaccessible_endpoints)
        unique_eps = sorted(self.api_client.telemetry.unique_endpoints)

        methodology_desc = (
            f"Layered pipeline: Layer A TransparentBaselineScorer (weights v{self.config.weights_version}) + "
            f"Layer B Supervised ({active_model.selected_model_name if active_model else 'disabled/untrained'}, "
            f"mode={scoring_mode}) + Layer C Unsupervised IsolationForest/LOF + Layer D Heterogeneous Graph."
        )

        known_limitations = [
            "Estimates reflect publicly observable correlations and must be treated as analytical leads, not definitive identity attribution.",
            "Unlinked alternative accounts that share no naming stems, friends, groups, or public assets with the target cannot be discovered via public APIs.",
            "Roblox restricts unauthenticated access to user badge lists (badges.roblox.com) and follower/following lists (friends.roblox.com) with HTTP 401.",
            "Accounts in the same small clan, roleplay division, or development team often share high ranks and mutual friends without sharing a single operator.",
        ]

        return AnalysisReport(
            schema_version=SCHEMA_VERSION,
            model_version=MODEL_VERSION,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            input_identifier=identifier,
            resolved_user_id=target_obs.user_id,
            resolved_username=target_obs.username,
            resolved_display_name=target_obs.display_name,
            status=status_val,  # type: ignore[arg-type]
            sources_queried_count=len(unique_eps),
            unique_endpoints_queried=unique_eps,
            inaccessible_endpoints=inaccessible_merged,
            candidates_discovered_count=discovered_count,
            candidates_analyzed_count=len(assessments),
            scoring_methodology=methodology_desc,
            threshold_status=(
                "validated" if self.config.threshold_status == "validated" else "provisional"
            ),
            assessments=assessments,
            graph_summary=graph_summary,
            data_freshness={
                "live_http_requests": self.api_client.telemetry.requests_sent,
                "cache_hits": self.api_client.telemetry.cache_hits,
                "retries_performed": self.api_client.telemetry.retries_performed,
                "rate_limits_encountered": self.api_client.telemetry.rate_limits_encountered,
                "http_status_counts": {
                    str(k): v for k, v in sorted(self.api_client.telemetry.status_code_counts.items())
                },
                "cache_ttl_seconds": self.config.cache_ttl_seconds if self.config.use_cache else 0,
            },
            known_limitations=known_limitations,
            runtime_seconds=elapsed,
            resource_usage={
                "rss_memory_mb": round(mem_end_mb, 2),
                "memory_delta_mb": round(mem_end_mb - mem_start_mb, 2),
                "logical_cpus": hw.logical_cpus,
                "physical_cpus": hw.physical_cpus,
                "available_ram_mb": hw.available_ram_mb,
                "gpu_available": hw.gpu_available,
                "gpu_name": hw.gpu_name,
                "worker_threads_used": workers,
            },
        )


def run_preflight(config: EngineConfig) -> int:
    """Verify configuration, local hardware, SQLite storage, and dependencies without network calls."""
    hw = detect_local_resources(config.max_concurrency)
    logger.info("Running preflight / dry-run verification")
    logger.info(
        "Hardware resources: OS=%s | Python=%s | CPUs=%d logical (%d physical) | RAM=%.1f/%.1f MB free | GPU=%s",
        hw.os_name,
        hw.python_version,
        hw.logical_cpus,
        hw.physical_cpus,
        hw.available_ram_mb,
        hw.total_ram_mb,
        hw.gpu_name if hw.gpu_available else "None (CPU-only mode active)",
    )
    with SQLiteStorage(
        config.db_path or Path(".alt_detector_cache/alt_detector.db"),
        default_ttl_seconds=config.cache_ttl_seconds,
        max_cache_entries=config.max_cache_entries,
    ) as storage:
        purged = storage.purge_expired()
        stats = storage.stats()
        logger.info(
            "Local SQLite storage verified at %s (cache_entries=%d, observations=%d, labeled_pairs=%d, expired_purged=%d)",
            config.db_path,
            stats["http_cache_entries"],
            stats["account_observations"],
            stats["labeled_pairs"],
            purged,
        )

    model_path = (config.model_dir or Path(".alt_detector_cache/models")) / "pairwise_model.joblib"
    if model_path.exists():
        loaded = SupervisedPairwiseModel.load(model_path)
        logger.info(
            "Local ML model verified: %s (mode=%s, schema=v%s)",
            loaded.selected_model_name,
            loaded.scoring_mode,
            loaded.feature_schema_version,
        )
    else:
        logger.info(
            "Local ML model artifact not present at %s (engine will run in transparent baseline mode)",
            model_path,
        )
    safe_write_stdout(
        json.dumps(
            {
                "preflight_status": "ok",
                "version": __version__,
                "hardware": {
                    "os": hw.os_name,
                    "python": hw.python_version,
                    "logical_cpus": hw.logical_cpus,
                    "physical_cpus": hw.physical_cpus,
                    "available_ram_mb": hw.available_ram_mb,
                    "gpu_available": hw.gpu_available,
                    "gpu_name": hw.gpu_name,
                },
                "storage": stats,
                "config": config.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alt_detector",
        description=(
            "Local evidence-based Roblox Alternative Account Detection Engine. "
            "Estimates whether publicly observable signals support a possible relationship between accounts."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Target Roblox username or numeric user ID (e.g., witrecx64 or 93910323)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="ACCOUNT",
        help="Optional additional usernames or user IDs to explicitly evaluate against the target",
    )
    parser.add_argument(
        "-o",
        "--json-out",
        type=Path,
        metavar="PATH",
        help="Export machine-readable JSON assessment report to PATH",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to local SQLite cache/storage file",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=18,
        metavar="N",
        help="Maximum number of discovered candidate accounts to analyze in full detail (default: 18)",
    )
    parser.add_argument(
        "--request-budget",
        type=int,
        default=110,
        metavar="N",
        help="Maximum outbound HTTP requests permitted per execution (default: 110)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Local cache expiration TTL in seconds (default: 3600)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing the local SQLite HTTP cache for this run",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run using only the Layer A TransparentBaselineScorer without ML blending",
    )
    parser.add_argument(
        "--preflight",
        "--dry-run",
        dest="preflight",
        action="store_true",
        help="Verify local hardware, dependencies, SQLite storage, and configuration without network calls",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run isolated real-account validation and grouped ML vs baseline benchmark suite",
    )
    parser.add_argument(
        "--delete-user",
        metavar="USER",
        help="Delete all locally cached records for the specified username or numeric user ID and exit",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all locally cached HTTP responses and account observations and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging and detailed feature/graph diagnostics in the report",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning standard POSIX exit codes."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_cli_logging(verbose=args.verbose)

    try:
        config = EngineConfig(
            db_path=args.db_path,
            use_cache=not args.no_cache,
            cache_ttl_seconds=args.cache_ttl,
            max_candidates=args.max_candidates,
            request_budget=args.request_budget,
            enable_ml=not args.baseline_only,
            verbose=args.verbose,
        )
    except ValueError as exc:
        logger.error("Invalid configuration parameter: %s", exc)
        return 2

    try:
        if args.delete_user:
            with SQLiteStorage(config.db_path or Path(".alt_detector_cache/alt_detector.db")) as storage:
                removed = storage.delete_account_records(args.delete_user)
            logger.info("Deleted %d local record(s) matching '%s'", removed, args.delete_user)
            safe_write_stdout(f"Deleted {removed} local record(s) for '{args.delete_user}'.")
            return 0

        if args.clear_cache:
            with SQLiteStorage(config.db_path or Path(".alt_detector_cache/alt_detector.db")) as storage:
                removed = storage.clear_all_cache()
            logger.info("Cleared %d total cached record(s) from local SQLite database", removed)
            safe_write_stdout(f"Cleared {removed} cached record(s) from local storage.")
            return 0

        if args.preflight:
            return run_preflight(config)

        if args.validate:
            from alt_detector.validation import run_real_world_validation

            summary = run_real_world_validation()
            safe_write_stdout(json.dumps(summary.to_dict(), indent=2))
            return 0

        if not args.target:
            parser.print_usage(sys.stderr)
            logger.error("A target Roblox username or numeric user ID is required (or use --preflight)")
            return 2

        with AltDetectionEngine(config) as engine:
            report = engine.analyze(args.target, explicit_compare=args.compare)
            formatted = format_terminal_report(report, verbose=config.verbose)
            safe_write_stdout(formatted)

            if args.json_out:
                saved_path = export_json_report(report, args.json_out)
                logger.info("Exported machine-readable JSON report to %s", saved_path)

        return 0

    except InvalidInputError as exc:
        logger.error("Invalid input: %s", exc)
        return 2
    except UserNotFoundError as exc:
        logger.error("Account not found: %s", exc)
        return 2
    except ProfileInaccessibleError as exc:
        logger.error("Profile inaccessible: %s", exc)
        return 1
    except AltDetectorError as exc:
        logger.error("Analysis error: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user (SIGINT); terminating cleanly")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
