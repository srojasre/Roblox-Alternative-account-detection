"""Terminal and JSON reporting for the Roblox Alt Account Detection Engine (Section 14)."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO
import sys

from alt_detector.model import BenchmarkComparisonReport
from alt_detector.schemas import AnalysisReport, CandidateAssessment


def format_terminal_report(
    report: AnalysisReport,
    *,
    verbose: bool = False,
    top_n: int = 8,
) -> str:
    """Format a concise, professional terminal report summarizing the analysis."""
    lines: list[str] = []
    sep = "=" * 78
    subsep = "-" * 78

    lines.append("")
    lines.append(sep)
    lines.append("ROBLOX ALTERNATIVE ACCOUNT DETECTION ENGINE - ASSESSMENT REPORT")
    lines.append(sep)
    lines.append(
        f"Target Input       : {report.input_identifier} -> "
        f"{report.resolved_username} (ID: {report.resolved_user_id}, Display: {report.resolved_display_name})"
    )
    lines.append(f"Analysis Status    : {report.status.upper()}")
    lines.append(
        f"Sources Queried    : {report.sources_queried_count} verified endpoint(s) "
        f"({', '.join(report.unique_endpoints_queried) if report.unique_endpoints_queried else 'none'})"
    )
    lines.append(
        f"Candidate Coverage : {report.candidates_discovered_count} discovered | "
        f"{report.candidates_analyzed_count} analyzed"
    )
    lines.append(
        f"Methodology        : {report.scoring_methodology} "
        f"[thresholds={report.threshold_status}, schema=v{report.schema_version}, model={report.model_version}]"
    )

    freshness = report.data_freshness
    lines.append(
        f"Data Freshness     : generated={report.generated_at} | "
        f"live_requests={freshness.get('live_http_requests', 0)}, "
        f"cache_hits={freshness.get('cache_hits', 0)}, "
        f"ttl={freshness.get('cache_ttl_seconds', 0)}s"
    )

    res = report.resource_usage
    lines.append(
        f"Runtime & Resources: {report.runtime_seconds:.2f}s elapsed | "
        f"RSS={res.get('rss_memory_mb', 0.0):.1f} MB (delta={res.get('memory_delta_mb', 0.0):+.1f} MB) | "
        f"CPU={res.get('logical_cpus', 1)} cores | GPU={'Yes' if res.get('gpu_available') else 'No (CPU mode)'}"
    )
    lines.append(subsep)

    # Filter or display candidates
    meaningful = [
        a for a in report.assessments if a.confidence_band != "Insufficient evidence"
    ]
    displayed = report.assessments[:top_n] if verbose else (
        meaningful[:top_n] if meaningful else report.assessments[: min(3, len(report.assessments))]
    )

    if not report.assessments or not meaningful:
        lines.append("CANDIDATE FINDINGS :")
        lines.append(
            "  No sufficiently supported account associations were identified from accessible "
            "public data."
        )
        lines.append(
            "  Note: Absence of public evidence does NOT imply that the target account has no "
            "alternative accounts (unlinked or private accounts cannot be discovered publicly)."
        )
        if displayed:
            lines.append("")
            lines.append("  Highest-scoring evaluated candidates (below Weak evidence threshold):")
    else:
        lines.append(
            f"LEADING CANDIDATE ASSESSMENTS (showing top {len(displayed)} of { len(report.assessments)}):"
        )

    for rank, cand in enumerate(displayed, start=1):
        lines.append("")
        ml_str = f", ML={cand.ml_score:.3f}" if cand.ml_score is not None else ""
        lines.append(
            f"  [{rank}] {cand.candidate_username} (ID: {cand.candidate_user_id}, "
            f"Display: {cand.candidate_display_name})"
        )
        lines.append(
            f"      Confidence   : {cand.confidence_band.upper()} "
            f"(Combined={cand.combined_score:.3f}, Baseline={cand.baseline_score:.3f}{ml_str})"
        )
        lines.append(f"      Interpretation: {cand.confidence_interpretation}")

        if cand.discovery_reasons:
            reasons_txt = "; ".join(
                f"{r.strategy}: {r.detail}" for r in cand.discovery_reasons[:2]
            )
            lines.append(f"      Discovery    : {reasons_txt}")

        if cand.supporting_evidence:
            lines.append("      Supporting Evidence:")
            for ev in cand.supporting_evidence[:3]:
                lines.append(
                    f"        + [{ev.strength.upper()}] {ev.signal_name} (+{ev.contribution:.3f}): "
                    f"{ev.explanation}"
                )
        else:
            lines.append("      Supporting Evidence: None above baseline threshold.")

        if cand.conflicting_evidence:
            lines.append("      Evidence Against / Counter-Indicators:")
            for ev in cand.conflicting_evidence[:3]:
                lines.append(f"        - [{ev.strength.upper()}] {ev.signal_name}: {ev.explanation}")

        if cand.alternative_explanations:
            lines.append(f"      Alt. Explanations  : {cand.alternative_explanations[0]}")

        if verbose:
            fam_str = ", ".join(f"{k}={v:.2f}" for k, v in cand.family_scores.items())
            lines.append(f"      Family Breakdown   : {fam_str}")
            if cand.graph_metrics:
                g_str = ", ".join(f"{k}={v:.3f}" for k, v in cand.graph_metrics.items())
                lines.append(f"      Graph Metrics      : {g_str}")
            if cand.unsupervised_signals:
                arch = cand.unsupervised_signals.get("profile_archetype", "unknown")
                iso = cand.unsupervised_signals.get("isolation_forest_anomaly_score", 0.0)
                lines.append(
                    f"      Unsupervised Layer : archetype={arch}, anomaly_index={iso}"
                )
            if cand.missing_evidence:
                miss_names = [m.signal_name for m in cand.missing_evidence[:8]]
                lines.append(
                    f"      Missing/Restricted : {', '.join(miss_names)} "
                    f"({len(cand.missing_evidence)} total unavailable signals)"
                )

    lines.append("")
    lines.append(subsep)
    lines.append("KNOWN LIMITATIONS & DATA AVAILABILITY NOTES:")
    for lim in report.known_limitations:
        lines.append(f"  * {lim}")
    if report.inaccessible_endpoints:
        lines.append("  * Restricted/Unauthenticated Endpoints Recorded:")
        for ep, reason in sorted(report.inaccessible_endpoints.items()):
            lines.append(f"      - {ep}: {reason}")
    lines.append(sep)
    return "\n".join(lines)


def export_json_report(report: AnalysisReport, output_path: Path | str) -> Path:
    """Write the versioned machine-readable AnalysisReport to a JSON file."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return target


def format_benchmark_report(report: BenchmarkComparisonReport) -> str:
    """Format the grouped cross-validation model vs baseline comparison table."""
    lines: list[str] = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("GROUP-ISOLATED EVALUATION BENCHMARK (ML vs TRANSPARENT BASELINE)")
    lines.append(sep)
    lines.append(
        f"Pairs Evaluated : {report.evaluated_pairs} (Positives={report.positive_pairs}, "
        f"Negatives={report.negative_pairs}, Uncertain Excluded={report.uncertain_excluded_pairs}, "
        f"Duplicates Removed={report.duplicates_removed})"
    )
    lines.append(
        f"Leakage Control : {report.unique_operator_clusters} distinct operator clusters "
        f"across {report.n_splits} group-isolated folds"
    )
    lines.append(f"Selection       : {report.selection_rationale}")
    lines.append("-" * 78)
    lines.append(
        f"{'Model Name':<26} {'Prec':>6} {'Rec':>6} {'F1':>6} {'PR-AUC':>8} "
        f"{'ROC-AUC':>8} {'Brier':>7} {'ECE':>6} {'CM [[TN,FP],[FN,TP]]'}"
    )
    lines.append("-" * 78)

    all_metrics = [report.baseline_metrics] + list(report.candidate_model_metrics.values())
    for m in all_metrics:
        lines.append(
            f"{m.model_name:<26} {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} "
            f"{m.pr_auc:>8.3f} {m.roc_auc:>8.3f} {m.brier_score:>7.3f} "
            f"{m.expected_calibration_error:>6.3f} {m.confusion_matrix}"
        )
    lines.append(sep)
    return "\n".join(lines)


def safe_write_stdout(text: str, stream: TextIO | None = None) -> None:
    """Write text safely to stdout even on Windows consoles with legacy codepages."""
    out = stream or sys.stdout
    try:
        out.write(text + "\n")
        out.flush()
    except UnicodeEncodeError:
        encoded = (text + "\n").encode("utf-8", errors="replace").decode("ascii", errors="replace")
        out.write(encoded)
        out.flush()
