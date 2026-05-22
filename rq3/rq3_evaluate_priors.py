#!/usr/bin/env python3
"""Evaluate pre-transfer priors against RQ3 standard scores (full pipeline)."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

_METRIC_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_METRIC_DIR))

from rq3_checkpoint_parse import canonical  # noqa: E402
from rq3_metric_analysis import (  # noqa: E402
    EPS,
    add_derived_prior_metrics,
    join_prior_rows,
    mean,
    median,
    metric_names_from_rows,
    pearson,
    read_prior_csv,
    score_hidden_size_granularity,
    score_prior_correlations,
    score_residual_correlations,
    score_within_compressor_target_rank,
    score_within_target_rank,
    spearman,
    write_csv,
    _float_or_none,
    _merge_discrete_priors,
)

DEFAULT_SCORES_DIR = _METRIC_DIR / "results" / "standard_scores"
DEFAULT_PRIORS = _METRIC_DIR / "results" / "priors" / "priors_consolidated_mem32.csv"

LOWER_IS_BETTER = {
    "rel_direction_l1_to_target",
    "rel_direction_rmse_to_target",
    "rel_magnitude_l1_to_target",
    "rel_magnitude_rmse_to_target",
    "memory_target_norm_gap",
    "anchor_direction_js_to_target",
    "target_direction_subspace_residual",
    "capacity_normalized_direction_rmse",
    "target_text_nll",
    "source_native_nll",
    "target_minus_source_nll",
    "abs_log_hidden_ratio",
    "pair_normalized_direction_rmse",
    "target_readability_z2_mean",
    "target_readability_tail_frac",
    "target_magnitude_tail_frac",
    "target_readability_cross_entropy",
    "source_to_target_anchor_kl",
    "source_target_direction_entropy_gap",
    "source_target_direction_dispersion_gap",
    "diagnostic_rel_direction_z2_mean",
    "diagnostic_rel_magnitude_z_abs_mean",
}

METRIC_TIERS = {
    "L0_baseline": [
        "target_hidden_size",
        "prior_target_param_b",
        "_z_target_hidden_size",
        "_z_prior_target_param_b",
    ],
    "L1_mechanism": [
        "rel_direction_rmse_to_target",
        "rel_direction_l1_to_target",
        "anchor_direction_js_to_target",
        "target_direction_subspace_residual",
        "rel_magnitude_rmse_to_target",
        "rel_magnitude_l1_to_target",
        "capacity_normalized_direction_rmse",
    ],
    "L1_pair_compatibility": [
        "pair_normalized_direction_rmse",
        "target_readability_z2_mean",
        "target_readability_tail_frac",
        "target_magnitude_tail_frac",
        "target_readability_cross_entropy",
        "source_to_target_anchor_kl",
        "target_readability_energy",
        "source_target_top_anchor_overlap",
        "source_target_direction_entropy_gap",
        "source_target_direction_dispersion_gap",
        "pair_geometry_compatibility_score",
    ],
    "L2_text": [
        "target_text_nll",
        "source_native_nll",
        "target_minus_source_nll",
    ],
    "L3_discrete": [
        "prior_same_model_family",
        "prior_log_param_ratio_target_over_source",
        "prior_param_tier_jump",
    ],
    "L4_composite": [
        "combined_hidden_direction_score",
        "combined_hidden_js_score",
        "combined_hidden_source_quality_score",
        "combined_param_direction_score",
        "combined_ratio_direction_score",
    ],
}

PRIMARY_PAIR_METRICS = [
    "pair_normalized_direction_rmse",
    "target_readability_z2_mean",
    "target_readability_tail_frac",
    "target_magnitude_tail_frac",
    "target_readability_cross_entropy",
    "source_to_target_anchor_kl",
    "target_readability_energy",
    "source_target_top_anchor_overlap",
    "source_target_direction_entropy_gap",
    "source_target_direction_dispersion_gap",
    "pair_geometry_compatibility_score",
]


def load_standard_scores(path: Path, exclude_targets: set[str]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if canonical(row["target"]) in exclude_targets:
                continue
            rows.append(
                {
                    "pair": row["pair"],
                    "encoder": row["encoder"],
                    "source": row["source"],
                    "target": row["target"],
                    "mem_tokens": int(row["mem_tokens"]),
                    "method": row.get("method", ""),
                    "origin_target_bleu": float(row["origin_target_bleu"]),
                    "transfer_bleu": float(row["transfer_bleu"]),
                    "transfer_accuracy": float(row.get("transfer_accuracy") or 0),
                    "transfer_retention": float(row["transfer_retention"]),
                }
            )
    if not rows:
        raise ValueError(f"No standard score rows after filtering: {path}")
    return rows


def prior_score(value: float, metric: str) -> float:
    """Convert metric to 'higher is better predicted retention' score."""
    if metric in LOWER_IS_BETTER:
        return -value
    return value


def score_ranking_tasks(
    standard_rows: list[dict[str, str | float]],
    prior_paths: list[Path],
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    joined, standard_by_pair = join_prior_rows(standard_rows, prior_paths)
    metric_names = metric_names_from_rows(joined)

    t1_detail: list[dict[str, str | float]] = []
    t2_detail: list[dict[str, str | float]] = []

    rows_by_target: dict[str, list[dict[str, str]]] = {}
    rows_by_compressor: dict[str, list[dict[str, str]]] = {}
    for row in joined:
        rows_by_target.setdefault(str(row["target"]), []).append(row)
        comp = f"{row['encoder']}:{row['source']}"
        rows_by_compressor.setdefault(comp, []).append(row)

    def eval_group(group_key: str, group_rows: list[dict[str, str]], task: str) -> None:
        if len(group_rows) < 3:
            return
        retentions = [standard_by_pair[str(r["pair"])] for r in group_rows]
        best_idx = max(range(len(retentions)), key=lambda i: retentions[i])
        out = t1_detail if task == "T1_select_source" else t2_detail
        for name in metric_names:
            scores: list[float] = []
            valid_idx: list[int] = []
            for i, row in enumerate(group_rows):
                v = _float_or_none(row, name)
                if v is None or math.isnan(v):
                    continue
                scores.append(prior_score(v, name))
                valid_idx.append(i)
            if len(valid_idx) < 3 or max(scores) - min(scores) <= EPS:
                continue
            prior_best_local = max(range(len(scores)), key=lambda j: scores[j])
            prior_idx = valid_idx[prior_best_local]
            hit = 1 if prior_idx == best_idx else 0
            regret = retentions[best_idx] - retentions[prior_idx]
            spearman_s = spearman(scores, [retentions[i] for i in valid_idx])
            out.append(
                {
                    "task": task,
                    "group": group_key,
                    "metric": name,
                    "n": len(valid_idx),
                    "top1_hit": hit,
                    "regret_at_1": regret,
                    "spearman": spearman_s,
                    "best_retention": retentions[best_idx],
                    "prior_pick_retention": retentions[prior_idx],
                }
            )

    for target, target_rows in sorted(rows_by_target.items()):
        eval_group(target, target_rows, "T1_select_source")
    for compressor, comp_rows in sorted(rows_by_compressor.items()):
        eval_group(compressor, comp_rows, "T2_select_target")

    def summarize(detail: list[dict[str, str | float]], task: str) -> list[dict[str, str | float]]:
        summary: list[dict[str, str | float]] = []
        for name in metric_names:
            sub = [r for r in detail if r["metric"] == name and r["task"] == task]
            if not sub:
                continue
            hits = [float(r["top1_hit"]) for r in sub]
            regrets = [float(r["regret_at_1"]) for r in sub]
            spears = [float(r["spearman"]) for r in sub if not math.isnan(float(r["spearman"]))]
            summary.append(
                {
                    "task": task,
                    "metric": name,
                    "tier": metric_tier(name),
                    "n_groups": len(sub),
                    "top1_hit_rate": mean(hits),
                    "mean_regret_at_1": mean(regrets),
                    "median_regret_at_1": median(regrets),
                    "mean_spearman": mean(spears) if spears else float("nan"),
                    "mean_abs_spearman": mean([abs(v) for v in spears]) if spears else float("nan"),
                }
            )
        return sorted(summary, key=lambda r: (-float(r["top1_hit_rate"]), -float(r["mean_abs_spearman"])))

    t1_summary = summarize(t1_detail, "T1_select_source")
    t2_summary = summarize(t2_detail, "T2_select_target")
    return t1_detail + t2_detail, t1_summary + t2_summary


def metric_tier(name: str) -> str:
    for tier, names in METRIC_TIERS.items():
        if name in names:
            return tier
    return "other"


def oriented_spearman(metric: str, spearman_value: float) -> float:
    """Convert raw Spearman to 'larger means better predicted retention' direction."""
    return -spearman_value if metric in LOWER_IS_BETTER else spearman_value


def primary_pair_metric_summary(
    correlation_rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    by_metric = {str(row["metric"]): row for row in correlation_rows}
    for metric in PRIMARY_PAIR_METRICS:
        row = by_metric.get(metric)
        if not row or not is_float(row.get("spearman")):
            continue
        raw_sp = float(row["spearman"])
        rows.append(
            {
                "metric": metric,
                "tier": metric_tier(metric),
                "raw_spearman": raw_sp,
                "oriented_spearman": oriented_spearman(metric, raw_sp),
                "n": row.get("n", ""),
            }
        )
    return sorted(rows, key=lambda row: -float(row["oriented_spearman"]))


def tier_summary_from_correlation(
    correlation_rows: list[dict[str, str | float]],
    within_target_summary: list[dict[str, str | float]] | None,
    within_compressor_summary: list[dict[str, str | float]] | None,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    corr_by_metric = {str(r["metric"]): r for r in correlation_rows}
    wt_by_metric = {str(r["metric"]): r for r in (within_target_summary or [])}
    wc_by_metric = {str(r["metric"]): r for r in (within_compressor_summary or [])}
    all_metrics = set(corr_by_metric) | set(wt_by_metric) | set(wc_by_metric)
    for metric in sorted(all_metrics):
        c = corr_by_metric.get(metric, {})
        wt = wt_by_metric.get(metric, {})
        wc = wc_by_metric.get(metric, {})
        rows.append(
            {
                "metric": metric,
                "tier": metric_tier(metric),
                "global_spearman": c.get("spearman", ""),
                "T1_mean_abs_spearman": wt.get("mean_abs_spearman", ""),
                "T2_mean_abs_spearman": wc.get("mean_abs_spearman", ""),
            }
        )
    return sorted(
        rows,
        key=lambda r: (
            {
                "L1_pair_compatibility": 0,
                "L1_mechanism": 1,
                "L2_text": 2,
                "L3_discrete": 3,
                "L4_composite": 4,
                "L0_baseline": 5,
            }.get(str(r["tier"]), 6),
            -abs(float(r["global_spearman"])) if is_float(r.get("global_spearman")) else 0,
            -float(r["T1_mean_abs_spearman"]) if is_float(r.get("T1_mean_abs_spearman")) else 0,
            -float(r["T2_mean_abs_spearman"]) if is_float(r.get("T2_mean_abs_spearman")) else 0,
        ),
    )


def is_float(value: str | float | None) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def write_rq3_report(
    standard_rows: list[dict[str, str | float]],
    scores_label: str,
    mem_tokens: int,
    correlation_rows: list[dict[str, str | float]] | None,
    ranking_summary: list[dict[str, str | float]] | None,
    tier_rows: list[dict[str, str | float]] | None,
    output_path: Path,
) -> None:
    retentions = [float(r["transfer_retention"]) for r in standard_rows]
    primary_pair_rows = primary_pair_metric_summary(correlation_rows or [])
    suite_scope = (
        "All evaluated targets are included on equal footing in this benchmark."
    )
    primary_scope = "over **all transfer pairs in the evaluated target pool**"
    lines = [
        f"# RQ3 Report ({scores_label}, mem{mem_tokens})",
        "",
        "## Question",
        "",
        "Can we predict `transfer_retention` before running transfer, using only the source compressor and raw target?",
        "",
        suite_scope,
        "",
        "### Sub-task T1: select source compressor (fixed encoder + target)",
        "",
        "Given multiple trained source compressors, which one will transfer best to a fixed target?",
        "",
        "### Sub-task T2: select target model (fixed encoder:source compressor)",
        "",
        "Given one source compressor, which target should we migrate to?",
        "",
        f"- Directions: {len(standard_rows)}",
        f"- Mean retention: {mean(retentions):.3f}",
        "",
        "## Primary All-Pair Metrics",
        "",
        f"This section evaluates the RQ3 question {primary_scope}. "
        "The primary metrics here are source-target pair compatibility metrics; "
        "`target_hidden_size` is treated only as a capacity baseline because it does not depend on the source-target pair.",
        "",
    ]
    if primary_pair_rows:
        lines.extend(
            [
                "| Pair metric | Raw Spearman | Oriented Spearman | Reading |",
                "|---|---:|---:|---|",
            ]
        )
        for row in primary_pair_rows:
            raw = float(row["raw_spearman"])
            oriented = float(row["oriented_spearman"])
            direction = "lower is better" if row["metric"] in LOWER_IS_BETTER else "higher is better"
            lines.append(f"| `{row['metric']}` | {raw:.3f} | {oriented:.3f} | {direction} |")
        lines.extend(
            [
                "",
                "Interpretation: the oriented Spearman column is the headline all-pair score. "
                "It asks whether a pre-transfer pair metric ranks higher-retention transfers ahead of lower-retention transfers across the whole dataset.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No pair compatibility metrics were found in the provided prior CSV. "
                "Use `priors_consolidated_mem*_with_pair_metrics.csv` for the primary all-pair analysis.",
                "",
            ]
        )
    lines.extend(
        [
        "## Metric tiers (storyline)",
        "",
        "- **Primary pair metrics**: source-target compatibility metrics above; these are the main all-pair transfer priors.",
        "- **L0 baseline**: `target_hidden_size` — coarse target capacity; useful baseline, not the main transfer metric.",
        "- **L1 mechanism**: direction/JS/subspace — geometry relative to target, useful for ranking subtasks.",
        "- **L2 text**: NLL features — source quality / target difficulty.",
        "- **L3 discrete**: family and parameter-ratio tags.",
        "- **L4 composite**: exploratory combined scores.",
        "",
        "T1/T2 ranking remains useful as a controlled follow-up, but it is not the primary all-pair metric analysis.",
        "",
        ]
    )
    if tier_rows:
        lines.extend(
            [
                "## Controlled / Baseline Metric Summary",
                "",
                "| Tier | Metric | Global Sp | T1 mean|Sp| | T2 mean|Sp| |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in tier_rows[:20]:
            g = row.get("global_spearman", "")
            t1 = row.get("T1_mean_abs_spearman", "")
            t2 = row.get("T2_mean_abs_spearman", "")
            gs = f"{float(g):.3f}" if is_float(g) else "n/a"
            ts1 = f"{float(t1):.3f}" if is_float(t1) else "n/a"
            ts2 = f"{float(t2):.3f}" if is_float(t2) else "n/a"
            lines.append(f"| {row['tier']} | `{row['metric']}` | {gs} | {ts1} | {ts2} |")

    if ranking_summary:
        lines.extend(["", "## Ranking task summary (top metrics)", ""])
        t1 = [r for r in ranking_summary if r.get("task") == "T1_select_source"]
        t2 = [r for r in ranking_summary if r.get("task") == "T2_select_target"]
        if t1:
            lines.append("")
            lines.append("### T1 — select source (top 8 by top-1 hit rate)")
            lines.append("")
            lines.append("| Metric | top-1 hit | mean |Spearman| | mean regret@1 |")
            lines.append("|---|---:|---:|---:|")
            for row in sorted(t1, key=lambda r: -float(r["top1_hit_rate"]))[:8]:
                lines.append(
                    f"| `{row['metric']}` | {float(row['top1_hit_rate']):.3f} | "
                    f"{float(row['mean_abs_spearman']):.3f} | {float(row['mean_regret_at_1']):.3f} |"
                )
        if t2:
            lines.append("")
            lines.append("### T2 — select target (top 8 by top-1 hit rate)")
            lines.append("")
            lines.append("| Metric | top-1 hit | mean |Spearman| | mean regret@1 |")
            lines.append("|---|---:|---:|---:|")
            for row in sorted(t2, key=lambda r: -float(r["top1_hit_rate"]))[:8]:
                lines.append(
                    f"| `{row['metric']}` | {float(row['top1_hit_rate']):.3f} | "
                    f"{float(row['mean_abs_spearman']):.3f} | {float(row['mean_regret_at_1']):.3f} |"
                )

    if correlation_rows:
        lines.extend(
            [
                "",
                "## All-Metric Global Correlation (Supplementary)",
                "",
                "This includes target-only capacity baselines such as `target_hidden_size`; those are useful controls but not the primary all-pair transfer metrics.",
                "",
                "| Metric | Spearman |",
                "|---|---:|",
            ]
        )
        for row in correlation_rows[:10]:
            lines.append(f"| `{row['metric']}` | {float(row['spearman']):.3f} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 prior evaluation on standard scores.")
    parser.add_argument("--mem-tokens", type=int, default=32)
    parser.add_argument(
        "--label",
        choices=["enc_conv", "converter_only"],
        default="enc_conv",
        help="Which standard score file to use.",
    )
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--prior-csv", type=Path, nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=_METRIC_DIR / "results")
    args = parser.parse_args()

    scores_path = args.scores_dir / f"{args.label}_mem{args.mem_tokens}.csv"
    standard_rows = load_standard_scores(scores_path, exclude_targets=set())
    n_scores = len(standard_rows)

    prior_path = (
        args.prior_csv[0]
        if args.prior_csv
        else DEFAULT_PRIORS.parent / f"priors_consolidated_mem{args.mem_tokens}.csv"
    )
    if args.prior_csv:
        prior_paths = list(args.prior_csv)
    elif prior_path.exists():
        prior_paths = [prior_path]
    else:
        legacy = sorted((_METRIC_DIR / "results").glob("priors_*_to_*.csv"))
        prior_paths = legacy
        if not prior_paths:
            raise FileNotFoundError("No prior CSV found; run server priors first.")

    analysis_dir = args.out_dir / f"analysis_mem{args.mem_tokens}" / "full" / args.label
    analysis_dir.mkdir(parents=True, exist_ok=True)

    write_csv(analysis_dir / "standard_scores_filtered.csv", standard_rows)

    joined_probe, _ = join_prior_rows(standard_rows, prior_paths)
    n_joined = len(joined_probe)
    if n_joined < n_scores:
        print(
            f"[warn] priors matched {n_joined}/{n_scores} directions; "
            "run run_rq3_priors_server.sh for missing encoder:source->target rows."
        )

    correlation_rows = score_prior_correlations(standard_rows, prior_paths)
    write_csv(analysis_dir / "prior_metric_correlations.csv", correlation_rows)
    primary_pair_rows = primary_pair_metric_summary(correlation_rows)
    if primary_pair_rows:
        write_csv(analysis_dir / "primary_pair_metric_summary.csv", primary_pair_rows)

    residual_rows = list(
        itertools.chain.from_iterable(
            score_residual_correlations(standard_rows, prior_paths, control_keys=controls)
            for controls in [("target",), ("target", "encoder")]
        )
    )
    write_csv(analysis_dir / "prior_metric_residual_correlations.csv", residual_rows)

    within_target_detail, within_target_summary = score_within_target_rank(standard_rows, prior_paths)
    write_csv(analysis_dir / "prior_metric_within_target_rank.csv", within_target_detail)
    write_csv(analysis_dir / "prior_metric_within_target_rank_summary.csv", within_target_summary)

    within_comp_detail, within_comp_summary = score_within_compressor_target_rank(standard_rows, prior_paths)
    write_csv(analysis_dir / "prior_metric_within_compressor_target_rank.csv", within_comp_detail)
    write_csv(analysis_dir / "prior_metric_within_compressor_target_rank_summary.csv", within_comp_summary)

    hidden_rows = score_hidden_size_granularity(standard_rows, prior_paths)
    write_csv(analysis_dir / "target_hidden_size_granularity.csv", hidden_rows)

    ranking_detail, ranking_summary = score_ranking_tasks(standard_rows, prior_paths)
    write_csv(analysis_dir / "prior_ranking_tasks_detail.csv", ranking_detail)
    write_csv(analysis_dir / "prior_ranking_tasks_summary.csv", ranking_summary)

    tier_rows = tier_summary_from_correlation(
        correlation_rows, within_target_summary, within_comp_summary
    )
    write_csv(analysis_dir / "metric_tier_summary.csv", tier_rows)

    write_rq3_report(
        standard_rows,
        args.label,
        args.mem_tokens,
        correlation_rows,
        ranking_summary,
        tier_rows,
        analysis_dir / "rq3_report.md",
    )

    print(f"Wrote analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
