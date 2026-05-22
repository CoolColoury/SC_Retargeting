#!/usr/bin/env python3
"""Build the RQ3 standard transfer scores and optionally score prior metrics.

Important constraint for RQ3:
    prior metrics must be computed before transfer, so this local script does
    not define priors from random/LS/converter transfer results.  It only builds
    the standard outcome from existing CSVs.  If a server-side prior CSV is
    provided, it joins by pair and reports correlations.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

_METRIC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_METRIC_DIR))
from discrete_priors import discrete_prior_features  # noqa: E402
from rq3_checkpoint_parse import (  # noqa: E402
    canonical,
    parse_ori_transfer_checkpoint,
    parse_origin_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]  # code_backup root
DEFAULT_EVAL_DIR = ROOT / "rq1" / "data"
DEFAULT_ORIGIN_EVAL_CSV = "origin_recovered_evals.csv"
DEFAULT_ORI_TRANSFER_EVAL_CSV = "ori_transfer_recovered_evals.csv"
NEW_EVAL_DIR = ROOT / "rq1" / "data"
NEW_ORIGIN_EVAL_CSV = "origin_evals.csv"
NEW_ORI_TRANSFER_EVAL_CSV = "ori_transfer_encoder_converter_evals.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MEM_TOKENS = 32
EPS = 1e-12

_EVAL_DIR: Path | None = None
_ORIGIN_EVAL_CSV: str | None = None
_ORI_TRANSFER_EVAL_CSV: str | None = None


def eval_dir() -> Path:
    return _EVAL_DIR if _EVAL_DIR is not None else DEFAULT_EVAL_DIR


def origin_eval_csv() -> str:
    return _ORIGIN_EVAL_CSV if _ORIGIN_EVAL_CSV is not None else DEFAULT_ORIGIN_EVAL_CSV


def ori_transfer_eval_csv() -> str:
    return (
        _ORI_TRANSFER_EVAL_CSV
        if _ORI_TRANSFER_EVAL_CSV is not None
        else DEFAULT_ORI_TRANSFER_EVAL_CSV
    )


def configure_eval_paths(
    eval_dir_arg: Path | None,
    origin_csv: str | None,
    ori_transfer_csv: str | None,
) -> None:
    global _EVAL_DIR, _ORIGIN_EVAL_CSV, _ORI_TRANSFER_EVAL_CSV
    if eval_dir_arg is None:
        _EVAL_DIR = None
        _ORIGIN_EVAL_CSV = origin_csv
        _ORI_TRANSFER_EVAL_CSV = ori_transfer_csv
        return
    _EVAL_DIR = eval_dir_arg
    if origin_csv is not None:
        _ORIGIN_EVAL_CSV = origin_csv
    elif eval_dir_arg == NEW_EVAL_DIR or "data_from_backups" in str(eval_dir_arg):
        _ORIGIN_EVAL_CSV = NEW_ORIGIN_EVAL_CSV
    else:
        _ORIGIN_EVAL_CSV = DEFAULT_ORIGIN_EVAL_CSV
    if ori_transfer_csv is not None:
        _ORI_TRANSFER_EVAL_CSV = ori_transfer_csv
    elif eval_dir_arg == NEW_EVAL_DIR or "data_from_backups" in str(eval_dir_arg):
        _ORI_TRANSFER_EVAL_CSV = NEW_ORI_TRANSFER_EVAL_CSV
    else:
        _ORI_TRANSFER_EVAL_CSV = DEFAULT_ORI_TRANSFER_EVAL_CSV


def safe_ratio(num: float, den: float) -> float:
    return num / den if den > EPS else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den > EPS else float("nan")


def rankdata(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and abs(order[j][1] - order[i][1]) <= EPS:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k][0]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def read_origin_bleu() -> dict[tuple[str, str, int], float]:
    origins: dict[tuple[str, str, int], float] = {}
    with (eval_dir() / origin_eval_csv()).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            encoder, target, mem = parse_origin_checkpoint(row["checkpoint_name"])
            origins[(encoder, target, mem)] = float(row["avg_bleu"])
    return origins


def build_standard_scores(mem_tokens: int) -> list[dict[str, str | float]]:
    origins = read_origin_bleu()
    rows_by_pair: dict[str, dict[str, str | float]] = {}
    with (eval_dir() / ori_transfer_eval_csv()).open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            encoder, source, target, method, mem = parse_ori_transfer_checkpoint(raw["checkpoint_name"])
            if mem != mem_tokens or method != "encoder_converter" or source == target:
                continue
            origin_bleu = origins.get((encoder, target, mem))
            if origin_bleu is None:
                continue
            transfer_bleu = float(raw["avg_bleu"])
            transfer_acc = float(raw["avg_accuracy"])
            pair = f"{encoder}:{source}->{target}"
            rows_by_pair[pair] = {
                "pair": pair,
                "encoder": encoder,
                "source": source,
                "target": target,
                "origin_target_bleu": origin_bleu,
                "transfer_bleu": transfer_bleu,
                "transfer_accuracy": transfer_acc,
                "transfer_retention": safe_ratio(transfer_bleu, origin_bleu),
            }
    return sorted(rows_by_pair.values(), key=lambda row: str(row["pair"]))


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_prior_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_float(value: str | None) -> bool:
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _merge_discrete_priors(row: dict[str, str]) -> None:
    src = (row.get("source") or "").strip()
    tgt = (row.get("target") or "").strip()
    if not src or not tgt:
        return
    for key, val in discrete_prior_features(src, tgt).items():
        if isinstance(val, float) and math.isnan(val):
            row[key] = ""
        elif isinstance(val, float) and val == int(val):
            row[key] = str(int(val))
        else:
            row[key] = str(val)


def _float_or_none(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    if not is_float(value):
        return None
    out = float(value)
    return None if math.isnan(out) else out


def _add_zscore_column(rows: list[dict[str, str]], src: str, dest: str, sign: float = 1.0) -> None:
    values = [_float_or_none(row, src) for row in rows]
    clean = [v for v in values if v is not None]
    if not clean:
        return
    mu = mean(clean)
    sd = math.sqrt(mean([(v - mu) ** 2 for v in clean]))
    if sd <= EPS:
        return
    for row, value in zip(rows, values):
        if value is not None:
            row[dest] = str(sign * (value - mu) / sd)


def add_derived_prior_metrics(rows: list[dict[str, str]]) -> None:
    """Add label-free composite metrics for analysis only."""
    _add_zscore_column(rows, "target_hidden_size", "_z_target_hidden_size", sign=1.0)
    _add_zscore_column(rows, "prior_target_param_b", "_z_prior_target_param_b", sign=1.0)
    _add_zscore_column(rows, "prior_log_param_ratio_target_over_source", "_z_log_param_ratio", sign=1.0)
    _add_zscore_column(rows, "rel_direction_rmse_to_target", "_z_neg_direction_rmse", sign=-1.0)
    _add_zscore_column(rows, "anchor_direction_js_to_target", "_z_neg_anchor_js", sign=-1.0)
    _add_zscore_column(rows, "rel_magnitude_rmse_to_target", "_z_neg_magnitude_rmse", sign=-1.0)
    _add_zscore_column(rows, "target_direction_subspace_residual", "_z_neg_subspace_residual", sign=-1.0)
    _add_zscore_column(rows, "pair_normalized_direction_rmse", "_z_neg_pair_norm_direction_rmse", sign=-1.0)
    _add_zscore_column(rows, "target_readability_z2_mean", "_z_neg_target_readability_z2", sign=-1.0)
    _add_zscore_column(rows, "target_readability_tail_frac", "_z_neg_target_readability_tail", sign=-1.0)
    _add_zscore_column(rows, "target_magnitude_tail_frac", "_z_neg_target_magnitude_tail", sign=-1.0)
    _add_zscore_column(rows, "target_readability_cross_entropy", "_z_neg_target_readability_ce", sign=-1.0)
    _add_zscore_column(rows, "source_to_target_anchor_kl", "_z_neg_source_to_target_kl", sign=-1.0)
    _add_zscore_column(rows, "target_readability_energy", "_z_target_readability_energy", sign=1.0)
    _add_zscore_column(rows, "source_target_top_anchor_overlap", "_z_top_anchor_overlap", sign=1.0)
    _add_zscore_column(rows, "source_target_direction_entropy_gap", "_z_neg_direction_entropy_gap", sign=-1.0)
    _add_zscore_column(rows, "source_target_direction_dispersion_gap", "_z_neg_direction_dispersion_gap", sign=-1.0)
    _add_zscore_column(rows, "source_native_nll", "_z_neg_source_native_nll", sign=-1.0)

    for row in rows:
        hidden = _float_or_none(row, "_z_target_hidden_size")
        param = _float_or_none(row, "_z_prior_target_param_b")
        ratio = _float_or_none(row, "_z_log_param_ratio")
        direction = _float_or_none(row, "_z_neg_direction_rmse")
        js = _float_or_none(row, "_z_neg_anchor_js")
        magnitude = _float_or_none(row, "_z_neg_magnitude_rmse")
        subspace = _float_or_none(row, "_z_neg_subspace_residual")
        pair_norm = _float_or_none(row, "_z_neg_pair_norm_direction_rmse")
        target_z2 = _float_or_none(row, "_z_neg_target_readability_z2")
        target_tail = _float_or_none(row, "_z_neg_target_readability_tail")
        magnitude_tail = _float_or_none(row, "_z_neg_target_magnitude_tail")
        target_ce = _float_or_none(row, "_z_neg_target_readability_ce")
        target_kl = _float_or_none(row, "_z_neg_source_to_target_kl")
        target_energy = _float_or_none(row, "_z_target_readability_energy")
        top_overlap = _float_or_none(row, "_z_top_anchor_overlap")
        source_quality = _float_or_none(row, "_z_neg_source_native_nll")

        if hidden is not None and direction is not None:
            row["combined_hidden_direction_score"] = str(hidden + direction)
        if hidden is not None and js is not None:
            row["combined_hidden_js_score"] = str(hidden + js)
        if hidden is not None and source_quality is not None:
            row["combined_hidden_source_quality_score"] = str(hidden + source_quality)
        if param is not None and direction is not None:
            row["combined_param_direction_score"] = str(param + direction)
        if ratio is not None and direction is not None:
            row["combined_ratio_direction_score"] = str(ratio + direction)
        pair_terms = [
            v
            for v in [
                direction,
                js,
                magnitude,
                subspace,
                pair_norm,
                target_z2,
                target_tail,
                magnitude_tail,
                target_ce,
                target_kl,
                target_energy,
                top_overlap,
            ]
            if v is not None
        ]
        if len(pair_terms) >= 2:
            row["pair_geometry_compatibility_score"] = str(mean(pair_terms))


def join_prior_rows(
    standard_rows: list[dict[str, str | float]],
    prior_paths: list[Path],
) -> tuple[list[dict[str, str]], dict[str, float]]:
    standard_by_pair = {str(row["pair"]): float(row["transfer_retention"]) for row in standard_rows}
    prior_rows: list[dict[str, str]] = []
    for prior_path in prior_paths:
        prior_rows.extend(read_prior_csv(prior_path))
    for row in prior_rows:
        _merge_discrete_priors(row)
    joined = [row for row in prior_rows if row.get("pair") in standard_by_pair]
    if not joined:
        raise ValueError(f"No prior rows match standard pairs in {prior_paths}")
    add_derived_prior_metrics(joined)
    return joined, standard_by_pair


def metric_names_from_rows(rows: list[dict[str, str]]) -> list[str]:
    return [
        name
        for name in rows[0].keys()
        if name not in {"pair", "encoder", "source", "target"} and any(is_float(row.get(name)) for row in rows)
    ]


def score_prior_correlations(standard_rows: list[dict[str, str | float]], prior_paths: list[Path]) -> list[dict[str, str | float]]:
    joined, standard_by_pair = join_prior_rows(standard_rows, prior_paths)
    metric_names = metric_names_from_rows(joined)

    out: list[dict[str, str | float]] = []
    for name in metric_names:
        xs: list[float] = []
        ys: list[float] = []
        for row in joined:
            value = row.get(name)
            if not is_float(value):
                continue
            x = float(value)
            if math.isnan(x):
                continue
            xs.append(x)
            ys.append(standard_by_pair[row["pair"]])
        out.append(
            {
                "metric": name,
                "n": len(xs),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
            }
        )
    out = [row for row in out if not math.isnan(float(row["pearson"])) and not math.isnan(float(row["spearman"]))]
    return sorted(out, key=lambda row: -abs(float(row["spearman"])) if not math.isnan(float(row["spearman"])) else -1.0)


def _residual_map(rows: list[dict[str, str]], value_by_id: dict[int, float], keys: tuple[str, ...]) -> dict[int, float]:
    groups: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        row_id = id(row)
        if row_id not in value_by_id:
            continue
        group_key = tuple(str(row[k]) for k in keys)
        groups.setdefault(group_key, []).append(value_by_id[row_id])
    means = {k: mean(vs) for k, vs in groups.items()}
    out: dict[int, float] = {}
    for row in rows:
        row_id = id(row)
        if row_id not in value_by_id:
            continue
        group_key = tuple(str(row[k]) for k in keys)
        out[row_id] = value_by_id[row_id] - means[group_key]
    return out


def score_residual_correlations(
    standard_rows: list[dict[str, str | float]],
    prior_paths: list[Path],
    control_keys: tuple[str, ...] = ("target",),
) -> list[dict[str, str | float]]:
    joined, standard_by_pair = join_prior_rows(standard_rows, prior_paths)
    metric_names = metric_names_from_rows(joined)

    y_raw = {id(row): standard_by_pair[str(row["pair"])] for row in joined}
    y_res = _residual_map(joined, y_raw, control_keys)

    out: list[dict[str, str | float]] = []
    for name in metric_names:
        x_raw: dict[int, float] = {}
        for row in joined:
            value = row.get(name)
            if not is_float(value):
                continue
            x = float(value)
            if math.isnan(x):
                continue
            x_raw[id(row)] = x
        x_res = _residual_map(joined, x_raw, control_keys)
        common_ids = [row_id for row_id in x_res if row_id in y_res]
        if len(common_ids) < 2:
            continue
        xs = [x_res[row_id] for row_id in common_ids]
        ys = [y_res[row_id] for row_id in common_ids]
        if max(xs) - min(xs) <= EPS:
            continue
        p = pearson(xs, ys)
        s = spearman(xs, ys)
        if math.isnan(p) or math.isnan(s):
            continue
        out.append(
            {
                "metric": name,
                "n": len(common_ids),
                "controls": "+".join(control_keys),
                "pearson_residual": p,
                "spearman_residual": s,
            }
        )
    return sorted(out, key=lambda row: -abs(float(row["spearman_residual"])))


def score_within_target_rank(
    standard_rows: list[dict[str, str | float]],
    prior_paths: list[Path],
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    joined, standard_by_pair = join_prior_rows(standard_rows, prior_paths)
    metric_names = metric_names_from_rows(joined)
    rows_by_target: dict[str, list[dict[str, str]]] = {}
    for row in joined:
        rows_by_target.setdefault(str(row["target"]), []).append(row)

    detail_rows: list[dict[str, str | float]] = []
    for target, target_rows in sorted(rows_by_target.items()):
        for name in metric_names:
            xs: list[float] = []
            ys: list[float] = []
            for row in target_rows:
                value = row.get(name)
                if not is_float(value):
                    continue
                x = float(value)
                if math.isnan(x):
                    continue
                xs.append(x)
                ys.append(standard_by_pair[str(row["pair"])])
            if len(xs) < 3 or max(xs) - min(xs) <= EPS:
                continue
            p = pearson(xs, ys)
            s = spearman(xs, ys)
            if math.isnan(p) or math.isnan(s):
                continue
            detail_rows.append(
                {
                    "target": target,
                    "metric": name,
                    "n": len(xs),
                    "pearson": p,
                    "spearman": s,
                }
            )

    summary_rows: list[dict[str, str | float]] = []
    for name in metric_names:
        metric_rows = [row for row in detail_rows if row["metric"] == name]
        if not metric_rows:
            continue
        spearmans = [float(row["spearman"]) for row in metric_rows]
        pearsons = [float(row["pearson"]) for row in metric_rows]
        summary_rows.append(
            {
                "metric": name,
                "n_targets": len(metric_rows),
                "mean_pearson": mean(pearsons),
                "median_pearson": median(pearsons),
                "mean_spearman": mean(spearmans),
                "median_spearman": median(spearmans),
                "mean_abs_spearman": mean([abs(v) for v in spearmans]),
                "positive_spearman_targets": sum(1 for v in spearmans if v > 0),
                "negative_spearman_targets": sum(1 for v in spearmans if v < 0),
            }
        )
    summary_rows = sorted(summary_rows, key=lambda row: -abs(float(row["mean_spearman"])))
    detail_rows = sorted(detail_rows, key=lambda row: (str(row["target"]), -abs(float(row["spearman"]))))
    return detail_rows, summary_rows


def score_within_compressor_target_rank(
    standard_rows: list[dict[str, str | float]],
    prior_paths: list[Path],
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    joined, standard_by_pair = join_prior_rows(standard_rows, prior_paths)
    metric_names = metric_names_from_rows(joined)
    rows_by_compressor: dict[str, list[dict[str, str]]] = {}
    for row in joined:
        compressor = f"{row['encoder']}:{row['source']}"
        rows_by_compressor.setdefault(compressor, []).append(row)

    detail_rows: list[dict[str, str | float]] = []
    for compressor, compressor_rows in sorted(rows_by_compressor.items()):
        for name in metric_names:
            xs: list[float] = []
            ys: list[float] = []
            for row in compressor_rows:
                x = _float_or_none(row, name)
                if x is None:
                    continue
                xs.append(x)
                ys.append(standard_by_pair[str(row["pair"])])
            if len(xs) < 3 or max(xs) - min(xs) <= EPS:
                continue
            p = pearson(xs, ys)
            s = spearman(xs, ys)
            if math.isnan(p) or math.isnan(s):
                continue
            detail_rows.append(
                {
                    "compressor": compressor,
                    "metric": name,
                    "n": len(xs),
                    "pearson": p,
                    "spearman": s,
                }
            )

    summary_rows: list[dict[str, str | float]] = []
    for name in metric_names:
        metric_rows = [row for row in detail_rows if row["metric"] == name]
        if not metric_rows:
            continue
        spearmans = [float(row["spearman"]) for row in metric_rows]
        pearsons = [float(row["pearson"]) for row in metric_rows]
        summary_rows.append(
            {
                "metric": name,
                "n_compressors": len(metric_rows),
                "mean_pearson": mean(pearsons),
                "median_pearson": median(pearsons),
                "mean_spearman": mean(spearmans),
                "median_spearman": median(spearmans),
                "mean_abs_spearman": mean([abs(v) for v in spearmans]),
                "positive_spearman_compressors": sum(1 for v in spearmans if v > 0),
                "negative_spearman_compressors": sum(1 for v in spearmans if v < 0),
            }
        )
    summary_rows = sorted(summary_rows, key=lambda row: -abs(float(row["mean_spearman"])))
    detail_rows = sorted(detail_rows, key=lambda row: (str(row["compressor"]), -abs(float(row["spearman"]))))
    return detail_rows, summary_rows


def score_hidden_size_granularity(
    standard_rows: list[dict[str, str | float]],
    prior_paths: list[Path],
) -> list[dict[str, str | float]]:
    joined, standard_by_pair = join_prior_rows(standard_rows, prior_paths)
    rows: list[dict[str, str | float]] = []
    groups: dict[float, list[dict[str, str]]] = {}
    for row in joined:
        hidden = _float_or_none(row, "target_hidden_size")
        if hidden is not None:
            groups.setdefault(hidden, []).append(row)

    for hidden, hidden_rows in sorted(groups.items()):
        retentions = [standard_by_pair[str(row["pair"])] for row in hidden_rows]
        targets = sorted({str(row["target"]) for row in hidden_rows})
        rows.append(
            {
                "target_hidden_size": hidden,
                "n_rows": len(hidden_rows),
                "n_targets": len(targets),
                "targets": " ".join(targets),
                "mean_retention": mean(retentions),
                "median_retention": median(retentions),
                "min_retention": min(retentions),
                "max_retention": max(retentions),
            }
        )

    all_hidden = []
    all_retention = []
    hidden_rank = []
    target_mean_rank = []
    target_mean: dict[str, float] = {}
    for target in sorted({str(row["target"]) for row in joined}):
        target_rows = [row for row in joined if str(row["target"]) == target]
        target_mean[target] = mean([standard_by_pair[str(row["pair"])] for row in target_rows])

    for row in joined:
        hidden = _float_or_none(row, "target_hidden_size")
        if hidden is None:
            continue
        all_hidden.append(hidden)
        all_retention.append(standard_by_pair[str(row["pair"])])
        hidden_rank.append(hidden)
        target_mean_rank.append(target_mean[str(row["target"])])

    if all_hidden:
        rows.insert(
            0,
            {
                "target_hidden_size": "ALL",
                "n_rows": len(all_hidden),
                "n_targets": len(target_mean),
                "targets": "all",
                "mean_retention": mean(all_retention),
                "median_retention": median(all_retention),
                "min_retention": min(all_retention),
                "max_retention": max(all_retention),
                "spearman_hidden_vs_rows": spearman(all_hidden, all_retention),
                "spearman_hidden_vs_target_means": spearman(
                    [next(_float_or_none(row, "target_hidden_size") for row in joined if str(row["target"]) == target) for target in sorted(target_mean)],
                    [target_mean[target] for target in sorted(target_mean)],
                ),
                "n_unique_hidden_values": len(groups),
            },
        )
    return rows


METRIC_DESCRIPTIONS = {
    "target_hidden_size": "Target decoder hidden dimension. It is a pure target-capacity prior and the strongest global coordinate.",
    "prior_target_param_b": "Nominal target parameter count in billions from short model names. It is a coarse target-scale prior.",
    "prior_log_param_ratio_target_over_source": "Log parameter ratio log(target_B/source_B). It adds source-size directionality to the target-scale prior.",
    "prior_same_model_family": "Binary source-target family match from short names (Llama/Qwen/Mistral). It is intentionally coarse.",
    "rel_direction_l1_to_target": "Mean absolute distance between source-memory anchor-direction profiles and the target raw anchor-profile mean.",
    "rel_direction_rmse_to_target": "RMSE version of the anchor-direction mismatch. Lower is the primary geometric compatibility hypothesis.",
    "capacity_normalized_direction_rmse": "Direction RMSE divided by log(target_hidden_size), intended to reduce the dominant target-capacity trend.",
    "anchor_direction_js_to_target": "Jensen-Shannon divergence between source-memory anchor-direction distributions and target raw manifold average distribution.",
    "target_direction_subspace_residual": "Residual energy after projecting source-memory direction profiles into the target raw anchor-profile PCA subspace.",
    "memory_target_norm_gap": "Absolute gap between source-memory log norm and target-anchor log norm.",
    "rel_magnitude_l1_to_target": "Mean absolute mismatch in anchor-relative log-norm profiles.",
    "rel_magnitude_rmse_to_target": "RMSE mismatch in anchor-relative log-norm profiles.",
    "target_text_nll": "Raw target LM negative log-likelihood on evaluation text. It measures target text difficulty, not memory compatibility.",
    "source_native_nll": "Native source compressor NLL on the same text.",
    "target_minus_source_nll": "Target raw text NLL minus source compressor native NLL.",
    "abs_log_hidden_ratio": "Absolute log hidden-size ratio between source decoder and target decoder.",
    "pair_normalized_direction_rmse": "Direction RMSE normalized by both source-memory and target-anchor direction dispersion.",
    "target_readability_z2_mean": "Mean squared z-score of source-memory direction profiles under the target readable direction profile.",
    "target_readability_tail_frac": "Fraction of source-memory direction coordinates with absolute target-readable z-score above 2.",
    "target_magnitude_tail_frac": "Fraction of source-memory magnitude coordinates with absolute target-readable z-score above 2.",
    "target_readability_cross_entropy": "Cross-entropy of source-memory anchor-direction mass under the target anchor-direction distribution.",
    "source_to_target_anchor_kl": "Asymmetric KL from source-memory anchor-direction mass to the target anchor-direction distribution.",
    "target_readability_energy": "Dot product between source-memory anchor-direction mass and target anchor-direction mass. Higher means source memories put more mass on target-preferred anchors.",
    "source_target_top_anchor_overlap": "Top-k anchor overlap between source-memory direction profiles and the target anchor distribution.",
    "source_target_direction_entropy_gap": "Absolute entropy gap between source-memory and target-anchor direction distributions.",
    "source_target_direction_dispersion_gap": "Absolute log dispersion ratio between source-memory and target-anchor direction profiles.",
    "combined_hidden_direction_score": "Label-free composite z(target_hidden_size) - z(rel_direction_rmse_to_target). Higher means larger target capacity and smaller direction mismatch.",
    "combined_hidden_js_score": "Label-free composite z(target_hidden_size) - z(anchor_direction_js_to_target). Higher means larger target capacity and smaller JS distance.",
    "combined_hidden_source_quality_score": "Label-free composite z(target_hidden_size) - z(source_native_nll). Higher means larger target capacity and better source native quality.",
    "combined_param_direction_score": "Label-free composite z(target_param_b) - z(rel_direction_rmse_to_target).",
    "combined_ratio_direction_score": "Label-free composite z(log target/source parameter ratio) - z(rel_direction_rmse_to_target).",
    "pair_geometry_compatibility_score": "Label-free average of oriented source-target pair geometry terms. Higher means better pair compatibility without using target capacity or source NLL.",
}


def _first_row(rows: list[dict[str, str | float]], metric: str) -> dict[str, str | float] | None:
    for row in rows:
        if row.get("metric") == metric:
            return row
    return None


def write_detailed_analysis_report(
    standard_rows: list[dict[str, str | float]],
    correlation_rows: list[dict[str, str | float]] | None,
    residual_rows: list[dict[str, str | float]] | None,
    within_summary_rows: list[dict[str, str | float]] | None,
    within_compressor_summary_rows: list[dict[str, str | float]] | None,
    hidden_granularity_rows: list[dict[str, str | float]] | None,
    mem_tokens: int,
    output_path: Path,
) -> None:
    hidden_all = hidden_granularity_rows[0] if hidden_granularity_rows else None
    lines = [
        "# RQ3 Detailed Metric Analysis Report",
        "",
        "## Objective",
        "",
        "RQ3 asks whether transfer performance can be predicted before running a transfer experiment. "
        f"The label is `transfer_retention = Enc-Conv BLEU / Origin(target) BLEU` at mem{mem_tokens}. "
        "All prior metrics here use only the trained source compressor, the raw target model, short model metadata, and evaluation text.",
        "",
        "## Recommended Use",
        "",
        "Use `target_hidden_size` as the simplest final global coordinate. "
        "It is not a mechanistic source-target compatibility measure, and its granularity is coarse. "
        "It should therefore be treated as a target-capacity baseline rather than a sufficient prior metric.",
        "",
        "Use `rel_direction_rmse_to_target` as a mechanistic companion rather than the main coordinate. "
        "It directly tests whether source memories have target-compatible anchor-relative direction structure, but its global correlation is weaker and much of its signal is target-dependent.",
        "",
        "Use within-target rank analysis to answer the narrower question: for a fixed target, which source transfers better? "
        "In the current results, the strongest within-target source-ranking signal is `source_native_nll` (lower is better), with `target_minus_source_nll` carrying the same information with the opposite sign inside a fixed target.",
        "",
        "Use within-compressor rank analysis to answer the other practical question: for a fixed compressor, which target should be selected? "
        "This is where target-capacity metrics and target-specific compatibility metrics should be evaluated.",
        "",
        "## Metric Definitions",
        "",
    ]
    for metric, desc in METRIC_DESCRIPTIONS.items():
        lines.append(f"- `{metric}`: {desc}")

    if correlation_rows:
        lines.extend(
            [
                "",
                "## Global Correlation Summary",
                "",
                "Global correlation mixes target capacity, source quality, and compatibility. It is appropriate for choosing one simple x-axis, but not sufficient to prove a compatibility mechanism.",
                "",
                "| Metric | N | Pearson | Spearman |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in correlation_rows[:12]:
            lines.append(f"| `{row['metric']}` | {row['n']} | {float(row['pearson']):.3f} | {float(row['spearman']):.3f} |")

    if hidden_granularity_rows:
        lines.extend(
            [
                "",
                "## Target Hidden Size Granularity Check",
                "",
                "`target_hidden_size` is a coarse variable with many tied rows. This section checks whether its high Spearman score mostly reflects target-level grouping.",
                "",
            ]
        )
        if hidden_all:
            lines.extend(
                [
                    f"- Unique hidden-size values: {hidden_all.get('n_unique_hidden_values')}.",
                    f"- Row-level Spearman(hidden, retention): {float(hidden_all.get('spearman_hidden_vs_rows', float('nan'))):.3f}.",
                    f"- Target-mean Spearman(hidden, mean retention): {float(hidden_all.get('spearman_hidden_vs_target_means', float('nan'))):.3f}.",
                    "",
                ]
            )
        lines.extend(
            [
                "| Hidden Size | Rows | Targets | Mean Retention | Min | Max |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in hidden_granularity_rows[1:]:
            lines.append(
                f"| {row['target_hidden_size']} | {row['n_rows']} | {row['n_targets']} | "
                f"{float(row['mean_retention']):.3f} | {float(row['min_retention']):.3f} | {float(row['max_retention']):.3f} |"
            )

    if residual_rows:
        target_rows = [row for row in residual_rows if row.get("controls") == "target"]
        target_encoder_rows = [row for row in residual_rows if row.get("controls") == "target+encoder"]
        lines.extend(
            [
                "",
                "## Residual Correlation Summary",
                "",
                "Residual correlation subtracts a categorical baseline before scoring metrics. "
                "The `target` control removes target-level capacity and target-specific difficulty; `target+encoder` also removes encoder-level baselines within each target.",
                "",
                "| Metric | Controls | Spearman(res) | Interpretation |",
                "|---|---|---:|---|",
            ]
        )
        for metric in [
            "anchor_direction_js_to_target",
            "rel_direction_rmse_to_target",
            "capacity_normalized_direction_rmse",
            "target_direction_subspace_residual",
            "prior_same_model_family",
            "source_native_nll",
        ]:
            for row in [_first_row(target_rows, metric), _first_row(target_encoder_rows, metric)]:
                if not row:
                    continue
                desc = "Residual signal after removing target baseline."
                if metric == "target_direction_subspace_residual":
                    desc = "Weak residual signal; mostly target-capacity confounded."
                elif metric == "anchor_direction_js_to_target":
                    desc = "Has residual rank signal, but direction is not aligned with the lower-is-better hypothesis."
                elif metric == "rel_direction_rmse_to_target":
                    desc = "Mechanistic signal weakens and flips after target control."
                lines.append(f"| `{metric}` | `{row['controls']}` | {float(row['spearman_residual']):.3f} | {desc} |")

    if within_summary_rows:
        lines.extend(
            [
                "",
                "## Within-Target Rank Summary",
                "",
                "This analysis computes rank correlation separately inside each target group, then averages across targets. "
                "It is the cleanest view for the question: holding the target fixed, which source transfers better?",
                "",
            "The strongest current within-target signal is source quality: lower `source_native_nll` ranks sources closer to the observed transfer-retention ordering than the anchor-geometry metrics do.",
            "",
                "| Metric | Targets | Mean Spearman | Median Spearman | Mean Abs Spearman | + Targets | - Targets |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in within_summary_rows[:15]:
            lines.append(
                f"| `{row['metric']}` | {row['n_targets']} | {float(row['mean_spearman']):.3f} | "
                f"{float(row['median_spearman']):.3f} | {float(row['mean_abs_spearman']):.3f} | "
                f"{row['positive_spearman_targets']} | {row['negative_spearman_targets']} |"
            )

    if within_compressor_summary_rows:
        lines.extend(
            [
                "",
                "## Within-Compressor Target-Selection Summary",
                "",
                "This analysis computes rank correlation separately for each fixed compressor (`encoder:source`) while varying the target. "
                "It directly addresses: given one compressor, which target model should be selected?",
                "",
                "| Metric | Compressors | Mean Spearman | Median Spearman | Mean Abs Spearman | + Compressors | - Compressors |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in within_compressor_summary_rows[:15]:
            lines.append(
                f"| `{row['metric']}` | {row['n_compressors']} | {float(row['mean_spearman']):.3f} | "
                f"{float(row['median_spearman']):.3f} | {float(row['mean_abs_spearman']):.3f} | "
                f"{row['positive_spearman_compressors']} | {row['negative_spearman_compressors']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The strongest global signal is target capacity (`target_hidden_size`). "
            "However, the granularity check makes clear that this is a coarse target-level baseline, not a fine-grained compatibility metric.",
            "",
            "The strongest within-target source-ranking signal is `source_native_nll`: sources whose own compressor has lower native NLL on the evaluation text tend to transfer better into a fixed target. "
            "This suggests separating the two RQ3 use cases: source native quality for selecting among compressors for one target, and target-side capacity/compatibility for selecting among targets for one compressor.",
            "",
            "The geometric metrics (`rel_direction_rmse_to_target`, `anchor_direction_js_to_target`) are better treated as mechanism probes or components in combined scores. "
            "Their standalone within-target rank signals are weaker and directionally mixed, so the conservative conclusion is that current anchor-profile distances are diagnostics rather than final predictors.",
            "",
            "The subspace residual is not recommended as a final coordinate in this implementation because its global signal is dominated by target-level effects and it contributes little after target control.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    standard_rows: list[dict[str, str | float]],
    correlation_rows: list[dict[str, str | float]] | None,
    residual_rows: list[dict[str, str | float]] | None,
    within_summary_rows: list[dict[str, str | float]] | None,
    within_compressor_summary_rows: list[dict[str, str | float]] | None,
    hidden_granularity_rows: list[dict[str, str | float]] | None,
    mem_tokens: int,
    output_path: Path,
) -> None:
    retentions = [float(row["transfer_retention"]) for row in standard_rows]
    low = sorted(standard_rows, key=lambda row: float(row["transfer_retention"]))[:6]
    high = sorted(standard_rows, key=lambda row: float(row["transfer_retention"]))[-6:]

    lines = [
        "# RQ3 Metric Analysis",
        "",
        "## Standard Score",
        "",
        f"`transfer_retention = Enc-Conv BLEU / Origin(target) BLEU` at mem{mem_tokens}.",
        "",
        f"- Rows: {len(standard_rows)} non-self transfer directions.",
        f"- Mean retention: {mean(retentions):.3f}.",
        "",
        "Lowest retention:",
    ]
    for row in low:
        lines.append(f"- `{row['pair']}`: retention={float(row['transfer_retention']):.3f}, BLEU={float(row['transfer_bleu']):.3f}.")
    lines.append("")
    lines.append("Highest retention:")
    for row in high:
        lines.append(f"- `{row['pair']}`: retention={float(row['transfer_retention']):.3f}, BLEU={float(row['transfer_bleu']):.3f}.")

    if correlation_rows:
        lines.extend(
            [
                "",
                "## Prior Correlations",
                "",
                "These metrics come from a server-side prior CSV and must use only source compressor + target raw model information.",
                "",
                "| Metric | N | Pearson | Spearman |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in correlation_rows:
            lines.append(
                f"| `{row['metric']}` | {row['n']} | {float(row['pearson']):.3f} | {float(row['spearman']):.3f} |"
            )
        if residual_rows:
            lines.extend(
                [
                    "",
                    "## Residual Correlations",
                    "",
                    "Residual analysis removes target-level baseline first, then correlates residual prior values with residual retention.",
                    "",
                    "| Metric | N | Controls | Pearson(res) | Spearman(res) |",
                    "|---|---:|---|---:|---:|",
                ]
            )
            for row in residual_rows:
                lines.append(
                    f"| `{row['metric']}` | {row['n']} | `{row['controls']}` | "
                    f"{float(row['pearson_residual']):.3f} | {float(row['spearman_residual']):.3f} |"
                )
        if within_summary_rows:
            lines.extend(
                [
                    "",
                    "## Within-Target Rank Summary",
                    "",
                    "Rank correlation is computed separately within each target group and then summarized across targets.",
                    "",
                    "| Metric | Targets | Mean Spearman | Median Spearman | Mean Abs Spearman | + Targets | - Targets |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in within_summary_rows:
                lines.append(
                    f"| `{row['metric']}` | {row['n_targets']} | {float(row['mean_spearman']):.3f} | "
                    f"{float(row['median_spearman']):.3f} | {float(row['mean_abs_spearman']):.3f} | "
                    f"{row['positive_spearman_targets']} | {row['negative_spearman_targets']} |"
                )
        if within_compressor_summary_rows:
            lines.extend(
                [
                    "",
                    "## Within-Compressor Target-Selection Summary",
                    "",
                    "Rank correlation is computed separately within each fixed compressor while target varies.",
                    "",
                    "| Metric | Compressors | Mean Spearman | Median Spearman | Mean Abs Spearman | + Compressors | - Compressors |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in within_compressor_summary_rows:
                lines.append(
                    f"| `{row['metric']}` | {row['n_compressors']} | {float(row['mean_spearman']):.3f} | "
                    f"{float(row['median_spearman']):.3f} | {float(row['mean_abs_spearman']):.3f} | "
                    f"{row['positive_spearman_compressors']} | {row['negative_spearman_compressors']} |"
                )
        if hidden_granularity_rows:
            all_row = hidden_granularity_rows[0]
            lines.extend(
                [
                    "",
                    "## Target Hidden Size Granularity",
                    "",
                    f"- Unique hidden-size values: {all_row.get('n_unique_hidden_values')}.",
                    f"- Row-level Spearman(hidden, retention): {float(all_row.get('spearman_hidden_vs_rows', float('nan'))):.3f}.",
                    f"- Target-mean Spearman(hidden, mean retention): {float(all_row.get('spearman_hidden_vs_target_means', float('nan'))):.3f}.",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "## Prior Correlations",
                "",
                "No prior CSV was provided. Run `compute_pretransfer_priors.py` on the server, concatenate its outputs, then pass it with `--prior_csv`.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def with_mem_suffix(filename: str, mem_tokens: int) -> Path:
    path = Path(filename)
    return OUT_DIR / f"{path.stem}_mem{mem_tokens}{path.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem_tokens",
        type=int,
        default=DEFAULT_MEM_TOKENS,
        help="Memory token setting used to build standard transfer scores (e.g., 8/16/32).",
    )
    parser.add_argument(
        "--residual_controls",
        nargs="*",
        default=["target"],
        help="Categorical controls for residual correlation, e.g. target or target encoder.",
    )
    parser.add_argument(
        "--prior_csv",
        type=Path,
        nargs="*",
        default=None,
        help="Optional one or more server-side prior CSVs to correlate with standard scores.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory with origin/ori_transfer eval CSVs (default: rq1/data).",
    )
    parser.add_argument(
        "--origin-csv",
        type=str,
        default=None,
        help="Origin eval CSV filename under --eval-dir.",
    )
    parser.add_argument(
        "--ori-transfer-csv",
        type=str,
        default=None,
        help="Ori-transfer eval CSV filename under --eval-dir.",
    )
    args = parser.parse_args()
    mem_tokens = args.mem_tokens
    configure_eval_paths(args.eval_dir, args.origin_csv, args.ori_transfer_csv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    standard_scores_path = with_mem_suffix("standard_transfer_scores.csv", mem_tokens)
    prior_corr_path = with_mem_suffix("prior_metric_correlations.csv", mem_tokens)
    prior_residual_path = with_mem_suffix("prior_metric_residual_correlations.csv", mem_tokens)
    within_target_path = with_mem_suffix("prior_metric_within_target_rank.csv", mem_tokens)
    within_target_summary_path = with_mem_suffix("prior_metric_within_target_rank_summary.csv", mem_tokens)
    within_compressor_path = with_mem_suffix("prior_metric_within_compressor_target_rank.csv", mem_tokens)
    within_compressor_summary_path = with_mem_suffix("prior_metric_within_compressor_target_rank_summary.csv", mem_tokens)
    hidden_granularity_path = with_mem_suffix("target_hidden_size_granularity.csv", mem_tokens)
    detailed_report_path = with_mem_suffix("rq3_detailed_analysis_report.md", mem_tokens)
    report_path = with_mem_suffix("rq3_metric_report.md", mem_tokens)

    standard_rows = build_standard_scores(mem_tokens)
    write_csv(standard_scores_path, standard_rows)

    correlation_rows = None
    residual_rows = None
    within_target_rows = None
    within_target_summary_rows = None
    within_compressor_rows = None
    within_compressor_summary_rows = None
    hidden_granularity_rows = None
    if args.prior_csv:
        correlation_rows = score_prior_correlations(standard_rows, args.prior_csv)
        write_csv(prior_corr_path, correlation_rows)
        control_options = [tuple(["target"])]
        if args.residual_controls:
            control_options.append(tuple(args.residual_controls))
        control_options = list(dict.fromkeys(control_options))
        residual_rows = list(
            itertools.chain.from_iterable(
                score_residual_correlations(standard_rows, args.prior_csv, control_keys=controls)
                for controls in control_options
            )
        )
        write_csv(prior_residual_path, residual_rows)
        within_target_rows, within_target_summary_rows = score_within_target_rank(standard_rows, args.prior_csv)
        write_csv(within_target_path, within_target_rows)
        write_csv(within_target_summary_path, within_target_summary_rows)
        within_compressor_rows, within_compressor_summary_rows = score_within_compressor_target_rank(standard_rows, args.prior_csv)
        write_csv(within_compressor_path, within_compressor_rows)
        write_csv(within_compressor_summary_path, within_compressor_summary_rows)
        hidden_granularity_rows = score_hidden_size_granularity(standard_rows, args.prior_csv)
        write_csv(hidden_granularity_path, hidden_granularity_rows)
        write_detailed_analysis_report(
            standard_rows,
            correlation_rows,
            residual_rows,
            within_target_summary_rows,
            within_compressor_summary_rows,
            hidden_granularity_rows,
            mem_tokens,
            detailed_report_path,
        )

    write_report(
        standard_rows,
        correlation_rows,
        residual_rows,
        within_target_summary_rows,
        within_compressor_summary_rows,
        hidden_granularity_rows,
        mem_tokens,
        report_path,
    )
    print(f"Wrote {standard_scores_path}")
    if correlation_rows is not None:
        print(f"Wrote {prior_corr_path}")
    if residual_rows is not None:
        print(f"Wrote {prior_residual_path}")
    if within_target_rows is not None:
        print(f"Wrote {within_target_path}")
        print(f"Wrote {within_target_summary_path}")
    if within_compressor_rows is not None:
        print(f"Wrote {within_compressor_path}")
        print(f"Wrote {within_compressor_summary_path}")
    if hidden_granularity_rows is not None:
        print(f"Wrote {hidden_granularity_path}")
    if within_target_rows is not None:
        print(f"Wrote {detailed_report_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
