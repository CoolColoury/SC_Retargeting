#!/usr/bin/env python3
"""Controlled robustness checks for RQ3 pair metrics.

This script asks whether the primary
pair metrics still predict transfer retention after controlling for target-only
capacity/text/family baselines.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # code_backup root
RESULTS_DIR = ROOT / "rq3" / "results"
PRIORS_DIR = ROOT / "rq3" / "results" / "priors"

LOWER_IS_BETTER = {
    "pair_normalized_direction_rmse",
    "target_readability_z2_mean",
    "target_readability_tail_frac",
    "target_magnitude_tail_frac",
    "target_readability_cross_entropy",
    "source_to_target_anchor_kl",
    "source_target_direction_entropy_gap",
    "source_target_direction_dispersion_gap",
    "target_text_nll",
    "source_native_nll",
    "target_hidden_size",
    "prior_target_param_b",
}

PRIMARY_PAIR_METRICS = [
    "target_readability_cross_entropy",
    "pair_normalized_direction_rmse",
    "source_to_target_anchor_kl",
    "pair_geometry_compatibility_score",
    "target_magnitude_tail_frac",
    "target_readability_tail_frac",
    "target_readability_z2_mean",
]

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    if value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def target_family(target: str) -> str:
    lowered = target.lower()
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("llama"):
        return "llama"
    if lowered.startswith("mistral"):
        return "mistral"
    return lowered.split("-")[0]


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranked[order[i:j]] = avg
        i = j
    return ranked


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(ranks(x), ranks(y))


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), controls])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def r2_score(values: np.ndarray, predictors: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(values)), predictors])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    pred = design @ beta
    total = float(((values - values.mean()) ** 2).sum())
    if total <= 1e-12:
        return float("nan")
    resid = float(((values - pred) ** 2).sum())
    return 1.0 - resid / total


def oriented(value: float, metric: str) -> float:
    return -value if metric in LOWER_IS_BETTER else value


def join_rows(mem: int, label: str) -> list[dict[str, str]]:
    score_path = RESULTS_DIR / f"analysis_mem{mem}" / "full" / label / "standard_scores_filtered.csv"
    prior_path = PRIORS_DIR / f"priors_consolidated_mem{mem}_with_pair_metrics.csv"
    priors = {row["pair"]: row for row in read_csv(prior_path)}
    rows: list[dict[str, str]] = []
    for score in read_csv(score_path):
        prior = priors.get(score["pair"])
        if not prior:
            continue
        joined = dict(prior)
        joined.update(score)
        rows.append(joined)
    return rows


def control_matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    numeric_controls = ["target_hidden_size", "target_text_nll", "prior_target_param_b"]
    families = sorted({target_family(str(row["target"])) for row in rows})
    # Drop one family level to avoid perfect collinearity with the intercept.
    family_controls = families[1:]
    cols: list[list[float]] = []
    names: list[str] = []
    for name in numeric_controls:
        vals = [as_float(row, name) for row in rows]
        if any(v is None for v in vals):
            continue
        arr = np.array([float(v) for v in vals], dtype=float)
        std = arr.std()
        if std > 1e-12:
            arr = (arr - arr.mean()) / std
        cols.append(arr.tolist())
        names.append(name)
    for family in family_controls:
        cols.append([1.0 if target_family(str(row["target"])) == family else 0.0 for row in rows])
        names.append(f"target_family={family}")
    return np.array(cols, dtype=float).T, names


def score_metric(rows: list[dict[str, str]], metric: str, controls: np.ndarray) -> dict[str, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    control_rows: list[np.ndarray] = []
    for i, row in enumerate(rows):
        x = as_float(row, metric)
        y = as_float(row, "transfer_retention")
        if x is None or y is None:
            continue
        xs.append(oriented(x, metric))
        ys.append(y)
        control_rows.append(controls[i])
    if len(xs) < 8:
        return None
    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    c_arr = np.array(control_rows, dtype=float)
    if float(x_arr.max() - x_arr.min()) <= 1e-12:
        return None
    raw_sp = spearman(x_arr, y_arr)
    raw_pr = pearson(x_arr, y_arr)
    rx = residualize(x_arr, c_arr)
    ry = residualize(y_arr, c_arr)
    base_r2 = r2_score(y_arr, c_arr)
    full_r2 = r2_score(y_arr, np.column_stack([c_arr, x_arr]))
    return {
        "n": float(len(x_arr)),
        "raw_spearman": raw_sp,
        "raw_pearson": raw_pr,
        "controlled_spearman": spearman(rx, ry),
        "controlled_pearson": pearson(rx, ry),
        "base_control_r2": base_r2,
        "metric_augmented_r2": full_r2,
        "delta_r2": full_r2 - base_r2,
    }


def leave_one_family(rows: list[dict[str, str]], metric: str) -> list[dict[str, str | float]]:
    out: list[dict[str, str | float]] = []
    for family in sorted({target_family(str(row["target"])) for row in rows}):
        sub = [row for row in rows if target_family(str(row["target"])) != family]
        if len(sub) < 8:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for row in sub:
            x = as_float(row, metric)
            y = as_float(row, "transfer_retention")
            if x is None or y is None:
                continue
            xs.append(oriented(x, metric))
            ys.append(y)
        if len(xs) >= 8 and max(xs) - min(xs) > 1e-12:
            out.append(
                {
                    "held_out_family": family,
                    "n": len(xs),
                    "metric": metric,
                    "oriented_spearman": spearman(np.array(xs), np.array(ys)),
                }
            )
    return out


def main() -> None:
    all_rows: list[dict[str, str | float]] = []
    leave_family_rows: list[dict[str, str | float]] = []
    top_metric_rows: list[dict[str, str | float]] = []
    for mem in [8, 16, 32]:
        for label in ["enc_conv", "converter_only"]:
            rows = join_rows(mem, label)
            controls, control_names = control_matrix(rows)
            for metric in PRIMARY_PAIR_METRICS:
                scored = score_metric(rows, metric, controls)
                if not scored:
                    continue
                all_rows.append(
                    {
                        "config": f"mem{mem}/{label}/full",
                        "metric": metric,
                        "tier": "pair_geometry",
                        "n": int(scored["n"]),
                        "raw_spearman": scored["raw_spearman"],
                        "controlled_spearman": scored["controlled_spearman"],
                        "raw_pearson": scored["raw_pearson"],
                        "controlled_pearson": scored["controlled_pearson"],
                        "base_control_r2": scored["base_control_r2"],
                        "metric_augmented_r2": scored["metric_augmented_r2"],
                        "delta_r2": scored["delta_r2"],
                        "controls": ";".join(control_names),
                    }
                )
            for metric in PRIMARY_PAIR_METRICS[:4]:
                for loo in leave_one_family(rows, metric):
                    loo["config"] = f"mem{mem}/{label}/full"
                    leave_family_rows.append(loo)

            pair_scores = [r for r in all_rows if r["config"] == f"mem{mem}/{label}/full"]
            if pair_scores:
                best_by_residual = max(pair_scores, key=lambda r: float(r["controlled_spearman"]))
                best_by_delta = max(pair_scores, key=lambda r: float(r["delta_r2"]))
                top_metric_rows.append(
                    {
                        "config": f"mem{mem}/{label}/full",
                        "best_residual_metric": best_by_residual["metric"],
                        "best_controlled_spearman": best_by_residual["controlled_spearman"],
                        "best_delta_r2_metric": best_by_delta["metric"],
                        "best_delta_r2": best_by_delta["delta_r2"],
                        "base_control_r2": best_by_delta["base_control_r2"],
                    }
                )

    write_csv(RESULTS_DIR / "rq3_controlled_metric_correlations.csv", all_rows)
    write_csv(RESULTS_DIR / "rq3_leave_one_family.csv", leave_family_rows)
    write_csv(RESULTS_DIR / "rq3_controlled_top_metrics.csv", top_metric_rows)

    lines = [
        "# RQ3 Controlled Robustness",
        "",
        "Main suite: full target pool. Controls: `target_hidden_size`, `target_text_nll`, "
        "`prior_target_param_b`, and target-family one-hot indicators.",
        "",
        "## Best Pair Metrics After Target-Only Controls",
        "",
        "| config | best residual metric | controlled Sp | best delta-R2 metric | delta R2 | base R2 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in top_metric_rows:
        lines.append(
            f"| {row['config']} | `{row['best_residual_metric']}` | "
            f"{float(row['best_controlled_spearman']):.3f} | `{row['best_delta_r2_metric']}` | "
            f"{float(row['best_delta_r2']):.3f} | {float(row['base_control_r2']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is intentionally stricter than the main RQ3 result. It tests whether pair metrics add signal after a "
            "target-only model already knows target size, target text NLL, target parameter count, and target family.",
            "",
            "The independent residual signal is modest. This means the paper should not claim that pair metrics fully "
            "dominate target difficulty after aggressive controls. The safer claim is that pair metrics are the main "
            "all-pair prior in the normal benchmark and retain some residual/incremental signal under target-only controls.",
            "",
            "## Leave-One-Target-Family Check",
            "",
            "The leave-one-family rows in `rq3_leave_one_family.csv` are a stricter stress test. They should be treated as "
            "supporting evidence because some held-out splits are small, but they help show whether the pair metrics survive beyond a single target family.",
        ]
    )
    (RESULTS_DIR / "rq3_controlled_robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'rq3_controlled_robustness.md'}")


if __name__ == "__main__":
    main()
