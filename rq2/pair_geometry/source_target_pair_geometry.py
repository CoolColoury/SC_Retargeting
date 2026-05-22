#!/usr/bin/env python3
"""Analyze directionality and pair-geometry asymmetry for RQ2.

This script focuses on matched reverse directions, e.g. encoder:A->B versus
encoder:B->A. It uses existing mem32 Enc-Conv retention and prior CSVs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # code_backup root
METRIC_DIR = ROOT / "rq3"
DEFAULT_SCORES = ROOT / "rq3" / "results" / "standard_scores" / "enc_conv_mem32.csv"
DEFAULT_PRIORS = ROOT / "rq3" / "results" / "priors" / "priors_consolidated_mem32.csv"
DEFAULT_OUT_DIR = ROOT / "rq2" / "results" / "pair_geometry"


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


def as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value else float("nan")


def make_pair(encoder: str, source: str, target: str) -> str:
    return f"{encoder}:{source}->{target}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--priors-csv", type=Path, default=DEFAULT_PRIORS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    scores = {row["pair"]: row for row in read_csv(args.scores_csv)}
    priors = {row["pair"]: row for row in read_csv(args.priors_csv)}

    rows: list[dict[str, str | float]] = []
    seen: set[tuple[str, str, str]] = set()
    for pair, row in scores.items():
        encoder = row["encoder"]
        source = row["source"]
        target = row["target"]
        key = tuple(sorted([source, target]) + [encoder])
        if key in seen:
            continue
        seen.add(key)
        reverse_pair = make_pair(encoder, target, source)
        if reverse_pair not in scores:
            continue

        fwd = scores[pair]
        rev = scores[reverse_pair]
        fwd_prior = priors.get(pair, {})
        rev_prior = priors.get(reverse_pair, {})

        fwd_ret = float(fwd["transfer_retention"])
        rev_ret = float(rev["transfer_retention"])
        fwd_rmse = as_float(fwd_prior, "rel_direction_rmse_to_target")
        rev_rmse = as_float(rev_prior, "rel_direction_rmse_to_target")
        fwd_js = as_float(fwd_prior, "anchor_direction_js_to_target")
        rev_js = as_float(rev_prior, "anchor_direction_js_to_target")
        fwd_norm_gap = as_float(fwd_prior, "memory_target_norm_gap")
        rev_norm_gap = as_float(rev_prior, "memory_target_norm_gap")

        rows.append(
            {
                "encoder": encoder,
                "source_a": source,
                "source_b": target,
                "pair_ab": pair,
                "pair_ba": reverse_pair,
                "retention_ab": fwd_ret,
                "retention_ba": rev_ret,
                "retention_abs_gap": abs(fwd_ret - rev_ret),
                "direction_rmse_ab": fwd_rmse,
                "direction_rmse_ba": rev_rmse,
                "direction_rmse_abs_gap": abs(fwd_rmse - rev_rmse),
                "anchor_js_ab": fwd_js,
                "anchor_js_ba": rev_js,
                "anchor_js_abs_gap": abs(fwd_js - rev_js),
                "norm_gap_ab": fwd_norm_gap,
                "norm_gap_ba": rev_norm_gap,
                "norm_gap_abs_gap": abs(fwd_norm_gap - rev_norm_gap),
            }
        )

    rows = sorted(rows, key=lambda r: -float(r["retention_abs_gap"]))
    write_csv(args.out_dir / "directionality_pair_geometry.csv", rows)

    lines = [
        "# Source-Target Pair Geometry",
        "",
        "Matched reverse directions at mem32 Enc-Conv.",
        "",
        "## Largest retention asymmetries",
        "",
        "| Encoder | A->B | B->A | Ret gap | RMSE gap | JS gap |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows[:12]:
        lines.append(
            f"| {row['encoder']} | {row['pair_ab']} ({float(row['retention_ab']):.3f}) | "
            f"{row['pair_ba']} ({float(row['retention_ba']):.3f}) | "
            f"{float(row['retention_abs_gap']):.3f} | {float(row['direction_rmse_abs_gap']):.3f} | "
            f"{float(row['anchor_js_abs_gap']):.3f} |"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pair_geometry_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'directionality_pair_geometry.csv'}")
    print(f"Wrote {args.out_dir / 'pair_geometry_report.md'}")


if __name__ == "__main__":
    main()
