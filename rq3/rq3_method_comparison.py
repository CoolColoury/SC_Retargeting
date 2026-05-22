#!/usr/bin/env python3
"""Compare Enc-Conv vs converter-only vs LS/random transfer outcomes."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

_METRIC_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES_DIR = _METRIC_DIR / "results" / "standard_scores"
DEFAULT_EVAL_DIR = ROOT / "rq1" / "data"
EPS = 1e-12


def load_scores(path: Path) -> dict[tuple[str, str, str, str, int], float]:
    out: dict[tuple[str, str, str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                row["encoder"],
                row["source"],
                row["target"],
                row.get("method", ""),
                int(row["mem_tokens"]),
            )
            out[key] = float(row["transfer_retention"])
    return out


def load_baseline_bleu(path: Path) -> dict[tuple[str, str, int], float]:
    """Map (source_model_short, target_model_short, mem) -> avg_bleu from ls/random CSVs."""
    out: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ckpt = row["checkpoint_name"]
            if "_mem" not in ckpt:
                continue
            mem = int(ckpt.split("_mem")[1].split("_")[0])
            parts = ckpt.split("_to_")
            if len(parts) < 2:
                continue
            src_part = parts[0].split("_")[-1] if "mem" in parts[0] else parts[0]
            tgt_part = parts[1].split("_mem")[0].split("_")[-1]
            # checkpoint like llama1b_to_qwen7b_mem16_len128_ds_4gpu_to_...
            if "_mem" in parts[1]:
                tgt_part = parts[1].split("_mem")[0]
                if "_to_" in ckpt:
                    segs = ckpt.split("_to_")
                    src_part = segs[0].replace("llama1b_to_", "").split("_mem")[0] if "llama1b" in ckpt else segs[0]
            bleu = float(row.get("avg_bleu") or row.get("avg_transfer_accuracy") or 0)
            if bleu > 0:
                out[(src_part, tgt_part, mem)] = bleu
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--mem-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=_METRIC_DIR / "results" / "rq3_when_enc_conv_needed.md",
    )
    args = parser.parse_args()

    enc = load_scores(args.scores_dir / f"enc_conv_mem{args.mem_tokens}.csv")
    conv = load_scores(args.scores_dir / f"converter_only_mem{args.mem_tokens}.csv")

    merged: list[dict[str, float | str]] = []
    for key, ret_enc in enc.items():
        encoder, source, target, method, mem = key
        ret_conv = conv.get((encoder, source, target, "converter_only", mem))
        if ret_conv is None:
            continue
        delta = ret_enc - ret_conv
        ratio = ret_enc / ret_conv if ret_conv > EPS else float("nan")
        merged.append(
            {
                "encoder": encoder,
                "source": source,
                "target": target,
                "mem_tokens": mem,
                "retention_enc_conv": ret_enc,
                "retention_converter_only": ret_conv,
                "delta_retention": delta,
                "ratio_enc_over_conv": ratio,
            }
        )

    by_target: dict[str, list[float]] = defaultdict(list)
    conv_sufficient = 0
    enc_needed = 0
    for row in merged:
        by_target[str(row["target"])].append(float(row["delta_retention"]))
        if float(row["retention_converter_only"]) >= 0.9 * float(row["retention_enc_conv"]):
            conv_sufficient += 1
        if float(row["delta_retention"]) > 0.15 or str(row["target"]) == "qwen3b":
            enc_needed += 1

    lines = [
        f"# When is Enc-Conv needed? (mem{args.mem_tokens})",
        "",
        f"Compared {len(merged)} directions with both Enc-Conv and converter-only scores.",
        "",
        "## Rule of thumb",
        "",
        "- **Converter-only sufficient**: `retention_converter_only >= 0.9 * retention_enc_conv` "
        f"({conv_sufficient}/{len(merged)} directions, {100*conv_sufficient/max(len(merged),1):.1f}%).",
        "- **Enc-Conv clearly helps**: `delta_retention > 0.15` or target is `qwen3b` "
        f"({enc_needed}/{len(merged)} directions).",
        "",
        "## Mean delta_retention (enc - conv) by target",
        "",
        "| Target | Mean delta | N |",
        "|---|---:|---:|",
    ]
    for target in sorted(by_target):
        vals = by_target[target]
        lines.append(f"| {target} | {mean(vals):.3f} | {len(vals)} |")

    lines.extend(
        [
            "",
            "## Largest Enc-Conv gains (top 10 delta)",
            "",
            "| Pair | ret_enc | ret_conv | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(merged, key=lambda r: -float(r["delta_retention"]))[:10]:
        pair = f"{row['encoder']}:{row['source']}->{row['target']}"
        lines.append(
            f"| {pair} | {float(row['retention_enc_conv']):.3f} | "
            f"{float(row['retention_converter_only']):.3f} | {float(row['delta_retention']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Where converter-only matches Enc-Conv (top 10 smallest delta)",
            "",
            "| Pair | ret_enc | ret_conv | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(merged, key=lambda r: float(r["delta_retention"]))[:10]:
        pair = f"{row['encoder']}:{row['source']}->{row['target']}"
        lines.append(
            f"| {pair} | {float(row['retention_enc_conv']):.3f} | "
            f"{float(row['retention_converter_only']):.3f} | {float(row['delta_retention']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Note on LS/random baselines",
            "",
            "See `rq1/data/ls_transfer_evals.csv` and `random_transfer_evals.csv` "
            "for baseline transfer quality; pairing keys differ from ori_transfer and require "
            "separate alignment for per-direction comparison.",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
