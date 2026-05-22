#!/usr/bin/env python3
"""Build RQ3 standard transfer scores from deduped eval CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

_METRIC_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(_METRIC_DIR))

from rq3_checkpoint_parse import (  # noqa: E402
    canonical,
    make_pair,
    parse_ori_transfer_checkpoint,
    parse_origin_checkpoint,
)

DEFAULT_EVAL_DIR = ROOT / "rq1" / "data"
DEFAULT_OUT_DIR = _METRIC_DIR / "results" / "standard_scores"
EPS = 1e-12

METHOD_FILES = {
    "encoder_converter": "ori_transfer_encoder_converter_evals.csv",
    "converter_only": "ori_transfer_converter_only_evals.csv",
}


def safe_ratio(num: float, den: float) -> float:
    return num / den if den > EPS else 0.0


def load_origin_bleu(eval_dir: Path) -> dict[tuple[str, str, int], float]:
    origins: dict[tuple[str, str, int], float] = {}
    path = eval_dir / "origin_evals.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            encoder, target, mem = parse_origin_checkpoint(row["checkpoint_name"])
            origins[(encoder, target, mem)] = float(row["avg_bleu"])
    return origins


def build_scores_for_method(
    eval_dir: Path,
    method: str,
    mem_tokens: int,
    origins: dict[tuple[str, str, int], float],
    exclude_targets: set[str],
    exclude_self: bool,
    min_origin_bleu: float | None,
) -> tuple[list[dict[str, str | float]], list[str]]:
    transfer_file = METHOD_FILES[method]
    path = eval_dir / transfer_file
    rows_out: list[dict[str, str | float]] = []
    warnings: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                encoder, source, target, parsed_method, mem = parse_ori_transfer_checkpoint(
                    raw["checkpoint_name"]
                )
            except ValueError as exc:
                warnings.append(f"parse_fail: {raw['checkpoint_name']}: {exc}")
                continue
            if parsed_method != method or mem != mem_tokens:
                continue
            if exclude_self and source == target:
                continue
            if target in exclude_targets:
                continue
            origin_bleu = origins.get((encoder, target, mem))
            if origin_bleu is None:
                warnings.append(
                    f"missing_origin: encoder={encoder} target={target} mem={mem} "
                    f"checkpoint={raw['checkpoint_name']}"
                )
                continue
            if min_origin_bleu is not None and origin_bleu < min_origin_bleu:
                continue
            transfer_bleu = float(raw["avg_bleu"])
            transfer_acc = float(raw.get("avg_accuracy") or 0)
            pair = make_pair(encoder, source, target)
            rows_out.append(
                {
                    "pair": pair,
                    "encoder": encoder,
                    "source": source,
                    "target": target,
                    "mem_tokens": mem,
                    "method": method,
                    "checkpoint_name": raw["checkpoint_name"],
                    "origin_target_bleu": origin_bleu,
                    "transfer_bleu": transfer_bleu,
                    "transfer_accuracy": transfer_acc,
                    "transfer_retention": safe_ratio(transfer_bleu, origin_bleu),
                }
            )

    rows_out.sort(key=lambda r: str(r["pair"]))
    return rows_out, warnings


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_report(
    path: Path,
    eval_dir: Path,
    all_warnings: dict[str, list[str]],
    counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RQ3 Standard Score Coverage Report",
        "",
        f"Eval dir: `{eval_dir}`",
        "",
        "## Row counts",
        "",
    ]
    for key, n in sorted(counts.items()):
        lines.append(f"- {key}: {n}")
    lines.append("")
    for label, warns in sorted(all_warnings.items()):
        lines.append(f"## Warnings: {label}")
        lines.append("")
        if not warns:
            lines.append("(none)")
        else:
            parse_fails = [w for w in warns if w.startswith("parse_fail")]
            missing = [w for w in warns if w.startswith("missing_origin")]
            lines.append(f"- parse failures: {len(parse_fails)}")
            lines.append(f"- missing origin: {len(missing)}")
            if missing[:20]:
                lines.append("")
                lines.append("### Sample missing origin (up to 20)")
                for w in missing[:20]:
                    lines.append(f"- {w}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RQ3 standard transfer score tables.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--mem-tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Memory token settings to export.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_FILES),
        default=list(METHOD_FILES),
    )
    parser.add_argument("--exclude-targets", nargs="*", default=[])
    parser.add_argument("--exclude-self", action="store_true", default=True)
    parser.add_argument("--no-exclude-self", action="store_false", dest="exclude_self")
    parser.add_argument("--min-origin-bleu", type=float, default=None)
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=_METRIC_DIR / "results" / "rq3_coverage_report.md",
    )
    args = parser.parse_args()

    exclude_targets = {canonical(t) for t in args.exclude_targets}
    origins = load_origin_bleu(args.eval_dir)
    all_warnings: dict[str, list[str]] = {}
    counts: dict[str, int] = {}

    for mem in args.mem_tokens:
        for method in args.methods:
            label = f"{method}_mem{mem}"
            rows, warns = build_scores_for_method(
                args.eval_dir,
                method,
                mem,
                origins,
                exclude_targets,
                args.exclude_self,
                args.min_origin_bleu,
            )
            all_warnings[label] = warns
            counts[label] = len(rows)
            method_slug = "enc_conv" if method == "encoder_converter" else "converter_only"
            out_path = args.out_dir / f"{method_slug}_mem{mem}.csv"
            write_csv(out_path, rows)
            print(f"Wrote {out_path} ({len(rows)} rows, {len(warns)} warnings)")

    write_coverage_report(args.coverage_report, args.eval_dir, all_warnings, counts)
    print(f"Wrote {args.coverage_report}")


if __name__ == "__main__":
    main()
