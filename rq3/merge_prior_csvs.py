#!/usr/bin/env python3
"""Merge per-direction prior CSVs into one consolidated file."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

_METRIC_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = _METRIC_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem", type=int, required=True, help="Memory token setting, e.g. 32.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--pattern",
        type=str,
        default="priors_*_to_*_mem{mem}.csv",
        help="Glob pattern relative to results-dir; {mem} is substituted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: results/rq3/priors/priors_consolidated_mem{N}.csv).",
    )
    args = parser.parse_args()

    pattern = args.pattern.format(mem=args.mem)
    paths = sorted(args.results_dir.glob(pattern))
    if not paths and args.mem == 32:
        # Legacy naming: priors_{encoder}_{source}_to_{target}.csv
        legacy = sorted(args.results_dir.glob("priors_*_to_*.csv"))
        paths = [p for p in legacy if "_mem" not in p.name]

    if not paths:
        raise FileNotFoundError(f"No prior CSVs matched {pattern} under {args.results_dir}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                rows.append(dict(row))

    if not rows:
        raise ValueError("No prior rows merged.")

    out = args.output or (_METRIC_DIR / "results" / "priors" / f"priors_consolidated_mem{args.mem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(paths)} files, {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
