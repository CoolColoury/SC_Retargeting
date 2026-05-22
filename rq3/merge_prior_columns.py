#!/usr/bin/env python3
"""Merge sidecar prior columns into consolidated RQ3 prior CSVs.

This is useful when new prior metrics are computed after the original prior
CSV already exists. Rows are joined by `pair`; sidecar columns overwrite base
columns with the same name.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

_METRIC_DIR = Path(__file__).resolve().parent


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="Existing consolidated prior CSV.")
    parser.add_argument(
        "--sidecar",
        type=Path,
        nargs="+",
        required=True,
        help="One or more sidecar CSVs or directories containing per-direction sidecar CSVs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="*.csv", help="Pattern for sidecar directories.")
    args = parser.parse_args()

    base_rows, base_fields = read_rows(args.base)
    sidecar_paths: list[Path] = []
    for item in args.sidecar:
        if item.is_dir():
            sidecar_paths.extend(sorted(item.glob(args.pattern)))
        else:
            sidecar_paths.append(item)

    sidecar_by_pair: dict[str, dict[str, str]] = {}
    sidecar_fields: list[str] = []
    for path in sidecar_paths:
        rows, fields = read_rows(path)
        for field in fields:
            if field not in {"pair", "encoder", "source", "target"} and field not in sidecar_fields:
                sidecar_fields.append(field)
        for row in rows:
            pair = row.get("pair")
            if pair:
                sidecar_by_pair.setdefault(pair, {}).update(row)

    output_fields = list(base_fields)
    for field in sidecar_fields:
        if field not in output_fields:
            output_fields.append(field)

    merged_rows: list[dict[str, str]] = []
    matched = 0
    for row in base_rows:
        merged = dict(row)
        sidecar = sidecar_by_pair.get(row.get("pair", ""))
        if sidecar:
            matched += 1
            for key, value in sidecar.items():
                if key not in {"pair", "encoder", "source", "target"}:
                    merged[key] = value
        merged_rows.append(merged)

    write_rows(args.output, merged_rows, output_fields)
    print(f"Merged {matched}/{len(base_rows)} base rows from {len(sidecar_paths)} sidecar files -> {args.output}")


if __name__ == "__main__":
    main()
