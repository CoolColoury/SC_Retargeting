#!/usr/bin/env python3
"""Recover text metrics from saved evaluation_results.json files.

This script does not load checkpoints or regenerate text. It recomputes BLEU
and ROUGE-L from the already saved `generated_text`. In `saved_cropped` mode it
uses the saved cropped reference text. In `shift_aligned` mode it tries to align
the reference window to the saved generated text after the known prefix
over-trimming bug (`n_mem_tokens + 1` tokens).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

try:
    from nltk.tokenize import wordpunct_tokenize as _wordpunct_tokenize
except Exception:  # pragma: no cover - server environments vary.
    _wordpunct_tokenize = None

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional for offline CSV-only use.
    AutoTokenizer = None


def tokenize(text: str) -> list[str]:
    if _wordpunct_tokenize is not None:
        return _wordpunct_tokenize(text or "")
    return TOKEN_RE.findall(text or "")


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def modified_precision(reference: list[str], prediction: list[str], n: int) -> float:
    if len(prediction) < n:
        return 0.0
    pred_counts = ngram_counts(prediction, n)
    ref_counts = ngram_counts(reference, n)
    clipped = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
    total = sum(pred_counts.values())
    return clipped / total if total else 0.0


def weighted_bleu(reference_text: str, prediction_text: str, weights: tuple[float, ...]) -> float:
    reference = tokenize(reference_text)
    prediction = tokenize(prediction_text)
    if not reference or not prediction:
        return 0.0

    positive_orders = [i + 1 for i, w in enumerate(weights) if w > 0]
    if not positive_orders:
        return 0.0

    precisions = {n: modified_precision(reference, prediction, n) for n in positive_orders}
    if any(precisions[n] <= 0.0 for n in positive_orders):
        return 0.0

    ref_len = len(reference)
    pred_len = len(prediction)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    log_precision = sum(weights[n - 1] * math.log(precisions[n]) for n in positive_orders)
    return brevity_penalty * math.exp(log_precision)


def rouge_l_f1(reference_text: str, prediction_text: str) -> float:
    reference = tokenize(reference_text)
    prediction = tokenize(prediction_text)
    if not reference or not prediction:
        return 0.0

    prev = [0] * (len(prediction) + 1)
    for ref_tok in reference:
        curr = [0]
        for j, pred_tok in enumerate(prediction, start=1):
            if ref_tok == pred_tok:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr

    lcs = prev[-1]
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def text_metrics(reference_text: str, prediction_text: str) -> dict[str, float | bool | int]:
    ref_words = (reference_text or "").lower().split()
    pred_words = (prediction_text or "").lower().split()
    ref_counter = Counter(ref_words)
    pred_counter = Counter(pred_words)
    overlap = sum((ref_counter & pred_counter).values())
    precision = overlap / len(pred_words) if pred_words else 0.0
    recall = overlap / len(ref_words) if ref_words else 0.0
    word_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "exact_match": reference_text.strip() == prediction_text.strip(),
        "char_similarity": difflib.SequenceMatcher(None, reference_text, prediction_text).ratio(),
        "word_precision": precision,
        "word_recall": recall,
        "word_f1": word_f1,
        "bleu": weighted_bleu(reference_text, prediction_text, (0.25, 0.25, 0.25, 0.25)),
        "bleu1": weighted_bleu(reference_text, prediction_text, (1.0, 0.0, 0.0, 0.0)),
        "bleu2": weighted_bleu(reference_text, prediction_text, (0.0, 1.0, 0.0, 0.0)),
        "bleu3": weighted_bleu(reference_text, prediction_text, (0.0, 0.0, 1.0, 0.0)),
        "bleu4": weighted_bleu(reference_text, prediction_text, (0.0, 0.0, 0.0, 1.0)),
        "rougeL": rouge_l_f1(reference_text, prediction_text),
        "reference_words": len(ref_words),
        "generated_words": len(pred_words),
    }


def load_eval(eval_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    with eval_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON: {eval_file}")

    results = data.get("results")
    if isinstance(results, list):
        nested_summary = data.get("summary")
        if isinstance(nested_summary, dict):
            summary = nested_summary
        else:
            summary = {k: v for k, v in data.items() if k != "results"}
        return summary, results, data

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    if results is None and isinstance(summary.get("results"), list):
        results = summary["results"]
    if not isinstance(results, list):
        raise ValueError(f"Cannot find results list: {eval_file}")

    return summary, results, data


def saved_reference(result: dict[str, Any]) -> str:
    """Use the same saved cropped-reference policy for every sample."""
    return (
        result.get("original_text_truncated")
        or result.get("truncated_text")
        or result.get("original_text")
        or ""
    )


def full_reference(result: dict[str, Any], dataset_index: dict[str, str] | None = None) -> str:
    text = result.get("text") or result.get("original_text")
    if text:
        return str(text)
    if dataset_index is not None:
        text_id = str(result.get("text_id", ""))
        if text_id in dataset_index:
            return dataset_index[text_id]
        match = re.search(r"(\d+)$", text_id)
        if match and match.group(1) in dataset_index:
            return dataset_index[match.group(1)]
    return saved_reference(result)


def load_dataset_index(dataset_path: str | None) -> dict[str, str] | None:
    if not dataset_path:
        return None
    path = Path(dataset_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    out: dict[str, str] = {}
    for idx, item in enumerate(data):
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            if item.get("id") is not None:
                out[str(item["id"])] = text
            out[str(idx)] = text
        else:
            out[str(idx)] = str(item)
    return out


def normalize_token_id(token_id: Any) -> int | None:
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        if not token_id:
            return None
        return int(token_id[0])
    return int(token_id)


def strip_special_ids(ids: list[int], tokenizer: Any) -> list[int]:
    bos = normalize_token_id(getattr(tokenizer, "bos_token_id", None))
    eos = normalize_token_id(getattr(tokenizer, "eos_token_id", None))
    pad = normalize_token_id(getattr(tokenizer, "pad_token_id", None))
    out = [int(tok) for tok in ids if pad is None or int(tok) != pad]
    if out and bos is not None and out[0] == bos:
        out = out[1:]
    if out and eos is not None and out[-1] == eos:
        out = out[:-1]
    return out


def infer_n_mem_tokens(summary: dict[str, Any], eval_file: Path) -> int | None:
    if summary.get("n_mem_tokens") not in (None, ""):
        try:
            return int(summary["n_mem_tokens"])
        except Exception:
            pass
    match = re.search(r"_mem(\d+)_", str(eval_file))
    return int(match.group(1)) if match else None


def target_model_name(summary: dict[str, Any]) -> str | None:
    for key in ("decoder_model", "tgt_model", "target_model"):
        value = summary.get(key)
        if value:
            return str(value)
    return None


def load_tokenizer(model_name: str | None, cache: dict[str, Any]) -> Any | None:
    if not model_name or AutoTokenizer is None:
        return None
    if model_name in cache:
        return cache[model_name]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        print(f"[WARN] cannot load tokenizer {model_name}: {exc}")
        tokenizer = None
    cache[model_name] = tokenizer
    return tokenizer


def should_shift_align(result: dict[str, Any], shift: int | None) -> bool:
    if shift is None or shift <= 0:
        return False
    try:
        original_count = int(result.get("original_token_count"))
        generated_count = int(result.get("generated_token_count"))
    except Exception:
        return False
    return original_count - generated_count == shift


def tokenizer_shift_reference(
    full_text: str,
    generated_text: str,
    result: dict[str, Any],
    tokenizer: Any,
    shift: int,
) -> str:
    max_length = None
    for key in ("target_length", "original_token_count", "num_tokens"):
        if result.get(key) not in (None, ""):
            try:
                max_length = int(result[key])
                break
            except Exception:
                pass

    token_kwargs = {"truncation": bool(max_length)}
    if max_length:
        token_kwargs["max_length"] = max_length
    ids = tokenizer(full_text, **token_kwargs)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ref_ids = strip_special_ids([int(tok) for tok in ids], tokenizer)

    generated_count = result.get("generated_token_count")
    try:
        window_len = int(generated_count)
    except Exception:
        gen_ids = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
        if gen_ids and isinstance(gen_ids[0], list):
            gen_ids = gen_ids[0]
        window_len = len(gen_ids)

    window_ids = ref_ids[shift : shift + window_len]
    return tokenizer.decode(window_ids, skip_special_tokens=True)


def word_window_shift_reference(full_text: str, generated_text: str, shift: int) -> str:
    ref_tokens = tokenize(full_text)
    gen_tokens = tokenize(generated_text)
    if not ref_tokens or not gen_tokens:
        return ""

    # Model-token shift is not exactly word-token shift. Search near the expected
    # offset to avoid overfitting across the whole document.
    center = min(shift, max(0, len(ref_tokens) - 1))
    lo = max(0, center - 8)
    hi = min(len(ref_tokens) - 1, center + 8)
    gen_set = Counter(gen_tokens)
    best_offset = center
    best_overlap = -1
    for offset in range(lo, hi + 1):
        window = ref_tokens[offset : offset + len(gen_tokens)]
        overlap = sum((Counter(window) & gen_set).values())
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
    return " ".join(ref_tokens[best_offset : best_offset + len(gen_tokens)])


def metric_reference_text(
    result: dict[str, Any],
    summary: dict[str, Any],
    eval_file: Path,
    metric_mode: str,
    tokenizer_cache: dict[str, Any],
    dataset_index: dict[str, str] | None,
) -> tuple[str, str, int | None]:
    if metric_mode == "saved_cropped":
        return saved_reference(result), "saved_cropped", None

    shift = infer_n_mem_tokens(summary, eval_file)
    shift = shift + 1 if shift is not None else None
    if not should_shift_align(result, shift):
        return saved_reference(result), "saved_cropped_no_shift_evidence", shift

    full_text = full_reference(result, dataset_index)
    generated_text = str(result.get("generated_text") or "")
    tokenizer = load_tokenizer(target_model_name(summary), tokenizer_cache)
    if tokenizer is not None:
        aligned = tokenizer_shift_reference(full_text, generated_text, result, tokenizer, int(shift))
        if aligned:
            return aligned, "shift_aligned_tokenizer", shift

    aligned = word_window_shift_reference(full_text, generated_text, int(shift))
    if aligned:
        return aligned, "shift_aligned_word_window", shift
    return saved_reference(result), "saved_cropped_alignment_failed", shift


def recover_eval(
    eval_file: Path,
    max_recovered_texts: int | None,
    metric_mode: str,
    tokenizer_cache: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary, results, raw_data = load_eval(eval_file)
    recovered_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float | bool | int]] = []
    max_texts = None if max_recovered_texts is None or max_recovered_texts <= 0 else int(max_recovered_texts)
    dataset_index = load_dataset_index(str(summary.get("dataset_path"))) if summary.get("dataset_path") else None

    for result in results:
        if not isinstance(result, dict):
            continue
        generated = result.get("generated_text") or ""
        reference, reference_kind, shift = metric_reference_text(
            result,
            summary,
            eval_file,
            metric_mode,
            tokenizer_cache,
            dataset_index,
        )
        recovered = dict(result)

        if generated and reference and (max_texts is None or len(metric_rows) < max_texts):
            metrics = text_metrics(reference, generated)
            recovered.update(metrics)
            recovered["metric_reference"] = reference_kind
            recovered["alignment_shift_tokens"] = shift
            recovered["aligned_reference_text"] = reference if metric_mode == "shift_aligned" else ""
            metric_rows.append(metrics)
        elif generated and reference:
            recovered["metric_reference"] = "skipped_by_max_recovered_texts"
        else:
            recovered["metric_reference"] = "missing"

        recovered_results.append(recovered)

    recovered_summary = dict(summary)
    recovered_summary["recovered_metric_source"] = (
        "saved_generated_text_vs_shift_aligned_reference"
        if metric_mode == "shift_aligned"
        else "saved_generated_text_vs_saved_cropped_reference"
    )
    recovered_summary["num_recovered_texts"] = len(metric_rows)
    if metric_rows:
        for key in [
            "exact_match",
            "char_similarity",
            "word_precision",
            "word_recall",
            "word_f1",
            "bleu",
            "bleu1",
            "bleu2",
            "bleu3",
            "bleu4",
            "rougeL",
        ]:
            recovered_summary[f"avg_{key}"] = float(mean(float(row[key]) for row in metric_rows))

    recovered_data = dict(raw_data)
    recovered_data["summary"] = recovered_summary
    recovered_data["results"] = recovered_results
    return recovered_summary, recovered_results, recovered_data


def format_float(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str], float_cols: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [format_float(row.get(h, "")) if h in float_cols else str(row.get(h, "")) for h in headers]
            )


def origin_eval_files(origin_base: Path) -> list[Path]:
    if not origin_base.exists():
        return []
    out: list[Path] = []
    for checkpoint_dir in sorted(origin_base.iterdir()):
        if not checkpoint_dir.is_dir() or checkpoint_dir.name == "transfer_compressor":
            continue
        eval_file = checkpoint_dir / "evaluation" / "evaluation_results.json"
        if eval_file.exists():
            out.append(eval_file)
    return out


def ori_transfer_eval_files(ori_transfer_base: Path) -> list[Path]:
    if not ori_transfer_base.exists():
        return []
    return sorted(ori_transfer_base.glob("*/evaluation/evaluation_results.json"))


def transfer_eval_files(transfer_base: Path, method: str) -> list[Path]:
    if not transfer_base.exists():
        return []

    out: list[Path] = []
    for checkpoint_dir in sorted(transfer_base.iterdir()):
        if not checkpoint_dir.is_dir():
            continue

        checkpoint_name = checkpoint_dir.name
        nested = (
            checkpoint_dir
            / "evaluation"
            / checkpoint_name
            / f"eval_transfer_compression_results_from_simple_compressor_using_{method}.json"
        )
        if nested.exists():
            out.append(nested)
            continue

        eval_dir = checkpoint_dir / "evaluation"
        if not eval_dir.is_dir():
            continue
        candidates = sorted(eval_dir.glob("eval_transfer_*.json"))
        if not candidates:
            for subdir in sorted(d for d in eval_dir.iterdir() if d.is_dir()):
                candidates.extend(sorted(subdir.glob("eval_transfer_*.json")))
        if candidates:
            out.append(candidates[0])

    return out


def maybe_write_sidecar(eval_file: Path, output_root: Path, base_dir: Path, data: dict[str, Any]) -> Path:
    rel = eval_file.relative_to(base_dir)
    sidecar = output_root / "sidecar_json" / base_dir.name / rel.with_name("evaluation_results.recovered_bleu.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return sidecar


def recover_origin(
    origin_base: Path,
    output_dir: Path,
    write_sidecars: bool,
    max_recovered_texts: int,
    metric_mode: str,
    tokenizer_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_file in origin_eval_files(origin_base):
        try:
            summary, _results, data = recover_eval(
                eval_file,
                max_recovered_texts=max_recovered_texts,
                metric_mode=metric_mode,
                tokenizer_cache=tokenizer_cache,
            )
        except Exception as exc:
            print(f"[WARN] skip origin {eval_file}: {exc}")
            continue

        checkpoint_dir = eval_file.parents[1]
        row: dict[str, Any] = {
            "checkpoint_name": checkpoint_dir.name,
            "checkpoint_path": str(checkpoint_dir),
            "source_model": summary.get("compressor_model", ""),
            "target_model": summary.get("decoder_model", ""),
            "model_pair": (
                f"{summary.get('compressor_model', '')}-{summary.get('decoder_model', '')}"
                if summary.get("compressor_model") and summary.get("decoder_model")
                else ""
            ),
            "n_mem_tokens": summary.get("n_mem_tokens", ""),
            "num_samples": summary.get("num_samples", ""),
            "num_recovered_texts": summary.get("num_recovered_texts", ""),
            "avg_loss": summary.get("avg_loss"),
            "avg_accuracy": summary.get("avg_accuracy"),
            "overall_accuracy": summary.get("overall_accuracy"),
            "avg_bleu": summary.get("avg_bleu"),
            "avg_bleu1": summary.get("avg_bleu1"),
            "avg_bleu2": summary.get("avg_bleu2"),
            "avg_bleu3": summary.get("avg_bleu3"),
            "avg_bleu4": summary.get("avg_bleu4"),
            "avg_rougeL": summary.get("avg_rougeL"),
            "metric_source": summary.get("recovered_metric_source", ""),
        }
        if write_sidecars:
            row["recovered_json"] = str(maybe_write_sidecar(eval_file, output_dir, origin_base, data))
        rows.append(row)
    return rows


def logical_ori_transfer_key(checkpoint_name: str) -> tuple[str, str, str, str, str] | None:
    match = re.match(
        r"^(?P<enc>[^_]+)_to_(?P<src>.+?)_to_(?P<tgt>.+?)_"
        r"(?P<mode>converter_only|encoder_converter)_mem(?P<mem>\d+)_",
        checkpoint_name,
    )
    if not match:
        return None
    data = match.groupdict()
    target = "mistral7b" if data["tgt"] == "mistral7binstruct" else data["tgt"]
    source = "mistral7b" if data["src"] == "mistral7binstruct" else data["src"]
    return data["enc"], source, target, data["mode"], data["mem"]


_ORI_TS = re.compile(r"_(\d{8})_(\d{6})$")


def ori_transfer_checkpoint_ts(checkpoint_name: str) -> int:
    """Sort key from trailing _YYYYMMDD_HHMMSS in ori_transfer output dir name."""
    m = _ORI_TS.search(checkpoint_name)
    if not m:
        return 0
    return int(m.group(1)) * 1_000_000 + int(m.group(2))


def recover_ori_transfer(
    ori_transfer_base: Path,
    output_dir: Path,
    write_sidecars: bool,
    max_recovered_texts: int,
    metric_mode: str,
    tokenizer_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Multiple timestamped dirs can share the same logical (enc, src, tgt, mode, mem).
    # Glob order is lexicographic, not chronological — keep the newest dir per logical key.
    best_eval: dict[tuple[str, str, str, str, str], Path] = {}
    unkeyed: list[Path] = []
    for eval_file in ori_transfer_eval_files(ori_transfer_base):
        checkpoint_dir = eval_file.parents[1]
        key = logical_ori_transfer_key(checkpoint_dir.name)
        if key is None:
            unkeyed.append(eval_file)
            continue
        ts = ori_transfer_checkpoint_ts(checkpoint_dir.name)
        prev = best_eval.get(key)
        if prev is None or ts > ori_transfer_checkpoint_ts(prev.parents[1].name):
            best_eval[key] = eval_file

    eval_files_to_process = sorted(best_eval.values(), key=lambda p: p.as_posix()) + sorted(
        unkeyed, key=lambda p: p.as_posix()
    )

    for eval_file in eval_files_to_process:
        try:
            summary, _results, data = recover_eval(
                eval_file,
                max_recovered_texts=max_recovered_texts,
                metric_mode=metric_mode,
                tokenizer_cache=tokenizer_cache,
            )
        except Exception as exc:
            print(f"[WARN] skip ori_transfer {eval_file}: {exc}")
            continue

        checkpoint_dir = eval_file.parents[1]
        key = logical_ori_transfer_key(checkpoint_dir.name)
        row: dict[str, Any] = {
            "checkpoint_name": checkpoint_dir.name,
            "checkpoint_path": str(checkpoint_dir),
            "train_mode": summary.get("train_mode", ""),
            "n_mem_tokens": summary.get("n_mem_tokens", ""),
            "num_samples": summary.get("num_samples", ""),
            "num_recovered_texts": summary.get("num_recovered_texts", ""),
            "avg_loss": summary.get("avg_loss"),
            "avg_accuracy": summary.get("avg_accuracy"),
            "overall_accuracy": summary.get("overall_accuracy"),
            "avg_bleu": summary.get("avg_bleu"),
            "avg_bleu1": summary.get("avg_bleu1"),
            "avg_bleu2": summary.get("avg_bleu2"),
            "avg_bleu3": summary.get("avg_bleu3"),
            "avg_bleu4": summary.get("avg_bleu4"),
            "avg_rougeL": summary.get("avg_rougeL"),
            "metric_source": summary.get("recovered_metric_source", ""),
        }
        if write_sidecars:
            row["recovered_json"] = str(maybe_write_sidecar(eval_file, output_dir, ori_transfer_base, data))
        rows.append(row)
    return rows


def recover_transfer_method(
    transfer_base: Path,
    method: str,
    output_dir: Path,
    write_sidecars: bool,
    max_recovered_texts: int,
    metric_mode: str,
    tokenizer_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_file in transfer_eval_files(transfer_base, method):
        try:
            summary, _results, data = recover_eval(
                eval_file,
                max_recovered_texts=max_recovered_texts,
                metric_mode=metric_mode,
                tokenizer_cache=tokenizer_cache,
            )
        except Exception as exc:
            print(f"[WARN] skip {method}_transfer {eval_file}: {exc}")
            continue

        actual_method = summary.get("transfer_method", "")
        if actual_method and actual_method != method:
            print(f"[WARN] skip {eval_file}: transfer_method={actual_method!r}, expected {method!r}")
            continue

        checkpoint_dir = eval_file.parents[2] if eval_file.parent.name == eval_file.parents[2].name else eval_file.parents[1]
        row: dict[str, Any] = {
            "checkpoint_name": checkpoint_dir.name,
            "checkpoint_path": str(checkpoint_dir),
            "eval_file": str(eval_file),
            "source_experiment": summary.get("source_experiment", ""),
            "source_model": summary.get("source_model", ""),
            "target_model": summary.get("target_model", ""),
            "transfer_method": summary.get("transfer_method", method),
            "dataset_path": summary.get("dataset_path", ""),
            "num_texts": summary.get("num_texts", ""),
            "num_recovered_texts": summary.get("num_recovered_texts", ""),
            "avg_transfer_accuracy": summary.get("avg_transfer_accuracy"),
            "avg_bleu": summary.get("avg_bleu"),
            "avg_bleu1": summary.get("avg_bleu1"),
            "avg_bleu2": summary.get("avg_bleu2"),
            "avg_bleu3": summary.get("avg_bleu3"),
            "avg_bleu4": summary.get("avg_bleu4"),
            "avg_rougeL": summary.get("avg_rougeL"),
            "metric_source": summary.get("recovered_metric_source", ""),
        }
        if write_sidecars:
            row["recovered_json"] = str(maybe_write_sidecar(eval_file, output_dir, transfer_base, data))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover BLEU/ROUGE from saved SimpleCompressor evaluation JSON files."
    )
    parser.add_argument("--origin_base", type=Path, default=Path("outputs/simple_compressor"))
    parser.add_argument(
        "--ori_transfer_base",
        type=Path,
        default=Path("outputs/simple_compressor/transfer_compressor/ori_transfer"),
    )
    parser.add_argument(
        "--ls_transfer_base",
        type=Path,
        default=Path("outputs/simple_compressor/transfer_compressor/ls_transfer"),
    )
    parser.add_argument(
        "--random_transfer_base",
        type=Path,
        default=Path("outputs/simple_compressor/transfer_compressor/random_transfer"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("eval_output_temp/recovered_bleu"))
    parser.add_argument(
        "--max_recovered_texts",
        type=int,
        default=50,
        help="Maximum saved generated texts to use per evaluation file (0 = all). Default: 50.",
    )
    parser.add_argument(
        "--metric_mode",
        choices=["saved_cropped", "shift_aligned"],
        default="saved_cropped",
        help=(
            "saved_cropped compares saved generated_text to the saved cropped reference. "
            "shift_aligned compares generated_text to a reference window shifted by n_mem_tokens + 1 "
            "when the saved token counts show the prefix-trimming bug."
        ),
    )
    parser.add_argument(
        "--write_sidecars",
        action="store_true",
        help="Also write recovered evaluation_results.recovered_bleu.json files under output_dir.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    tokenizer_cache: dict[str, Any] = {}
    origin_rows = recover_origin(
        args.origin_base,
        output_dir,
        args.write_sidecars,
        args.max_recovered_texts,
        args.metric_mode,
        tokenizer_cache,
    )
    ori_rows = recover_ori_transfer(
        args.ori_transfer_base,
        output_dir,
        args.write_sidecars,
        args.max_recovered_texts,
        args.metric_mode,
        tokenizer_cache,
    )
    ls_rows = recover_transfer_method(
        args.ls_transfer_base,
        "ls",
        output_dir,
        args.write_sidecars,
        args.max_recovered_texts,
        args.metric_mode,
        tokenizer_cache,
    )
    random_rows = recover_transfer_method(
        args.random_transfer_base,
        "random",
        output_dir,
        args.write_sidecars,
        args.max_recovered_texts,
        args.metric_mode,
        tokenizer_cache,
    )

    float_cols = {
        "avg_loss",
        "avg_accuracy",
        "overall_accuracy",
        "avg_transfer_accuracy",
        "avg_bleu",
        "avg_bleu1",
        "avg_bleu2",
        "avg_bleu3",
        "avg_bleu4",
        "avg_rougeL",
    }

    origin_headers = [
        "checkpoint_name",
        "source_model",
        "target_model",
        "model_pair",
        "n_mem_tokens",
        "num_samples",
        "num_recovered_texts",
        "avg_loss",
        "avg_accuracy",
        "overall_accuracy",
        "avg_bleu",
        "avg_bleu1",
        "avg_bleu2",
        "avg_bleu3",
        "avg_bleu4",
        "avg_rougeL",
        "metric_source",
        "checkpoint_path",
    ]
    ori_headers = [
        "checkpoint_name",
        "train_mode",
        "n_mem_tokens",
        "num_samples",
        "num_recovered_texts",
        "avg_loss",
        "avg_accuracy",
        "overall_accuracy",
        "avg_bleu",
        "avg_bleu1",
        "avg_bleu2",
        "avg_bleu3",
        "avg_bleu4",
        "avg_rougeL",
        "metric_source",
        "checkpoint_path",
    ]
    transfer_headers = [
        "checkpoint_name",
        "source_experiment",
        "source_model",
        "target_model",
        "transfer_method",
        "dataset_path",
        "num_texts",
        "num_recovered_texts",
        "avg_transfer_accuracy",
        "avg_bleu",
        "avg_bleu1",
        "avg_bleu2",
        "avg_bleu3",
        "avg_bleu4",
        "avg_rougeL",
        "metric_source",
        "checkpoint_path",
        "eval_file",
    ]
    if args.write_sidecars:
        origin_headers.append("recovered_json")
        ori_headers.append("recovered_json")
        transfer_headers.append("recovered_json")

    write_csv(output_dir / "origin_recovered_evals.csv", origin_rows, origin_headers, float_cols)
    write_csv(output_dir / "ori_transfer_recovered_evals.csv", ori_rows, ori_headers, float_cols)
    write_csv(output_dir / "ls_transfer_recovered_evals.csv", ls_rows, transfer_headers, float_cols)
    write_csv(output_dir / "random_transfer_recovered_evals.csv", random_rows, transfer_headers, float_cols)

    print(f"Origin evals recovered: {len(origin_rows)}")
    print(f"Ori-transfer evals recovered: {len(ori_rows)}")
    print(f"LS-transfer evals recovered: {len(ls_rows)}")
    print(f"Random-transfer evals recovered: {len(random_rows)}")
    print(f"Wrote: {output_dir / 'origin_recovered_evals.csv'}")
    print(f"Wrote: {output_dir / 'ori_transfer_recovered_evals.csv'}")
    print(f"Wrote: {output_dir / 'ls_transfer_recovered_evals.csv'}")
    print(f"Wrote: {output_dir / 'random_transfer_recovered_evals.csv'}")


if __name__ == "__main__":
    main()
