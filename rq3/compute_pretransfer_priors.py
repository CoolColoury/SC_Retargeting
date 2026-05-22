#!/usr/bin/env python3
"""Compute RQ3 transfer priors from pre-transfer information only.

Allowed inputs for a direction such as ``gpt2:llama1b -> llama8b``:
    1. the trained source compressor ``gpt2_to_llama1b``;
    2. the target raw model ``llama8b``;
    3. evaluation texts.

The script does not load a target compressor and does not use transfer metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]  # code_backup root
_METRIC_TEST_DIR = Path(__file__).resolve().parent  # rq3/
sys.path.append(str(_METRIC_TEST_DIR))
sys.path.append(str(ROOT / "src" / "soft_compress" / "simple_compressor"))
from discrete_priors import discrete_prior_features  # noqa: E402
from simple_compressor import SimpleCompressor  # noqa: E402


EPS = 1e-8
IDENTITY_COLUMNS = ["pair", "encoder", "source", "target", "n_samples", "n_anchors"]
PAIR_METRIC_COLUMNS = [
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
    "diagnostic_source_direction_entropy_mean",
    "diagnostic_target_direction_entropy",
    "diagnostic_source_direction_std_mean",
    "diagnostic_target_direction_std_mean",
]


def js_divergence_from_logits(
    source_logits: torch.Tensor,
    target_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Jensen-Shannon divergence between source profile logits and target profile distribution."""
    source_probs = F.softmax(source_logits.float() / temperature, dim=-1)
    target_probs = target_probs.float().unsqueeze(0).expand_as(source_probs)
    mixture = 0.5 * (source_probs + target_probs)
    source_kl = source_probs * (torch.log(source_probs.clamp_min(EPS)) - torch.log(mixture.clamp_min(EPS)))
    target_kl = target_probs * (torch.log(target_probs.clamp_min(EPS)) - torch.log(mixture.clamp_min(EPS)))
    return 0.5 * (source_kl.sum(dim=-1) + target_kl.sum(dim=-1))


def subspace_residual_ratio(x: torch.Tensor, mean_vec: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Fraction of centered energy outside the target anchor-profile PCA subspace."""
    centered = x.float() - mean_vec.float()
    if basis.numel() == 0:
        return torch.ones(centered.shape[0], device=centered.device)
    projected = (centered @ basis) @ basis.T
    residual = centered - projected
    num = (residual**2).sum(dim=-1)
    den = (centered**2).sum(dim=-1).clamp_min(EPS)
    return num / den


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Entropy over the last dimension."""
    p = probs.float().clamp_min(EPS)
    return -(p * torch.log(p)).sum(dim=-1)


def topk_anchor_overlap(source_logits: torch.Tensor, target_probs: torch.Tensor, k: int) -> torch.Tensor:
    """Fraction of source top-k anchors that are also target top-k anchors."""
    k = max(1, min(k, source_logits.shape[-1], target_probs.shape[-1]))
    source_top = torch.topk(source_logits.float(), k=k, dim=-1).indices
    target_top = set(torch.topk(target_probs.float(), k=k, dim=-1).indices.detach().cpu().tolist())
    overlaps: list[float] = []
    for row in source_top.detach().cpu().tolist():
        overlaps.append(sum(1 for idx in row if idx in target_top) / float(k))
    return torch.tensor(overlaps, device=source_logits.device)


def load_texts(path: Path, max_samples: int) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data if item.get("text")]
    return texts[:max_samples]


def get_vocab(tokenizer) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    return {tok: int(idx) for tok, idx in vocab.items()}


def build_common_anchor_ids(src_tokenizer, tgt_tokenizer, max_anchors: int) -> tuple[list[int], list[int], list[str]]:
    src_vocab = get_vocab(src_tokenizer)
    tgt_vocab = get_vocab(tgt_tokenizer)
    special = set(src_tokenizer.all_special_tokens) | set(tgt_tokenizer.all_special_tokens)
    tokens = [
        tok
        for tok in sorted(set(src_vocab) & set(tgt_vocab))
        if tok not in special and len(tok.strip()) > 0
    ]
    if len(tokens) > max_anchors:
        step = len(tokens) / max_anchors
        tokens = [tokens[int(i * step)] for i in range(max_anchors)]
    if len(tokens) < 32:
        raise ValueError(f"Too few common anchor tokens: {len(tokens)}")
    return [src_vocab[tok] for tok in tokens], [tgt_vocab[tok] for tok in tokens], tokens


def relative_profile(x: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cosine-to-anchor profile and log-norm-to-anchor profile."""
    x = x.float()
    anchors = anchors.float()
    direction = F.normalize(x, dim=-1) @ F.normalize(anchors, dim=-1).T
    x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    anchor_norm = torch.linalg.norm(anchors, dim=-1).unsqueeze(0)
    magnitude = torch.log(x_norm + EPS) - torch.log(anchor_norm + EPS)
    return direction, magnitude


def summarize_target_anchor_manifold(
    tgt_anchors: torch.Tensor,
    subspace_rank: int,
    js_temperature: float,
) -> dict[str, torch.Tensor | float]:
    direction, magnitude = relative_profile(tgt_anchors, tgt_anchors)
    direction_std_raw = direction.std(dim=0)
    magnitude_std_raw = magnitude.std(dim=0)
    # A few anchor columns can have very small variance and make z-scores explode.
    # Keep z-score metrics diagnostic-only and prefer raw L1/RMSE distances.
    direction_std = direction_std_raw.clamp_min(1e-3)
    magnitude_std = magnitude_std_raw.clamp_min(1e-3)
    direction_mean = direction.mean(dim=0)
    centered_direction = direction - direction_mean
    max_rank = max(0, min(subspace_rank, centered_direction.shape[0] - 1, centered_direction.shape[1]))
    if max_rank > 0:
        _, _, vh = torch.linalg.svd(centered_direction.float(), full_matrices=False)
        direction_basis = vh[:max_rank].T.contiguous()
    else:
        direction_basis = torch.empty(direction.shape[1], 0, device=direction.device)
    target_direction_probs = F.softmax(direction_mean.float() / js_temperature, dim=-1)
    target_direction_entropy = entropy_from_probs(target_direction_probs).item()

    return {
        "direction_mean": direction_mean,
        "direction_std": direction_std,
        "magnitude_mean": magnitude.mean(dim=0),
        "magnitude_std": magnitude_std,
        "direction_subspace_basis": direction_basis,
        "target_direction_probs": target_direction_probs,
        "target_direction_entropy": target_direction_entropy,
        "target_direction_std_mean": direction_std_raw.mean().item(),
        "target_direction_std_min": direction_std_raw.min().item(),
        "target_magnitude_std_mean": magnitude_std_raw.mean().item(),
        "target_magnitude_std_min": magnitude_std_raw.min().item(),
        "target_anchor_log_norm_mean": torch.log(torch.linalg.norm(tgt_anchors.float(), dim=-1) + EPS).mean().item(),
        "target_anchor_log_norm_std": torch.log(torch.linalg.norm(tgt_anchors.float(), dim=-1) + EPS).std().item(),
    }


def target_text_nll(text: str, target_model, target_tokenizer, max_length: int, device: str) -> float:
    inputs = target_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    if input_ids.numel() < 2:
        return float("nan")
    with torch.inference_mode():
        out = target_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    return float(out.loss.item())


def source_native_nll(text: str, source_compressor: SimpleCompressor, max_length: int, device: str) -> float:
    comp = source_compressor.compressor_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    dec = source_compressor.decoder_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        out = source_compressor(
            compress_input_ids=comp["input_ids"].to(device),
            compress_attention_mask=comp["attention_mask"].to(device),
            decoder_input_ids=dec["input_ids"].to(device),
            decoder_attention_mask=dec["attention_mask"].to(device),
        )
    return float(out.loss.item())


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def compute_priors(args: argparse.Namespace) -> dict[str, str | float]:
    device = args.device
    texts = load_texts(args.dataset, args.max_samples)

    source_compressor = SimpleCompressor.from_pretrained(
        checkpoint_path=str(args.source_checkpoint),
        compressor_model_name=str(args.compressor_model),
        decoder_model_name=str(args.source_decoder_model),
        n_mem_tokens=args.n_mem_tokens,
        dtype=torch.bfloat16,
        device=device,
    ).to(device)
    source_compressor.eval()

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    target_model.eval()

    src_ids, tgt_ids, anchor_tokens = build_common_anchor_ids(
        source_compressor.decoder_tokenizer,
        target_tokenizer,
        args.max_anchors,
    )
    src_anchors = source_compressor.decoder.get_input_embeddings().weight[src_ids].detach().to(device)
    tgt_anchors = target_model.get_input_embeddings().weight[tgt_ids].detach().to(device)
    target_stats = summarize_target_anchor_manifold(tgt_anchors, args.subspace_rank, args.js_temperature)

    direction_l1: list[float] = []
    direction_rmse: list[float] = []
    direction_z2: list[float] = []
    magnitude_abs: list[float] = []
    magnitude_rmse: list[float] = []
    magnitude_z_abs: list[float] = []
    direction_js: list[float] = []
    direction_subspace_residual: list[float] = []
    target_readability_z2: list[float] = []
    target_readability_tail_frac: list[float] = []
    target_magnitude_tail_frac: list[float] = []
    source_direction_entropy: list[float] = []
    source_direction_std_mean: list[float] = []
    target_readability_cross_entropy: list[float] = []
    source_to_target_anchor_kl: list[float] = []
    target_readability_energy: list[float] = []
    top_anchor_overlap: list[float] = []
    memory_log_norms: list[float] = []
    target_nlls: list[float] = []
    source_nlls: list[float] = []

    for text in texts:
        comp = source_compressor.compressor_tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            memory = source_compressor.compress(
                comp["input_ids"].to(device),
                comp["attention_mask"].to(device),
            )[0]

        src_dir, src_mag = relative_profile(memory, src_anchors)
        dir_delta = src_dir - target_stats["direction_mean"]
        mag_delta = src_mag - target_stats["magnitude_mean"]
        dir_z = (src_dir - target_stats["direction_mean"]) / target_stats["direction_std"]
        mag_z = (src_mag - target_stats["magnitude_mean"]) / target_stats["magnitude_std"]
        js = js_divergence_from_logits(src_dir, target_stats["target_direction_probs"], args.js_temperature)
        residual = subspace_residual_ratio(
            src_dir,
            target_stats["direction_mean"],
            target_stats["direction_subspace_basis"],
        )
        src_probs = F.softmax(src_dir.float() / args.js_temperature, dim=-1)
        tgt_probs = target_stats["target_direction_probs"].float().unsqueeze(0).expand_as(src_probs)
        cross_entropy = -(src_probs * torch.log(tgt_probs.clamp_min(EPS))).sum(dim=-1)
        src_kl = src_probs * (torch.log(src_probs.clamp_min(EPS)) - torch.log(tgt_probs.clamp_min(EPS)))
        readability_energy = (src_probs * tgt_probs).sum(dim=-1)
        overlap = topk_anchor_overlap(src_dir, target_stats["target_direction_probs"], args.top_k_anchors)

        direction_z2.append(float((dir_z**2).mean().item()))
        target_readability_z2.append(float((dir_z**2).mean().item()))
        target_readability_tail_frac.append(float((dir_z.abs() > 2.0).float().mean().item()))
        target_magnitude_tail_frac.append(float((mag_z.abs() > 2.0).float().mean().item()))
        direction_l1.append(float(dir_delta.abs().mean().item()))
        direction_rmse.append(float(torch.sqrt((dir_delta**2).mean()).item()))
        magnitude_abs.append(float(mag_delta.abs().mean().item()))
        magnitude_rmse.append(float(torch.sqrt((mag_delta**2).mean()).item()))
        magnitude_z_abs.append(float(mag_z.abs().mean().item()))
        direction_js.append(float(js.mean().item()))
        direction_subspace_residual.append(float(residual.mean().item()))
        source_direction_entropy.append(float(entropy_from_probs(src_probs).mean().item()))
        source_direction_std_mean.append(float(src_dir.std(dim=0).mean().item()))
        target_readability_cross_entropy.append(float(cross_entropy.mean().item()))
        source_to_target_anchor_kl.append(float(src_kl.sum(dim=-1).mean().item()))
        target_readability_energy.append(float(readability_energy.mean().item()))
        top_anchor_overlap.append(float(overlap.mean().item()))
        memory_log_norms.extend(torch.log(torch.linalg.norm(memory.float(), dim=-1) + EPS).detach().cpu().tolist())

        if args.compute_text_nll:
            target_nlls.append(target_text_nll(text, target_model, target_tokenizer, args.max_length, device))
            source_nlls.append(source_native_nll(text, source_compressor, args.max_length, device))

    memory_log_norm_mean = mean(memory_log_norms)
    target_anchor_log_norm_mean = float(target_stats["target_anchor_log_norm_mean"])
    source_direction_std_mean_value = mean(source_direction_std_mean)
    target_direction_std_mean_value = float(target_stats["target_direction_std_mean"])

    row: dict[str, str | float] = {
        "pair": f"{args.encoder_name}:{args.source_name}->{args.target_name}",
        "encoder": args.encoder_name,
        "source": args.source_name,
        "target": args.target_name,
        "n_samples": len(texts),
        "n_anchors": len(anchor_tokens),
        "rel_direction_l1_to_target": mean(direction_l1),
        "rel_direction_rmse_to_target": mean(direction_rmse),
        "rel_magnitude_l1_to_target": mean(magnitude_abs),
        "rel_magnitude_rmse_to_target": mean(magnitude_rmse),
        "memory_target_norm_gap": abs(memory_log_norm_mean - target_anchor_log_norm_mean),
        "anchor_direction_js_to_target": mean(direction_js),
        "target_direction_subspace_residual": mean(direction_subspace_residual),
        "capacity_normalized_direction_rmse": mean(direction_rmse) / math.log(float(target_model.config.hidden_size) + EPS),
        "pair_normalized_direction_rmse": mean(direction_rmse)
        / (source_direction_std_mean_value + target_direction_std_mean_value + EPS),
        "target_readability_z2_mean": mean(target_readability_z2),
        "target_readability_tail_frac": mean(target_readability_tail_frac),
        "target_magnitude_tail_frac": mean(target_magnitude_tail_frac),
        "target_readability_cross_entropy": mean(target_readability_cross_entropy),
        "source_to_target_anchor_kl": mean(source_to_target_anchor_kl),
        "target_readability_energy": mean(target_readability_energy),
        "source_target_top_anchor_overlap": mean(top_anchor_overlap),
        "source_target_direction_entropy_gap": abs(
            mean(source_direction_entropy) - float(target_stats["target_direction_entropy"])
        ),
        "source_target_direction_dispersion_gap": abs(
            math.log((source_direction_std_mean_value + EPS) / (target_direction_std_mean_value + EPS))
        ),
        "diagnostic_rel_direction_z2_mean": mean(direction_z2),
        "diagnostic_rel_magnitude_z_abs_mean": mean(magnitude_z_abs),
        "diagnostic_source_direction_entropy_mean": mean(source_direction_entropy),
        "diagnostic_target_direction_entropy": float(target_stats["target_direction_entropy"]),
        "diagnostic_source_direction_std_mean": source_direction_std_mean_value,
        "diagnostic_target_direction_std_mean": float(target_stats["target_direction_std_mean"]),
        "diagnostic_target_direction_std_min": float(target_stats["target_direction_std_min"]),
        "diagnostic_target_magnitude_std_mean": float(target_stats["target_magnitude_std_mean"]),
        "diagnostic_target_magnitude_std_min": float(target_stats["target_magnitude_std_min"]),
        "target_text_nll": mean(target_nlls),
        "source_native_nll": mean(source_nlls),
        "target_minus_source_nll": mean(target_nlls) - mean(source_nlls) if target_nlls and source_nlls else float("nan"),
        "source_hidden_size": int(source_compressor.decoder.config.hidden_size),
        "target_hidden_size": int(target_model.config.hidden_size),
        "abs_log_hidden_ratio": abs(
            math.log((float(source_compressor.decoder.config.hidden_size) + EPS) / (float(target_model.config.hidden_size) + EPS))
        ),
    }
    row.update(discrete_prior_features(args.source_name, args.target_name))
    return row


def write_one_row(path: Path, row: dict[str, str | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def select_output_columns(row: dict[str, str | float], mode: str) -> dict[str, str | float]:
    if mode == "all":
        return row
    if mode == "pair_only":
        keep = IDENTITY_COLUMNS + [name for name in PAIR_METRIC_COLUMNS if name in row]
        return {name: row[name] for name in keep if name in row}
    raise ValueError(f"Unknown output column mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name", required=True, help="Short encoder name, e.g. gpt2.")
    parser.add_argument("--source_name", required=True, help="Short source decoder name, e.g. llama1b.")
    parser.add_argument("--target_name", required=True, help="Short target raw model name, e.g. llama8b.")
    parser.add_argument("--source_checkpoint", type=Path, required=True)
    parser.add_argument("--compressor_model", type=Path, required=True)
    parser.add_argument("--source_decoder_model", type=Path, required=True)
    parser.add_argument("--target_model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_mem_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_anchors", type=int, default=512)
    parser.add_argument("--top_k_anchors", type=int, default=32)
    parser.add_argument("--subspace_rank", type=int, default=64)
    parser.add_argument("--js_temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_text_nll", action="store_true")
    parser.add_argument(
        "--output_columns",
        choices=["all", "pair_only"],
        default="all",
        help="Write all prior columns or only source-target pair compatibility columns.",
    )
    args = parser.parse_args()

    row = compute_priors(args)
    write_one_row(args.output, select_output_columns(row, args.output_columns))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
