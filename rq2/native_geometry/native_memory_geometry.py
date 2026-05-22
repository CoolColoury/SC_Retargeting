#!/usr/bin/env python3
"""Summarize native SimpleCompressor memory geometry for RQ2.

This diagnostic uses target-native compressors (encoder -> target) and compares
their compressed memory manifolds with label-free geometry statistics. It is
intended to complement perturbation sensitivity: perturbation gives local
readable-region width, while this script describes the geometry of the native
memory distribution itself.
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

ROOT = Path(__file__).resolve().parents[2]  # code_backup root
sys.path.append(str(ROOT / "src" / "soft_compress" / "simple_compressor"))
from simple_compressor import SimpleCompressor  # noqa: E402

EPS = 1e-8


def load_texts(path: Path, max_samples: int) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data if item.get("text")][:max_samples]


def get_vocab(tokenizer) -> dict[str, int]:
    return {tok: int(idx) for tok, idx in tokenizer.get_vocab().items()}


def build_anchor_ids(tokenizer, max_anchors: int) -> tuple[list[int], list[str]]:
    vocab = get_vocab(tokenizer)
    special = set(tokenizer.all_special_tokens)
    tokens = [tok for tok in sorted(vocab) if tok not in special and tok.strip()]
    if len(tokens) > max_anchors:
        step = len(tokens) / max_anchors
        tokens = [tokens[int(i * step)] for i in range(max_anchors)]
    if len(tokens) < 32:
        raise ValueError(f"Too few anchor tokens: {len(tokens)}")
    return [vocab[tok] for tok in tokens], tokens


def relative_profile(memory: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mem = F.normalize(memory.float(), dim=-1)
    anc = F.normalize(anchors.float(), dim=-1)
    direction = mem @ anc.T
    mem_norm = torch.linalg.norm(memory.float(), dim=-1, keepdim=True)
    anc_norm = torch.linalg.norm(anchors.float(), dim=-1).clamp_min(EPS).unsqueeze(0)
    magnitude = torch.log((mem_norm / anc_norm).clamp_min(EPS))
    return direction, magnitude


def entropy_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = F.softmax(logits.float() / temperature, dim=-1).clamp_min(EPS)
    return -(probs * torch.log(probs)).sum(dim=-1)


def effective_rank(values: torch.Tensor) -> float:
    # Participation ratio of covariance eigenvalues; stable for comparing
    # memory-manifold dimensionality across hidden sizes.
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        return float("nan")
    cov = centered.T @ centered / float(centered.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    denom = float((eig**2).sum().item())
    if denom <= EPS:
        return 0.0
    return float((eig.sum().item() ** 2) / denom)


def offdiag_cosine_mean(memory: torch.Tensor) -> float:
    # Average cosine similarity between memory-token slots. High values imply
    # redundant/collapsed memory-token geometry.
    x = F.normalize(memory.float(), dim=-1)
    sims = x @ x.transpose(0, 1)
    n = sims.shape[0]
    if n <= 1:
        return float("nan")
    mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
    return float(sims[mask].mean().item())


def summarize_run(
    encoder: str,
    target: str,
    checkpoint: Path,
    compressor_model: str,
    decoder_model: str,
    texts: list[str],
    n_mem_tokens: int,
    max_length: int,
    max_anchors: int,
    temperature: float,
    device: str,
) -> dict[str, str | float]:
    model = SimpleCompressor.from_pretrained(
        checkpoint_path=str(checkpoint),
        compressor_model_name=compressor_model,
        decoder_model_name=decoder_model,
        n_mem_tokens=n_mem_tokens,
        dtype=torch.bfloat16,
        device=device,
    ).to(device)
    model.eval()
    model.decoder.eval()

    anchor_ids, _ = build_anchor_ids(model.decoder_tokenizer, max_anchors)
    anchors = model.decoder.get_input_embeddings().weight[torch.tensor(anchor_ids, device=device)].detach()

    memories: list[torch.Tensor] = []
    direction_entropy: list[float] = []
    top1_mass: list[float] = []
    top5_mass: list[float] = []
    direction_std: list[float] = []
    magnitude_std: list[float] = []
    token_cos: list[float] = []

    for text in texts:
        encoded = model.compressor_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            memory = model.compress(encoded["input_ids"], encoded["attention_mask"])[0]

        direction, magnitude = relative_profile(memory, anchors)
        probs = F.softmax(direction.float() / temperature, dim=-1)
        memories.append(memory.detach().float().cpu())
        direction_entropy.append(float(entropy_from_logits(direction, temperature).mean().item()))
        top_values = torch.topk(probs, k=min(5, probs.shape[-1]), dim=-1).values
        top1_mass.append(float(top_values[:, 0].mean().item()))
        top5_mass.append(float(top_values.sum(dim=-1).mean().item()))
        direction_std.append(float(direction.std(dim=0).mean().item()))
        magnitude_std.append(float(magnitude.std(dim=0).mean().item()))
        token_cos.append(offdiag_cosine_mean(memory))

    all_memory = torch.cat(memories, dim=0)
    token_norms = torch.linalg.norm(all_memory.float(), dim=-1)

    return {
        "encoder": encoder,
        "target": target,
        "n_samples": len(texts),
        "n_mem_tokens": n_mem_tokens,
        "hidden_size": int(model.decoder.config.hidden_size),
        "memory_norm_mean": float(token_norms.mean().item()),
        "memory_norm_std": float(token_norms.std().item()),
        "memory_norm_cv": float((token_norms.std() / token_norms.mean().clamp_min(EPS)).item()),
        "memory_effective_rank": effective_rank(all_memory),
        "memory_token_cosine_mean": sum(token_cos) / max(len(token_cos), 1),
        "anchor_direction_entropy_mean": sum(direction_entropy) / max(len(direction_entropy), 1),
        "anchor_top1_mass_mean": sum(top1_mass) / max(len(top1_mass), 1),
        "anchor_top5_mass_mean": sum(top5_mass) / max(len(top5_mass), 1),
        "anchor_direction_std_mean": sum(direction_std) / max(len(direction_std), 1),
        "anchor_magnitude_std_mean": sum(magnitude_std) / max(len(magnitude_std), 1),
    }


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_mem_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_anchors", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runs",
        default="llama1b->llama1b,llama1b->llama3b,llama1b->qwen1.5b,llama1b->qwen3b,llama1b->qwen7b,gpt2->llama3b,gpt2->qwen1.5b,gpt2->qwen3b,gpt2->qwen7b",
        help="Comma-separated encoder->target native compressors.",
    )
    args = parser.parse_args()

    model_paths = {
        "gpt2": "${MODELS_DIR}/gpt2",
        "llama1b": "${MODELS_DIR}/Llama-3.2-1B-Instruct",
        "llama3b": "${MODELS_DIR}/Llama-3.2-3B-Instruct",
        "llama8b": "${MODELS_DIR}/Llama-3-8B-Instruct",
        "mistral7b": "${MODELS_DIR}/Mistral-7B-Instruct",
        "qwen1.5b": "${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct",
        "qwen3b": "${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct",
        "qwen7b": "${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct",
    }

    texts = load_texts(args.dataset, args.max_samples)
    rows: list[dict[str, str | float]] = []
    for spec in [x.strip() for x in args.runs.split(",") if x.strip()]:
        encoder, target = spec.split("->", 1)
        checkpoint = ROOT / "outputs" / "simple_compressor" / f"{encoder}_to_{target}_mem{args.n_mem_tokens}_len128_ds_4gpu"
        if not checkpoint.exists():
            print(f"[skip] missing checkpoint: {checkpoint}")
            continue
        print(f"[run] {encoder}->{target}")
        rows.append(
            summarize_run(
                encoder,
                target,
                checkpoint,
                model_paths[encoder],
                model_paths[target],
                texts,
                args.n_mem_tokens,
                args.max_length,
                args.max_anchors,
                args.temperature,
                args.device,
            )
        )

    if not rows:
        raise ValueError("No native geometry rows produced.")
    write_csv(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
