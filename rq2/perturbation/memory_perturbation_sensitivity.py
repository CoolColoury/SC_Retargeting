#!/usr/bin/env python3
"""Estimate local readable-region width by corrupting compressed memory vectors.

This server-side diagnostic loads a trained SimpleCompressor checkpoint, adds
noise/control corruptions to its compressed memory vectors, and measures
teacher-forced reconstruction loss/accuracy under the frozen decoder. It does
not train.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # code_backup root
sys.path.append(str(ROOT / "src" / "soft_compress" / "simple_compressor"))
from simple_compressor import SimpleCompressor  # noqa: E402


def load_texts(path: Path, max_samples: int) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data if item.get("text")][:max_samples]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_token_id(token_id: int | list[int] | tuple[int, ...] | None) -> int | None:
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        return int(token_id[0]) if token_id else None
    return int(token_id)


def validate_checkpoint_name(checkpoint: Path, expected_encoder: str | None, expected_decoder: str | None) -> None:
    if not expected_encoder and not expected_decoder:
        return

    prefix = f"{expected_encoder}_to_" if expected_encoder else None
    decoder_marker = f"_to_{expected_decoder}_mem" if expected_decoder else None
    if (prefix and not checkpoint.name.startswith(prefix)) or (
        decoder_marker and decoder_marker not in checkpoint.name
    ):
        raise ValueError(
            "Checkpoint/model mismatch: "
            f"expected encoder={expected_encoder!r}, decoder={expected_decoder!r}, "
            f"got checkpoint name {checkpoint.name!r}. "
            "For native perturbation sensitivity, use the target decoder's own "
            "SimpleCompressor checkpoint."
        )


def build_decoder_batch(
    model: SimpleCompressor,
    memory: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = memory.shape[0]
    # Match SimpleCompressor.forward exactly. Qwen2.5 can set config.bos_token_id
    # to PAD while tokenizer.bos_token_id is None; training used tokenizer EOS.
    bos_id = normalize_token_id(model.decoder_tokenizer.bos_token_id)
    if bos_id is None:
        bos_id = normalize_token_id(model.decoder_tokenizer.eos_token_id)
    if bos_id is None:
        raise ValueError("Decoder tokenizer has neither bos_token_id nor eos_token_id.")
    bos_ids = torch.full((batch_size, 1), bos_id, dtype=decoder_input_ids.dtype, device=decoder_input_ids.device)
    bos_embedding = model.decoder.get_input_embeddings()(bos_ids)
    target_embeddings = model.decoder.get_input_embeddings()(decoder_input_ids)
    inputs_embeds = torch.cat([bos_embedding, memory, target_embeddings], dim=1)

    bos_mask = torch.ones(batch_size, 1, dtype=decoder_attention_mask.dtype, device=decoder_input_ids.device)
    memory_mask = torch.ones(batch_size, memory.shape[1], dtype=decoder_attention_mask.dtype, device=decoder_input_ids.device)
    attention_mask = torch.cat([bos_mask, memory_mask, decoder_attention_mask], dim=1)

    prefix_labels = torch.full(
        (batch_size, memory.shape[1] + 1),
        -100,
        dtype=decoder_input_ids.dtype,
        device=decoder_input_ids.device,
    )
    target_labels = decoder_input_ids.clone()
    target_labels[decoder_attention_mask == 0] = -100
    labels = torch.cat([prefix_labels, target_labels], dim=1)
    return inputs_embeds, attention_mask, labels


def shifted_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if mask.sum().item() == 0:
        return 0, 0
    preds = shift_logits.argmax(dim=-1)
    correct = ((preds == shift_labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


def perturb_memory(memory: torch.Tensor, mode: str, sigma: float, generator: torch.Generator) -> torch.Tensor:
    if mode == "gaussian" and sigma == 0:
        return memory

    if mode == "zero":
        return torch.zeros_like(memory)

    if mode == "shuffle":
        order = torch.randperm(memory.shape[1], device=memory.device, generator=generator)
        return memory[:, order, :]

    rms = torch.sqrt((memory.float() ** 2).mean(dim=-1, keepdim=True)).clamp_min(1e-8)
    noise = torch.randn(memory.shape, device=memory.device, dtype=memory.dtype, generator=generator)

    if mode == "gaussian":
        return memory + noise * rms.to(memory.dtype) * sigma

    if mode == "random":
        return noise * rms.to(memory.dtype)

    raise ValueError(f"Unknown perturbation mode: {mode}")


def evaluate_level(
    model: SimpleCompressor,
    texts: list[str],
    mode: str,
    sigma: float,
    max_length: int,
    device: str,
    trials: int,
    seed: int,
) -> dict[str, float]:
    loss_sum = 0.0
    n_loss = 0
    correct = 0
    total = 0
    generator = torch.Generator(device=device)

    for trial in range(trials):
        generator.manual_seed(seed + trial)
        for text in texts:
            comp = model.compressor_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
            dec = model.decoder_tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
            comp = {k: v.to(device) for k, v in comp.items()}
            dec = {k: v.to(device) for k, v in dec.items()}

            with torch.inference_mode():
                memory = model.compress(comp["input_ids"], comp["attention_mask"])
                memory = perturb_memory(memory, mode, sigma, generator)
                inputs_embeds, attention_mask, labels = build_decoder_batch(
                    model,
                    memory,
                    dec["input_ids"],
                    dec["attention_mask"],
                )
                outputs = model.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
            loss_sum += float(outputs.loss.item())
            n_loss += 1
            c, t = shifted_accuracy(outputs.logits, labels)
            correct += c
            total += t

    return {
        "mode": mode,
        "sigma": sigma,
        "n_examples": len(texts) * trials,
        "loss": loss_sum / max(n_loss, 1),
        "accuracy": correct / total if total else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compressor_model", type=str, required=True)
    parser.add_argument("--decoder_model", type=str, required=True)
    parser.add_argument(
        "--expected_checkpoint_encoder",
        type=str,
        default=None,
        help="Optional short encoder key used to verify checkpoint name, e.g. llama1b.",
    )
    parser.add_argument(
        "--expected_checkpoint_decoder",
        type=str,
        default=None,
        help="Optional short decoder key used to verify checkpoint name, e.g. qwen3b.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_mem_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--noise_levels", default="0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.4,3.2")
    parser.add_argument(
        "--perturbation_modes",
        default="gaussian",
        help="Comma-separated modes: gaussian,zero,random,shuffle. Controls are evaluated once per listed mode.",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    validate_checkpoint_name(args.checkpoint, args.expected_checkpoint_encoder, args.expected_checkpoint_decoder)

    texts = load_texts(args.dataset, args.max_samples)
    model = SimpleCompressor.from_pretrained(
        checkpoint_path=str(args.checkpoint),
        compressor_model_name=str(args.compressor_model),
        decoder_model_name=str(args.decoder_model),
        n_mem_tokens=args.n_mem_tokens,
        dtype=torch.bfloat16,
        device=args.device,
    ).to(args.device)
    model.eval()
    model.decoder.eval()

    rows: list[dict[str, float | str]] = []
    clean_acc = None

    modes = parse_str_list(args.perturbation_modes)
    if "gaussian" not in modes:
        modes = ["gaussian", *modes]

    for sigma in parse_float_list(args.noise_levels):
        result = evaluate_level(model, texts, "gaussian", sigma, args.max_length, args.device, args.trials, args.seed)
        if sigma == 0:
            clean_acc = result["accuracy"]
        result["relative_accuracy"] = result["accuracy"] / clean_acc if clean_acc else float("nan")
        rows.append(result)

    for mode in modes:
        if mode == "gaussian":
            continue
        result = evaluate_level(model, texts, mode, math.nan, args.max_length, args.device, args.trials, args.seed)
        result["relative_accuracy"] = result["accuracy"] / clean_acc if clean_acc else float("nan")
        rows.append(result)

    write_csv(args.output, rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
