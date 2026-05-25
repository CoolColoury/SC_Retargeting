"""Shared anchor construction for linear alignment (main-table LS + ablations)."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_converter import Converter


def get_vocab(tokenizer) -> dict[str, int]:
    vocab = tokenizer.get_vocab()
    return {tok: int(idx) for tok, idx in vocab.items()}


def list_common_anchor_tokens(src_tokenizer, tgt_tokenizer) -> list[str]:
    """Sorted shared non-special tokenizer items (RQ3-style, before subsampling)."""
    src_vocab = get_vocab(src_tokenizer)
    tgt_vocab = get_vocab(tgt_tokenizer)
    special = set(src_tokenizer.all_special_tokens) | set(tgt_tokenizer.all_special_tokens)
    return [
        tok
        for tok in sorted(set(src_vocab) & set(tgt_vocab))
        if tok not in special and len(tok.strip()) > 0
    ]


def _subsample_indices(src_indices: list[int], tgt_indices: list[int], max_anchors: int):
    if len(src_indices) <= max_anchors:
        return src_indices, tgt_indices
    step = len(src_indices) / max_anchors
    picked = [int(i * step) for i in range(max_anchors)]
    return [src_indices[i] for i in picked], [tgt_indices[i] for i in picked]


class _VocabAnchorHelper(Converter):
    """Thin wrapper to reuse base_converter._vocab_filter for vocab_100k anchors."""

    def __init__(self, src_model_path: str, tgt_model_path: str):
        super().__init__(src_model_path, tgt_model_path, None, "anchor_helper")


def build_vocab_file_anchor_indices(
    src_model_path: str,
    tgt_model_path: str,
    common_vocab: str,
    max_anchors: Optional[int] = None,
) -> tuple[list[int], list[int], int]:
    """Match eval_ls_transfer.sh / main-table LS: vocab file + single-token filter."""
    helper = _VocabAnchorHelper(src_model_path, tgt_model_path)
    src_indices, tgt_indices = helper._vocab_filter(common_vocab)
    if not src_indices:
        raise ValueError(f"No vocab-file anchors for {src_model_path} -> {tgt_model_path}")
    if max_anchors is not None:
        src_indices, tgt_indices = _subsample_indices(src_indices, tgt_indices, max_anchors)
    return src_indices, tgt_indices, len(src_indices)


def count_vocab_file_anchors(
    src_model_path: str,
    tgt_model_path: str,
    common_vocab: str,
    max_anchors: Optional[int] = None,
) -> int:
    _, _, n = build_vocab_file_anchor_indices(
        src_model_path, tgt_model_path, common_vocab, max_anchors=max_anchors
    )
    return n


def load_model_embeddings(model_path: str) -> torch.Tensor:
    model_name_lower = model_path.lower()
    if not any(tag in model_name_lower for tag in ("qwen", "mistral", "llama", "gpt2")):
        raise NotImplementedError(f"Unsupported model path: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    embeddings = model.get_input_embeddings().weight.detach().cpu()
    del model
    return embeddings


def get_vocab_file_anchor_embeddings(
    src_model_path: str,
    tgt_model_path: str,
    common_vocab: str,
    max_anchors: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (src_anchor_embeds, tgt_anchor_embeds, n_anchors) using vocab_100k logic."""
    src_ids, tgt_ids, n = build_vocab_file_anchor_indices(
        src_model_path, tgt_model_path, common_vocab, max_anchors=max_anchors
    )
    src_embeddings = load_model_embeddings(src_model_path)
    tgt_embeddings = load_model_embeddings(tgt_model_path)
    return src_embeddings[src_ids], tgt_embeddings[tgt_ids], n


# --- RQ3-style helpers (tokenizer intersection; used by priors, not main-table LS) ---

def build_rq3_anchor_ids(
    src_tokenizer,
    tgt_tokenizer,
    max_anchors: int,
) -> tuple[list[int], list[int], list[str]]:
    src_vocab = get_vocab(src_tokenizer)
    tgt_vocab = get_vocab(tgt_tokenizer)
    tokens = list_common_anchor_tokens(src_tokenizer, tgt_tokenizer)
    if len(tokens) > max_anchors:
        step = len(tokens) / max_anchors
        tokens = [tokens[int(i * step)] for i in range(max_anchors)]
    if len(tokens) < 32:
        raise ValueError(f"Too few RQ3 anchor tokens: {len(tokens)} (max_anchors={max_anchors})")
    return [src_vocab[tok] for tok in tokens], [tgt_vocab[tok] for tok in tokens], tokens
