"""Decoder prefix token ids aligned with SimpleCompressor.forward."""

from __future__ import annotations

from typing import Any


def normalize_token_id(token_id: Any) -> int | None:
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        if len(token_id) == 0:
            return None
        return int(token_id[0])
    return int(token_id)


def decoder_bos_token_id(tokenizer: Any, decoder_config: Any | None = None) -> int:
    """Match SimpleCompressor: tokenizer BOS, else tokenizer EOS (not config.bos).

    Qwen2.5 sets config.bos_token_id to PAD while tokenizer.bos is None; native
    training uses tokenizer EOS (151645) as the prefix embedding.
    """
    bos_id = normalize_token_id(getattr(tokenizer, "bos_token_id", None))
    if bos_id is None:
        bos_id = normalize_token_id(getattr(tokenizer, "eos_token_id", None))
    if bos_id is not None:
        return bos_id
    if decoder_config is not None:
        eos_id = normalize_token_id(getattr(decoder_config, "eos_token_id", None))
        if eos_id is not None:
            return eos_id
    raise ValueError("Cannot determine decoder prefix token id from tokenizer or config.eos")


def decoder_eos_token_id(tokenizer: Any, decoder_config: Any | None = None) -> int:
    """EOS for generation: prefer model.config.eos, else tokenizer eos."""
    if decoder_config is not None:
        eos_id = normalize_token_id(getattr(decoder_config, "eos_token_id", None))
        if eos_id is not None:
            return eos_id
    eos_id = normalize_token_id(getattr(tokenizer, "eos_token_id", None))
    if eos_id is not None:
        return eos_id
    raise ValueError("Cannot determine decoder eos token id")
