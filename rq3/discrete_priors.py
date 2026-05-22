"""Discrete pre-transfer priors from short model names (source, target) only.

These features do not use transfer outcomes, target compressors, or random/LS
runs. They are coarse architecture tags derived from the same ``source_name`` /
``target_name`` strings already passed to ``compute_pretransfer_priors.py``.
"""
from __future__ import annotations

import math

# Billions of parameters for canonical short names used in this project.
_PARAM_BILLIONS: dict[str, float] = {
    "llama1b": 1.0,
    "llama3b": 3.0,
    "llama8b": 8.0,
    "mistral7b": 7.0,
    "qwen1.5b": 1.5,
    "qwen3b": 3.0,
    "qwen7b": 7.0,
}


def model_family(short_name: str) -> str:
    n = (short_name or "").strip().lower()
    if n.startswith("llama"):
        return "llama"
    if n.startswith("qwen"):
        return "qwen"
    if n.startswith("mistral"):
        return "mistral"
    return "unknown"


def param_billions(short_name: str) -> float | None:
    key = (short_name or "").strip().lower()
    return _PARAM_BILLIONS.get(key)


def discrete_prior_features(source_name: str, target_name: str) -> dict[str, float]:
    """Return numeric 0/1 and log-scale features for correlation tables."""
    src_f = model_family(source_name)
    tgt_f = model_family(target_name)
    same = 1.0 if src_f == tgt_f and src_f != "unknown" else 0.0
    src_b = param_billions(source_name)
    tgt_b = param_billions(target_name)

    out: dict[str, float] = {
        "prior_same_model_family": same,
        "prior_cross_model_family": 1.0 - same,
    }

    if src_b is not None and tgt_b is not None:
        out["prior_target_param_b"] = tgt_b
        out["prior_source_param_b"] = src_b
        out["prior_log_param_ratio_target_over_source"] = math.log((tgt_b + 1e-9) / (src_b + 1e-9))
        # Ordinal "size jump" bucket: 0 same tier (~ratio in [0.85, 1.15]), 1 adjacent, 2 larger jump.
        ratio = tgt_b / src_b if src_b > 0 else float("nan")
        if not math.isnan(ratio):
            if 0.85 <= ratio <= 1.15:
                out["prior_param_tier_jump"] = 0.0
            elif ratio < 2.5:
                out["prior_param_tier_jump"] = 1.0
            else:
                out["prior_param_tier_jump"] = 2.0
        else:
            out["prior_param_tier_jump"] = float("nan")
    else:
        out["prior_target_param_b"] = float("nan")
        out["prior_source_param_b"] = float("nan")
        out["prior_log_param_ratio_target_over_source"] = float("nan")
        out["prior_param_tier_jump"] = float("nan")

    return out
