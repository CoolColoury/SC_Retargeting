"""Shared checkpoint name parsing for RQ3 and backups export."""
from __future__ import annotations

import re

ALIASES = {
    "gpt2": "gpt2",
    "llama1b": "llama1b",
    "llama3b": "llama3b",
    "llama8b": "llama8b",
    "mistral7b": "mistral7b",
    "mistral7binstruct": "mistral7b",
    "qwen1.5b": "qwen1.5b",
    "qwen3b": "qwen3b",
    "qwen7b": "qwen7b",
}


def canonical(name: str) -> str:
    key = name.strip()
    if key not in ALIASES:
        raise ValueError(f"Unknown model alias: {key}")
    return ALIASES[key]


def parse_origin_checkpoint(name: str) -> tuple[str, str, int]:
    base = name.split("_mem", 1)[0]
    mem_match = re.search(r"_mem(\d+)_", name)
    if not mem_match:
        raise ValueError(f"Cannot parse mem from origin checkpoint: {name}")
    encoder, target = base.split("_to_", 1)
    return canonical(encoder), canonical(target), int(mem_match.group(1))


def parse_ori_transfer_checkpoint(name: str) -> tuple[str, str, str, str, int]:
    pattern = re.compile(
        r"^(?P<encoder>[^_]+)_to_(?P<source>[^_]+)_to_(?P<target>[^_]+)_"
        r"(?P<method>converter_only|encoder_converter)_mem(?P<mem>\d+)_"
    )
    match = pattern.match(name)
    if not match:
        raise ValueError(f"Cannot parse transfer checkpoint: {name}")
    return (
        canonical(match.group("encoder")),
        canonical(match.group("source")),
        canonical(match.group("target")),
        match.group("method"),
        int(match.group("mem")),
    )


def make_pair(encoder: str, source: str, target: str) -> str:
    return f"{canonical(encoder)}:{canonical(source)}->{canonical(target)}"


def logical_ori_transfer_key(checkpoint_name: str) -> tuple[str, str, str, str, str] | None:
    match = re.match(
        r"^(?P<enc>[^_]+)_to_(?P<src>.+?)_to_(?P<tgt>.+?)_"
        r"(?P<mode>converter_only|encoder_converter)_mem(?P<mem>\d+)_",
        checkpoint_name,
    )
    if not match:
        return None
    data = match.groupdict()
    return (
        canonical(data["enc"]),
        canonical(data["src"]),
        canonical(data["tgt"]),
        data["mode"],
        data["mem"],
    )


_TIMESTAMP_RE = re.compile(r"_(\d{8})_(\d{6})(?:_|$)")


def checkpoint_timestamp_score(checkpoint_name: str) -> int:
    match = _TIMESTAMP_RE.search(checkpoint_name)
    if not match:
        return 0
    return int(match.group(1)) * 1_000_000 + int(match.group(2))
