#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

N_MEM_TOKENS="${N_MEM_TOKENS:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
TEMPERATURE="${TEMPERATURE:-0.1}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-${DATA_ROOT}/fineweb_test.json}"
RUNS="${RUNS:-llama1b->llama1b,llama1b->llama3b,llama1b->qwen1.5b,llama1b->qwen3b,llama1b->qwen7b,gpt2->llama3b,gpt2->qwen1.5b,gpt2->qwen3b,gpt2->qwen7b}"
OUT_DIR="${SCRIPT_DIR}/../results/native_geometry"
OUTPUT="${OUTPUT:-${OUT_DIR}/native_memory_geometry_mem${N_MEM_TOKENS}.csv}"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"

python "${SCRIPT_DIR}/native_memory_geometry.py" \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --n_mem_tokens "$N_MEM_TOKENS" \
  --max_samples "$MAX_SAMPLES" \
  --max_length "$MAX_LENGTH" \
  --max_anchors "$MAX_ANCHORS" \
  --temperature "$TEMPERATURE" \
  --device "$DEVICE" \
  --runs "$RUNS"
