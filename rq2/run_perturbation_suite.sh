#!/usr/bin/env bash
set -euo pipefail

# RQ2: Gaussian / control perturbations on native SimpleCompressor memory vectors.
# Requires GPU, trained checkpoints, and ${DATA_ROOT}/fineweb_test.json.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PERTURB_SCRIPT="${SCRIPT_DIR}/perturbation/memory_perturbation_sensitivity.py"
OUT_DIR="${SCRIPT_DIR}/results/perturbation"

N_MEM="${N_MEM_TOKENS:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_LENGTH="${MAX_LENGTH:-128}"
TRIALS="${TRIALS:-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
NOISE_LEVELS="${NOISE_LEVELS:-0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.4,3.2}"
PERTURBATION_MODES="${PERTURBATION_MODES:-gaussian,zero,random,shuffle}"
DATASET="${DATASET:-${DATA_ROOT}/fineweb_test.json}"
MODELS="${MODELS_DIR:?Set MODELS_DIR to HuggingFace model root}"

declare -A COMPRESSOR_MODELS=(
  ["gpt2"]="${MODELS}/gpt2"
  ["llama1b"]="${MODELS}/Llama-3.2-1B-Instruct"
  ["llama3b"]="${MODELS}/Llama-3.2-3B-Instruct"
  ["llama8b"]="${MODELS}/Meta-Llama-3-8B-Instruct"
  ["mistral7b"]="${MODELS}/Mistral-7B-Instruct-v0.3"
  ["qwen1.5b"]="${MODELS}/Qwen2.5-1.5B-Instruct"
  ["qwen3b"]="${MODELS}/Qwen2.5-3B-Instruct"
  ["qwen7b"]="${MODELS}/Qwen2.5-7B-Instruct"
)

DEFAULT_RUNS="llama1b->llama1b,llama1b->llama3b,llama1b->qwen3b,llama1b->qwen7b,gpt2->llama3b,gpt2->qwen3b"
IFS=',' read -r -a RUN_SPECS <<< "${RUNS:-$DEFAULT_RUNS}"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

run_group() {
  local spec="$1"
  local encoder="${spec%%->*}"
  local target="${spec##*->}"

  compressor_model="${COMPRESSOR_MODELS[$encoder]:-}"
  decoder_model="${COMPRESSOR_MODELS[$target]:-}"
  if [ -z "$compressor_model" ] || [ -z "$decoder_model" ]; then
    echo "[skip] ${spec} (unknown model: encoder=${encoder} target=${target})"
    return 0
  fi

  ckpt="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}/${encoder}_to_${target}_mem${N_MEM}_len${MAX_LENGTH}"
  out_csv="${OUT_DIR}/${encoder}_to_${target}_mem${N_MEM}_native_bosfix_wide_noise.csv"

  echo "==> ${spec} mem=${N_MEM}"
  python "$PERTURB_SCRIPT" \
    --checkpoint "$ckpt" \
    --compressor_model "$compressor_model" \
    --decoder_model "$decoder_model" \
    --dataset "$DATASET" \
    --output "$out_csv" \
    --n_mem_tokens "$N_MEM" \
    --max_samples "$MAX_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --noise_levels "$NOISE_LEVELS" \
    --perturbation_modes "$PERTURBATION_MODES" \
    --trials "$TRIALS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --expected_encoder "$encoder" \
    --expected_decoder "$target"
}

for spec in "${RUN_SPECS[@]}"; do
  run_group "$spec"
done

echo "Wrote CSVs under ${OUT_DIR}"
