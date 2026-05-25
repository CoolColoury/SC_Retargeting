#!/usr/bin/env bash
set -euo pipefail

# Unified server runner for RQ2 perturbation sensitivity experiments.
#
# Important: this script perturbs a decoder's own trained SimpleCompressor
# checkpoint, i.e. checkpoint = ${encoder}_to_${target}. It does not evaluate
# cross-decoder transferred memory. At sigma=0, native checkpoints should match
# the usual compressor evaluation accuracy.
#
# Example:
#   bash experiments/soft_compress/analysis/run_perturbation_suite.sh
#
# Optional overrides:
#   N_MEM_TOKENS=32 MAX_SAMPLES=128 TRIALS=3 \
#   NOISE_LEVELS="0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,1.2,1.6,2.4,3.2" \
#   RUNS="llama1b->qwen1.5b,gpt2->qwen1.5b" \
#   PERTURBATION_MODES="gaussian,zero,random,shuffle" \
#   bash experiments/soft_compress/analysis/run_perturbation_suite.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_DIR}")/../..")"
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
DATASET="${DATASET:-${DATA_ROOT}"

declare -A COMPRESSOR_MODELS=(
  ["gpt2"]="${DATA_ROOT}"
  ["llama1b"]="${DATA_ROOT}"
  ["llama3b"]="${DATA_ROOT}"
  ["llama8b"]="${DATA_ROOT}"
  ["mistral7b"]="${DATA_ROOT}"
  ["qwen1.5b"]="${DATA_ROOT}"
  ["qwen3b"]="${DATA_ROOT}"
  ["qwen7b"]="${DATA_ROOT}"
)

# encoder->target (do NOT name this GROUPS — bash reserves GROUPS for gid list)
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
    return 1
  fi

  checkpoint="outputs/simple_compressor/${encoder}_to_${target}_mem${N_MEM}_len128_ds_4gpu"
  output="${OUT_DIR}/${encoder}_to_${target}_mem${N_MEM}_native_bosfix_wide_noise.csv"

  if [ ! -d "$checkpoint" ]; then
    echo "[skip] ${spec} (missing checkpoint: ${checkpoint})"
    return 1
  fi

  echo "=== ${spec} mem${N_MEM} checkpoint=${checkpoint} ==="
  python "$PERTURB_SCRIPT" \
    --checkpoint "$checkpoint" \
    --compressor_model "$compressor_model" \
    --decoder_model "$decoder_model" \
    --expected_checkpoint_encoder "$encoder" \
    --expected_checkpoint_decoder "$target" \
    --dataset "$DATASET" \
    --output "$output" \
    --n_mem_tokens "$N_MEM" \
    --max_samples "$MAX_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --noise_levels "$NOISE_LEVELS" \
    --perturbation_modes "$PERTURBATION_MODES" \
    --trials "$TRIALS" \
    --seed "$SEED" \
    --device "$DEVICE"
  return 0
}

total=0
ran=0
skipped=0

for spec in "${RUN_SPECS[@]}"; do
  total=$((total + 1))
  if run_group "$spec"; then
    ran=$((ran + 1))
  else
    skipped=$((skipped + 1))
  fi
done

echo ""
echo "Perturbation suite finished."
echo "planned=${total} ran=${ran} skipped=${skipped}"
echo "Outputs under ${OUT_DIR}/"