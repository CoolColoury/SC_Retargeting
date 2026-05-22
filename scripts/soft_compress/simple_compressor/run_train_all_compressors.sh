#!/usr/bin/env bash
#
# Paper bundle: train all native SimpleCompressor checkpoints used in the work.
# Combines the former run_all.sh / run_compressor_selected grid with the
# Mistral-target rows from run_train_mistral_compressors.sh.
#
# Usage:
#   export PROJECT_ROOT=$(pwd)   # repository root
#   export MODELS_DIR=...        # local HF checkout root
#   export DATA_ROOT=...         # contains fineweb_test.json
#   bash scripts/soft_compress/simple_compressor/run_train_all_compressors.sh
#
# Optional env:
#   MEM_LIST="8 16 32"           # memory-token sizes (default: 8 16 32)
#   SEGMENT_LENGTH=128           # segment length (default: 128)
#   EPOCHS BATCH_SIZE            # forwarded to run_compressor_single.sh
#   SKIP_EXISTING=1              # skip run if checkpoint already exists (default: 1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${PROJECT_ROOT}"

PATH_TO_SCRIPT="scripts/soft_compress/simple_compressor"
TRAIN_ONE="${PATH_TO_SCRIPT}/run_compressor_single.sh"

MEM_LIST="${MEM_LIST:-8 16 32}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-128}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

COMPRESSORS=(gpt2 llama1b llama3b)
DECODERS=(llama1b llama3b llama8b qwen1.5b qwen3b qwen7b mistral7b)

if [[ ! -f "${TRAIN_ONE}" ]]; then
  echo "Missing ${TRAIN_ONE}" >&2
  exit 1
fi

run_one() {
  local c="$1" d="$2" m="$3"
  local save_dir="outputs/simple_compressor/${c}_to_${d}_mem${m}_len${SEGMENT_LENGTH}_ds_4gpu"
  if [[ "${SKIP_EXISTING}" == "1" ]] && { [[ -f "${save_dir}/pytorch_model.bin" ]] || [[ -f "${save_dir}/model.safetensors" ]]; }; then
    echo "[SKIP] ${c} -> ${d} mem=${m} (${save_dir})"
    return 0
  fi
  echo "[RUN ] ${c} -> ${d} mem=${m} len=${SEGMENT_LENGTH}"
  bash "${TRAIN_ONE}" "${c}" "${d}" "${m}" "${SEGMENT_LENGTH}"
}

echo "=========================================="
echo "Train all paper SimpleCompressors"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MEM_LIST=${MEM_LIST}  SEGMENT_LENGTH=${SEGMENT_LENGTH}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "=========================================="

for c in "${COMPRESSORS[@]}"; do
  for d in "${DECODERS[@]}"; do
    for m in ${MEM_LIST}; do
      run_one "${c}" "${d}" "${m}"
    done
  done
done

echo "Done."
