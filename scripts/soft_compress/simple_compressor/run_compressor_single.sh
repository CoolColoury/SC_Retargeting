#!/bin/bash

# Generic training script for a single compressor configuration
# Usage: bash run_compressor_single.sh <compressor_model> <decoder_model> <n_mem_tokens> <segment_length>
# Example: bash run_compressor_single.sh gpt2 llama1b 32 128

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <compressor_model> <decoder_model> <n_mem_tokens> <segment_length>"
    echo "Example: $0 gpt2 llama1b 32 128"
    exit 1
fi

COMPRESSOR_MODEL_NAME="$1"
DECODER_MODEL_NAME="$2"
N_MEM_TOKENS="$3"
SEGMENT_LENGTH="$4"

PROJECT_ROOT="."
TRAIN_SCRIPT="${PROJECT_ROOT}/src/soft_compress/simple_compressor/train.py"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/scripts/soft_compress/simple_compressor/ds_config_zero1_bf16.json"

# Model name to path mapping
declare -A MODEL_PATHS=(
    ["gpt2"]="${MODELS_DIR}/gpt2"
    ["llama1b"]="${MODELS_DIR}/Llama-3.2-1B-Instruct"
    ["llama3b"]="${MODELS_DIR}/Llama-3.2-3B-Instruct"
    ["llama8b"]="${MODELS_DIR}/Llama-3-8B-Instruct"
    ["qwen1.5b"]="${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct"
    ["qwen3b"]="${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct"
    ["qwen7b"]="${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct"
    ["mistral7b"]="${MODELS_DIR}/Mistral-7B-Instruct"
)

# Model name to display name mapping (for output directory)
declare -A MODEL_DISPLAY_NAMES=(
    ["gpt2"]="gpt2"
    ["llama1b"]="llama1b"
    ["llama3b"]="llama3b"
    ["llama8b"]="llama8b"
    ["qwen1.5b"]="qwen1.5b"
    ["qwen3b"]="qwen3b"
    ["qwen7b"]="qwen7b"
    ["mistral7b"]="mistral7b"
)

# Validate model names
if [ -z "${MODEL_PATHS[$COMPRESSOR_MODEL_NAME]}" ]; then
    echo "Error: Unknown compressor model name: ${COMPRESSOR_MODEL_NAME}"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

if [ -z "${MODEL_PATHS[$DECODER_MODEL_NAME]}" ]; then
    echo "Error: Unknown decoder model name: ${DECODER_MODEL_NAME}"
    echo "Available models: ${!MODEL_PATHS[@]}"
    exit 1
fi

COMPRESSOR_MODEL="${MODEL_PATHS[$COMPRESSOR_MODEL_NAME]}"
DECODER_MODEL="${MODEL_PATHS[$DECODER_MODEL_NAME]}"
COMPRESSOR_DISPLAY="${MODEL_DISPLAY_NAMES[$COMPRESSOR_MODEL_NAME]}"
DECODER_DISPLAY="${MODEL_DISPLAY_NAMES[$DECODER_MODEL_NAME]}"

# Training configuration
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-4}"

# Data paths
TRAIN_DATA="datasets/simple_mem_1M/fineweb_train.json"
VALID_DATA="${DATA_ROOT}/fineweb_test.json"

# GPU configuration (4 GPUs: 0,1,2,3)
NUM_GPUS=4
GPU_LIST="4,5,6,7"
MASTER_PORT=${MASTER_PORT:-29500}

# Output directory
SAVE_DIR="outputs/simple_compressor/${COMPRESSOR_DISPLAY}_to_${DECODER_DISPLAY}_mem${N_MEM_TOKENS}_len${SEGMENT_LENGTH}_ds_4gpu"

echo "=========================================="
echo "Training Compressor: ${COMPRESSOR_MODEL_NAME} -> ${DECODER_MODEL_NAME}"
echo "=========================================="
echo "Compressor: ${COMPRESSOR_MODEL}"
echo "Decoder: ${DECODER_MODEL}"
echo "N_MEM_TOKENS: ${N_MEM_TOKENS}"
echo "SEGMENT_LENGTH: ${SEGMENT_LENGTH}"
echo "Output: ${SAVE_DIR}"
echo "=========================================="

# Run training with DeepSpeed
# Resolve DeepSpeed launcher from current Python env first.
if python -c "import deepspeed" >/dev/null 2>&1; then
    DEEPSPEED_LAUNCHER=(python -m deepspeed.launcher.runner)
elif command -v deepspeed >/dev/null 2>&1; then
    DEEPSPEED_LAUNCHER=(deepspeed)
else
    echo "Error: DeepSpeed launcher not found. Please install in current environment."
    exit 1
fi
echo "DeepSpeed launcher: ${DEEPSPEED_LAUNCHER[*]}"

"${DEEPSPEED_LAUNCHER[@]}" --include localhost:${GPU_LIST} --master_port ${MASTER_PORT} \
    "${TRAIN_SCRIPT}" \
    --compress_model "${COMPRESSOR_MODEL}" \
    --decoder_model "${DECODER_MODEL}" \
    --embed_len "${N_MEM_TOKENS}" \
    --segment_length "${SEGMENT_LENGTH}" \
    --train_data_dir "${TRAIN_DATA}" \
    --valid_data_dir "${VALID_DATA}" \
    --output_dir "${SAVE_DIR}" \
    --logging_dir "${SAVE_DIR}/logs" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LR}" \
    --save_strategy steps \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10 \
    --bf16 \
    --dataloader_num_workers 8 \
    --save_total_limit 1 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --gradient_accumulation_steps 1

echo ""
echo "=========================================="
echo "Training completed: ${COMPRESSOR_MODEL_NAME} -> ${DECODER_MODEL_NAME}"
echo "Output: ${SAVE_DIR}"
echo "=========================================="

