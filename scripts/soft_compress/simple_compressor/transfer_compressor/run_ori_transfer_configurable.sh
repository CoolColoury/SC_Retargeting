#!/bin/bash

# Training script for OriTransfer Compressor (Configurable Version)
# Example: Transfer from gpt2-to-llama1b to gpt2-to-llama8b
#
# This script demonstrates direct continuation training of Simple Compressor
# without adding a projector layer.
#
# Two training modes:
# 1. converter_only: Only train Converter (freeze Encoder)
# 2. encoder_converter: Train both Encoder and Converter
#
# Usage:
#   # Method 1: Set environment variables
#   export TRAIN_MODE="converter_only"
#   export SRC_COMPRESSOR_PATH="outputs/simple_compressor/gpt2_to_llama1b_mem32_len128_ds_4gpu"
#   export SRC_DECODER_MODEL_PATH="${MODELS_DIR}/gpt2"
#   export TGT_MODEL_PATH="${MODELS_DIR}/Llama-3-8B-Instruct"
#   export EMBED_LEN=32
#   export SEGMENT_LENGTH=128
#   bash run_ori_transfer_configurable.sh [num_samples] [epochs] [batch_size] [lr]
#
#   # Method 2: Pass as arguments (in order: train_mode, src_compressor_path, src_decoder_path, tgt_model_path, embed_len, segment_length)
#   bash run_ori_transfer_configurable.sh 1000000 1 16 0.0001 converter_only "outputs/..." "/home/..." "/home/..." 32 128

set -e

# ============================================================================
# Configuration - Can be overridden by environment variables or command line args
# ============================================================================

# Training parameters (positional arguments)
NUM_SAMPLES="${1:-1000000}"          # Default: 1000000 samples
EPOCHS="${2:-1}"                     # Default: 1 epoch
BATCH_SIZE="${3:-16}"                # Default: batch size 16
LEARNING_RATE="${4:-0.0001}"         # Default: 1e-4

# Model configuration (can be set via env vars or positional args 5-10)
# Priority: Command line args > Environment variables > Defaults
if [ -n "$5" ]; then
    TRAIN_MODE="$5"
elif [ -z "$TRAIN_MODE" ]; then
    TRAIN_MODE="converter_only"      # Default: converter_only
fi

if [ -n "$6" ]; then
    SRC_COMPRESSOR_PATH="$6"
elif [ -z "$SRC_COMPRESSOR_PATH" ]; then
    SRC_COMPRESSOR_PATH="outputs/simple_compressor/gpt2_to_llama1b_mem32_len128_ds_4gpu"
fi

if [ -n "$7" ]; then
    SRC_DECODER_MODEL_PATH="$7"
elif [ -z "$SRC_DECODER_MODEL_PATH" ]; then
    SRC_DECODER_MODEL_PATH="${MODELS_DIR}/gpt2"
fi

if [ -n "$8" ]; then
    TGT_MODEL_PATH="$8"
elif [ -z "$TGT_MODEL_PATH" ]; then
    TGT_MODEL_PATH="${MODELS_DIR}/Llama-3-8B-Instruct"
fi

if [ -n "$9" ]; then
    EMBED_LEN="$9"
elif [ -z "$EMBED_LEN" ]; then
    EMBED_LEN=32
fi

if [ -n "${10}" ]; then
    SEGMENT_LENGTH="${10}"
elif [ -z "$SEGMENT_LENGTH" ]; then
    SEGMENT_LENGTH=128
fi

# GPU configuration (can be overridden by environment variables)
NUM_GPUS="${NUM_GPUS:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Validate training mode early
if [[ "${TRAIN_MODE}" != "converter_only" && "${TRAIN_MODE}" != "encoder_converter" ]]; then
    echo "Error: TRAIN_MODE must be 'converter_only' or 'encoder_converter'. Got: ${TRAIN_MODE}"
    exit 1
fi

# Data paths
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${PROJECT_ROOT}/datasets/simple_mem_1M/fineweb_train.json}"
VALID_DATA_DIR="${VALID_DATA_DIR:-${PROJECT_ROOT}/${DATA_ROOT}/fineweb_test.json}"

# Eval samples (fixed small subset by default)
VALID_DATA_SAMPLES="${VALID_DATA_SAMPLES:-1000}"

# Training hyperparameters
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"

# Output directory
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-outputs/simple_compressor/transfer_compressor/ori_transfer}"

# Extract source encoder and decoder, and target model names for output directory naming
# If OUTPUT_DIR_NAME is set, use it; otherwise assemble from SRC_COMPRESSOR_PATH and TGT_MODEL_PATH
if [ -z "$OUTPUT_DIR_NAME" ]; then
    # Try to parse SRC_COMPRESSOR_BASENAME format: {src_encoder}_to_{src_decoder}_mem...
    SRC_COMPRESSOR_BASENAME=$(basename "${SRC_COMPRESSOR_PATH}")
    if [[ "$SRC_COMPRESSOR_BASENAME" =~ ^([^_]+)_to_([^_]+)_mem ]]; then
        SRC_ENCODER_SHORT="${BASH_REMATCH[1]}"
        SRC_DECODER_SHORT="${BASH_REMATCH[2]}"
    else
        # fallback
        SRC_ENCODER_SHORT="gpt2"
        SRC_DECODER_SHORT="llama1b"
    fi

    # Determine short name for TGT_MODEL_PATH (from path)
    TGT_MODEL_BASENAME=$(basename "${TGT_MODEL_PATH}")
    case "${TGT_MODEL_BASENAME}" in
        "Llama-3-8B-Instruct"|"Llama-3.2-8B-Instruct")
            TGT_MODEL_SHORT="llama8b"
            ;;
        "Qwen2.5-8B-Instruct"|"Qwen-8B-Instruct")
            TGT_MODEL_SHORT="qwen8b"
            ;;
        "Llama-3.2-1B-Instruct")
            TGT_MODEL_SHORT="llama1b"
            ;;
        "Llama-3.2-3B-Instruct")
            TGT_MODEL_SHORT="llama3b"
            ;;
        "Qwen2.5-1.5B-Instruct")
            TGT_MODEL_SHORT="qwen1.5b"
            ;;
        "Qwen2.5-3B-Instruct")
            TGT_MODEL_SHORT="qwen3b"
            ;;
        "Qwen2.5-7B-Instruct")
            TGT_MODEL_SHORT="qwen7b"
            ;;
        "Mistral-7B-Instruct")
            TGT_MODEL_SHORT="mistral7binstruct"
            ;;
        *)
            # Fallback: use lowercase basename with special chars removed
            TGT_MODEL_SHORT=$(echo "${TGT_MODEL_BASENAME}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
            ;;
    esac
    OUTPUT_DIR_NAME="${SRC_ENCODER_SHORT}_to_${SRC_DECODER_SHORT}_to_${TGT_MODEL_SHORT}"
fi

OUTPUT_DIR="${OUTPUT_DIR_BASE}/${OUTPUT_DIR_NAME}_${TRAIN_MODE}_mem${EMBED_LEN}_len${SEGMENT_LENGTH}"

# DeepSpeed config
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/soft_compress/simple_compressor/ds_config_zero1_bf16.json}"

# Random seed
RANDOM_SEED="${RANDOM_SEED:-42}"

# Script paths
PROJECT_ROOT="${PROJECT_ROOT:-.}"
TRAIN_SCRIPT="${PROJECT_ROOT}/src/soft_compress/transfer_compressor/train_ori_transfer.py"

# Save total limit (can be overridden)
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"

# ============================================================================
# Validation
# ============================================================================

echo "=================================="
echo "OriTransfer Compressor Training"
echo "=================================="
echo "Train dataset: ${TRAIN_DATA_DIR}"
echo "Valid dataset: ${VALID_DATA_DIR}"
echo "Source compressor: ${SRC_COMPRESSOR_PATH}"
echo "Source decoder: ${SRC_DECODER_MODEL_PATH}"
echo "Target model: ${TGT_MODEL_PATH}"
echo "Train mode: ${TRAIN_MODE}"
echo "Memory tokens: ${EMBED_LEN}"
echo "Segment length: ${SEGMENT_LENGTH}"
echo "=================================="
echo "Training parameters:"
echo "  - Num samples: ${NUM_SAMPLES}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "  - Grad accum: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  - Warmup ratio: ${WARMUP_RATIO}"
echo "=================================="
echo "GPU configuration:"
echo "  - GPUs: ${GPU_LIST}"
echo "  - Master port: ${MASTER_PORT}"
echo "=================================="

# Check if datasets exist
if [ ! -f "${TRAIN_DATA_DIR}" ]; then
    echo "Error: Train dataset not found: ${TRAIN_DATA_DIR}"
    exit 1
fi

if [ ! -f "${VALID_DATA_DIR}" ]; then
    echo "Warning: Valid dataset not found: ${VALID_DATA_DIR}"
    echo "Will use training data for validation"
    VALID_DATA_DIR="${TRAIN_DATA_DIR}"
fi

# Check if source compressor exists
if [ ! -d "${SRC_COMPRESSOR_PATH}" ]; then
    echo "Error: Source compressor not found: ${SRC_COMPRESSOR_PATH}"
    echo "Please train a compressor first or update SRC_COMPRESSOR_PATH"
    exit 1
fi

# Check if training script exists
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Error: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# Check if DeepSpeed config exists
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "Error: DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
    exit 1
fi

# Output directory (timestamped like run_e2e_test.sh)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR}_${NUM_SAMPLES}samples_${TIMESTAMP}"

echo "Output directory: ${OUTPUT_DIR}"
echo "=================================="

# ============================================================================
# Run Training
# ============================================================================

# Create output directory
mkdir -p $OUTPUT_DIR

# =============================================================================
# Step 1: Train OriTransfer Compressor (using DeepSpeed)
# =============================================================================

echo ""
echo "=========================================="
echo "Step 1: Training OriTransfer Compressor (DeepSpeed)"
echo "=========================================="
echo "Using ${NUM_GPUS} GPUs with DeepSpeed"
echo "Architecture: Simple Compressor continuation (no extra projector)"
echo "Mode: ${TRAIN_MODE}"
echo ""

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

# Run training with DeepSpeed
"${DEEPSPEED_LAUNCHER[@]}" --include localhost:${GPU_LIST} --master_port ${MASTER_PORT} \
    "${TRAIN_SCRIPT}" \
    --src_compressor_path "${SRC_COMPRESSOR_PATH}" \
    --src_decoder_model_path "${SRC_DECODER_MODEL_PATH}" \
    --tgt_model_path "${TGT_MODEL_PATH}" \
    --embed_len ${EMBED_LEN} \
    --segment_length ${SEGMENT_LENGTH} \
    --train_mode "${TRAIN_MODE}" \
    --train_data_dir "${TRAIN_DATA_DIR}" \
    --valid_data_dir "${VALID_DATA_DIR}" \
    --train_data_samples ${NUM_SAMPLES} \
    --valid_data_samples ${VALID_DATA_SAMPLES} \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.01 \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 2000 \
    --save_strategy steps \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --eval_strategy steps \
    --eval_steps 500 \
    --bf16 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --random_seed ${RANDOM_SEED} \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --gradient_checkpointing False

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "✓ Training completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}"

# Remove step checkpoints to save disk (root model weights remain in OUTPUT_DIR)
if [ "${REMOVE_CHECKPOINTS_AFTER_TRAIN:-1}" != "0" ]; then
    shopt -s nullglob
    for ckpt_dir in "${OUTPUT_DIR}"/checkpoint-*; do
        if [ -d "${ckpt_dir}" ]; then
            echo "Removing DeepSpeed/HF checkpoint dir to save space: ${ckpt_dir}"
            rm -rf "${ckpt_dir}"
        fi
    done
    shopt -u nullglob
fi

# =============================================================================
# Step 2: Display Training Metrics
# =============================================================================

echo ""
echo "=========================================="
echo "Step 2: Training Metrics"
echo "=========================================="
echo ""

METRICS_FILE="${OUTPUT_DIR}/metrics.json"
if [ -f "${METRICS_FILE}" ]; then
    echo "Final Metrics:"
    cat "${METRICS_FILE}"
    echo ""
else
    echo "Warning: Metrics file not found: ${METRICS_FILE}"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Model architecture: Simple Compressor continuation (no extra projector)"
echo "Training mode: ${TRAIN_MODE}"
echo "Training configuration:"
echo "  - Samples: ${NUM_SAMPLES}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo ""
echo "Model paths:"
echo "  - Source compressor: ${SRC_COMPRESSOR_PATH}"
echo "  - Source decoder: ${SRC_DECODER_MODEL_PATH}"
echo "  - Target model: ${TGT_MODEL_PATH}"
echo "  - Embed len: ${EMBED_LEN}"
echo "  - Segment length: ${SEGMENT_LENGTH}"
echo ""
echo "Output files:"
echo "  - Metrics: ${OUTPUT_DIR}/metrics.json"
echo "  - Final weights: ${OUTPUT_DIR}/pytorch_model.bin (or model.safetensors)"
echo ""
echo "=========================================="
echo "✓ All tasks completed!"
echo "=========================================="
