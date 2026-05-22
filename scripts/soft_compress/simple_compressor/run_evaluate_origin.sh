#!/bin/bash

# Evaluate a trained SimpleCompressor model
# Usage: bash run_evaluate_origin.sh <compressor_model> <decoder_model> <n_mem_tokens> <segment_length> [checkpoint_dir]
# Example: bash run_evaluate_origin.sh gpt2 llama1b 8 128 outputs/simple_compressor/gpt2_to_llama1b_mem8_len128_ds_4gpu

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <compressor_model> <decoder_model> <n_mem_tokens> <segment_length> [checkpoint_dir]"
    echo "Example: $0 gpt2 llama1b 8 128 outputs/simple_compressor/gpt2_to_llama1b_mem8_len128_ds_4gpu"
    echo ""
    echo "Arguments:"
    echo "  compressor_model: Model used in source compressor (e.g., gpt2)"
    echo "  decoder_model: Decoder model used in source compressor training (e.g., llama1b)"
    echo "  n_mem_tokens: Number of memory tokens (e.g., 8)"
    echo "  segment_length: Maximum sequence length (e.g., 128)"
    echo "  checkpoint_dir: (Optional) Checkpoint directory path (default: auto-construct from model names)"
    exit 1
fi

COMPRESSOR_MODEL_NAME="$1"
DECODER_MODEL_NAME="$2"
N_MEM_TOKENS="$3"
SEGMENT_LENGTH="$4"
CHECKPOINT_DIR="${5:-}"  # Optional: if not provided, will auto-construct

PROJECT_ROOT="."
EVAL_SCRIPT="${PROJECT_ROOT}/src/soft_compress/evaluation/test_simple_compressor.py"

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

# Auto-construct checkpoint_dir if not provided
if [ -z "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_DIR="outputs/simple_compressor/${COMPRESSOR_DISPLAY}_to_${DECODER_DISPLAY}_mem${N_MEM_TOKENS}_len${SEGMENT_LENGTH}_ds_4gpu"
fi

# Check if checkpoint directory exists
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint directory not found: ${CHECKPOINT_DIR}"
    echo "Please train the compressor first or provide a valid checkpoint directory."
    exit 1
fi

# Infer n_mem_tokens from checkpoint directory name if it exists
# This ensures we use the correct n_mem_tokens even if user provided wrong value
if [ -d "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_DIR_NAME=$(basename "${CHECKPOINT_DIR}")
    if [[ "${CHECKPOINT_DIR_NAME}" =~ _mem([0-9]+)_ ]]; then
        INFERRED_N_MEM_TOKENS="${BASH_REMATCH[1]}"
        if [ "${INFERRED_N_MEM_TOKENS}" != "${N_MEM_TOKENS}" ]; then
            echo "Warning: n_mem_tokens mismatch!"
            echo "  Provided: ${N_MEM_TOKENS}"
            echo "  Inferred from directory: ${INFERRED_N_MEM_TOKENS}"
            echo "  Using inferred value: ${INFERRED_N_MEM_TOKENS}"
            N_MEM_TOKENS="${INFERRED_N_MEM_TOKENS}"
        fi
    fi
fi

# Check if model.safetensors or pytorch_model.bin exists
if [ ! -f "${CHECKPOINT_DIR}/model.safetensors" ] && [ ! -f "${CHECKPOINT_DIR}/pytorch_model.bin" ]; then
    echo "Error: Model checkpoint not found in: ${CHECKPOINT_DIR}"
    echo "Expected either model.safetensors or pytorch_model.bin"
    exit 1
fi

# Data paths (aligned with eval_ori_transfer.sh)
TEST_DATA_PATH="${TEST_DATA_PATH:-${DATA_ROOT}/fineweb_test.json}"

# Output directory (save evaluation results in the checkpoint directory)
OUTPUT_DIR="${CHECKPOINT_DIR}/evaluation"
OUTPUT_PATH="${OUTPUT_DIR}/evaluation_results.json"

# Evaluation settings (aligned with transfer eval defaults)
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"  # 0 = greedy decoding
GENERATE_TEXT="${GENERATE_TEXT:-1}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-50}"
DEVICE="${DEVICE:-cuda}"

echo "=========================================="
echo "Evaluating SimpleCompressor"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Compressor model: ${COMPRESSOR_MODEL_NAME} -> ${COMPRESSOR_MODEL}"
echo "Decoder model: ${DECODER_MODEL_NAME} -> ${DECODER_MODEL}"
echo "Memory tokens: ${N_MEM_TOKENS}"
echo "Segment length: ${SEGMENT_LENGTH}"
echo "Test data: ${TEST_DATA_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Generate text: ${GENERATE_TEXT}"
echo "Generate samples: ${GENERATE_SAMPLES}"
echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run evaluation
EVAL_ARGS=(
    --model_path "${CHECKPOINT_DIR}"
    --compressor_model "${COMPRESSOR_MODEL}"
    --decoder_model "${DECODER_MODEL}"
    --n_mem_tokens ${N_MEM_TOKENS}
    --test_data "${TEST_DATA_PATH}"
    --output_path "${OUTPUT_PATH}"
    --max_samples ${MAX_SAMPLES}
    --max_length ${SEGMENT_LENGTH}
    --max_new_tokens ${MAX_NEW_TOKENS}
    --temperature ${TEMPERATURE}
    --generate_samples ${GENERATE_SAMPLES}
    --device "${DEVICE}"
)

if [ "${GENERATE_TEXT}" = "0" ] || [ "${GENERA cTE _TEXT}" = "false" ] || [ "${GENERATE_TEXT}" = "no" ]; then
    EVAL_ARGS+=(--no_generate)
fi

python "${EVAL_SCRIPT}" \
    "${EVAL_ARGS[@]}"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"
echo "=========================================="
