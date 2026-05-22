#!/bin/bash

# Script to run both converter_only and encoder_converter training modes
# This script uses the configurable script with different settings for each mode

set -e
# Pipelines: without this, $? after `cmd | tee log` is tee's exit code (usually 0).
set -o pipefail

PATH_TO_SCRIPT="scripts/soft_compress/simple_compressor/transfer_compressor"

# ============================================================================
# Configuration - Modify these variables as needed
# ============================================================================

# Model paths (can be overridden by environment variables)
SRC_COMPRESSOR_PATH="${SRC_COMPRESSOR_PATH:-outputs/simple_compressor/gpt2_to_llama1b_mem32_len128_ds_4gpu}"
SRC_DECODER_MODEL_PATH="${SRC_DECODER_MODEL_PATH:-${MODELS_DIR}/gpt2}"
TGT_MODEL_PATH="${TGT_MODEL_PATH:-${MODELS_DIR}/Llama-3-8B-Instruct}"

# Model config (can be overridden by environment variables)
EMBED_LEN="${EMBED_LEN:-32}"
SEGMENT_LENGTH="${SEGMENT_LENGTH:-128}"

# Training parameters (can be overridden by command line arguments)
NUM_SAMPLES="${1:-1000000}"
EPOCHS="${2:-1}"
BATCH_SIZE="${3:-16}"
LEARNING_RATE="${4:-0.0001}"

# GPU configuration for each run
# First run: GPUs 0,1,2,3
# Second run: GPUs 4,5,6,7
GPU_LIST_1="${GPU_LIST_1:-0,1,2,3}"
GPU_LIST_2="${GPU_LIST_2:-4,5,6,7}"
MASTER_PORT_1="${MASTER_PORT_1:-29500}"
MASTER_PORT_2="${MASTER_PORT_2:-29501}"

# Execution mode:
# - auto (default): if only 4 visible GPUs, run sequentially; otherwise parallel.
# - force sequential: RUN_MODES_SEQUENTIAL=1
# - force parallel: RUN_MODES_SEQUENTIAL=0
RUN_MODES_SEQUENTIAL="${RUN_MODES_SEQUENTIAL:-auto}"

is_port_free() {
    local port="$1"
    python - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    print("0")
else:
    print("1")
finally:
    s.close()
PY
}

find_free_port_from() {
    local start_port="$1"
    local p="${start_port}"
    local max_tries=200
    local i=0
    while [ "${i}" -lt "${max_tries}" ]; do
        if [ "$(is_port_free "${p}")" = "1" ]; then
            echo "${p}"
            return 0
        fi
        p=$((p + 1))
        i=$((i + 1))
    done
    echo ""
}

# ============================================================================
# Run Training
# ============================================================================

echo "=========================================="
echo "Running Two Training Modes"
echo "=========================================="
echo "Configuration:"
echo "  - Source compressor: ${SRC_COMPRESSOR_PATH}"
echo "  - Source decoder: ${SRC_DECODER_MODEL_PATH}"
echo "  - Target model: ${TGT_MODEL_PATH}"
echo "  - Embed len: ${EMBED_LEN}"
echo "  - Segment length: ${SEGMENT_LENGTH}"
echo "  - Num samples: ${NUM_SAMPLES}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo "=========================================="

# =============================================================================
# Step 1 & 2: Run both modes (parallel or sequential)
# =============================================================================

echo ""
echo "=========================================="
echo "Running both training modes"
echo "=========================================="
echo "Mode 1 (converter_only): candidate GPUs ${GPU_LIST_1}, Port ${MASTER_PORT_1}"
echo "Mode 2 (encoder_converter): candidate GPUs ${GPU_LIST_2}, Port ${MASTER_PORT_2}"

AUTO_VISIBLE_GPU_COUNT=0
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    AUTO_VISIBLE_GPU_COUNT=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi

SEQUENTIAL_MODE=0
if [ "${RUN_MODES_SEQUENTIAL}" = "1" ] || [ "${RUN_MODES_SEQUENTIAL}" = "true" ] || [ "${RUN_MODES_SEQUENTIAL}" = "yes" ]; then
    SEQUENTIAL_MODE=1
elif [ "${RUN_MODES_SEQUENTIAL}" = "0" ] || [ "${RUN_MODES_SEQUENTIAL}" = "false" ] || [ "${RUN_MODES_SEQUENTIAL}" = "no" ]; then
    SEQUENTIAL_MODE=0
else
    # auto mode
    if [ "${AUTO_VISIBLE_GPU_COUNT}" -gt 0 ] && [ "${AUTO_VISIBLE_GPU_COUNT}" -le 4 ]; then
        SEQUENTIAL_MODE=1
    fi
fi

if [ "${SEQUENTIAL_MODE}" -eq 1 ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPU_LIST_1="${CUDA_VISIBLE_DEVICES}"
        GPU_LIST_2="${CUDA_VISIBLE_DEVICES}"
    fi
    echo "Execution mode: SEQUENTIAL"
    echo "  Reason: RUN_MODES_SEQUENTIAL=${RUN_MODES_SEQUENTIAL}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
else
    echo "Execution mode: PARALLEL"
    echo "  Reason: RUN_MODES_SEQUENTIAL=${RUN_MODES_SEQUENTIAL}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
echo ""

# Resolve MASTER_PORT conflicts (common in shared servers).
RESOLVED_PORT_1="${MASTER_PORT_1}"
RESOLVED_PORT_2="${MASTER_PORT_2}"

if [ "$(is_port_free "${RESOLVED_PORT_1}")" != "1" ]; then
    alt="$(find_free_port_from "$((MASTER_PORT_1 + 1))")"
    if [ -n "${alt}" ]; then
        echo "[port] MASTER_PORT_1=${MASTER_PORT_1} in use, switch to ${alt}"
        RESOLVED_PORT_1="${alt}"
    else
        echo "Error: MASTER_PORT_1=${MASTER_PORT_1} is occupied and no free port found nearby"
        exit 1
    fi
fi

if [ "${SEQUENTIAL_MODE}" -eq 1 ]; then
    # sequential only needs one port in practice; keep both consistent
    RESOLVED_PORT_2="${RESOLVED_PORT_1}"
else
    if [ "${RESOLVED_PORT_2}" = "${RESOLVED_PORT_1}" ] || [ "$(is_port_free "${RESOLVED_PORT_2}")" != "1" ]; then
        alt2="$(find_free_port_from "$((MASTER_PORT_2 + 1))")"
        if [ -n "${alt2}" ] && [ "${alt2}" != "${RESOLVED_PORT_1}" ]; then
            echo "[port] MASTER_PORT_2=${MASTER_PORT_2} unavailable/conflict, switch to ${alt2}"
            RESOLVED_PORT_2="${alt2}"
        else
            alt2="$(find_free_port_from "$((RESOLVED_PORT_1 + 1))")"
            if [ -n "${alt2}" ] && [ "${alt2}" != "${RESOLVED_PORT_1}" ]; then
                echo "[port] MASTER_PORT_2 fallback -> ${alt2}"
                RESOLVED_PORT_2="${alt2}"
            else
                echo "Error: cannot find a distinct free MASTER_PORT_2"
                exit 1
            fi
        fi
    fi
fi
echo "Resolved ports: mode1=${RESOLVED_PORT_1}, mode2=${RESOLVED_PORT_2}"
echo ""

# Create temporary files to capture exit status
TMP_DIR=$(mktemp -d)
CONVERTER_ONLY_LOG="${TMP_DIR}/converter_only.log"
ENCODER_CONVERTER_LOG="${TMP_DIR}/encoder_converter.log"
CONVERTER_ONLY_PID_FILE="${TMP_DIR}/converter_only.pid"
ENCODER_CONVERTER_PID_FILE="${TMP_DIR}/encoder_converter.pid"

# Function to run converter_only mode
run_converter_only() {
    export TRAIN_MODE="converter_only"
    export SRC_COMPRESSOR_PATH="${SRC_COMPRESSOR_PATH}"
    export SRC_DECODER_MODEL_PATH="${SRC_DECODER_MODEL_PATH}"
    export TGT_MODEL_PATH="${TGT_MODEL_PATH}"
    export EMBED_LEN="${EMBED_LEN}"
    export SEGMENT_LENGTH="${SEGMENT_LENGTH}"
    export GPU_LIST="${GPU_LIST_1}"
    export MASTER_PORT="${RESOLVED_PORT_1}"
    export SAVE_TOTAL_LIMIT="1"
    
    echo "[converter_only] Starting training on GPUs ${GPU_LIST_1}, port ${MASTER_PORT}..."
    bash ${PATH_TO_SCRIPT}/run_ori_transfer_configurable.sh \
        "${NUM_SAMPLES}" \
        "${EPOCHS}" \
        "${BATCH_SIZE}" \
        "${LEARNING_RATE}" 2>&1 | tee "${CONVERTER_ONLY_LOG}"
    # With pipefail, $? is the first failing command in the pipe (the training script).
    EXIT_CODE=$?
    echo $EXIT_CODE > "${TMP_DIR}/converter_only.exit"
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[converter_only] Training failed with exit code $EXIT_CODE"
    else
        echo "[converter_only] Training completed successfully!"
    fi
    return $EXIT_CODE
}

# Function to run encoder_converter mode
run_encoder_converter() {
    export TRAIN_MODE="encoder_converter"
    export SRC_COMPRESSOR_PATH="${SRC_COMPRESSOR_PATH}"
    export SRC_DECODER_MODEL_PATH="${SRC_DECODER_MODEL_PATH}"
    export TGT_MODEL_PATH="${TGT_MODEL_PATH}"
    export EMBED_LEN="${EMBED_LEN}"
    export SEGMENT_LENGTH="${SEGMENT_LENGTH}"
    export GPU_LIST="${GPU_LIST_2}"
    export MASTER_PORT="${RESOLVED_PORT_2}"
    export SAVE_TOTAL_LIMIT="1"
    
    echo "[encoder_converter] Starting training on GPUs ${GPU_LIST_2}, port ${MASTER_PORT}..."
    bash ${PATH_TO_SCRIPT}/run_ori_transfer_configurable.sh \
        "${NUM_SAMPLES}" \
        "${EPOCHS}" \
        "${BATCH_SIZE}" \
        "${LEARNING_RATE}" 2>&1 | tee "${ENCODER_CONVERTER_LOG}"
    EXIT_CODE=$?
    echo $EXIT_CODE > "${TMP_DIR}/encoder_converter.exit"
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[encoder_converter] Training failed with exit code $EXIT_CODE"
    else
        echo "[encoder_converter] Training completed successfully!"
    fi
    return $EXIT_CODE
}

if [ "${SEQUENTIAL_MODE}" -eq 1 ]; then
    echo "Running sequentially: converter_only -> encoder_converter"
    set +e
    run_converter_only
    CONVERTER_ONLY_EXIT=$?
    if [ "${CONVERTER_ONLY_EXIT}" -eq 0 ]; then
        run_encoder_converter
        ENCODER_CONVERTER_EXIT=$?
    else
        ENCODER_CONVERTER_EXIT=1
    fi
    set -e
else
    # Run both modes in parallel
    run_converter_only &
    CONVERTER_ONLY_PID=$!
    echo $CONVERTER_ONLY_PID > "${CONVERTER_ONLY_PID_FILE}"

    run_encoder_converter &
    ENCODER_CONVERTER_PID=$!
    echo $ENCODER_CONVERTER_PID > "${ENCODER_CONVERTER_PID_FILE}"

    echo "Started both training processes:"
    echo "  - converter_only: PID ${CONVERTER_ONLY_PID}"
    echo "  - encoder_converter: PID ${ENCODER_CONVERTER_PID}"
    echo "Waiting for both to complete..."
    echo ""

    # Wait for both processes to complete
    # Temporarily disable set -e to allow wait to capture exit codes
    set +e
    wait $CONVERTER_ONLY_PID
    CONVERTER_ONLY_EXIT=$?

    wait $ENCODER_CONVERTER_PID
    ENCODER_CONVERTER_EXIT=$?
    set -e
fi

# Check exit status from files (more reliable)
echo ""
echo "=========================================="
echo "Training Results"
echo "=========================================="
if [ -f "${TMP_DIR}/converter_only.exit" ]; then
    CONVERTER_ONLY_EXIT=$(cat "${TMP_DIR}/converter_only.exit")
fi
if [ -f "${TMP_DIR}/encoder_converter.exit" ]; then
    ENCODER_CONVERTER_EXIT=$(cat "${TMP_DIR}/encoder_converter.exit")
fi

if [ $CONVERTER_ONLY_EXIT -ne 0 ]; then
    echo "✗ converter_only training failed (exit code: ${CONVERTER_ONLY_EXIT})"
    echo "Log saved to: ${CONVERTER_ONLY_LOG}"
else
    echo "✓ converter_only training completed successfully"
fi

if [ $ENCODER_CONVERTER_EXIT -ne 0 ]; then
    echo "✗ encoder_converter training failed (exit code: ${ENCODER_CONVERTER_EXIT})"
    echo "Log saved to: ${ENCODER_CONVERTER_LOG}"
else
    echo "✓ encoder_converter training completed successfully"
fi

# Clean up temporary files
rm -rf "${TMP_DIR}"

# Exit with error if either failed
if [ $CONVERTER_ONLY_EXIT -ne 0 ] || [ $ENCODER_CONVERTER_EXIT -ne 0 ]; then
    echo ""
    echo "Error: At least one training mode failed"
    exit 1
fi

echo ""
echo "✓ Both training modes completed successfully!"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✓ Both training modes completed successfully!"
echo ""
echo "Training configuration used:"
echo "  - Source compressor: ${SRC_COMPRESSOR_PATH}"
echo "  - Source decoder: ${SRC_DECODER_MODEL_PATH}"
echo "  - Target model: ${TGT_MODEL_PATH}"
echo "  - Embed len: ${EMBED_LEN}"
echo "  - Segment length: ${SEGMENT_LENGTH}"
echo "  - Num samples: ${NUM_SAMPLES}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Learning rate: ${LEARNING_RATE}"
echo ""
echo "=========================================="
echo "✓ All tasks completed!"
echo "=========================================="
