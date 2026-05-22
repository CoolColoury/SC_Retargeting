#!/bin/bash

# Re-run OriTransfer training for transfers where source decoder family is in
# {llama, qwen, mistral}, and target model is in:
# {llama1b, llama3b, llama8b, qwen1.5b, qwen3b, qwen7b, mistral7binstruct}.
#
# Low-disk mode:
# - After EACH pair training finishes, evaluate its checkpoints.
# - Once eval result exists, remove checkpoint model files.
#
# Usage:
#   bash rerun_transfer_all.sh [num_samples] [epochs] [batch_size] [lr]
#
# Common env controls:
#   SKIP_EXISTING=1          # skip pairs that already have both mode results
#   MEM_FILTER=32            # only run mem32
#   MEM_FILTER=8,16,32       # run selected memory token sizes
#   PLAN_ONLY=1              # print estimation and exit (do not run training)
#   MINUTES_PER_RUN=35       # used for wall-time estimation

set -e

SH_PATH="scripts/soft_compress/simple_compressor/transfer_compressor"
RUN_TWO_SCRIPT="${SH_PATH}/run_ori_two.sh"
EVAL_SCRIPT="src/soft_compress/transfer_compressor/evaluate_ori_transfer.py"

NUM_SAMPLES="${1:-100000}"
EPOCHS="${2:-1}"
BATCH_SIZE="${3:-16}"
LEARNING_RATE="${4:-0.0001}"

COMPRESSOR_BASE_DIR="${COMPRESSOR_BASE_DIR:-outputs/simple_compressor}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-outputs/simple_compressor/transfer_compressor/ori_transfer}"

# Force re-run by default.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Optional memory-token filter (empty/all => no filter).
MEM_FILTER="${MEM_FILTER:-all}"

# Plan/estimate only mode
PLAN_ONLY="${PLAN_ONLY:-0}"
MINUTES_PER_RUN="${MINUTES_PER_RUN:-35}"
WARN_HOURS_THRESHOLD="${WARN_HOURS_THRESHOLD:-24}"

# Make sure intermediate checkpoints are removed.
export REMOVE_CHECKPOINTS_AFTER_TRAIN="${REMOVE_CHECKPOINTS_AFTER_TRAIN:-1}"

# Eval configuration (aligned with eval_ori_transfer_all.sh)
TEST_DATA_PATH="${TEST_DATA_PATH:-${PROJECT_ROOT}/${DATA_ROOT}/fineweb_test.json}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-128}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_GPU_ID="${EVAL_GPU_ID:-7}"
EVAL_GENERATE_TEXT="${EVAL_GENERATE_TEXT:-1}"
EVAL_GENERATE_SAMPLES="${EVAL_GENERATE_SAMPLES:-50}"
DELETE_MODEL_FILES_AFTER_EVAL="${DELETE_MODEL_FILES_AFTER_EVAL:-1}"

if [ ! -f "${RUN_TWO_SCRIPT}" ]; then
    echo "Error: Script not found: ${RUN_TWO_SCRIPT}"
    exit 1
fi
if [ ! -d "${COMPRESSOR_BASE_DIR}" ]; then
    echo "Error: Compressor base dir not found: ${COMPRESSOR_BASE_DIR}"
    exit 1
fi
if [ ! -f "${EVAL_SCRIPT}" ]; then
    echo "Error: Eval script not found: ${EVAL_SCRIPT}"
    exit 1
fi

is_true() {
    local v="$1"
    [ "${v}" = "1" ] || [ "${v}" = "true" ] || [ "${v}" = "yes" ]
}

map_model_short_to_path() {
    local model_short="$1"
    case "${model_short}" in
        gpt2) echo "${MODELS_DIR}/gpt2" ;;
        llama1b) echo "${MODELS_DIR}/Llama-3.2-1B-Instruct" ;;
        llama3b) echo "${MODELS_DIR}/Llama-3.2-3B-Instruct" ;;
        llama8b) echo "${MODELS_DIR}/Llama-3-8B-Instruct" ;;
        qwen1.5b) echo "${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct" ;;
        qwen3b) echo "${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct" ;;
        qwen7b) echo "${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct" ;;
        mistral7binstruct|mistral7b|mistral) echo "${MODELS_DIR}/Mistral-7B-Instruct" ;;
        *) echo "" ;;
    esac
}

get_family() {
    local model_short="$1"
    if [[ "${model_short}" =~ ^llama ]]; then
        echo "llama"
    elif [[ "${model_short}" =~ ^qwen ]]; then
        echo "qwen"
    elif [[ "${model_short}" =~ ^mistral ]]; then
        echo "mistral"
    else
        echo "unknown"
    fi
}

tgt_path_to_short() {
    local tgt_path="$1"
    case "$(basename "${tgt_path}")" in
        "Llama-3.2-1B-Instruct") echo "llama1b" ;;
        "Llama-3.2-3B-Instruct") echo "llama3b" ;;
        "Llama-3-8B-Instruct"|"Llama-3.2-8B-Instruct") echo "llama8b" ;;
        "Qwen2.5-1.5B-Instruct") echo "qwen1.5b" ;;
        "Qwen2.5-3B-Instruct") echo "qwen3b" ;;
        "Qwen2.5-7B-Instruct") echo "qwen7b" ;;
        "Mistral-7B-Instruct") echo "mistral7binstruct" ;;
        *) echo "" ;;
    esac
}

mem_allowed() {
    local mem="$1"
    if [ -z "${MEM_FILTER}" ] || [ "${MEM_FILTER}" = "all" ] || [ "${MEM_FILTER}" = "*" ]; then
        return 0
    fi
    local normalized
    normalized=",${MEM_FILTER// /},"
    [[ "${normalized}" == *",${mem},"* ]]
}

dir_has_final_weights() {
    local d="$1"
    [ -f "${d}/pytorch_model.bin" ] || [ -f "${d}/model.safetensors" ]
}

dir_eval_done() {
    local d="$1"
    [ -f "${d}/evaluation/evaluation_results.json" ]
}

dir_is_complete() {
    local d="$1"
    dir_has_final_weights "${d}" || dir_eval_done "${d}"
}

delete_model_files() {
    local d="$1"
    local deleted=0
    for f in "${d}/pytorch_model.bin" "${d}/model.safetensors"; do
        if [ -f "${f}" ]; then
            rm -f "${f}"
            echo "    Removed model file: ${f}"
            deleted=1
        fi
    done
    if [ "${deleted}" -eq 0 ]; then
        echo "    No model file to remove under: ${d}"
    fi
}

evaluate_checkpoint() {
    local checkpoint_dir="$1"
    local src_compressor="$2"
    local src_decoder_model="$3"
    local tgt_model="$4"
    local mem_tokens="$5"
    local train_mode="$6"

    local output_dir="${checkpoint_dir}/evaluation"
    local output_file="${output_dir}/evaluation_results.json"
    mkdir -p "${output_dir}"

    if [ -f "${output_file}" ]; then
        echo "    Eval exists, skipping run: ${output_file}"
        return 0
    fi

    local eval_args=(
        --compressor_checkpoint "${checkpoint_dir}"
        --src_compressor_path "${src_compressor}"
        --src_decoder_model_path "${src_decoder_model}"
        --tgt_model_path "${tgt_model}"
        --n_mem_tokens "${mem_tokens}"
        --train_mode "${train_mode}"
        --test_data_path "${TEST_DATA_PATH}"
        --max_samples "${EVAL_MAX_SAMPLES}"
        --output_path "${output_file}"
        --max_length "${EVAL_MAX_LENGTH}"
        --max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
        --temperature "${EVAL_TEMPERATURE}"
        --batch_size "${EVAL_BATCH_SIZE}"
        --device cuda
    )

    if [ "${EVAL_GENERATE_TEXT}" = "0" ] || [ "${EVAL_GENERATE_TEXT}" = "false" ] || [ "${EVAL_GENERATE_TEXT}" = "no" ]; then
        eval_args+=(--no_generate)
    else
        if [ "${EVAL_GENERATE_SAMPLES}" != "0" ]; then
            eval_args+=(--generate_samples "${EVAL_GENERATE_SAMPLES}")
        fi
    fi

    echo "    Evaluating: $(basename "${checkpoint_dir}")"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_ID}" python "${EVAL_SCRIPT}" "${eval_args[@]}"
}

latest_checkpoint_dir_for_mode() {
    local pair_name="$1"
    local mode="$2"
    local mem="$3"
    local seg="$4"
    local samples="$5"
    local pattern="${OUTPUT_DIR_BASE}/${pair_name}_${mode}_mem${mem}_len${seg}_${samples}samples_"*
    local matches
    shopt -s nullglob
    matches=( ${pattern} )
    shopt -u nullglob
    if [ ${#matches[@]} -eq 0 ]; then
        echo ""
        return 0
    fi
    ls -dt "${matches[@]}" 2>/dev/null | head -1
}

evaluate_and_cleanup_pair() {
    local pair_name="$1"
    local src_compressor="$2"
    local src_decoder_model="$3"
    local tgt_model="$4"
    local mem="$5"
    local seg="$6"
    local samples="$7"

    local modes=("converter_only" "encoder_converter")
    for mode in "${modes[@]}"; do
        local ckpt_dir
        ckpt_dir="$(latest_checkpoint_dir_for_mode "${pair_name}" "${mode}" "${mem}" "${seg}" "${samples}")"
        if [ -z "${ckpt_dir}" ] || [ ! -d "${ckpt_dir}" ]; then
            echo "    [WARN] No checkpoint dir found for mode=${mode}, pair=${pair_name}"
            continue
        fi

        if ! dir_has_final_weights "${ckpt_dir}" && [ ! -f "${ckpt_dir}/evaluation/evaluation_results.json" ]; then
            echo "    [WARN] No model weights and no eval file: ${ckpt_dir}"
            continue
        fi

        evaluate_checkpoint "${ckpt_dir}" "${src_compressor}" "${src_decoder_model}" "${tgt_model}" "${mem}" "${mode}"

        if is_true "${DELETE_MODEL_FILES_AFTER_EVAL}"; then
            if [ -f "${ckpt_dir}/evaluation/evaluation_results.json" ]; then
                delete_model_files "${ckpt_dir}"
            fi
        fi
    done
}

ori_pair_both_modes_done() {
    local enc_short="$1"
    local dec_short="$2"
    local tgt_short="$3"
    local mem="$4"
    local seg="$5"
    local samples="$6"

    local name="${enc_short}_to_${dec_short}_to_${tgt_short}"
    local pat_conv="${OUTPUT_DIR_BASE}/${name}_converter_only_mem${mem}_len${seg}_${samples}samples_"*
    local pat_enc="${OUTPUT_DIR_BASE}/${name}_encoder_converter_mem${mem}_len${seg}_${samples}samples_"*

    local found_conv=0
    local found_enc=0

    shopt -s nullglob
    for d in ${pat_conv}; do
        if [ -d "${d}" ] && dir_is_complete "${d}"; then
            found_conv=1
            break
        fi
    done
    for d in ${pat_enc}; do
        if [ -d "${d}" ] && dir_is_complete "${d}"; then
            found_enc=1
            break
        fi
    done
    shopt -u nullglob

    [ "${found_conv}" -eq 1 ] && [ "${found_enc}" -eq 1 ]
}

tgt_models=(
    ${MODELS_DIR}/Llama-3.2-1B-Instruct
    ${MODELS_DIR}/Llama-3.2-3B-Instruct
    ${MODELS_DIR}/Llama-3-8B-Instruct
    ${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct
    ${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct
    ${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct
    ${MODELS_DIR}/Mistral-7B-Instruct
)

echo "=========================================="
echo "Re-run OriTransfer: llama/qwen/mistral transfer (all pairs)"
echo "=========================================="
echo "Num samples: ${NUM_SAMPLES}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR: ${LEARNING_RATE}"
echo "SKIP_EXISTING: ${SKIP_EXISTING}"
echo "MEM_FILTER: ${MEM_FILTER}"
echo "PLAN_ONLY: ${PLAN_ONLY}"
echo "MINUTES_PER_RUN (estimate): ${MINUTES_PER_RUN}"
echo "Immediate eval after each pair: yes"
echo "EVAL_GENERATE_TEXT: ${EVAL_GENERATE_TEXT}"
echo "EVAL_GENERATE_SAMPLES: ${EVAL_GENERATE_SAMPLES}"
echo "DELETE_MODEL_FILES_AFTER_EVAL: ${DELETE_MODEL_FILES_AFTER_EVAL}"
echo "OUTPUT_DIR_BASE: ${OUTPUT_DIR_BASE}"
echo "=========================================="
echo ""

mapfile -t ALL_COMPRESSORS < <(find "${COMPRESSOR_BASE_DIR}" -maxdepth 1 -type d -name "*_to_*_mem*_len*_ds_4gpu" | sort)
if [ ${#ALL_COMPRESSORS[@]} -eq 0 ]; then
    echo "Error: No compressor directories found under ${COMPRESSOR_BASE_DIR}"
    exit 1
fi

# Pre-calculate planned runs for estimation and phased plan.
EST_RUNS=0
EST_MEM8=0
EST_MEM16=0
EST_MEM32=0
EST_SKIPPED=0

for src_compressor_dir in "${ALL_COMPRESSORS[@]}"; do
    src_name="$(basename "${src_compressor_dir}")"
    if [[ ! "${src_name}" =~ ^([^_]+)_to_([^_]+)_mem([0-9]+)_len([0-9]+)_ds_4gpu$ ]]; then
        continue
    fi
    src_encoder_short="${BASH_REMATCH[1]}"
    src_decoder_short="${BASH_REMATCH[2]}"
    embed_len="${BASH_REMATCH[3]}"
    seg_len="${BASH_REMATCH[4]}"

    if ! mem_allowed "${embed_len}"; then
        continue
    fi

    src_decoder_model_path="$(map_model_short_to_path "${src_decoder_short}")"
    if [ -z "${src_decoder_model_path}" ]; then
        continue
    fi

    src_family="$(get_family "${src_decoder_short}")"
    if [ "${src_family}" != "llama" ] && [ "${src_family}" != "qwen" ] && [ "${src_family}" != "mistral" ]; then
        continue
    fi

    for tgt_model in "${tgt_models[@]}"; do
        tgt_short="$(tgt_path_to_short "${tgt_model}")"
        if [ -z "${tgt_short}" ]; then
            continue
        fi
        if is_true "${SKIP_EXISTING}" && ori_pair_both_modes_done "${src_encoder_short}" "${src_decoder_short}" "${tgt_short}" "${embed_len}" "${seg_len}" "${NUM_SAMPLES}"; then
            EST_SKIPPED=$((EST_SKIPPED + 1))
            continue
        fi
        EST_RUNS=$((EST_RUNS + 1))
        case "${embed_len}" in
            8) EST_MEM8=$((EST_MEM8 + 1)) ;;
            16) EST_MEM16=$((EST_MEM16 + 1)) ;;
            32) EST_MEM32=$((EST_MEM32 + 1)) ;;
        esac
    done
done

EST_TOTAL_MIN=$((EST_RUNS * MINUTES_PER_RUN))
EST_HOURS=$((EST_TOTAL_MIN / 60))
EST_REMAIN_MIN=$((EST_TOTAL_MIN % 60))
EST_MEM32_MIN=$((EST_MEM32 * MINUTES_PER_RUN))
EST_MEM32_H=$((EST_MEM32_MIN / 60))
EST_MEM32_RM=$((EST_MEM32_MIN % 60))

echo "Planned run estimation:"
echo "  - Planned pairs to run: ${EST_RUNS}"
echo "  - Skipped by existing results: ${EST_SKIPPED}"
echo "  - Breakdown by mem: mem8=${EST_MEM8}, mem16=${EST_MEM16}, mem32=${EST_MEM32}"
echo "  - Estimated total time: ~${EST_HOURS}h ${EST_REMAIN_MIN}m  (minutes_per_run=${MINUTES_PER_RUN})"
echo "  - Phase-1 (mem32 only) estimate: ~${EST_MEM32_H}h ${EST_MEM32_RM}m"
echo ""

if [ "${EST_RUNS}" -eq 0 ]; then
    echo "No run needed under current filters."
    exit 0
fi

if [ "${PLAN_ONLY}" = "1" ]; then
    echo "PLAN_ONLY=1 set; exit after estimation."
    exit 0
fi

if [ $((EST_TOTAL_MIN / 60)) -ge "${WARN_HOURS_THRESHOLD}" ]; then
    echo "------------------------------------------"
    echo "Suggested staged plan (long run detected)"
    echo "------------------------------------------"
    echo "Stage 1 (recommended first): mem32 only"
    echo "  MEM_FILTER=32 bash ${SH_PATH}/rerun_transfer_all.sh ${NUM_SAMPLES} ${EPOCHS} ${BATCH_SIZE} ${LEARNING_RATE}"
    echo "Stage 2: mem16"
    echo "  MEM_FILTER=16 bash ${SH_PATH}/rerun_transfer_all.sh ${NUM_SAMPLES} ${EPOCHS} ${BATCH_SIZE} ${LEARNING_RATE}"
    echo "Stage 3: mem8"
    echo "  MEM_FILTER=8 bash ${SH_PATH}/rerun_transfer_all.sh ${NUM_SAMPLES} ${EPOCHS} ${BATCH_SIZE} ${LEARNING_RATE}"
    echo "------------------------------------------"
    echo ""
fi

TOTAL_RUNS=0
SKIPPED_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

for src_compressor_dir in "${ALL_COMPRESSORS[@]}"; do
    src_name="$(basename "${src_compressor_dir}")"
    if [[ ! "${src_name}" =~ ^([^_]+)_to_([^_]+)_mem([0-9]+)_len([0-9]+)_ds_4gpu$ ]]; then
        continue
    fi

    src_encoder_short="${BASH_REMATCH[1]}"
    src_decoder_short="${BASH_REMATCH[2]}"
    embed_len="${BASH_REMATCH[3]}"
    seg_len="${BASH_REMATCH[4]}"

    if ! mem_allowed "${embed_len}"; then
        continue
    fi

    src_decoder_model_path="$(map_model_short_to_path "${src_decoder_short}")"
    if [ -z "${src_decoder_model_path}" ]; then
        continue
    fi

    src_family="$(get_family "${src_decoder_short}")"
    if [ "${src_family}" != "llama" ] && [ "${src_family}" != "qwen" ] && [ "${src_family}" != "mistral" ]; then
        continue
    fi

    for tgt_model in "${tgt_models[@]}"; do
        tgt_short="$(tgt_path_to_short "${tgt_model}")"
        if [ -z "${tgt_short}" ]; then
            continue
        fi

        if is_true "${SKIP_EXISTING}" && ori_pair_both_modes_done "${src_encoder_short}" "${src_decoder_short}" "${tgt_short}" "${embed_len}" "${seg_len}" "${NUM_SAMPLES}"; then
            SKIPPED_RUNS=$((SKIPPED_RUNS + 1))
            continue
        fi

        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        pair_name="${src_encoder_short}_to_${src_decoder_short}_to_${tgt_short}"
        echo ""
        echo "------------------------------------------"
        echo "[$TOTAL_RUNS] Running: ${src_name} -> $(basename "${tgt_model}")"
        echo "  src_encoder_short: ${src_encoder_short}"
        echo "  src_decoder_short: ${src_decoder_short}"
        echo "  tgt_short:         ${tgt_short}"
        echo "  EMBED_LEN:         ${embed_len}"
        echo "  SEGMENT_LENGTH:    ${seg_len}"
        echo "------------------------------------------"

        set +e
        SRC_COMPRESSOR_PATH="${src_compressor_dir}" \
        SRC_DECODER_MODEL_PATH="${src_decoder_model_path}" \
        TGT_MODEL_PATH="${tgt_model}" \
        EMBED_LEN="${embed_len}" \
        SEGMENT_LENGTH="${seg_len}" \
        OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE}" \
        bash "${RUN_TWO_SCRIPT}" "${NUM_SAMPLES}" "${EPOCHS}" "${BATCH_SIZE}" "${LEARNING_RATE}"
        exit_code=$?
        set -e

        if [ ${exit_code} -eq 0 ]; then
            SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
            evaluate_and_cleanup_pair "${pair_name}" "${src_compressor_dir}" "${src_decoder_model_path}" "${tgt_model}" "${embed_len}" "${seg_len}" "${NUM_SAMPLES}"
        else
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
    done
done

echo ""
echo "=========================================="
echo "Rerun transfer summary"
echo "=========================================="
echo "Total runs: ${TOTAL_RUNS}"
echo "Skipped: ${SKIPPED_RUNS}"
echo "Successful: ${SUCCESS_RUNS}"
echo "Failed: ${FAILED_RUNS}"
echo "=========================================="

if [ ${FAILED_RUNS} -gt 0 ]; then
    exit 1
fi

exit 0

