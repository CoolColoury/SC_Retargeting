#!/bin/bash

# Batch evaluation script for transfer between all compressors
# Defaults to LS method, can be overridden via env vars
# Only keeps final evaluation results, removes intermediate generation files

# Base directory containing all compressor models
COMPRESSOR_BASE_DIR="${COMPRESSOR_BASE_DIR:-outputs/simple_compressor}"

# Test data configuration
TEST_DATA_PATH="${TEST_DATA_PATH:-${PROJECT_ROOT}/${DATA_ROOT}/fineweb_test.json}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

# Transfer settings
MAX_LENGTH="${MAX_LENGTH:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"  # Batch size for compression
TRANSFER_BATCH_SIZE="${TRANSFER_BATCH_SIZE:-256}"  # Batch size for transfer conversion

# Evaluation settings
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"  # Batch size for evaluation
GENERATE_TEXT="${GENERATE_TEXT:-1}"
GENERATE_SAMPLES="${GENERATE_SAMPLES:-50}"

# Common vocab (required for LS converter)
COMMON_VOCAB="${COMMON_VOCAB:-${PROJECT_ROOT:-.}/scripts/soft_compress/simple_compressor/data/vocab_100k.txt}"

# Converter settings (overridable)
CONVERTER_TYPE="${CONVERTER_TYPE:-ls}"
CONVERTER_KWARGS="${CONVERTER_KWARGS:-{}}"
TRANSFER_TITLE="${TRANSFER_TITLE:-Transfer}"

# GPU configuration (align default with rerun_transfer_all.sh)
GPU_ID="${GPU_ID:-0}"

# Scripts
TRANSFER_SCRIPT="src/soft_compress/transfer_compressor/transfer.py"
EVAL_SCRIPT="src/soft_compress/transfer_compressor/evaluate_transfer.py"

# Output base directory
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-outputs/simple_compressor/transfer_compressor/ls_transfer}"

# OriTransfer results directory (for filtering)
ORI_TRANSFER_DIR="${ORI_TRANSFER_DIR:-outputs/simple_compressor/transfer_compressor/ori_transfer}"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "${TRANSFER_TITLE} Batch Evaluation"
echo "========================================"
echo "Compressor base dir: ${COMPRESSOR_BASE_DIR}"
echo "Test data: ${TEST_DATA_PATH}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Output base dir: ${OUTPUT_BASE_DIR}"
echo "Converter type: ${CONVERTER_TYPE}"
echo "Compression batch size: ${BATCH_SIZE}"
echo "Transfer batch size: ${TRANSFER_BATCH_SIZE}"
echo "Eval batch size: ${EVAL_BATCH_SIZE}"
echo "========================================"
echo ""

# Function to parse compressor directory name and extract info
# Format: {source}_to_{target}_mem{n}_len{len}_ds_4gpu
normalize_model_short() {
    local short="$1"
    case "$short" in
        mistral7binstruct|mistral7b|mistral) echo "mistral7b" ;;
        *) echo "$short" ;;
    esac
}

candidate_model_shorts() {
    local short="$1"
    case "$short" in
        mistral7binstruct|mistral7b|mistral) echo "mistral7b mistral7binstruct" ;;
        *) echo "$short" ;;
    esac
}

parse_compressor_dir() {
    local dir_name=$1
    
    # Extract source and target models
    if [[ $dir_name =~ ^([^_]+)_to_([^_]+)_mem([0-9]+)_len([0-9]+) ]]; then
        src_model_name="$(normalize_model_short "${BASH_REMATCH[1]}")"
        tgt_model_name="$(normalize_model_short "${BASH_REMATCH[2]}")"
        mem_tokens="${BASH_REMATCH[3]}"
        segment_len="${BASH_REMATCH[4]}"
    else
        echo "Error: Could not parse compressor directory name: $dir_name" >&2
        return 1
    fi
    
    # Map model names to actual model paths
    case "$src_model_name" in
        gpt2)
            src_model_path="${MODELS_DIR}/gpt2"
            ;;
        llama1b)
            src_model_path="${MODELS_DIR}/Llama-3.2-1B-Instruct"
            ;;
        llama3b)
            src_model_path="${MODELS_DIR}/Llama-3.2-3B-Instruct"
            ;;
        llama8b)
            src_model_path="${MODELS_DIR}/Llama-3-8B-Instruct"
            ;;
        qwen1.5b)
            src_model_path="${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct"
            ;;
        qwen3b)
            src_model_path="${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct"
            ;;
        qwen7b)
            src_model_path="${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct"
            ;;
        mistral7b)
            src_model_path="${MODELS_DIR}/Mistral-7B-Instruct"
            ;;
        *)
            echo "Warning: Unknown source model '$src_model_name', using as-is" >&2
            src_model_path="$src_model_name"
            ;;
    esac
    
    case "$tgt_model_name" in
        llama1b)
            tgt_model_path="${MODELS_DIR}/Llama-3.2-1B-Instruct"
            ;;
        llama3b)
            tgt_model_path="${MODELS_DIR}/Llama-3.2-3B-Instruct"
            ;;
        llama8b)
            tgt_model_path="${MODELS_DIR}/Llama-3-8B-Instruct"
            ;;
        qwen1.5b)
            tgt_model_path="${MODELS_DIR}/Qwen/Qwen2.5-1.5B-Instruct"
            ;;
        qwen3b)
            tgt_model_path="${MODELS_DIR}/Qwen/Qwen2.5-3B-Instruct"
            ;;
        qwen7b)
            tgt_model_path="${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct"
            ;;
        mistral7b)
            tgt_model_path="${MODELS_DIR}/Mistral-7B-Instruct"
            ;;
        *)
            echo "Warning: Unknown target model '$tgt_model_name', using as-is" >&2
            tgt_model_path="$tgt_model_name"
            ;;
    esac
    
    echo "$src_model_name|$tgt_model_name|$mem_tokens|$segment_len|$src_model_path|$tgt_model_path"
}

# Find all compressor directories
if [ ! -d "$COMPRESSOR_BASE_DIR" ]; then
    echo -e "${RED}Error: Directory $COMPRESSOR_BASE_DIR not found${NC}"
    exit 1
fi

# Get all compressor directories (exclude subdirectories like transfer_compressor, metrics_pngs, etc.)
ALL_COMPRESSORS=($(find "$COMPRESSOR_BASE_DIR" -maxdepth 1 -type d -name "*_mem*_len*_ds_4gpu" | sort))

if [ ${#ALL_COMPRESSORS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No compressor directories found in $COMPRESSOR_BASE_DIR${NC}"
    exit 1
fi

# Parse ori_transfer directory to get valid transfer pairs
# Format: {src_compressor}_to_{tgt_decoder}_{train_mode}_mem{n}_len{len}_*
declare -A VALID_TRANSFER_PAIRS

if [ -d "$ORI_TRANSFER_DIR" ]; then
    echo -e "${BLUE}Scanning ori_transfer directory for valid transfer pairs...${NC}"
    total_ori_dirs=0
    parsed_dirs=0
    
    for ori_dir in "$ORI_TRANSFER_DIR"/*; do
        if [ ! -d "$ori_dir" ]; then
            continue
        fi
        
        total_ori_dirs=$((total_ori_dirs + 1))
        ori_name=$(basename "$ori_dir")
        
        # Parse: {src_compressor}_to_{tgt_decoder}_{train_mode}_mem{n}_len{len}_*
        # Example: gpt2_to_llama1b_to_llama8b_converter_only_mem8_len128_*
        if [[ $ori_name =~ ^([^_]+_to_[^_]+)_to_([^_]+)_(converter_only|encoder_converter)_mem([0-9]+)_len([0-9]+) ]]; then
            parsed_dirs=$((parsed_dirs + 1))
            src_compressor_name="${BASH_REMATCH[1]}"
            tgt_decoder_name="$(normalize_model_short "${BASH_REMATCH[2]}")"
            train_mode="${BASH_REMATCH[3]}"
            mem_tokens="${BASH_REMATCH[4]}"
            
            # Find matching source compressor: {src_compressor}_mem{n}_len{l}_ds_4gpu
            src_compressor_basename=""
            for comp in "${ALL_COMPRESSORS[@]}"; do
                comp_name=$(basename "$comp")
                if [[ "$comp_name" =~ ^${src_compressor_name}_mem${mem_tokens}_len128_ds_4gpu$ ]]; then
                    src_compressor_basename="$comp_name"
                    break
                fi
            done
            
            # Find matching target compressor: {src_encoder}_to_{tgt_decoder}_mem{n}_len{l}_ds_4gpu
            # Extract src_encoder from src_compressor_name (e.g., "gpt2" from "gpt2_to_llama1b")
            src_encoder=$(echo "$src_compressor_name" | cut -d'_' -f1)
            tgt_compressor_basename=""
            tried_tgt_names=()
            for tgt_short_candidate in $(candidate_model_shorts "${tgt_decoder_name}"); do
                tgt_compressor_name="${src_encoder}_to_${tgt_short_candidate}_mem${mem_tokens}_len128_ds_4gpu"
                tried_tgt_names+=("${tgt_compressor_name}")
                for comp in "${ALL_COMPRESSORS[@]}"; do
                    comp_name=$(basename "$comp")
                    if [ "$comp_name" == "$tgt_compressor_name" ]; then
                        tgt_compressor_basename="$comp_name"
                        break 2
                    fi
                done
            done
            
            if [ -n "$src_compressor_basename" ] && [ -n "$tgt_compressor_basename" ]; then
                pair_key="${src_compressor_basename}|${tgt_compressor_basename}"
                VALID_TRANSFER_PAIRS["$pair_key"]=1
                echo "  Found: $src_compressor_basename -> $tgt_compressor_basename (from $train_mode, mem${mem_tokens})"
            else
                if [ -z "$src_compressor_basename" ]; then
                    echo -e "${YELLOW}    Warning: Source compressor not found: ${src_compressor_name}_mem${mem_tokens}_len128_ds_4gpu${NC}"
                fi
                if [ -z "$tgt_compressor_basename" ]; then
                    echo -e "${YELLOW}    Warning: Target compressor not found: ${tried_tgt_names[*]}${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}    Warning: Could not parse directory name: $ori_name${NC}"
        fi
    done
    
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo "  Total ori_transfer directories: $total_ori_dirs"
    echo "  Successfully parsed: $parsed_dirs"
    echo "  Unique transfer pairs found: ${#VALID_TRANSFER_PAIRS[@]}"
    
    if [ ${#VALID_TRANSFER_PAIRS[@]} -gt 0 ]; then
        echo ""
        echo -e "${BLUE}Unique transfer pairs:${NC}"
        for pair_key in $(printf '%s\n' "${!VALID_TRANSFER_PAIRS[@]}" | sort); do
            IFS='|' read -r src_name tgt_name <<< "$pair_key"
            echo "  - $src_name -> $tgt_name"
        done
    fi
    echo ""
fi

# If no valid pairs found from ori_transfer, use all compressors (backward compatibility)
if [ ${#VALID_TRANSFER_PAIRS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No valid transfer pairs found in ori_transfer, using all compressors${NC}"
    COMPRESSORS=("${ALL_COMPRESSORS[@]}")
    TOTAL_PAIRS=$((${#COMPRESSORS[@]} * ${#COMPRESSORS[@]}))
else
    echo -e "${BLUE}Found ${#VALID_TRANSFER_PAIRS[@]} valid transfer pair(s) from ori_transfer${NC}"
    echo ""
    # Store all compressors for lookup
    COMPRESSORS=("${ALL_COMPRESSORS[@]}")
    TOTAL_PAIRS=${#VALID_TRANSFER_PAIRS[@]}
fi

echo -e "${BLUE}Total compressors available: ${#COMPRESSORS[@]}${NC}"
echo -e "${BLUE}Total transfer pairs to evaluate: ${TOTAL_PAIRS}${NC}"
echo ""

# Summary tracking
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
FAILED_PAIRS=()
SKIPPED_PAIRS=()

# Process each source compressor
pair_idx=0
for src_compressor_dir in "${COMPRESSORS[@]}"; do
    src_compressor_name=$(basename "$src_compressor_dir")
    
    # Parse source compressor info
    src_config=$(parse_compressor_dir "$src_compressor_name")
    if [ $? -ne 0 ]; then
        echo -e "${RED}⚠ Error parsing source compressor: $src_compressor_name${NC}"
        continue
    fi
    
    IFS='|' read -r src_model_name src_tgt_model_name src_mem_tokens src_segment_len src_model_path src_tgt_model_path <<< "$src_config"
    
    echo ""
    echo "========================================"
    echo -e "${BLUE}Source Compressor: $src_compressor_name${NC}"
    echo "========================================"
    echo "  Compressor (encoder): $src_model_path"
    echo "  Decoder: $src_tgt_model_path"
    echo "  Memory tokens: $src_mem_tokens"
    echo "  Segment length: $src_segment_len"
    echo ""
    
    # For each target compressor
    for tgt_compressor_dir in "${COMPRESSORS[@]}"; do
        tgt_compressor_name=$(basename "$tgt_compressor_dir")
        
        # If using filtered pairs, check if this pair is valid
        if [ ${#VALID_TRANSFER_PAIRS[@]} -gt 0 ]; then
            pair_key="${src_compressor_name}|${tgt_compressor_name}"
            if [ -z "${VALID_TRANSFER_PAIRS[$pair_key]}" ]; then
                continue  # Skip if not in valid pairs
            fi
        fi
        
        pair_idx=$((pair_idx + 1))
        
        # Parse target compressor info
        tgt_config=$(parse_compressor_dir "$tgt_compressor_name")
        if [ $? -ne 0 ]; then
            echo -e "${RED}⚠ Error parsing target compressor: $tgt_compressor_name${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_PAIRS+=("$src_compressor_name -> $tgt_compressor_name (parse error)")
            continue
        fi
        
        IFS='|' read -r tgt_model_name tgt_tgt_model_name tgt_mem_tokens tgt_segment_len tgt_model_path tgt_tgt_model_path <<< "$tgt_config"
        
        # Skip if source and target are the same
        if [ "$src_compressor_name" == "$tgt_compressor_name" ]; then
            echo -e "${YELLOW}[$pair_idx/$TOTAL_PAIRS] Skipping: $src_compressor_name -> $tgt_compressor_name (same compressor)${NC}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi
        
        # Skip if memory tokens don't match (for now, we only transfer same mem_tokens)
        if [ "$src_mem_tokens" != "$tgt_mem_tokens" ]; then
            echo -e "${YELLOW}[$pair_idx/$TOTAL_PAIRS] Skipping: $src_compressor_name -> $tgt_compressor_name (different mem_tokens: $src_mem_tokens vs $tgt_mem_tokens)${NC}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi
        
        echo ""
        echo "----------------------------------------"
        echo -e "${BLUE}[$pair_idx/$TOTAL_PAIRS] Transfer: $src_compressor_name -> $tgt_compressor_name${NC}"
        echo "----------------------------------------"
        
        # Create output directory for this transfer pair
        transfer_output_dir="${OUTPUT_BASE_DIR}/${src_compressor_name}_to_${tgt_compressor_name}"
        eval_output_dir="${transfer_output_dir}/evaluation"
        
        # Check if evaluation results already exist
        # evaluate_transfer.py saves files as eval_transfer_*.json
        existing_eval_file=$(find "${eval_output_dir}" -name "eval_transfer_*.json" -type f 2>/dev/null | head -1)
        if [ -n "$existing_eval_file" ] && [ -f "$existing_eval_file" ]; then
            echo -e "${YELLOW}⏭ Evaluation results already exist${NC}"
            echo -e "${YELLOW}  Results: ${existing_eval_file}${NC}"
            echo -e "${YELLOW}  Skipping transfer and evaluation${NC}"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            SKIPPED_PAIRS+=("$src_compressor_name -> $tgt_compressor_name")
            continue
        fi
        
        # Step 1: Transfer compression
        echo "Step 1: Transferring compression (${CONVERTER_TYPE} method)..."
        transfer_results_file="${transfer_output_dir}/compression_results.json"
        
        mkdir -p "$transfer_output_dir"
        
        # For converter methods, src_model is source decoder and tgt_model is target decoder
        # The compressed vectors are in the source decoder's embedding space
        CUDA_VISIBLE_DEVICES=${GPU_ID} python "${TRANSFER_SCRIPT}" \
            --dataset "${TEST_DATA_PATH}" \
            --src_compressor "${src_compressor_dir}" \
            --src_model "${src_tgt_model_path}" \
            --tgt_model "${tgt_tgt_model_path}" \
            --output_dir "${transfer_output_dir}" \
            --n_mem_tokens ${src_mem_tokens} \
            --max_length ${MAX_LENGTH} \
            --num_samples ${NUM_SAMPLES} \
            --converter_type "${CONVERTER_TYPE}" \
            --common_vocab "${COMMON_VOCAB}" \
            --converter_kwargs "${CONVERTER_KWARGS}" \
            --device cuda \
            --batch_size ${BATCH_SIZE} \
            --transfer_batch_size ${TRANSFER_BATCH_SIZE} \
            > "${transfer_output_dir}/transfer.log" 2>&1
        
        transfer_exit_code=$?
        if [ $transfer_exit_code -ne 0 ]; then
            echo -e "${RED}✗ Transfer failed (exit code: $transfer_exit_code)${NC}"
            echo -e "${RED}  Check log: ${transfer_output_dir}/transfer.log${NC}"
            echo -e "${YELLOW}  Last 20 lines of log:${NC}"
            tail -20 "${transfer_output_dir}/transfer.log" | sed 's/^/    /'
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_PAIRS+=("$src_compressor_name -> $tgt_compressor_name (transfer failed)")
            continue
        fi
        
        # Find the actual results file (transfer.py may create files with timestamps)
        # Look for the most recent compression_results file
        actual_results_file=$(find "${transfer_output_dir}" -name "compression_results*.json" -type f | sort -r | head -1)
        if [ -z "$actual_results_file" ] || [ ! -f "$actual_results_file" ]; then
            echo -e "${RED}✗ Transfer results file not found${NC}"
            echo -e "${RED}  Searched in: ${transfer_output_dir}${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_PAIRS+=("$src_compressor_name -> $tgt_compressor_name (results file not found)")
            continue
        fi
        
        echo -e "${GREEN}✓ Transfer completed${NC}"
        echo "  Results: $actual_results_file"
        
        # Step 2: Evaluate transfer results
        echo ""
        echo "Step 2: Evaluating transfer results..."
        
        mkdir -p "$eval_output_dir"
        
        CUDA_VISIBLE_DEVICES=${GPU_ID} python "${EVAL_SCRIPT}" \
            --results_path "${actual_results_file}" \
            --dataset_path "${TEST_DATA_PATH}" \
            --output_dir "${eval_output_dir}" \
            --device cuda \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --generate_text "${GENERATE_TEXT}" \
            --generate_samples "${GENERATE_SAMPLES}" \
            > "${eval_output_dir}/evaluation.log" 2>&1
        
        eval_exit_code=$?
        
        # Find the actual evaluation results file (evaluate_transfer.py saves as eval_transfer_*.json)
        actual_eval_file=$(find "${eval_output_dir}" -name "eval_transfer_*.json" -type f 2>/dev/null | head -1)
        
        if [ $eval_exit_code -ne 0 ] || [ -z "$actual_eval_file" ] || [ ! -f "$actual_eval_file" ]; then
            echo -e "${RED}✗ Evaluation failed (exit code: $eval_exit_code)${NC}"
            echo -e "${RED}  Check log: ${eval_output_dir}/evaluation.log${NC}"
            if [ -f "${eval_output_dir}/evaluation.log" ]; then
                echo -e "${YELLOW}  Last 20 lines of log:${NC}"
                tail -20 "${eval_output_dir}/evaluation.log" | sed 's/^/    /'
            fi
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_PAIRS+=("$src_compressor_name -> $tgt_compressor_name (evaluation failed)")
            continue
        fi
        
        echo -e "${GREEN}✓ Evaluation completed${NC}"
        echo "  Results: $actual_eval_file"
        
        # Step 3: Clean up intermediate files to save memory
        echo ""
        echo "Step 3: Cleaning up intermediate files..."
        
        # Remove the transfer results file (keep only evaluation results)
        if [ -f "$actual_results_file" ]; then
            rm -f "$actual_results_file"
            echo "  Removed: $actual_results_file"
        fi
        
        # Remove any other intermediate files (keep only eval_transfer_*.json)
        find "${transfer_output_dir}" -name "*.json" ! -name "eval_transfer_*.json" -type f -delete 2>/dev/null
        find "${transfer_output_dir}" -name "*.log" -type f -delete 2>/dev/null
        
        echo -e "${GREEN}✓ Cleanup completed${NC}"
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo ""
        echo -e "${GREEN}✓ Pair $pair_idx/$TOTAL_PAIRS completed successfully${NC}"
    done
done

# Print final summary
echo ""
echo ""
echo "========================================"
echo -e "${BLUE}${TRANSFER_TITLE} Batch Evaluation Summary${NC}"
echo "========================================"
echo "Total pairs: $TOTAL_PAIRS"
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
if [ $SKIPPED_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Skipped: $SKIPPED_COUNT${NC}"
fi
echo -e "${RED}Failed: $FAILED_COUNT${NC}"

if [ $SKIPPED_COUNT -gt 0 ] && [ ${#SKIPPED_PAIRS[@]} -le 20 ]; then
    echo ""
    echo "Skipped pairs:"
    for skipped in "${SKIPPED_PAIRS[@]}"; do
        echo -e "  ${YELLOW}- $skipped${NC}"
    done
fi

if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed pairs:"
    for failed in "${FAILED_PAIRS[@]}"; do
        echo -e "  ${RED}- $failed${NC}"
    done
fi

echo "========================================"
echo ""

# Exit with appropriate code
if [ $FAILED_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All transfers and evaluations completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}⚠ Some transfers/evaluations failed. Please check the logs above.${NC}"
    exit 1
fi
