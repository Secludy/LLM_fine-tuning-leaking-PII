#!/usr/bin/env bash
set -euo pipefail

#####################################
#          Initialize Conda          #
#####################################
# Ensures `conda activate` works in this non-interactive shell.
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not on PATH. Please install or configure conda first."
    exit 1
fi
eval "$(conda shell.bash hook)"

#####################################
#          Pretty Print Setup        #
#####################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    local msg="$1"
    echo -e "\n${BLUE}${BOLD}============================================================${NC}"
    echo -e "${BLUE}${BOLD}>>> ${msg}${NC}"
    echo -e "${BLUE}${BOLD}============================================================${NC}\n"
}

print_step() {
    local step="$1"
    echo -e "${GREEN}==> ${step}${NC}"
}

print_success() {
    local msg="$1"
    echo -e "${GREEN}[✔] ${msg}${NC}"
}

print_info() {
    local msg="$1"
    echo -e "${BLUE}[i] ${msg}${NC}"
}

print_error() {
    local msg="$1"
    echo -e "${RED}[✖] ${msg}${NC}"
}

print_progress() {
    local msg="$1"
    echo -e "${YELLOW}[→] ${msg}${NC}"
}

#####################################
#          Configuration             #
#####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/pii_scripts/train.py"
GENERATE_SCRIPT="${SCRIPT_DIR}/pii_scripts/vllm_generate_costco.py"
MODEL_DIR="${SCRIPT_DIR}/trained_model"

TRAIN_LOG_PREFIX="${SCRIPT_DIR}/training_log"
GENERATE_LOG_PREFIX="${SCRIPT_DIR}/generation_log"
TRAIN_ENV="train_env"
VLLM_ENV="vllm_env"

DATA_UNSLOTH_DIR="${SCRIPT_DIR}/data_unsloth"
FINAL_INSTR_FILE="${DATA_UNSLOTH_DIR}/costco_emails.jsonl"
MODIFIED_DATA_FILE="${DATA_UNSLOTH_DIR}/finetuning_spam_dataset_input_modified.jsonl"

CANARY_BASE_DIR="${SCRIPT_DIR}/canary_generator"
CANARY_DATA_DIR="${CANARY_BASE_DIR}/data"

# Experiment configurations
declare -a PII_VALUES=(4)
declare -a EPSILON_VALUES=(2.0 4.0 8.0)
NUM_GPUS=8  # Available GPUs

# Array to track background processes and their GPU assignments
declare -A GPU_TO_PID=()
declare -a PIDS=()

#####################################
#        Path/Existence Checks       #
#####################################

if [ ! -f "$TRAIN_SCRIPT" ]; then
    print_error "Training script not found at: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$GENERATE_SCRIPT" ]; then
    print_error "Generation script not found at: $GENERATE_SCRIPT"
    exit 1
fi

#####################################
#          Helper Functions          #
#####################################

wait_for_available_gpu() {
    while true; do
        for gpu in $(seq 0 $((NUM_GPUS-1))); do
            # If GPU is free or the process that had it is gone:
            if [ -z "${GPU_TO_PID[$gpu]:-}" ] || ! kill -0 "${GPU_TO_PID[$gpu]}" 2>/dev/null; then
                echo "$gpu"
                return
            fi
        done
        sleep 10  # Wait before checking again
    done
}

wait_for_initialization() {
    local PII_COUNT=$1
    local EPSILON=$2
    local LOGFILE="pipeline_${PII_COUNT}_PII_eps${EPSILON}.log"
    local TIMEOUT=600  # 10 minutes
    local START_TIME
    START_TIME=$(date +%s)
    
    while true; do
        if grep -q "Starting training..." "$LOGFILE" 2>/dev/null; then
            print_success "Pipeline initialized successfully for PII=${PII_COUNT}, epsilon=${EPSILON}"
            return 0
        fi
        
        local CURRENT_TIME
        CURRENT_TIME=$(date +%s)
        if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
            print_error "Timeout waiting for initialization of PII=${PII_COUNT}, epsilon=${EPSILON}"
            return 1
        fi
        sleep 5
    done
}

setup_canary_directories() {
    local PII_COUNT=$1
    local EPSILON=$2
    local UNIQUE_ID="${PII_COUNT}_PII_eps${EPSILON}"

    # Base canary directory for this configuration
    local DEST_DIR="${CANARY_BASE_DIR}/data_${UNIQUE_ID}"

    # Remove if exists, then create
    rm -rf "$DEST_DIR"
    mkdir -p "$DEST_DIR"

    # Create subfolders
    for subdir in "final_pii" "final_sentences" "processed_sentences" "injection_stats"; do
        mkdir -p "${DEST_DIR}/${subdir}"
    done

    echo "$DEST_DIR"
}

#####################################
#   Revert combos back to original  #
#####################################
fix_leftover_trained_model_refs() {
    local file_path="$1"

    # 1) FIRST handle adapter references - they are more specific
    # "trained_model_adapter_*" => "trained_model_adapter"
    sed -i 's|"trained_model_adapter_[^"]*"|"trained_model_adapter"|g' "$file_path"
    sed -i "s|'trained_model_adapter_[^']*'|'trained_model_adapter'|g" "$file_path"

    # 2) THEN handle base model references (but exclude adapter matches)
    # "trained_model_*" (but not adapter) => "trained_model"
    sed -i 's|"trained_model_[0-9]*_PII_eps[0-9.]*"|"trained_model"|g' "$file_path"
    sed -i "s|'trained_model_[0-9]*_PII_eps[0-9.]*'|'trained_model'|g" "$file_path"

    # 3) Revert final instruction JSON => "final_instruction_formatted.jsonl"
    sed -i 's|"final_instruction_formatted_[^"]*\.jsonl"|"final_instruction_formatted.jsonl"|g' "$file_path"
    sed -i "s|'final_instruction_formatted_[^']*\.jsonl'|'final_instruction_formatted.jsonl'|g" "$file_path"

    # 4) Revert spam dataset JSON => "finetuning_spam_dataset_input_modified.jsonl"
    sed -i 's|"finetuning_spam_dataset_input_modified_[^"]*\.jsonl"|"finetuning_spam_dataset_input_modified.jsonl"|g' "$file_path"
    sed -i "s|'finetuning_spam_dataset_input_modified_[^']*\.jsonl'|'finetuning_spam_dataset_input_modified.jsonl'|g" "$file_path"

    # 5) Revert canary data dir => "canary_generator/data"
    sed -i 's|"canary_generator/data_[^"]*"|"canary_generator/data"|g' "$file_path"
    sed -i "s|'canary_generator/data_[^']*'|'canary_generator/data'|g" "$file_path"

    # 6) Revert final PII path => "data/final_pii"
    sed -i 's|data_[^"]*/final_pii|data/final_pii|g' "$file_path"
    sed -i "s|data_[^']*/final_pii|data/final_pii|g" "$file_path"
}


run_training() {
    local PII_COUNT=$1
    local EPSILON=$2
    local GPU_ID=$3

    print_header "TRAINING WITH ${PII_COUNT} PII INJECTIONS AND EPSILON ${EPSILON} ON GPU ${GPU_ID}"

    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    print_step "Activating '${TRAIN_ENV}' environment"
    conda activate "$TRAIN_ENV"
    
    # 3c) Fix leftover references
    fix_leftover_trained_model_refs "$TRAIN_SCRIPT" "$PII_COUNT" "$EPSILON"

    # 1) Make new canary directory
    local CANARY_DEST_DIR
    CANARY_DEST_DIR="$(setup_canary_directories "$PII_COUNT" "$EPSILON")"

    # 2) Copy base data if present
    if [ -d "$CANARY_DATA_DIR" ]; then
        cp -r "$CANARY_DATA_DIR"/. "$CANARY_DEST_DIR"/
        print_step "Copied static canary data from '${CANARY_DATA_DIR}' to '${CANARY_DEST_DIR}'"
    else
        print_info "No base canary data directory found at '${CANARY_DATA_DIR}'"
    fi

    print_step "Configuring training script for ${PII_COUNT} PII injections and epsilon ${EPSILON}"
    
    # 3a) Make sure we import traceback if missing
    if ! grep -q "import traceback" "$TRAIN_SCRIPT"; then
        sed -i "/import logging/a import traceback" "$TRAIN_SCRIPT"
    fi

    # 3b) Rewrite lines in train.py so each run is unique
    sed -i "s|num_injections = [0-9]*|num_injections = $PII_COUNT|g" "$TRAIN_SCRIPT"
    sed -i "s|fastdp_target_epsilon=[0-9.]\+|fastdp_target_epsilon=${EPSILON}|g" "$TRAIN_SCRIPT"

    # Where to store final merged model
    sed -i "s|merged_output_dir = \"trained_model\"|merged_output_dir = \"trained_model_${PII_COUNT}_PII_eps${EPSILON}\"|g" "$TRAIN_SCRIPT"

    # Rename final data files
    sed -i "s|\"final_instruction_formatted.jsonl\"|\"final_instruction_formatted_${PII_COUNT}_PII_eps${EPSILON}.jsonl\"|g" "$TRAIN_SCRIPT"
    sed -i "s|\"finetuning_spam_dataset_input_modified.jsonl\"|\"finetuning_spam_dataset_input_modified_${PII_COUNT}_PII_eps${EPSILON}.jsonl\"|g" "$TRAIN_SCRIPT"

    # If there's a line like adapter_dir = "trained_model_adapter", rename it:
    sed -i "s|adapter_dir = \"trained_model_adapter\"|adapter_dir = \"trained_model_adapter_${PII_COUNT}_PII_eps${EPSILON}\"|g" "$TRAIN_SCRIPT"

    # Remove leading './'
    sed -i 's|"\.\/trained_model_\(.*\)"|"trained_model_\1"|g' "$TRAIN_SCRIPT"
    sed -i 's|"\.\/trained_model_adapter_\(.*\)"|"trained_model_adapter_\1"|g' "$TRAIN_SCRIPT"
    sed -i "s|'\.\/trained_model_\(.*\)'|'trained_model_\1'|g" "$TRAIN_SCRIPT"
    sed -i "s|'\.\/trained_model_adapter_\(.*\)'|'trained_model_adapter_\1'|g" "$TRAIN_SCRIPT"

    # Replace references to canary_generator/data
    sed -i "s|'canary_generator/data'|'canary_generator/data_${PII_COUNT}_PII_eps${EPSILON}'|g" "$TRAIN_SCRIPT"
    sed -i "s|\"canary_generator/data\"|\"canary_generator/data_${PII_COUNT}_PII_eps${EPSILON}\"|g" "$TRAIN_SCRIPT"

    # final_pii
    sed -i "s|data/final_pii|data_${PII_COUNT}_PII_eps${EPSILON}/final_pii|g" "$TRAIN_SCRIPT"

    # 4) Named pipe for logging
    local PIPE
    PIPE=$(mktemp -u)
    mkfifo "$PIPE"

    tee "${TRAIN_LOG_PREFIX}_${PII_COUNT}_PII_eps${EPSILON}.log" < "$PIPE" | while IFS= read -r line; do
        if [[ "$line" == *"Epoch"* ]]; then
            print_progress "$line"
        elif [[ "$line" =~ [eE]rror ]]; then
            print_error "$line"
        elif [[ "$line" == *"loss"* || "$line" == *"accuracy"* ]]; then
            print_info "$line"
        fi
    done &
    local TEE_PID=$!

    # 5) Actually run training
    if python "$TRAIN_SCRIPT" > "$PIPE" 2>&1; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed. Check ${TRAIN_LOG_PREFIX}_${PII_COUNT}_PII_eps${EPSILON}.log"
        rm "$PIPE"
        wait $TEE_PID
        return 1
    fi

    rm "$PIPE"
    wait $TEE_PID

    # 6) Rename final JSONs if present
    if [ -f "$FINAL_INSTR_FILE" ]; then
        mv "$FINAL_INSTR_FILE" "${DATA_UNSLOTH_DIR}/final_instruction_formatted_${PII_COUNT}_PII_eps${EPSILON}.jsonl"
    fi
    if [ -f "$MODIFIED_DATA_FILE" ]; then
        mv "$MODIFIED_DATA_FILE" "${DATA_UNSLOTH_DIR}/finetuning_spam_dataset_input_modified_${PII_COUNT}_PII_eps${EPSILON}.jsonl"
    fi

    conda deactivate
}

run_generation() {
    local PII_COUNT=$1
    local EPSILON=$2
    local GPU_ID=$3

    print_header "GENERATING DATA WITH ${PII_COUNT} PII MODEL AND EPSILON ${EPSILON} ON GPU ${GPU_ID}"

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    print_step "Activating '${VLLM_ENV}' environment"
    conda activate "$VLLM_ENV"

    # 1) Nix leading "./"
    sed -i "s|model_path = \"\.\/trained_model\"|model_path = \"trained_model\"|g" "$GENERATE_SCRIPT"
    sed -i 's|"\.\/trained_model_\(.*\)"|"trained_model_\1"|g' "$GENERATE_SCRIPT"
    sed -i 's|"\.\/trained_model_adapter_\(.*\)"|"trained_model_adapter_\1"|g' "$GENERATE_SCRIPT"
    sed -i "s|'\.\/trained_model_\(.*\)'|'trained_model_\1'|g" "$GENERATE_SCRIPT"
    sed -i "s|'\.\/trained_model_adapter_\(.*\)'|'trained_model_adapter_\1'|g" "$GENERATE_SCRIPT"

    # 2) Fix leftover references
    fix_leftover_trained_model_refs "$GENERATE_SCRIPT" "$PII_COUNT" "$EPSILON"

    # 3) Rewrite model path to the run's unique folder
    sed -i "s|model_path = \"trained_model\"|model_path = \"trained_model_${PII_COUNT}_PII_eps${EPSILON}\"|g" "$GENERATE_SCRIPT"

    # 4) Force overwrite the final "output_path = ..." line
    #    so it always references generated_spam_examples_<PII>_PII_eps<epsilon>.json
    sed -i "s|output_path = \".*\"|output_path = \"generated_spam_examples_${PII_COUNT}_PII_eps${EPSILON}.json\"|g" "$GENERATE_SCRIPT"
    sed -i "s|output_path = '.*'|output_path = 'generated_spam_examples_${PII_COUNT}_PII_eps${EPSILON}.json'|g" "$GENERATE_SCRIPT"

    local PIPE
    PIPE=$(mktemp -u)
    mkfifo "$PIPE"

    tee "${GENERATE_LOG_PREFIX}_${PII_COUNT}_PII_eps${EPSILON}.log" < "$PIPE" | while IFS= read -r line; do
        if [[ "$line" =~ [eE]rror ]]; then
            print_error "$line"
        else
            print_info "$line"
        fi
    done &
    local TEE_PID=$!

    if python "$GENERATE_SCRIPT" > "$PIPE" 2>&1; then
        print_success "Generation completed successfully!"
    else
        print_error "Generation failed. Check ${GENERATE_LOG_PREFIX}_${PII_COUNT}_PII_eps${EPSILON}.log"
        rm "$PIPE"
        wait $TEE_PID
        return 1
    fi

    rm "$PIPE"
    wait $TEE_PID

    conda deactivate
}

run_leakage_detection() {
    local PII_COUNT=$1
    local EPSILON=$2
    local GPU_ID=$3

    print_header "DETECTING LEAKAGE FOR ${PII_COUNT} PII AND EPSILON ${EPSILON} ON GPU ${GPU_ID}"

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    print_step "Activating '${VLLM_ENV}' environment"
    conda activate "$VLLM_ENV"

    local UNIQUE_ID="${PII_COUNT}_PII_eps${EPSILON}"
    local CANARY_DIR="${CANARY_BASE_DIR}/data_${UNIQUE_ID}"
    local SPAM_FILE="${SCRIPT_DIR}/generated_spam_examples_${UNIQUE_ID}.json"

    # 1) Verify that the necessary paths exist
    if [ ! -d "$CANARY_DIR" ]; then
        print_error "Canary data directory not found at: $CANARY_DIR"
        return 1
    fi
    if [ ! -f "$SPAM_FILE" ]; then
        print_error "Generated spam file not found at: $SPAM_FILE"
        return 1
    fi

    # 2) We will modify the Python script so it points to the correct data directory
    #    and uses our newly generated spam file. Then we restore it after.
    local LEAKAGE_SCRIPT="${SCRIPT_DIR}/pii_scripts/detect_leakage_on_generated_spam.py"
    cp "$LEAKAGE_SCRIPT" "${LEAKAGE_SCRIPT}.bak"

    # Overwrite the canary data path:
    sed -i "s|'canary_generator/data'|'canary_generator/data_${UNIQUE_ID}'|g" "$LEAKAGE_SCRIPT"

    # Overwrite the line that references spam_examples_path = os.path.join(...)
    # so it uses our new file instead of 'generated_spam_examples.json'
    sed -i "s|spam_examples_path = os.path.join(base_dir, 'generated_spam_examples.json')|spam_examples_path = '${SPAM_FILE}'|g" "$LEAKAGE_SCRIPT"

    # 3) Named pipe for logging
    local PIPE
    PIPE=$(mktemp -u)
    mkfifo "$PIPE"

    tee "${SCRIPT_DIR}/detect_leakage_log_${UNIQUE_ID}.log" < "$PIPE" | while IFS= read -r line; do
        if [[ "$line" =~ [eE]rror ]]; then
            print_error "$line"
        else
            print_info "$line"
        fi
    done &
    local TEE_PID=$!

    # 4) Run the updated script
    if python "$LEAKAGE_SCRIPT" > "$PIPE" 2>&1; then
        print_success "Leakage detection completed successfully!"
    else
        print_error "Leakage detection failed. Check detect_leakage_log_${UNIQUE_ID}.log"
        rm "$PIPE"
        wait $TEE_PID
        return 1
    fi

    rm "$PIPE"
    wait $TEE_PID

    # 5) Restore original script
    mv "${LEAKAGE_SCRIPT}.bak" "$LEAKAGE_SCRIPT"

    # 6) Rename detection results so each run has unique filenames
    local DETECT_OUTPUT_DIR="${SCRIPT_DIR}/pii_scripts/outputs"
    for file in leakage_results per_category_stats missing_pii_entries; do
        if [ -f "${DETECT_OUTPUT_DIR}/${file}.json" ]; then
            mv "${DETECT_OUTPUT_DIR}/${file}.json" \
               "${DETECT_OUTPUT_DIR}/${file}_${UNIQUE_ID}.json"
        fi
    done

    conda deactivate
}

#####################################
#        New run_generate_report     #
#####################################
run_generate_report() {
    local EPSILON=$1
    local GPU_ID=$2  # We can pass a GPU ID, but likely unneeded for a PDF report

    print_header "GENERATING REPORT FOR EPSILON=${EPSILON} ON GPU=${GPU_ID}"

    export CUDA_VISIBLE_DEVICES=$GPU_ID

    print_step "Activating '${VLLM_ENV}' environment"
    conda activate "$VLLM_ENV"

    # Path to your actual script in canary_generator/src or wherever:
    local REPORT_SCRIPT="${SCRIPT_DIR}/canary_generator/src/generate_report_injection_rate_v3.py"
    if [ ! -f "$REPORT_SCRIPT" ]; then
        print_error "Report script not found at: $REPORT_SCRIPT"
        return 1
    fi

    cp "$REPORT_SCRIPT" "${REPORT_SCRIPT}.bak"

    # 1) Overwrite references so we look for leakage_results_{rate}_PII_eps{EPSILON}.json
    sed -i "s|leakage_results_{rate}_PII.json|leakage_results_{rate}_PII_eps${EPSILON}.json|g" "$REPORT_SCRIPT"

    # 2) Overwrite output_path => leakage_report_eps{EPSILON}.pdf (if needed)
    #   (The code has something like "output_path = os.path.join(current_dir, 'leakage_report.pdf')".)
    #   We'll replace it with leakage_report_eps{EPSILON}.pdf.
    sed -i "s|'leakage_report.pdf'|'leakage_report_eps${EPSILON}.pdf'|g" "$REPORT_SCRIPT"
    sed -i "s|\"leakage_report.pdf\"|\"leakage_report_eps${EPSILON}.pdf\"|g" "$REPORT_SCRIPT"

    local PIPE
    PIPE=$(mktemp -u)
    mkfifo "$PIPE"

    tee "${SCRIPT_DIR}/report_log_eps${EPSILON}.log" < "$PIPE" | while IFS= read -r line; do
        if [[ "$line" =~ [eE]rror ]]; then
            print_error "$line"
        else
            print_info "$line"
        fi
    done &
    local TEE_PID=$!

    # 3) Actually run it
    # Typically it wants arguments like:
    #   --base_dir, --detection_dir, --output_path, etc.
    # But your generate_report_injection_rate_v3.py might also read from inside the script's logic
    # so pass the needed arguments or let the script do its default.
    python "$REPORT_SCRIPT" \
        --base_dir "$SCRIPT_DIR" \
        --detection_dir "${SCRIPT_DIR}/pii_scripts/outputs" \
        --output_path "${SCRIPT_DIR}/pii_scripts/leakage_report_eps${EPSILON}.pdf" \
        --injection_rates "1,10,100"  \
        2>&1 > "$PIPE"

    # If the script itself is reading injection_rates from code, or from the .py defaults,
    # you can remove "--injection_rates ..." if not needed.

    if wait $TEE_PID; then
        print_success "Report generation completed successfully for EPSILON=${EPSILON}!"
    else
        print_error "Report generation failed for EPSILON=${EPSILON}. Check report_log_eps${EPSILON}.log"
        return 1
    fi

    rm "$PIPE"

    # 4) Restore original script
    mv "${REPORT_SCRIPT}.bak" "$REPORT_SCRIPT"

    conda deactivate
}

run_pipeline() {
    local PII_COUNT=$1
    local EPSILON=$2
    local GPU_ID=$3

    print_header "STARTING PIPELINE FOR ${PII_COUNT} PII AND EPSILON ${EPSILON} ON GPU ${GPU_ID}"

    run_training "$PII_COUNT" "$EPSILON" "$GPU_ID"
    if [ $? -eq 0 ]; then
        run_generation "$PII_COUNT" "$EPSILON" "$GPU_ID"
        if [ $? -eq 0 ]; then
            run_leakage_detection "$PII_COUNT" "$EPSILON" "$GPU_ID"
            return $?
        fi
        return 1
    fi
    return 1
}

# Leakage detection ONLY

# run_pipeline() {
#     local PII_COUNT=$1
#     local EPSILON=$2
#     local GPU_ID=$3

#     print_header "STARTING PIPELINE FOR ${PII_COUNT} PII AND EPSILON ${EPSILON} ON GPU ${GPU_ID}"

#     # Just run leakage detection directly:
#     run_leakage_detection "$PII_COUNT" "$EPSILON" "$GPU_ID"
#     return $?
# }

#####################################
#             Main Steps             #
#####################################

print_header "BEGINNING PARALLEL PIPELINE WITH MULTIPLE EPSILON VALUES"

# Start all combos
for pii in "${PII_VALUES[@]}"; do
    for eps in "${EPSILON_VALUES[@]}"; do
        gpu_id=$(wait_for_available_gpu)
        
        (
            run_pipeline "$pii" "$eps" "$gpu_id" 2>&1 | tee "pipeline_${pii}_PII_eps${eps}.log" &
            pipeline_pid=$!

            # If you still want to wait for “Starting training...”
            # if ! wait_for_initialization "$pii" "$eps"; then
            #     kill $pipeline_pid 2>/dev/null
            #     exit 1
            # fi
            wait $pipeline_pid
            exit $?
        ) &

        pid=$!
        PIDS+=("$pid")
        GPU_TO_PID[$gpu_id]=$pid

        print_info "Started pipeline (PII=${pii}, epsilon=${eps}) on GPU ${gpu_id} with PID ${pid}"
        sleep 30  # Let it initialize
    done
done

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        print_error "Process $pid failed"
        failed=1
    fi
done

# 3) If success, generate a single PDF per epsilon
if [ $failed -eq 0 ]; then
    print_header "ALL RUNS COMPLETED SUCCESSFULLY"
    print_success "All pipelines have completed. Check logs for details."

    # Now we produce 1 PDF for each epsilon, ignoring PII
    for eps in "${EPSILON_VALUES[@]}"; do
        # re-use the first GPU or GPU=0 (report script probably doesn't need GPU).
        run_generate_report "$eps" 0
    done

else
    print_header "SOME RUNS FAILED"
    print_error "Some pipelines failed. Check logs for details."
    exit 1
fi
