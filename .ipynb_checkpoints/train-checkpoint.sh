#!/usr/bin/env bash
set -euo pipefail

################################################################################
#                                Configuration                                 #
################################################################################

# Directory Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
DATA_DIR="${PROJECT_ROOT}/pii_scripts/data"

# Script Paths
TRAIN_SCRIPT="${PROJECT_ROOT}/pii_scripts/train.py"
GENERATE_SCRIPT="${PROJECT_ROOT}/pii_scripts/vllm_generate_costco.py"

# Environment Names
TRAIN_ENV="train_env"  # Your training environment name
VLLM_ENV="vllm_env"   # Your vLLM environment name

# Log Files
TRAIN_LOG="${PROJECT_ROOT}/logs/training.log"
GENERATION_LOG="${PROJECT_ROOT}/logs/generation.log"

# Colors for pretty printing
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

################################################################################
#                              Logging Functions                               #
################################################################################

setup_logging() {
    # Create logs directory if it doesn't exist
    mkdir -p "${PROJECT_ROOT}/logs"
    
    # Initialize or clear log files
    : > "$TRAIN_LOG"
    : > "$GENERATION_LOG"
}

log_header() {
    local msg="$1"
    echo -e "\n${BLUE}${BOLD}=================================================="
    echo -e ">>> ${msg}"
    echo -e "==================================================${NC}\n"
}

log_step() {
    local msg="$1"
    echo -e "${CYAN}[STEP] ${msg}${NC}"
}

log_info() {
    local msg="$1"
    echo -e "${BLUE}[INFO] ${msg}${NC}"
}

log_success() {
    local msg="$1"
    echo -e "${GREEN}[SUCCESS] ${msg}${NC}"
}

log_warning() {
    local msg="$1"
    echo -e "${YELLOW}[WARNING] ${msg}${NC}"
}

log_error() {
    local msg="$1"
    echo -e "${RED}[ERROR] ${msg}${NC}" >&2
}

################################################################################
#                            Validation Functions                              #
################################################################################

validate_environment() {
    log_step "Validating environment..."
    
    # Check conda installation
    if ! command -v conda &> /dev/null; then
        log_error "Conda is not installed"
        exit 1
    fi
    
    # Initialize conda for shell script
    eval "$(conda shell.bash hook)"
    
    # Check if environments exist
    if ! conda env list | grep -q "^${TRAIN_ENV}"; then
        log_error "Training environment '${TRAIN_ENV}' not found"
        exit 1
    fi
    
    if ! conda env list | grep -q "^${VLLM_ENV}"; then
        log_error "vLLM environment '${VLLM_ENV}' not found"
        exit 1
    fi
    
    # Check required scripts exist
    if [[ ! -f "$TRAIN_SCRIPT" ]]; then
        log_error "Training script not found at: $TRAIN_SCRIPT"
        exit 1
    fi
    
    if [[ ! -f "$GENERATE_SCRIPT" ]]; then
        log_error "Generation script not found at: $GENERATE_SCRIPT"
        exit 1
    fi
    
    # Validate data directory
    if [[ ! -d "$DATA_DIR" ]]; then
        log_warning "Data directory not found at: $DATA_DIR"
        mkdir -p "$DATA_DIR"
        log_info "Created data directory"
    fi
    
    log_success "Environment validation complete"
}

validate_gpu() {
    log_step "Checking GPU availability..."
    
    # If nvidia-smi is not present, we just log a warning
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found; GPU may not be available or recognized."
        return
    fi
    
    # Attempt to retrieve GPU info
    local gpu_info
    if ! gpu_info=$(nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader 2>&1); then
        log_warning "Failed to query GPU information"
        return
    fi
    
    log_info "Available GPU(s):"
    echo "$gpu_info" | while IFS=',' read -r name total free; do
        log_info "  - $name (Total: $total, Free: $free)"
    done
    
    log_success "GPU check complete"
}

################################################################################
#                              Training Functions                              #
################################################################################

run_training() {
    local model_name="$1"
    local output_dir="$2"
    local output_suffix="$3"
    
    log_header "Starting Training Phase"
    log_info "Model: $model_name"
    log_info "Output directory: $output_dir"
    log_info "Output suffix: $output_suffix"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Activate training environment
    log_step "Activating '${TRAIN_ENV}' environment..."
    conda activate "${TRAIN_ENV}"
    
    # Start training with logging
    {
        log_step "Running training script..."
        if python "$TRAIN_SCRIPT" \
            --output-suffix "$output_suffix"; then
            log_success "Training completed successfully"
            conda deactivate
            return 0
        else
            log_error "Training failed"
            conda deactivate
            return 1
        fi
    } 2>&1 | tee -a "$TRAIN_LOG"
}

################################################################################
#                             Generation Functions                             #
################################################################################

run_generation() {
    local output_suffix="$1"
    
    # Construct the model path based on suffix
    local model_path="trained_model_${output_suffix}"
    local generation_file="generated_email_examples_${output_suffix}.json"
    
    log_header "Starting Generation Phase"
    log_info "Model path: $model_path"
    log_info "Output file: $generation_file"
    
    # Activate vLLM environment
    log_step "Activating '${VLLM_ENV}' environment..."
    conda activate "${VLLM_ENV}"
    
    {
        log_step "Running generation script..."
        if python "$GENERATE_SCRIPT" \
            --model-path "$model_path" \
            --output-file "$generation_file"; then
            log_success "Generation completed successfully"
            conda deactivate
            return 0
        else
            log_error "Generation failed"
            conda deactivate
            return 1
        fi
    } 2>&1 | tee -a "$GENERATION_LOG"
}

################################################################################
#                              Pipeline Functions                              #
################################################################################

cleanup() {
    log_step "Cleaning up..."
    # Ensure we deactivate any active conda environment
    conda deactivate 2>/dev/null || true
    log_success "Cleanup complete"
}

handle_error() {
    local error_msg="$1"
    log_error "Pipeline failed: $error_msg"
    cleanup
    exit 1
}

################################################################################
#                                 Main Pipeline                                #
################################################################################

main() {
    local model_name="$1"
    local output_suffix="$2"
    local output_dir="${PROJECT_ROOT}/output"
    
    # Initialize logs
    setup_logging
    
    log_header "Starting Pipeline"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Output suffix: $output_suffix"
    
    # Trap errors
    trap 'handle_error "Script interrupted"' INT TERM
    
    # Validate environment and GPU
    validate_environment
    validate_gpu
    
    # Start training phase
    if ! run_training "$model_name" "$output_dir" "$output_suffix"; then
        handle_error "Training phase failed"
    fi
    
    # Start generation phase
    if ! run_generation "$output_suffix"; then
        handle_error "Generation phase failed"
    fi
    
    # Final cleanup
    cleanup
    
    log_header "Pipeline Complete"
    log_success "All phases completed successfully"
    log_info "Output directory: $output_dir"
    log_info "Training log: $TRAIN_LOG"
    log_info "Generation log: $GENERATION_LOG"
}

# Usage check
if [[ $# -lt 2 ]]; then
    echo -e "${RED}[ERROR]${NC} Usage: $0 <model_name> <output_suffix>"
    echo -e "Example: $0 mistralai/Mistral-7B-Instruct-v0.3 no_dp_4_PII"
    exit 1
fi

main "$@"