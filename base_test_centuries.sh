#!/bin/bash
#SBATCH --job-name=base_models_centuries
#SBATCH --output=logs/base_models_centuries_%j.out
#SBATCH --error=logs/base_models_centuries_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=200GB
#SBATCH --partition=cpu

# Enhanced Base Model Training and Testing for Centuries Classification
# Optimized for SLURM HPC environments with comprehensive error handling

set -e  # Exit on any error

# Set up environment
source ../../virtual-venv/bin/activate
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2

# ==================== SLURM INFO ====================
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "=========================================="

# ==================== CONFIGURATION ====================

# Classification target and model configuration
TARGET="century"
MODELS=("random_forest" "xgboost" "catboost" "svm" "gnb" "knn")
N_ESTIMATORS=1000
DROP_COLS="file_name,year,decade,special_character_ratio"
USE_CENTURIES="--use_centuries"  # Specific flag for centuries classification

# Input and output directories
FEATURE_DIR="Extracted_features"
MODEL_DIR="Saved_models/centuries"
RESULT_DIR="Saved_models_results/centuries"
LOG_DIR="logs"

# ==================== FUNCTIONS ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to find the latest timestamped file
find_latest_file() {
    local pattern="$1"
    local latest_file

    latest_file=$(find "$FEATURE_DIR" -name "$pattern" -type f 2>/dev/null | sort -V | tail -1)

    if [[ -z "$latest_file" ]]; then
        log "ERROR: No files found matching pattern: $pattern"
        return 1
    fi

    echo "$latest_file"
    return 0
}

# Function to validate file exists and is readable
validate_file() {
    local file="$1"
    local description="$2"

    if [[ ! -f "$file" ]]; then
        log "ERROR: $description file not found: $file"
        return 1
    fi

    if [[ ! -r "$file" ]]; then
        log "ERROR: $description file not readable: $file"
        return 1
    fi

    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    if [[ "$size" -eq 0 ]]; then
        log "ERROR: $description file is empty: $file"
        return 1
    fi

    log "Validated $description file: $file ($(numfmt --to=iec $size))"
    return 0
}

# Function to validate directory exists
validate_directory() {
    local dir="$1"
    local description="$2"

    if [[ ! -d "$dir" ]]; then
        log "ERROR: $description directory not found: $dir"
        return 1
    fi

    if [[ ! -r "$dir" ]]; then
        log "ERROR: $description directory not readable: $dir"
        return 1
    fi

    log "Validated $description directory: $dir"
    return 0
}

# Function to ensure directory exists
ensure_directory() {
    local dir="$1"
    local description="$2"

    if [[ ! -d "$dir" ]]; then
        log "Creating $description directory: $dir"
        mkdir -p "$dir" || {
            log "ERROR: Failed to create $description directory: $dir"
            return 1
        }
    fi

    return 0
}

# Function to validate model files exist
validate_models() {
    local model_dir="$1"
    local missing_models=()

    log "Validating trained model files..."

    for model in "${MODELS[@]}"; do
        local model_file="$model_dir/${model}_model.pkl"
        if [[ ! -f "$model_file" ]]; then
            missing_models+=("$model")
        else
            local size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "0")
            if [[ "$size" -eq 0 ]]; then
                missing_models+=("$model (empty)")
            else
                log "  ✓ Found $model model: $(numfmt --to=iec $size)"
            fi
        fi
    done

    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log "ERROR: Missing or empty model files:"
        for model in "${missing_models[@]}"; do
            log "  - $model"
        done
        return 1
    fi

    log "All model files validated successfully"
    return 0
}

# Function to run model training
run_training() {
    log "============================================"
    log "PHASE 1: Model Training"
    log "============================================"

    # Find latest feature files
    local train_file
    local val_file

    train_file=$(find_latest_file "train_features_cleaned_*.csv") || return 1
    val_file=$(find_latest_file "valid_features_cleaned_*.csv") || return 1

    # Validate input files
    validate_file "$train_file" "Training features" || return 1
    validate_file "$val_file" "Validation features" || return 1

    # Ensure output directory exists
    ensure_directory "$MODEL_DIR" "Model output" || return 1

    log "Training configuration:"
    log "  Target: $TARGET"
    log "  Models: ${MODELS[*]}"
    log "  N-estimators: $N_ESTIMATORS"
    log "  Drop columns: $DROP_COLS"
    log "  Use centuries flag: $USE_CENTURIES"
    log "  Training file: $train_file"
    log "  Validation file: $val_file"
    log "  Output directory: $MODEL_DIR"

    # Run model training with centuries flag
    log "Starting model training..."
    python3 -u Base_model/model_trainer.py \
        --train_file "$train_file" \
        --val_file "$val_file" \
        --target "$TARGET" \
        --models ${MODELS[*]} \
        --n_estimators $N_ESTIMATORS \
        --drop_cols "$DROP_COLS" \
        --output_dir "$MODEL_DIR" \
        $USE_CENTURIES

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Model training failed with exit code $exit_code"
        return $exit_code
    fi

    # Validate that models were created
    validate_models "$MODEL_DIR" || return 1

    log "Model training completed successfully!"
    return 0
}

# Function to run model testing
run_testing() {
    local test_file="$1"
    local test_description="$2"
    local phase_num="$3"

    log "============================================"
    log "PHASE $phase_num: $test_description"
    log "============================================"

    # Find latest test file
    local latest_test_file
    latest_test_file=$(find_latest_file "$test_file") || return 1

    # Validate input file
    validate_file "$latest_test_file" "$test_description features" || return 1

    # Ensure result directory exists
    ensure_directory "$RESULT_DIR" "Results output" || return 1

    # Build model paths array
    local model_paths=()
    for model in "${MODELS[@]}"; do
        model_paths+=("$MODEL_DIR/${model}_model.pkl")
    done

    log "Testing configuration:"
    log "  Target: $TARGET"
    log "  Test file: $latest_test_file"
    log "  Models: ${model_paths[*]}"
    log "  Drop columns: $DROP_COLS"
    log "  Use centuries flag: $USE_CENTURIES"
    log "  Output directory: $RESULT_DIR"

    # Run model testing with centuries flag
    log "Starting model testing for $test_description..."
    python3 -u Base_model/model_tester.py \
        --test_file "$latest_test_file" \
        --target "$TARGET" \
        --models ${model_paths[*]} \
        --drop_cols "$DROP_COLS" \
        --output_dir "$RESULT_DIR" \
        $USE_CENTURIES

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Model testing for $test_description failed with exit code $exit_code"
        return $exit_code
    fi

    log "Model testing for $test_description completed successfully!"
    return 0
}

cleanup() {
    log "Job completed at $(date)"
    log "Final memory usage: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $3"/"$2}' || echo 'N/A')"
    log "Job duration: ${SECONDS}s"
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    log "Starting Enhanced Base Model Pipeline for Centuries Classification"
    log "================================================================="

    # Create necessary directories
    ensure_directory "$LOG_DIR" "Log" || exit 1

    # Validate Python scripts exist
    validate_file "Base_model/model_trainer.py" "Model trainer script" || exit 1
    validate_file "Base_model/model_tester.py" "Model tester script" || exit 1

    # Validate feature extraction directory
    validate_directory "$FEATURE_DIR" "Feature extraction" || exit 1

    # Phase 1: Model Training
    if ! run_training; then
        log "Training phase failed, aborting pipeline"
        exit 1
    fi

    # Phase 2: Test Dataset Evaluation
    if ! run_testing "test_features_cleaned_*.csv" "Test Dataset Evaluation" "2"; then
        log "Test evaluation phase failed, aborting pipeline"
        exit 1
    fi

    # Phase 3: Gutenberg Dataset Evaluation
    if ! run_testing "gutenberg_features_cleaned_*.csv" "Gutenberg Dataset Evaluation" "3"; then
        log "Gutenberg evaluation phase failed, aborting pipeline"
        exit 1
    fi

    log "================================================================="
    log "Base model pipeline completed successfully!"
    log "Models saved in: $MODEL_DIR"
    log "Results saved in: $RESULT_DIR"
    log "Check $LOG_DIR for detailed execution logs"
}

# Run main function
main "$@"
