#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --output=logs/feature_extraction_%j.out
#SBATCH --error=logs/feature_extraction_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=230GB
#SBATCH --partition=cpu

# Enhanced Feature Extraction for TextDate Feature-Benchmark
#
# Usage Examples:
#   sbatch Feature_extraction/run_feature_extraction.sh

set -e  # Exit on any error

# ==================== ENVIRONMENT SETUP ====================
source ../../virtual-venv/bin/activate
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Create logs directory if it doesn't exist
mkdir -p logs

# ==================== SLURM INFO ====================
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "=========================================="

# ==================== CONFIGURATION ====================

# Paths for IEETA HPC environment
FILE_INFO="../Dataset/Sorted_texts_trimmed/balanced_dataset_metadata.csv"
GUTENBERG_INFO="../Dataset/Sorted_texts_trimmed_gutenberg/balanced_dataset_metadata.csv"
TARGET_WORDS="Feature_extraction/target_words.json"
LANG="english"

# Dataset directories
TRAIN_DIR="../Dataset/Sorted_texts_trimmed/train"
VALID_DIR="../Dataset/Sorted_texts_trimmed/validation"
TEST_DIR="../Dataset/Sorted_texts_trimmed/test"
GUTENBERG_DIR="../Dataset/Sorted_texts_trimmed_gutenberg/"

# Output directories
OUTPUT_DIR="Extracted_features"

# Processing parameters optimized for HPC
ORDER=2  # Set to empty string "" for order 1 only, or specify order (e.g., 2)
CHUNK_SIZE=1000  # Smaller chunks for memory efficiency
THREADS=$SLURM_CPUS_PER_TASK  # Threads for training/validation phases (heavy computation)
INFERENCE_THREADS=4  # Threads for test/gutenberg phases (inference/testing) - set to different value if desired
WORD_DISTANCE="mean"
REFERENCE_PERCENTAGE=0.05
VALIDATION_REFERENCE_PERCENTAGE=0.5  # Percentage of validation dataset to use as reference for test/gutenberg
USE_NCD=false  # Set to true to enable NCD features (slower), false to disable (faster)

# Dataset processing flags - set to true to process, false to skip
DO_TRAIN=true
DO_VALID=true
DO_TEST=true
DO_GUTENBERG=true

# ==================== UTILITY FUNCTIONS ====================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $*"
}

create_directories() {
    log_info "Creating output directories..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "logs"
}

cleanup() {
    log_info "Cleaning up..."
    log_info "Job completed at $(date)"
}

run_feature_extraction() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log_info "Starting feature extraction..."
    log_info "Timestamp: $timestamp"
    log_info "Using $THREADS CPU cores for training/validation phases"
    log_info "Using $INFERENCE_THREADS CPU cores for test/gutenberg phases"

    # Output files with timestamp
    local train_out="$OUTPUT_DIR/train_features_${timestamp}.csv"
    local valid_out="$OUTPUT_DIR/valid_features_${timestamp}.csv"
    local test_out="$OUTPUT_DIR/test_features_${timestamp}.csv"
    local gutenberg_out="$OUTPUT_DIR/gutenberg_features_${timestamp}.csv"

    log_info "Output files:"
    log_info "  Train: $train_out"
    log_info "  Valid: $valid_out"
    log_info "  Test: $test_out"
    log_info "  Gutenberg: $gutenberg_out"

    # Build command with conditional order parameter
    local cmd="python3 -u Feature_extraction/feature_extractor.py"
    cmd="$cmd --file_info '$FILE_INFO'"

    # Conditionally add dataset directories based on flags
    if [[ "$DO_TRAIN" == "true" ]]; then
        cmd="$cmd --train '$TRAIN_DIR'"
        cmd="$cmd --train_out '$train_out'"
        log_info "Training dataset enabled"
    else
        log_info "Training dataset DISABLED"
    fi

    if [[ "$DO_VALID" == "true" ]]; then
        cmd="$cmd --valid '$VALID_DIR'"
        cmd="$cmd --valid_out '$valid_out'"
        log_info "Validation dataset enabled"
    else
        log_info "Validation dataset DISABLED"
    fi

    if [[ "$DO_TEST" == "true" ]]; then
        cmd="$cmd --test '$TEST_DIR'"
        cmd="$cmd --test_out '$test_out'"
        log_info "Test dataset enabled"
    else
        log_info "Test dataset DISABLED"
    fi

    if [[ "$DO_GUTENBERG" == "true" ]]; then
        cmd="$cmd --gutenberg '$GUTENBERG_DIR'"
        cmd="$cmd --gutenberg_out '$gutenberg_out'"
        cmd="$cmd --gutenberg_info '$GUTENBERG_INFO'"
        log_info "Gutenberg dataset enabled"
    else
        log_info "Gutenberg dataset DISABLED"
    fi

    # Add validation directory for test/gutenberg reference if needed
    if [[ "$DO_TEST" == "true" || "$DO_GUTENBERG" == "true" ]]; then
        if [[ "$DO_VALID" == "false" ]]; then
            # If validation is disabled but test/gutenberg is enabled, provide validation directory
            cmd="$cmd --valid '$VALID_DIR'"
            log_info "Adding validation directory for test/gutenberg reference (validation processing disabled)"
        fi
    fi

    cmd="$cmd --target_words '$TARGET_WORDS'"
    cmd="$cmd --lang '$LANG'"

    # Only add order parameter if ORDER is not empty
    if [[ -n "$ORDER" ]]; then
        cmd="$cmd --order '$ORDER'"
        log_info "Using Markov order: $ORDER"
    else
        log_info "Using order 1 only (no higher order features)"
    fi

    cmd="$cmd --word_distance '$WORD_DISTANCE'"
    cmd="$cmd --chunk_size '$CHUNK_SIZE'"
    cmd="$cmd --threads '$THREADS'"
    cmd="$cmd --inference_threads '$INFERENCE_THREADS'"
    cmd="$cmd --create_references"
    cmd="$cmd --reference_percentage '$REFERENCE_PERCENTAGE'"
    cmd="$cmd --validation_reference_percentage '$VALIDATION_REFERENCE_PERCENTAGE'"
    cmd="$cmd --lowercase"

    # Add --no-ncd flag if NCD is disabled
    if [[ "$USE_NCD" == "false" ]]; then
        cmd="$cmd --no-ncd"
        log_info "NCD features disabled for faster processing"
    else
        log_info "NCD features enabled (slower processing)"
    fi

    # Run feature extraction with full output logging
    eval $cmd

    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log_success "Feature extraction completed!"
    else
        log_error " Feature extraction failed with exit code $exit_code"
        exit $exit_code
    fi

    return $exit_code
}

run_cleaning() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log_info "Starting data cleaning..."

    # Find the latest extraction files
    local latest_train=$(ls -t "$OUTPUT_DIR"/train_features_*.csv | head -1)
    local latest_valid=$(ls -t "$OUTPUT_DIR"/valid_features_*.csv | head -1)
    local latest_test=$(ls -t "$OUTPUT_DIR"/test_features_*.csv | head -1)
    local latest_gutenberg=$(ls -t "$OUTPUT_DIR"/gutenberg_features_*.csv | head -1)

    if [[ -z "$latest_train" || -z "$latest_valid" || -z "$latest_test" ]]; then
        log_error " Could not find all required feature files"
        return 1
    fi

    # Create cleaned versions
    local clean_train="$OUTPUT_DIR/train_features_cleaned_${timestamp}.csv"
    local clean_valid="$OUTPUT_DIR/valid_features_cleaned_${timestamp}.csv"
    local clean_test="$OUTPUT_DIR/test_features_cleaned_${timestamp}.csv"
    local clean_gutenberg="$OUTPUT_DIR/gutenberg_features_cleaned_${timestamp}.csv"

    log_info "Cleaning files..."

    # Run cleaning with logging
    python3 -u Feature_extraction/clean_csv.py \
        --text_info "$FILE_INFO" \
        --gutenberg_text_info "$GUTENBERG_INFO" \
        --input_train_csv "$latest_train" \
        --input_validation_csv "$latest_valid" \
        --input_test_csv "$latest_test" \
        --input_gutenberg_csv "$latest_gutenberg" \
        --output_train_csv "$clean_train" \
        --output_validation_csv "$clean_valid" \
        --output_test_csv "$clean_test" \
        --output_gutenberg_csv "$clean_gutenberg" \
        --lang "$LANG" \
        --target_words_json "$TARGET_WORDS"

    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log_success "Data cleaning completed!"
        log_info "Final cleaned files:"
        log_info "  Train: $clean_train"
        log_info "  Valid: $clean_valid"
        log_info "  Test: $clean_test"
        log_info "  Gutenberg: $clean_gutenberg"
    else
        log_error " Data cleaning failed with exit code $exit_code"
        return $exit_code
    fi

    return 0
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    log_info "Starting Feature Extraction Pipeline on IEETA HPC"
    log_info "==========================================================="

    create_directories

    # Run extraction
    if run_feature_extraction; then
        log_info "Feature extraction phase completed"
    else
        log_info "Feature extraction phase failed"
        exit 1
    fi

    # Run cleaning
    if run_cleaning; then
        log_info "Data cleaning phase completed"
    else
        log_info "Data cleaning phase failed"
        exit 1
    fi

    log_info "==========================================================="
    log_success "Feature extraction pipeline completed!"
    log_info "Check $OUTPUT_DIR for final feature files"
    log_info "Check logs/ for detailed execution logs"

    # Final resource summary
    log_info "Final memory usage: $(free -h | grep '^Mem:' | awk '{print $3"/"$2}')"
    log_info "Job duration: $SECONDS seconds"
}

# Run main function
main "$@"