#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --output=logs/feature_extraction_%j.out
#SBATCH --error=logs/feature_extraction_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=230GB
#SBATCH --partition=cpu

# Enhanced Feature Extraction for IEETA HPC (SLURM)
# This script runs feature extraction with NRC, NCD, and reference models

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
ORDER=""  # Set to empty string "" for order 1 only, or specify order (e.g., 2)
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

# ==================== FUNCTIONS ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

create_directories() {
    log "Creating output directories..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "logs"
}

cleanup() {
    log "Cleaning up..."
    log "Job completed at $(date)"
}

run_feature_extraction() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log "Starting feature extraction..."
    log "Timestamp: $timestamp"
    log "Using $THREADS CPU cores for training/validation phases"
    log "Using $INFERENCE_THREADS CPU cores for test/gutenberg phases"

    # Output files with timestamp
    local train_out="$OUTPUT_DIR/train_features_${timestamp}.csv"
    local valid_out="$OUTPUT_DIR/valid_features_${timestamp}.csv"
    local test_out="$OUTPUT_DIR/test_features_${timestamp}.csv"
    local gutenberg_out="$OUTPUT_DIR/gutenberg_features_${timestamp}.csv"

    log "Output files:"
    log "  Train: $train_out"
    log "  Valid: $valid_out"
    log "  Test: $test_out"
    log "  Gutenberg: $gutenberg_out"

    # Build command with conditional order parameter
    local cmd="python3 -u Feature_extraction/feature_extractor.py"
    cmd="$cmd --file_info '$FILE_INFO'"

    # Conditionally add dataset directories based on flags
    if [[ "$DO_TRAIN" == "true" ]]; then
        cmd="$cmd --train '$TRAIN_DIR'"
        cmd="$cmd --train_out '$train_out'"
        log "Training dataset enabled"
    else
        log "Training dataset DISABLED"
    fi

    if [[ "$DO_VALID" == "true" ]]; then
        cmd="$cmd --valid '$VALID_DIR'"
        cmd="$cmd --valid_out '$valid_out'"
        log "Validation dataset enabled"
    else
        log "Validation dataset DISABLED"
    fi

    if [[ "$DO_TEST" == "true" ]]; then
        cmd="$cmd --test '$TEST_DIR'"
        cmd="$cmd --test_out '$test_out'"
        log "Test dataset enabled"
    else
        log "Test dataset DISABLED"
    fi

    if [[ "$DO_GUTENBERG" == "true" ]]; then
        cmd="$cmd --gutenberg '$GUTENBERG_DIR'"
        cmd="$cmd --gutenberg_out '$gutenberg_out'"
        cmd="$cmd --gutenberg_info '$GUTENBERG_INFO'"
        log "Gutenberg dataset enabled"
    else
        log "Gutenberg dataset DISABLED"
    fi

    # Add validation directory for test/gutenberg reference if needed
    if [[ "$DO_TEST" == "true" || "$DO_GUTENBERG" == "true" ]]; then
        if [[ "$DO_VALID" == "false" ]]; then
            # If validation is disabled but test/gutenberg is enabled, provide validation directory
            cmd="$cmd --valid '$VALID_DIR'"
            log "Adding validation directory for test/gutenberg reference (validation processing disabled)"
        fi
    fi

    cmd="$cmd --target_words '$TARGET_WORDS'"
    cmd="$cmd --lang '$LANG'"

    # Only add order parameter if ORDER is not empty
    if [[ -n "$ORDER" ]]; then
        cmd="$cmd --order '$ORDER'"
        log "Using Markov order: $ORDER"
    else
        log "Using order 1 only (no higher order features)"
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
        log "NCD features disabled for faster processing"
    else
        log "NCD features enabled (slower processing)"
    fi

    # Run feature extraction with full output logging
    eval $cmd

    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log "Feature extraction completed successfully!"
    else
        log "ERROR: Feature extraction failed with exit code $exit_code"
        exit $exit_code
    fi


log "==================== SLURM JOB INFO ===================="
log "SLURM Job Information - $(date)"
log "Job ID: $SLURM_JOB_ID"
log "Job Name: $SLURM_JOB_NAME"
log "Node: $SLURM_NODELIST"
log "CPUs: $SLURM_CPUS_PER_TASK"
log "Memory: ${SLURM_MEM_PER_NODE}MB"
log "Partition: $SLURM_JOB_PARTITION"
log ""
log "Feature Extraction Parameters:"
log "File Info: $FILE_INFO"
log "Target Words: $TARGET_WORDS"
log "Language: $LANG"
log "Markov Order: ${ORDER:-1 only}"
log "Chunk Size: $CHUNK_SIZE"
log "Training/Validation Threads: $THREADS"
log "Test/Gutenberg Threads: $INFERENCE_THREADS"
log "Word Distance: $WORD_DISTANCE"
log "Reference Percentage: $REFERENCE_PERCENTAGE"
log "Validation Reference Percentage: $VALIDATION_REFERENCE_PERCENTAGE"
log "Use NCD: $USE_NCD"
log "Lowercase: Yes"
log "Create References: Yes"
log ""
log "Dataset Processing Status:"
log "Train: $DO_TRAIN"
log "Valid: $DO_VALID"
log "Test: $DO_TEST"
log "Gutenberg: $DO_GUTENBERG"
log ""
log "Dataset Directories:"
log "Train: $TRAIN_DIR"
log "Valid: $VALID_DIR"
log "Test: $TEST_DIR"
log "Gutenberg: $GUTENBERG_DIR"
log ""
log "Output Files:"
log "Train: $train_out"
log "Valid: $valid_out"
log "Test: $test_out"
log "Gutenberg: $gutenberg_out"
log ""
log "Features Extracted:"
log "- Compression ratios (order 1$([ -n "$ORDER" ] && echo " and $ORDER" || echo ""))"
log "- NRC with external reference models"
log "- NCD (Normalized Compression Distance)"
log "- Entropy ratios"
log "- Linguistic features (readability, word lengths, etc.)"
log "- Target word distance features"
log ""
log "Reference Strategy:"
log "- Train/Valid: Use 1% of own files as reference"
log "- Test/Gutenberg: Use entire validation dataset as reference"
log ""
log "Start Time: $(date)"
log "=========================================="

    return $exit_code
}

run_cleaning() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')

    log "Starting data cleaning..."

    # Find the latest extraction files
    local latest_train=$(ls -t "$OUTPUT_DIR"/train_features_*.csv | head -1)
    local latest_valid=$(ls -t "$OUTPUT_DIR"/valid_features_*.csv | head -1)
    local latest_test=$(ls -t "$OUTPUT_DIR"/test_features_*.csv | head -1)
    local latest_gutenberg=$(ls -t "$OUTPUT_DIR"/gutenberg_features_*.csv | head -1)

    if [[ -z "$latest_train" || -z "$latest_valid" || -z "$latest_test" ]]; then
        log "ERROR: Could not find all required feature files"
        return 1
    fi

    # Create cleaned versions
    local clean_train="$OUTPUT_DIR/train_features_cleaned_${timestamp}.csv"
    local clean_valid="$OUTPUT_DIR/valid_features_cleaned_${timestamp}.csv"
    local clean_test="$OUTPUT_DIR/test_features_cleaned_${timestamp}.csv"
    local clean_gutenberg="$OUTPUT_DIR/gutenberg_features_cleaned_${timestamp}.csv"

    log "Cleaning files..."

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
        log "Data cleaning completed successfully!"
        log "Final cleaned files:"
        log "  Train: $clean_train"
        log "  Valid: $clean_valid"
        log "  Test: $clean_test"
        log "  Gutenberg: $clean_gutenberg"
    else
        log "ERROR: Data cleaning failed with exit code $exit_code"
        return $exit_code
    fi

    return 0
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    log "Starting Feature Extraction Pipeline on IEETA HPC"
    log "==========================================================="

    create_directories

    # Run extraction
    if run_feature_extraction; then
        log "Feature extraction phase completed"
    else
        log "Feature extraction phase failed"
        exit 1
    fi

    # Run cleaning
    if run_cleaning; then
        log "Data cleaning phase completed"
    else
        log "Data cleaning phase failed"
        exit 1
    fi

    log "==========================================================="
    log "Feature extraction pipeline completed successfully!"
    log "Check $OUTPUT_DIR for final feature files"
    log "Check logs/ for detailed execution logs"

    # Final resource summary
    log "Final memory usage: $(free -h | grep '^Mem:' | awk '{print $3"/"$2}')"
    log "Job duration: $SECONDS seconds"
}

# Run main function
main "$@"