#!/bin/bash
#SBATCH --job-name=run_all_binary_features
#SBATCH --output=logs/run_all_binary_features_%j.out
#SBATCH --error=logs/run_all_binary_features_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=64GB
#SBATCH --partition=cpu

# Consolidated Binary Model Pipeline - All Feature Types
# Runs all 4 feature types (Compression, Lexical-Structure, Readability, Distance Features)
# sequentially for both decades and centuries classification
# Optimized for SLURM HPC environments with comprehensive error handling

set -e  # Exit on any error

# Set up environment
cd /data/home/paulojnpinto02/TextDate/Feature-Benchmark
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2

# ==================== SLURM INFO ====================
echo "========================================================================"
echo "CONSOLIDATED BINARY MODEL PIPELINE - ALL FEATURE TYPES"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "========================================================================"

# ==================== CONFIGURATION ====================

# Feature type configurations (DROP_COLS for each feature type)
declare -A FEATURE_CONFIGS
FEATURE_CONFIGS[compression]="file_name year century special_character_ratio Avg_Word_Length Lexical_Richness Avg_Sentence_Length Punctuation_Density Syllable_Per_Word Digit_Ratio Flesch_Readability Stopword_Ratio by and the at in with a is to of as on an that for it was"
FEATURE_CONFIGS[lexical_structure]="file_name year century decade special_character_ratio Compression_Ratio_Order_1 NRC_Order_1 Entropy_Ratio_Order_1 Shannon_Entropy Flesch_Readability"
FEATURE_CONFIGS[readability]="file_name year decade special_character_ratio Compression_Ratio_Order_1 NRC_Order_1 Entropy_Ratio_Order_1 Shannon_Entropy Avg_Word_Length Lexical_Richness Avg_Sentence_Length Punctuation_Density Syllable_Per_Word Digit_Ratio Stopword_Ratio by and the at in with a is to of as on an that for it was"
FEATURE_CONFIGS[distance]="file_name year century decade special_character_ratio Compression_Ratio_Order_1 NRC_Order_1 Entropy_Ratio_Order_1 Shannon_Entropy Avg_Word_Length Lexical_Richness Avg_Sentence_Length Punctuation_Density Syllable_Per_Word Digit_Ratio Flesch_Readability Stopword_Ratio"

# Binary model configurations
BINARY_MODELS=("random_forest" "xgboost" "catboost" "random")

# Execution order
FEATURE_TYPES=("compression" "lexical_structure" "readability" "distance")
CLASSIFICATION_TARGETS=("decades" "centuries")

# Input and output directories
FEATURE_DIR="Extracted_features"
LOG_DIR="logs"

# Timing variables
OVERALL_START_TIME=$SECONDS
declare -A FEATURE_TIMINGS

# ==================== FUNCTIONS ====================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_phase() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
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

    log "✓ Validated $description file: $file ($(numfmt --to=iec $size))"
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

    log "✓ Validated $description directory: $dir"
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
        log "✓ Created directory: $dir"
    else
        log "✓ Directory exists: $dir"
    fi

    return 0
}

# Function to validate base model files exist
validate_base_models() {
    local feature_type="$1"
    local target="$2"
    local model_source_dir="Saved_models/$feature_type/$target"

    log "Validating base model files for $feature_type - $target..."

    # Check for required model files (excluding 'random' which doesn't need a file)
    local required_models=("random_forest" "xgboost" "catboost")
    local missing_models=()

    for model in "${required_models[@]}"; do
        local model_file="$model_source_dir/${model}_model.pkl"
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
        log "ERROR: Missing or empty base model files for $feature_type - $target:"
        for model in "${missing_models[@]}"; do
            log "  - $model"
        done
        return 1
    fi

    log "✓ All base model files validated successfully for $feature_type - $target"
    return 0
}

# Function to get feature type display name
get_feature_display_name() {
    local feature_type="$1"
    case "$feature_type" in
        "compression") echo "Compression Features" ;;
        "lexical_structure") echo "Lexical-Structure Features" ;;
        "readability") echo "Readability Features" ;;
        "distance") echo "Distance Features" ;;
        *) echo "Unknown Features" ;;
    esac
}

# Function to get proper target name for directory structure
get_target_name() {
    local target="$1"
    case "$target" in
        "decades") echo "decades" ;;
        "centuries") echo "centuries" ;;
        *) echo "$target" ;;
    esac
}

# Function to run binary testing phase
run_binary_testing() {
    local feature_type="$1"
    local target="$2"
    local test_file="$3"
    local test_description="$4"
    local output_suffix="$5"
    local feature_display_name="$6"

    local target_name=$(get_target_name "$target")
    local model_source_dir="Saved_models/$feature_type/$target_name"
    local output_dir="Saved_models_binary/$feature_type/$target_name"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    log_phase "BINARY TESTING: $feature_display_name - ${target^} Classification - $test_description"

    # Validate test file
    validate_file "$test_file" "$test_description features" || return 1

    # Ensure output directory exists
    ensure_directory "$output_dir" "Binary results output" || return 1

    log "Binary testing configuration:"
    log "  Feature type: $feature_display_name"
    log "  Target: $target"
    log "  Test file: $test_file"
    log "  Model source: $model_source_dir"
    log "  Output directory: $output_dir"
    log "  Drop columns: $drop_cols"

    # Build model paths for binary testing
    local model_paths=()
    model_paths+=("${model_source_dir}/random_forest_model.pkl")
    model_paths+=("${model_source_dir}/xgboost_model.pkl")
    model_paths+=("${model_source_dir}/catboost_model.pkl")
    model_paths+=("random")  # Random baseline

    log "  Models: ${model_paths[*]}"

    # Run binary model testing
    log "Starting binary model testing for $feature_display_name - $target - $test_description..."
    python Binary_model/binary_model_tester.py \
        --test_file "$test_file" \
        --target "$target" \
        --models ${model_paths[*]} \
        --drop_cols $drop_cols \
        --output_file "binary_results_${output_suffix}.csv" \
        --output_dir "$output_dir"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Binary testing for $feature_display_name - $target - $test_description failed with exit code $exit_code"
        return $exit_code
    fi

    log "✓ Binary testing for $feature_display_name - $target - $test_description completed successfully!"
    return 0
}

# Function to run complete binary workflow for a feature type and target
run_complete_binary_workflow() {
    local feature_type="$1"
    local target="$2"
    local feature_display_name=$(get_feature_display_name "$feature_type")
    local target_name=$(get_target_name "$target")

    local workflow_start_time=$SECONDS

    log_phase "STARTING COMPLETE BINARY WORKFLOW: $feature_display_name - ${target^} Classification"

    # Validate that base models exist for this feature type and target
    if ! validate_base_models "$feature_type" "$target_name"; then
        log "ERROR: Base models validation failed for $feature_display_name - $target, aborting workflow"
        log "NOTE: Base models must be trained first using the base model pipeline"
        return 1
    fi

    # Find latest feature files
    local valid_file
    local test_file
    local gutenberg_file

    valid_file=$(find_latest_file "valid_features_cleaned_*.csv") || return 1
    test_file=$(find_latest_file "test_features_cleaned_*.csv") || return 1
    gutenberg_file=$(find_latest_file "gutenberg_features_cleaned_*.csv") || return 1

    # Phase 1: Validation Data Testing
    if ! run_binary_testing "$feature_type" "$target" "$valid_file" "Validation Dataset Evaluation" "validation" "$feature_display_name"; then
        log "ERROR: Validation testing failed for $feature_display_name - $target, aborting workflow"
        return 1
    fi

    # Phase 2: Test Data Testing
    if ! run_binary_testing "$feature_type" "$target" "$test_file" "Test Dataset Evaluation" "test" "$feature_display_name"; then
        log "ERROR: Test evaluation failed for $feature_display_name - $target, aborting workflow"
        return 1
    fi

    # Phase 3: Gutenberg Data Testing
    if ! run_binary_testing "$feature_type" "$target" "$gutenberg_file" "Gutenberg Dataset Evaluation" "gutenberg" "$feature_display_name"; then
        log "ERROR: Gutenberg evaluation failed for $feature_display_name - $target, aborting workflow"
        return 1
    fi

    local workflow_duration=$((SECONDS - workflow_start_time))
    FEATURE_TIMINGS["${feature_type}_${target}"]=$workflow_duration

    log_phase "COMPLETED BINARY WORKFLOW: $feature_display_name - ${target^} Classification (${workflow_duration}s)"
    return 0
}

# Cleanup function
cleanup() {
    local total_duration=$((SECONDS - OVERALL_START_TIME))

    log_phase "FINAL SUMMARY"
    log "Job completed at $(date)"
    log "Final memory usage: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $3"/"$2}' || echo 'N/A')"
    log "Total job duration: ${total_duration}s ($(date -d@${total_duration} -u +%H:%M:%S))"

    log ""
    log "Individual workflow timings:"
    for feature_type in "${FEATURE_TYPES[@]}"; do
        for target in "${CLASSIFICATION_TARGETS[@]}"; do
            local key="${feature_type}_${target}"
            if [[ -n "${FEATURE_TIMINGS[$key]:-}" ]]; then
                local display_name=$(get_feature_display_name "$feature_type")
                local duration=${FEATURE_TIMINGS[$key]}
                log "  $display_name - ${target^}: ${duration}s ($(date -d@${duration} -u +%H:%M:%S))"
            fi
        done
    done

    log ""
    log "Binary model results saved in respective directories:"
    log "  Results: Saved_models_binary/[feature_type]/[target]/"
    log "  Logs: $LOG_DIR/"
    log ""
    log "NOTE: This pipeline uses pre-trained base models from:"
    log "  Base models: Saved_models/[feature_type]/[target]/"
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    log_phase "INITIALIZING CONSOLIDATED BINARY MODEL PIPELINE"

    # Create necessary directories
    ensure_directory "$LOG_DIR" "Log" || exit 1

    # Validate Python scripts exist
    validate_file "Binary_model/binary_model_tester.py" "Binary model tester script" || exit 1

    # Validate feature extraction directory
    validate_directory "$FEATURE_DIR" "Feature extraction" || exit 1

    log "Pipeline configuration:"
    log "  Feature types: ${FEATURE_TYPES[*]}"
    log "  Classification targets: ${CLASSIFICATION_TARGETS[*]}"
    log "  Binary models per workflow: ${BINARY_MODELS[*]}"
    log "  Total workflows to execute: $((${#FEATURE_TYPES[@]} * ${#CLASSIFICATION_TARGETS[@]}))"

    # Execute all workflows sequentially
    local workflow_count=0
    local total_workflows=$((${#FEATURE_TYPES[@]} * ${#CLASSIFICATION_TARGETS[@]}))

    for feature_type in "${FEATURE_TYPES[@]}"; do
        for target in "${CLASSIFICATION_TARGETS[@]}"; do
            workflow_count=$((workflow_count + 1))
            local feature_display_name=$(get_feature_display_name "$feature_type")

            log_phase "BINARY WORKFLOW $workflow_count/$total_workflows: $feature_display_name - ${target^} Classification"

            if ! run_complete_binary_workflow "$feature_type" "$target"; then
                log "ERROR: Binary workflow $workflow_count failed, aborting entire pipeline"
                exit 1
            fi

            # Memory status after each workflow
            log "Memory after workflow $workflow_count: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $3"/"$2}' || echo 'N/A')"
        done
    done

    log_phase "ALL BINARY WORKFLOWS COMPLETED SUCCESSFULLY!"
    log "Successfully completed all $total_workflows binary workflows across 4 feature types and 2 classification targets"
}

# Run main function
main "$@"