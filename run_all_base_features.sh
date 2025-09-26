#!/bin/bash
#SBATCH --job-name=run_all_base_features
#SBATCH --output=logs/run_all_base_features_%j.out
#SBATCH --error=logs/run_all_base_features_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=200GB
#SBATCH --partition=cpu

# Consolidated Base Model Pipeline - All Feature Types
# Runs all 4 feature types (Compression, Lexical-Structure, Readability, Distance Features)
# sequentially for both decades and centuries classification
# Optimized for SLURM HPC environments with comprehensive error handling

set -e  # Exit on any error

# Set up environment
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2

# ==================== SLURM INFO ====================
echo "========================================================================"
echo "CONSOLIDATED BASE MODEL PIPELINE - ALL FEATURE TYPES"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "========================================================================"

# ==================== CONFIGURATION ====================

# Model configuration
MODELS=("random_forest" "xgboost" "catboost" "svm" "gnb" "knn")
N_ESTIMATORS=1000

# Feature type configurations
declare -A FEATURE_CONFIGS
FEATURE_CONFIGS[compression]="file_name,year,century,special_character_ratio,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio,by,and,the,at,in,with,a,is,to,of,as,on,an,that,for,it,was"
FEATURE_CONFIGS[lexical_structure]="file_name,year,century,special_character_ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Flesch_Readability"
FEATURE_CONFIGS[readability]="file_name,year,century,special_character_ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Stopword_Ratio,by,and,the,at,in,with,a,is,to,of,as,on,an,that,for,it,was"
FEATURE_CONFIGS[distance]="file_name,year,century,special_character_ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio"

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
                log "  Found $model model: $(numfmt --to=iec $size)"
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

# Function to run model training
run_training() {
    local feature_type="$1"
    local target="$2"
    local feature_display_name="$3"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    local model_dir="Saved_models/$feature_type/$target"

    log_phase "TRAINING: $feature_display_name - ${target^} Classification"

    # Find latest feature files
    local train_file
    local val_file

    train_file=$(find_latest_file "train_features_cleaned_*.csv") || return 1
    val_file=$(find_latest_file "valid_features_cleaned_*.csv") || return 1

    # Validate input files
    validate_file "$train_file" "Training features" || return 1
    validate_file "$val_file" "Validation features" || return 1

    # Ensure output directory exists
    ensure_directory "$model_dir" "Model output" || return 1

    log "Training configuration:"
    log "  Feature type: $feature_display_name"
    log "  Target: $target"
    log "  Models: ${MODELS[*]}"
    log "  N-estimators: $N_ESTIMATORS"
    log "  Drop columns: $drop_cols"
    log "  Training file: $train_file"
    log "  Validation file: $val_file"
    log "  Output directory: $model_dir"

    # Run model training
    log "Starting model training for $feature_display_name - $target..."
    python3 -u Base_model/model_trainer.py \
        --train_file "$train_file" \
        --val_file "$val_file" \
        --target "$target" \
        --models ${MODELS[*]} \
        --n_estimators $N_ESTIMATORS \
        --drop_cols "$drop_cols" \
        --output_dir "$model_dir"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Model training for $feature_display_name - $target failed with exit code $exit_code"
        return $exit_code
    fi

    # Validate that models were created
    validate_models "$model_dir" || return 1

    log "Model training for $feature_display_name - $target completed successfully!"
    return 0
}

# Function to run model testing
run_testing() {
    local feature_type="$1"
    local target="$2"
    local test_file_pattern="$3"
    local test_description="$4"
    local feature_display_name="$5"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    local model_dir="Saved_models/$feature_type/$target"
    local result_dir="Saved_models_results/$feature_type/$target"

    log_phase "TESTING: $feature_display_name - ${target^} Classification - $test_description"

    # Find latest test file
    local latest_test_file
    latest_test_file=$(find_latest_file "$test_file_pattern") || return 1

    # Validate input file
    validate_file "$latest_test_file" "$test_description features" || return 1

    # Ensure result directory exists
    ensure_directory "$result_dir" "Results output" || return 1

    # Build model paths array
    local model_paths=()
    for model in "${MODELS[@]}"; do
        model_paths+=("$model_dir/${model}_model.pkl")
    done

    log "Testing configuration:"
    log "  Feature type: $feature_display_name"
    log "  Target: $target"
    log "  Test file: $latest_test_file"
    log "  Models: ${model_paths[*]}"
    log "  Drop columns: $drop_cols"
    log "  Output directory: $result_dir"

    # Run model testing
    log "Starting model testing for $feature_display_name - $target - $test_description..."
    python3 -u Base_model/model_tester.py \
        --test_file "$latest_test_file" \
        --target "$target" \
        --models ${model_paths[*]} \
        --drop_cols "$drop_cols" \
        --output_dir "$result_dir"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Model testing for $feature_display_name - $target - $test_description failed with exit code $exit_code"
        return $exit_code
    fi

    log "Model testing for $feature_display_name - $target - $test_description completed successfully!"
    return 0
}

# Function to run complete workflow for a feature type and target
run_complete_workflow() {
    local feature_type="$1"
    local target="$2"
    local feature_display_name=$(get_feature_display_name "$feature_type")

    local workflow_start_time=$SECONDS

    log_phase "STARTING COMPLETE WORKFLOW: $feature_display_name - ${target^} Classification"

    # Phase 1: Training
    if ! run_training "$feature_type" "$target" "$feature_display_name"; then
        log "ERROR: Training failed for $feature_display_name - $target, aborting workflow"
        return 1
    fi

    # Phase 2: Test Dataset Evaluation
    if ! run_testing "$feature_type" "$target" "test_features_cleaned_*.csv" "Test Dataset Evaluation" "$feature_display_name"; then
        log "ERROR: Test evaluation failed for $feature_display_name - $target, aborting workflow"
        return 1
    fi

    # Phase 3: Gutenberg Dataset Evaluation
    if ! run_testing "$feature_type" "$target" "gutenberg_features_cleaned_*.csv" "Gutenberg Dataset Evaluation" "$feature_display_name"; then
        log "ERROR: Gutenberg evaluation failed for $feature_display_name - $target, aborting workflow"
        return 1
    fi

    local workflow_duration=$((SECONDS - workflow_start_time))
    FEATURE_TIMINGS["${feature_type}_${target}"]=$workflow_duration

    log_phase "COMPLETED WORKFLOW: $feature_display_name - ${target^} Classification (${workflow_duration}s)"
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
    log "Models and results saved in respective directories:"
    log "  Models: Saved_models/[feature_type]/[target]/"
    log "  Results: Saved_models_results/[feature_type]/[target]/"
    log "  Logs: $LOG_DIR/"
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    log_phase "INITIALIZING CONSOLIDATED BASE MODEL PIPELINE"

    # Create necessary directories
    ensure_directory "$LOG_DIR" "Log" || exit 1

    # Validate Python scripts exist
    validate_file "Base_model/model_trainer.py" "Model trainer script" || exit 1
    validate_file "Base_model/model_tester.py" "Model tester script" || exit 1

    # Validate feature extraction directory
    validate_directory "$FEATURE_DIR" "Feature extraction" || exit 1

    log "Pipeline configuration:"
    log "  Feature types: ${FEATURE_TYPES[*]}"
    log "  Classification targets: ${CLASSIFICATION_TARGETS[*]}"
    log "  Models per workflow: ${MODELS[*]}"
    log "  Total workflows to execute: $((${#FEATURE_TYPES[@]} * ${#CLASSIFICATION_TARGETS[@]}))"

    # Execute all workflows sequentially
    local workflow_count=0
    local total_workflows=$((${#FEATURE_TYPES[@]} * ${#CLASSIFICATION_TARGETS[@]}))

    for feature_type in "${FEATURE_TYPES[@]}"; do
        for target in "${CLASSIFICATION_TARGETS[@]}"; do
            workflow_count=$((workflow_count + 1))
            local feature_display_name=$(get_feature_display_name "$feature_type")

            log_phase "WORKFLOW $workflow_count/$total_workflows: $feature_display_name - ${target^} Classification"

            if ! run_complete_workflow "$feature_type" "$target"; then
                log "ERROR: Workflow $workflow_count failed, aborting entire pipeline"
                exit 1
            fi

            # Memory status after each workflow
            log "Memory after workflow $workflow_count: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $3"/"$2}' || echo 'N/A')"
        done
    done

    log_phase "ALL WORKFLOWS COMPLETED SUCCESSFULLY!"
    log "Successfully completed all $total_workflows workflows across 4 feature types and 2 classification targets"
}

# Run main function
main "$@"