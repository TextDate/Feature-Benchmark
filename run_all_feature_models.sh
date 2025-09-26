#!/bin/bash
#SBATCH --job-name=all_feature_models
#SBATCH --output=logs/all_feature_models_%j.out
#SBATCH --error=logs/all_feature_models_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=200GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00

# Master SLURM Script for All Feature-Specific Base and Binary Models
# Runs all 4 feature types (Compression, Lexical-Structure, Readability, Distance)
# For both decades and centuries classification with base and binary models
# Optimized for SLURM HPC environments with comprehensive error handling

set -e  # Exit on any error

# Set up environment
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2

# ==================== SLURM INFO ====================
echo "======================================================================="
echo "MASTER FEATURE MODEL PIPELINE - ALL FEATURE TYPES"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "======================================================================="

# ==================== GLOBAL CONFIGURATION ====================

# Model configuration
MODELS=("random_forest" "xgboost" "catboost" "svm" "gnb" "knn")
N_ESTIMATORS=1000

# Feature directories
FEATURE_DIR="Extracted_features"
LOG_DIR="logs"

# Feature type configurations (DROP_COLS for each feature type)
declare -A FEATURE_CONFIGS
FEATURE_CONFIGS[compression]="file_name,year,decade,century,special_character_ratio,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio,by,and,the,at,in,with,a,is,to,of,as,on,an,that,for,it,was"
FEATURE_CONFIGS[lexical_structure]="file_name,year,decade,century,special_character_ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Flesch_Readability"
FEATURE_CONFIGS[readability]="file_name,year,decade,century,special_character_ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Stopword_Ratio,by,and,the,at,in,with,a,is,to,of,as,on,an,that,for,it,was"
FEATURE_CONFIGS[distance]="file_name,year,decade,century,special_character_ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio"

# Execution order and progress tracking
TOTAL_PHASES=32  # 4 features × 2 time_scales × 2 model_types × 2 workflows (training+testing vs testing only)
CURRENT_PHASE=0
OVERALL_START_TIME=$(date +%s)

# ==================== UTILITY FUNCTIONS ====================

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local elapsed=$(($(date +%s) - OVERALL_START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    echo "[$timestamp] [+${hours}h${minutes}m${seconds}s] $1"
}

progress_update() {
    CURRENT_PHASE=$((CURRENT_PHASE + 1))
    local percentage=$((CURRENT_PHASE * 100 / TOTAL_PHASES))
    log "========== PROGRESS: $CURRENT_PHASE/$TOTAL_PHASES ($percentage%) =========="
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

    log " Validated $description file: $file ($(numfmt --to=iec $size))"
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
    local feature_type="$2"
    local missing_models=()

    log "Validating trained model files for $feature_type..."

    for model in "${MODELS[@]}"; do
        local model_file="$model_dir/${model}_model.pkl"
        if [[ ! -f "$model_file" ]]; then
            missing_models+=("$model")
        else
            local size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "0")
            if [[ "$size" -eq 0 ]]; then
                missing_models+=("$model (empty)")
            else
                log "   Found $model model: $(numfmt --to=iec $size)"
            fi
        fi
    done

    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log "ERROR: Missing or empty model files for $feature_type:"
        for model in "${missing_models[@]}"; do
            log "  - $model"
        done
        return 1
    fi

    log "All model files validated successfully for $feature_type"
    return 0
}

# Function to run model training
run_base_training() {
    local feature_type="$1"
    local time_scale="$2"  # decades or centuries
    local target="$3"      # decade or century

    local model_dir="Saved_models/$feature_type/$time_scale"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    progress_update
    log "============================================"
    log "PHASE $CURRENT_PHASE: Base Model Training"
    log "Feature Type: $feature_type"
    log "Time Scale: $time_scale"
    log "Target: $target"
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
    ensure_directory "$model_dir" "Model output" || return 1

    log "Training configuration:"
    log "  Target: $target"
    log "  Feature type: $feature_type"
    log "  Models: ${MODELS[*]}"
    log "  N-estimators: $N_ESTIMATORS"
    log "  Drop columns: $drop_cols"
    log "  Training file: $train_file"
    log "  Validation file: $val_file"
    log "  Output directory: $model_dir"

    # Prepare command arguments
    local cmd_args=(
        "--train_file" "$train_file"
        "--val_file" "$val_file"
        "--target" "$target"
        "--models" "${MODELS[@]}"
        "--n_estimators" "$N_ESTIMATORS"
        "--drop_cols" "$drop_cols"
        "--output_dir" "$model_dir"
    )

    # Add --use_centuries flag for century targets
    if [[ "$target" == "century" ]]; then
        cmd_args+=("--use_centuries")
    fi

    # Run model training
    log "Starting model training for $feature_type - $time_scale..."
    python3 -u Base_model/model_trainer.py "${cmd_args[@]}"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Model training failed with exit code $exit_code"
        return $exit_code
    fi

    # Validate that models were created
    validate_models "$model_dir" "$feature_type" || return 1

    log " Base model training completed for $feature_type - $time_scale!"
    return 0
}

# Function to run base model testing
run_base_testing() {
    local feature_type="$1"
    local time_scale="$2"  # decades or centuries
    local target="$3"      # decade or century
    local test_file_pattern="$4"
    local test_description="$5"

    local model_dir="Saved_models/$feature_type/$time_scale"
    local result_dir="Saved_models_results/$feature_type/$time_scale"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    log "--------------------------------------------"
    log "Base Model Testing: $test_description"
    log "Feature Type: $feature_type - $time_scale"
    log "--------------------------------------------"

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
    log "  Target: $target"
    log "  Feature type: $feature_type"
    log "  Test file: $latest_test_file"
    log "  Models: ${model_paths[*]}"
    log "  Drop columns: $drop_cols"
    log "  Output directory: $result_dir"

    # Prepare command arguments
    local cmd_args=(
        "--test_file" "$latest_test_file"
        "--target" "$target"
        "--models" "${model_paths[@]}"
        "--drop_cols" "$drop_cols"
        "--output_dir" "$result_dir"
    )

    # Add --use_centuries flag for century targets
    if [[ "$target" == "century" ]]; then
        cmd_args+=("--use_centuries")
    fi

    # Run model testing
    log "Starting model testing for $test_description - $feature_type - $time_scale..."
    python3 -u Base_model/model_tester.py "${cmd_args[@]}"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Model testing for $test_description failed with exit code $exit_code"
        return $exit_code
    fi

    log " Base model testing completed for $test_description - $feature_type - $time_scale!"
    return 0
}

# Function to run binary model testing
run_binary_testing() {
    local feature_type="$1"
    local time_scale="$2"  # decades or centuries
    local target="$3"      # decade or century

    local model_source_dir="Saved_models/$feature_type/$time_scale"
    local output_dir="Saved_models_binary/$feature_type/$time_scale"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    progress_update
    log "============================================"
    log "PHASE $CURRENT_PHASE: Binary Model Testing"
    log "Feature Type: $feature_type"
    log "Time Scale: $time_scale"
    log "Target: $target"
    log "============================================"

    # Find latest timestamped feature files
    log "Finding latest feature files..."
    local valid_file
    local test_file
    local gutenberg_file

    valid_file=$(find_latest_file "valid_features_cleaned_*.csv") || return 1
    test_file=$(find_latest_file "test_features_cleaned_*.csv") || return 1
    gutenberg_file=$(find_latest_file "gutenberg_features_cleaned_*.csv") || return 1

    # Validate all input files exist
    log "Validating input files..."
    validate_file "$valid_file" "Validation features" || return 1
    validate_file "$test_file" "Test features" || return 1
    validate_file "$gutenberg_file" "Gutenberg features" || return 1

    # Validate model files exist
    log "Validating model files..."
    validate_file "${model_source_dir}/random_forest_model.pkl" "Random Forest model" || return 1
    validate_file "${model_source_dir}/xgboost_model.pkl" "XGBoost model" || return 1
    validate_file "${model_source_dir}/catboost_model.pkl" "CatBoost model" || return 1

    # Create output directory
    ensure_directory "$output_dir" "Binary model output" || return 1

    # Run binary testing for each dataset
    local datasets=("$valid_file:Validation Data:validation" "$test_file:Test Data:test" "$gutenberg_file:Gutenberg Data:gutenberg")

    for dataset in "${datasets[@]}"; do
        IFS=':' read -r dataset_file dataset_description output_suffix <<< "$dataset"

        log "--------------------------------------------"
        log "Running Binary Model Testing for $dataset_description"
        log "Feature Type: $feature_type - $time_scale"
        log "--------------------------------------------"

        python Binary_model/binary_model_tester.py \
            --test_file "$dataset_file" \
            --target "$target" \
            --models "${model_source_dir}/random_forest_model.pkl" "${model_source_dir}/xgboost_model.pkl" "${model_source_dir}/catboost_model.pkl" "random" \
            --drop_cols $drop_cols \
            --output_file "binary_results_${output_suffix}.csv" \
            --output_dir "$output_dir"

        local exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            log "ERROR: Binary model testing for $dataset_description failed with exit code $exit_code"
            return $exit_code
        fi

        log " $dataset_description binary testing completed successfully"
    done

    log " Binary model testing completed for $feature_type - $time_scale!"
    return 0
}

# Function to run complete workflow for a feature type and time scale
run_feature_workflow() {
    local feature_type="$1"
    local time_scale="$2"
    local target="$3"

    log ""
    log "######################################################################"
    log "STARTING WORKFLOW: $feature_type - $time_scale ($target)"
    log "######################################################################"

    # Phase 1: Base Model Training
    if ! run_base_training "$feature_type" "$time_scale" "$target"; then
        log "ERROR: Base model training failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    # Phase 2: Base Model Testing (Test Dataset)
    if ! run_base_testing "$feature_type" "$time_scale" "$target" "test_features_cleaned_*.csv" "Test Dataset Evaluation"; then
        log "ERROR: Base model test evaluation failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    # Phase 3: Base Model Testing (Gutenberg Dataset)
    if ! run_base_testing "$feature_type" "$time_scale" "$target" "gutenberg_features_cleaned_*.csv" "Gutenberg Dataset Evaluation"; then
        log "ERROR: Base model Gutenberg evaluation failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    # Phase 4: Binary Model Testing (All datasets)
    if ! run_binary_testing "$feature_type" "$time_scale" "$target"; then
        log "ERROR: Binary model testing failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    log "######################################################################"
    log "COMPLETED WORKFLOW: $feature_type - $time_scale ($target)"
    log "######################################################################"
    log ""

    return 0
}

cleanup() {
    local end_time=$(date +%s)
    local total_duration=$((end_time - OVERALL_START_TIME))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))

    log "======================================================================="
    log "JOB COMPLETION SUMMARY"
    log "======================================================================="
    log "Job completed at $(date)"
    log "Final memory usage: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $3"/"$2}' || echo 'N/A')"
    log "Total job duration: ${hours}h ${minutes}m ${seconds}s"
    log "Phases completed: $CURRENT_PHASE/$TOTAL_PHASES"
    log "======================================================================="
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    log "Starting Master Feature Model Pipeline - ALL FEATURE TYPES"
    log "======================================================================="

    # Create necessary directories
    ensure_directory "$LOG_DIR" "Log" || exit 1

    # Validate Python scripts exist
    validate_file "Base_model/model_trainer.py" "Model trainer script" || exit 1
    validate_file "Base_model/model_tester.py" "Model tester script" || exit 1
    validate_file "Binary_model/binary_model_tester.py" "Binary model tester script" || exit 1

    # Validate feature extraction directory
    if [[ ! -d "$FEATURE_DIR" ]]; then
        log "ERROR: Feature extraction directory not found: $FEATURE_DIR"
        exit 1
    fi

    log "Validated feature extraction directory: $FEATURE_DIR"

    # Define execution order: Feature Type -> Time Scale -> Target
    # Order: Compression, Lexical-Structure, Readability, Distance
    # Each with decades then centuries
    local workflows=(
        "compression:decades:decade"
        "compression:centuries:century"
        "lexical_structure:decades:decade"
        "lexical_structure:centuries:century"
        "readability:decades:decade"
        "readability:centuries:century"
        "distance:decades:decade"
        "distance:centuries:century"
    )

    log "Execution order planned:"
    for i in "${!workflows[@]}"; do
        IFS=':' read -r feature_type time_scale target <<< "${workflows[$i]}"
        log "  $((i+1)). $feature_type - $time_scale ($target)"
    done

    # Execute all workflows sequentially
    for workflow in "${workflows[@]}"; do
        IFS=':' read -r feature_type time_scale target <<< "$workflow"

        if ! run_feature_workflow "$feature_type" "$time_scale" "$target"; then
            log "ERROR: Workflow failed for $feature_type - $time_scale, aborting entire pipeline"
            exit 1
        fi
    done

    log "======================================================================="
    log "ALL FEATURE MODEL PIPELINES COMPLETED SUCCESSFULLY"
    log "======================================================================="
    log ""
    log "Summary of completed workflows:"
    for workflow in "${workflows[@]}"; do
        IFS=':' read -r feature_type time_scale target <<< "$workflow"
        log "   $feature_type - $time_scale ($target)"
        log "    - Base models: Saved_models/$feature_type/$time_scale/"
        log "    - Base results: Saved_models_results/$feature_type/$time_scale/"
        log "    - Binary results: Saved_models_binary/$feature_type/$time_scale/"
    done
    log ""
    log "Check $LOG_DIR for detailed execution logs"
    log "======================================================================="
}

# Run main function
main "$@"