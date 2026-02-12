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

# Train and test all models (base + binary) for all feature subsets
# Runs: compression, lexical_structure, readability, distance, neologism, final_model, optimal
# Both decades and centuries classification
#
# Usage:
#   sbatch run_all_feature_models.sh
#   sbatch run_all_feature_models.sh -f compression

set -e  # Exit on any error

# ==================== ENVIRONMENT SETUP ====================
source ../../virtual-venv/bin/activate
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Create logs directory if it doesn't exist
mkdir -p logs

# ==================== SLURM INFO ====================
echo "======================================================================="
echo "FEATURE MODEL PIPELINE"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK | Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start: $(date)"
echo "======================================================================="

# ==================== GLOBAL CONFIGURATION ====================

# Model configuration
MODELS=("random_forest" "xgboost" "catboost" "svm" "gnb" "knn")
N_ESTIMATORS=1000

# Binary model configuration
TOP_K_DECADES=10    # Top-K for decade classification (43 classes)
TOP_K_CENTURIES=1   # Top-K for century classification (5 classes)

# Feature directories
FEATURE_DIR="Extracted_features"
LOG_DIR="logs"

# Feature configurations (columns to drop - everything else is used)
declare -A FEATURE_CONFIGS

FEATURE_CONFIGS[compression]="file_name,year,decade,century,Special_Character_Ratio,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio,at,and,by,the,for,a,of,with,to,as,on,in,an,that,it,is,was,contains_early_modern_words,contains_industrial_words,contains_late_industrial_words,contains_early_20th_words,contains_post_war_words,contains_late_modern_words,contains_digital_dawn_words,contains_digital_native_words,contains_modern_vocabulary,contains_historical_vocabulary,vocabulary_modernity_score"

FEATURE_CONFIGS[lexical_structure]="file_name,year,decade,century,Special_Character_Ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Compression_Ratio_Order_2,NRC_Order_2,Entropy_Ratio_Order_2,Flesch_Readability,Stopword_Ratio,at,and,by,the,for,a,of,with,to,as,on,in,an,that,it,is,was,contains_early_modern_words,contains_industrial_words,contains_late_industrial_words,contains_early_20th_words,contains_post_war_words,contains_late_modern_words,contains_digital_dawn_words,contains_digital_native_words,contains_modern_vocabulary,contains_historical_vocabulary,vocabulary_modernity_score"

FEATURE_CONFIGS[readability]="file_name,year,decade,century,Special_Character_Ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Compression_Ratio_Order_2,NRC_Order_2,Entropy_Ratio_Order_2,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,at,and,by,the,for,a,of,with,to,as,on,in,an,that,it,is,was,contains_early_modern_words,contains_industrial_words,contains_late_industrial_words,contains_early_20th_words,contains_post_war_words,contains_late_modern_words,contains_digital_dawn_words,contains_digital_native_words,contains_modern_vocabulary,contains_historical_vocabulary,vocabulary_modernity_score"

FEATURE_CONFIGS[distance]="file_name,year,decade,century,Special_Character_Ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Compression_Ratio_Order_2,NRC_Order_2,Entropy_Ratio_Order_2,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio,contains_early_modern_words,contains_industrial_words,contains_late_industrial_words,contains_early_20th_words,contains_post_war_words,contains_late_modern_words,contains_digital_dawn_words,contains_digital_native_words,contains_modern_vocabulary,contains_historical_vocabulary,vocabulary_modernity_score"

FEATURE_CONFIGS[neologism]="file_name,year,decade,century,Special_Character_Ratio,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Compression_Ratio_Order_2,NRC_Order_2,Entropy_Ratio_Order_2,Avg_Word_Length,Lexical_Richness,Avg_Sentence_Length,Punctuation_Density,Syllable_Per_Word,Digit_Ratio,Flesch_Readability,Stopword_Ratio,at,and,by,the,for,a,of,with,to,as,on,in,an,that,it,is,was"

FEATURE_CONFIGS[final_model]="file_name,year,decade,century,Special_Character_Ratio"

FEATURE_CONFIGS[optimal]="file_name,year,decade,century,Compression_Ratio_Order_1,NRC_Order_1,Entropy_Ratio_Order_1,Shannon_Entropy,Avg_Word_Length,Compression_Ratio_Order_2,Entropy_Ratio_Order_2,Flesch_Readability,Stopword_Ratio,of,the,a,and,in,with,to,is,at,an,that,for,was,it,as,contains_early_modern_words,contains_industrial_words,contains_early_20th_words,contains_post_war_words,contains_late_modern_words,contains_digital_dawn_words,contains_digital_native_words,contains_modern_vocabulary,contains_historical_vocabulary,vocabulary_modernity_score"

# Execution order and progress tracking
TOTAL_PHASES=28  # 7 feature types × 4 workflows (decades/centuries × base/binary)
CURRENT_PHASE=0
OVERALL_START_TIME=$(date +%s)

# ==================== UTILITY FUNCTIONS ====================

log_info() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local elapsed=$(($(date +%s) - OVERALL_START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    echo "[$timestamp] [+${hours}h${minutes}m${seconds}s] INFO: $*"
}

log_error() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] ERROR: $*" >&2
}

log_success() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] SUCCESS: $*"
}

progress_update() {
    CURRENT_PHASE=$((CURRENT_PHASE + 1))
    local percentage=$((CURRENT_PHASE * 100 / TOTAL_PHASES))
    log_info "========== PROGRESS: $CURRENT_PHASE/$TOTAL_PHASES ($percentage%) =========="
}

# Function to find the latest timestamped file
find_latest_file() {
    local pattern="$1"
    local latest_file

    latest_file=$(find "$FEATURE_DIR" -name "$pattern" -type f 2>/dev/null | sort -V | tail -1)

    if [[ -z "$latest_file" ]]; then
        log_error " No files found matching pattern: $pattern"
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
        log_error " $description file not found: $file"
        return 1
    fi

    if [[ ! -r "$file" ]]; then
        log_error " $description file not readable: $file"
        return 1
    fi

    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
    if [[ "$size" -eq 0 ]]; then
        log_error " $description file is empty: $file"
        return 1
    fi

    log_info " Validated $description file: $file ($(numfmt --to=iec $size))"
    return 0
}

# Function to ensure directory exists
ensure_directory() {
    local dir="$1"
    local description="$2"

    if [[ ! -d "$dir" ]]; then
        log_info "Creating $description directory: $dir"
        mkdir -p "$dir" || {
            log_error " Failed to create $description directory: $dir"
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

    log_info "Validating trained model files for $feature_type..."

    for model in "${MODELS[@]}"; do
        local model_file="$model_dir/${model}_model.pkl"
        if [[ ! -f "$model_file" ]]; then
            missing_models+=("$model")
        else
            local size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "0")
            if [[ "$size" -eq 0 ]]; then
                missing_models+=("$model (empty)")
            else
                log_info "   Found $model model: $(numfmt --to=iec $size)"
            fi
        fi
    done

    if [[ ${#missing_models[@]} -gt 0 ]]; then
        log_error " Missing or empty model files for $feature_type:"
        for model in "${missing_models[@]}"; do
            log_info "  - $model"
        done
        return 1
    fi

    log_info "All model files validated successfully for $feature_type"
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
    log_info "============================================"
    log_info "PHASE $CURRENT_PHASE: Base Model Training"
    log_info "Feature Type: $feature_type"
    log_info "Time Scale: $time_scale"
    log_info "Target: $target"
    log_info "============================================"

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

    log_info "Training configuration:"
    log_info "  Target: $target"
    log_info "  Feature type: $feature_type"
    log_info "  Models: ${MODELS[*]}"
    log_info "  N-estimators: $N_ESTIMATORS"
    log_info "  Drop columns: $drop_cols"
    log_info "  Training file: $train_file"
    log_info "  Validation file: $val_file"
    log_info "  Output directory: $model_dir"

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
    log_info "Starting model training for $feature_type - $time_scale..."
    python3 -u Base_model/model_trainer.py train "${cmd_args[@]}"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error " Model training failed with exit code $exit_code"
        return $exit_code
    fi

    # Validate that models were created
    validate_models "$model_dir" "$feature_type" || return 1

    log_success "Base model training completed for $feature_type - $time_scale!"
    return 0
}

# Function to run base model testing
run_base_testing() {
    local feature_type="$1"
    local time_scale="$2"  # decades or centuries
    local target="$3"      # decade or century
    local test_file_pattern="$4"
    local test_description="$5"
    local dataset_suffix="$6"  # dataset identifier (test, gutenberg, etc.)

    local model_dir="Saved_models/$feature_type/$time_scale"
    local result_dir="Results/saved_models_results/$feature_type/$time_scale"
    local drop_cols="${FEATURE_CONFIGS[$feature_type]}"

    log_info "--------------------------------------------"
    log_info "Base Model Testing: $test_description"
    log_info "Feature Type: $feature_type - $time_scale"
    log_info "--------------------------------------------"

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

    log_info "Testing configuration:"
    log_info "  Target: $target"
    log_info "  Feature type: $feature_type"
    log_info "  Test file: $latest_test_file"
    log_info "  Models: ${model_paths[*]}"
    log_info "  Drop columns: $drop_cols"
    log_info "  Output directory: $result_dir"

    # Prepare command arguments
    local cmd_args=(
        "--test_file" "$latest_test_file"
        "--target" "$target"
        "--models" "${model_paths[@]}"
        "--drop_cols" "$drop_cols"
        "--output_dir" "$result_dir"
    )

    # Add dataset suffix if provided
    if [[ -n "$dataset_suffix" ]]; then
        cmd_args+=("--dataset_suffix" "$dataset_suffix")
    fi

    # Add --use_centuries flag for century targets
    if [[ "$target" == "century" ]]; then
        cmd_args+=("--use_centuries")
    fi

    # Run model testing
    log_info "Starting model testing for $test_description - $feature_type - $time_scale..."
    python3 -u Base_model/model_tester.py test "${cmd_args[@]}"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error " Model testing for $test_description failed with exit code $exit_code"
        return $exit_code
    fi

    log_success "Base model testing completed for $test_description - $feature_type - $time_scale!"
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
    log_info "============================================"
    log_info "PHASE $CURRENT_PHASE: Binary Model Testing"
    log_info "Feature Type: $feature_type"
    log_info "Time Scale: $time_scale"
    log_info "Target: $target"
    log_info "============================================"

    # Find latest timestamped feature files
    log_info "Finding latest feature files..."
    local valid_file
    local test_file
    local gutenberg_file

    valid_file=$(find_latest_file "valid_features_cleaned_*.csv") || return 1
    test_file=$(find_latest_file "test_features_cleaned_*.csv") || return 1
    gutenberg_file=$(find_latest_file "gutenberg_features_cleaned_*.csv") || return 1

    # Validate all input files exist
    log_info "Validating input files..."
    validate_file "$valid_file" "Validation features" || return 1
    validate_file "$test_file" "Test features" || return 1
    validate_file "$gutenberg_file" "Gutenberg features" || return 1

    # Validate model files exist
    log_info "Validating model files..."
    validate_file "${model_source_dir}/random_forest_model.pkl" "Random Forest model" || return 1
    validate_file "${model_source_dir}/xgboost_model.pkl" "XGBoost model" || return 1
    validate_file "${model_source_dir}/catboost_model.pkl" "CatBoost model" || return 1

    # Create output directory
    ensure_directory "$output_dir" "Binary model output" || return 1

    # Run binary testing for each dataset
    local datasets=("$valid_file:Validation Data:validation" "$test_file:Test Data:test" "$gutenberg_file:Gutenberg Data:gutenberg")

    for dataset in "${datasets[@]}"; do
        IFS=':' read -r dataset_file dataset_description output_suffix <<< "$dataset"

        log_info "--------------------------------------------"
        log_info "Running Binary Model Testing for $dataset_description"
        log_info "Feature Type: $feature_type - $time_scale"
        log_info "--------------------------------------------"

        # Set top_k based on target type
        local top_k
        if [[ "$target" == "century" ]]; then
            top_k="$TOP_K_CENTURIES"
            log_info "Using top-$top_k approach for century classification"
        else
            top_k="$TOP_K_DECADES"
            log_info "Using top-$top_k approach for decade classification"
        fi

        python Binary_model/binary_model_tester.py \
            --test_file "$dataset_file" \
            --target "$target" \
            --models "${model_source_dir}/random_forest_model.pkl" "${model_source_dir}/xgboost_model.pkl" "${model_source_dir}/catboost_model.pkl" "random" \
            --drop_cols $drop_cols \
            --output_file "binary_results_${output_suffix}.csv" \
            --output_dir "$output_dir" \
            --top_k "$top_k"

        local exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            log_error " Binary model testing for $dataset_description failed with exit code $exit_code"
            return $exit_code
        fi

        log_success "$dataset_description binary testing completed"
    done

    log_success "Binary model testing completed for $feature_type - $time_scale!"
    return 0
}

# Function to run complete workflow for a feature type and time scale
run_feature_workflow() {
    local feature_type="$1"
    local time_scale="$2"
    local target="$3"

    log_info ""
    log_info "######################################################################"
    log_info "STARTING WORKFLOW: $feature_type - $time_scale ($target)"
    log_info "######################################################################"

    # Phase 1: Base Model Training
    if ! run_base_training "$feature_type" "$time_scale" "$target"; then
        log_error " Base model training failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    # Phase 2: Base Model Testing (Test Dataset)
    if ! run_base_testing "$feature_type" "$time_scale" "$target" "test_features_cleaned_*.csv" "Test Dataset Evaluation" "test"; then
        log_error " Base model test evaluation failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    # Phase 3: Base Model Testing (Gutenberg Dataset)
    if ! run_base_testing "$feature_type" "$time_scale" "$target" "gutenberg_features_cleaned_*.csv" "Gutenberg Dataset Evaluation" "gutenberg"; then
        log_error " Base model Gutenberg evaluation failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    # Phase 4: Binary Model Testing (All datasets)
    if ! run_binary_testing "$feature_type" "$time_scale" "$target"; then
        log_error " Binary model testing failed for $feature_type - $time_scale, aborting workflow"
        return 1
    fi

    log_info "######################################################################"
    log_info "COMPLETED WORKFLOW: $feature_type - $time_scale ($target)"
    log_info "######################################################################"
    log_info ""

    return 0
}

cleanup() {
    local end_time=$(date +%s)
    local total_duration=$((end_time - OVERALL_START_TIME))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))

    log_info "======================================================================="
    log_info "JOB COMPLETION SUMMARY"
    log_info "======================================================================="
    log_info "Job completed at $(date)"
    log_info "Final memory show_help: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $3"/"$2}' || echo 'N/A')"
    log_info "Total job duration: ${hours}h ${minutes}m ${seconds}s"
    log_info "Phases completed: $CURRENT_PHASE/$TOTAL_PHASES"
    log_info "======================================================================="
}

# ==================== ARGUMENT PARSING ====================

show_help() {
    cat << EOF
Usage: sbatch [SLURM_OPTIONS] $0 [SCRIPT_OPTIONS]

Script Options:
  -f, --features FEATURES    Comma-separated list of feature types to run
                             Available: compression,lexical_structure,readability,distance,neologism,final_model,optimal
                             Default: all
  -h, --help                 Show this help message

Examples:
  sbatch $0                                         # Run all feature types
  sbatch $0 -f compression                          # Run only compression features
  sbatch $0 -f compression,readability              # Run compression and readability
  sbatch $0 -f final_model                          # Run only the final combined model
  sbatch $0 -f optimal                              # Run only the optimal 16-feature model

SLURM Examples:
  sbatch --job-name=compression $0 -f compression   # Custom job name
  sbatch --mem=100GB $0 -f final_model              # Custom memory allocation

EOF
}

parse_arguments() {
    SELECTED_FEATURES="all"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--features)
                SELECTED_FEATURES="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ==================== MAIN EXECUTION ====================

main() {
    # Set up cleanup trap
    trap cleanup EXIT

    # Parse command line arguments
    parse_arguments "$@"

    log_info "Starting Feature Model Pipeline"
    log_info "Selected features: $SELECTED_FEATURES"
    log_info "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
    log_info "======================================================================="

    # Create necessary directories
    ensure_directory "$LOG_DIR" "Log" || exit 1

    # Validate Python scripts exist
    validate_file "Base_model/model_trainer.py" "Model trainer script" || exit 1
    validate_file "Base_model/model_tester.py" "Model tester script" || exit 1
    validate_file "Binary_model/binary_model_tester.py" "Binary model tester script" || exit 1

    # Validate feature extraction directory
    if [[ ! -d "$FEATURE_DIR" ]]; then
        log_error " Feature extraction directory not found: $FEATURE_DIR"
        exit 1
    fi

    log_info "Validated feature extraction directory: $FEATURE_DIR"

    # Define all possible workflows
    local all_workflows=(
        "compression:decades:decade"
        "compression:centuries:century"
        "lexical_structure:decades:decade"
        "lexical_structure:centuries:century"
        "readability:decades:decade"
        "readability:centuries:century"
        "distance:decades:decade"
        "distance:centuries:century"
        "neologism:decades:decade"
        "neologism:centuries:century"
        "final_model:decades:decade"
        "final_model:centuries:century"
        "optimal:decades:decade"
        "optimal:centuries:century"
    )

    # Filter workflows based on selected features
    local workflows=()
    if [[ "$SELECTED_FEATURES" == "all" ]]; then
        workflows=("${all_workflows[@]}")
    else
        IFS=',' read -ra SELECTED_ARRAY <<< "$SELECTED_FEATURES"
        for workflow in "${all_workflows[@]}"; do
            IFS=':' read -r feature_type time_scale target <<< "$workflow"
            for selected in "${SELECTED_ARRAY[@]}"; do
                selected=$(echo "$selected" | xargs)  # Trim whitespace
                if [[ "$feature_type" == "$selected" ]]; then
                    workflows+=("$workflow")
                    break
                fi
            done
        done
    fi

    # Update total phases based on selected workflows
    CURRENT_PHASE=0

    # Validate that we have workflows to run
    if [[ ${#workflows[@]} -eq 0 ]]; then
        log_error " No valid feature types selected or found"
        log_info "Available feature types: compression, lexical_structure, readability, distance, neologism, final_model, optimal"
        exit 1
    fi

    log_info "Execution order planned (${#workflows[@]} workflows):"
    for i in "${!workflows[@]}"; do
        IFS=':' read -r feature_type time_scale target <<< "${workflows[$i]}"
        log_info "  $((i+1)). $feature_type - $time_scale ($target)"
    done

    # Execute all workflows sequentially
    for workflow in "${workflows[@]}"; do
        IFS=':' read -r feature_type time_scale target <<< "$workflow"

        if ! run_feature_workflow "$feature_type" "$time_scale" "$target"; then
            log_error " Workflow failed for $feature_type - $time_scale, aborting entire pipeline"
            exit 1
        fi
    done

    log_info "======================================================================="
    log_info "SELECTED FEATURE PIPELINES COMPLETED SUCCESSFULLY"
    log_info "======================================================================="
    log_info ""
    log_info "Summary of completed workflows:"
    for workflow in "${workflows[@]}"; do
        IFS=':' read -r feature_type time_scale target <<< "$workflow"
        log_info "   $feature_type - $time_scale ($target)"
        log_info "    - Base models: Saved_models/$feature_type/$time_scale/"
        log_info "    - Base results: Results/saved_models_results/$feature_type/$time_scale/"
        log_info "    - Binary results: Saved_models_binary/$feature_type/$time_scale/"
    done
    log_info ""
    log_info "Check $LOG_DIR for detailed execution logs"
    log_info "======================================================================="
}

# Run main function
main "$@"