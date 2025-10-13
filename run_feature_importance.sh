#!/bin/bash

#SBATCH --job-name=feature_importance
#SBATCH --output=logs/feature_importance_%j.out
#SBATCH --error=logs/feature_importance_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --partition=cpu


# Feature importance analysis script

# Load required modules

# Activate virtual environment
source ../../virtual-venv/bin/activate

# Default parameters
MODEL_DIR="Saved_models"
TEST_DATA_DIR="Extracted_features"
OUTPUT_DIR="feature_importance_results"
TARGET_COL="century"
DROP_COLS="text,file_name,id"

# Model filtering (default: all feature types with catboost and xgboost)
MODEL_FILTER="optimal,compression,lexical_structure,readability,distance,neologism,final_model"

# Analysis flags (all enabled by default)
RUN_TREE=true
RUN_PERMUTATION=true
RUN_PCA=true
RUN_SHAP=true
RUN_CORRELATION=true


print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -m, --model-dir DIR         Model directory (default: $MODEL_DIR)
    -t, --test-dir DIR          Test data directory (default: $TEST_DATA_DIR)
    -o, --output-dir DIR        Output directory (default: $OUTPUT_DIR)
    --target COL                Target column name (default: $TARGET_COL)
    --drop COLS                 Comma-separated columns to drop (default: $DROP_COLS)
    --models FILTER             Model filter pattern (default: $MODEL_FILTER)
                               Examples: "catboost", "final_model/.*catboost", "compression/.*xgboost", "all"

Analysis options (disable with --no-X):
    --tree                      Enable tree-based importance (default: enabled)
    --permutation              Enable permutation importance (default: enabled)
    --pca                      Enable PCA analysis (default: enabled)
    --shap                     Enable SHAP analysis (default: enabled)
    --correlation              Enable correlation analysis (default: enabled)

    --no-tree                  Disable tree-based importance
    --no-permutation           Disable permutation importance
    --no-pca                   Disable PCA analysis
    --no-shap                  Disable SHAP analysis
    --no-correlation           Disable correlation analysis


    -h, --help                 Show this help message

Examples:
    # Run all analyses on CatBoost models only (default)
    $0

    # Run only on RandomForest models
    $0 --models random_forest

    # Run on XGBoost and CatBoost models
    $0 --models "xgboost,catboost"

    # Run on all models (including slow SVM, KNN)
    $0 --models all

    # Run only SHAP and correlation analysis on CatBoost
    $0 --no-tree --no-permutation --no-pca

EOF
}

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -t|--test-dir)
            TEST_DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --target)
            TARGET_COL="$2"
            shift 2
            ;;
        --drop)
            DROP_COLS="$2"
            shift 2
            ;;
        --models)
            MODEL_FILTER="$2"
            shift 2
            ;;
        --tree)
            RUN_TREE=true
            shift
            ;;
        --no-tree)
            RUN_TREE=false
            shift
            ;;
        --permutation)
            RUN_PERMUTATION=true
            shift
            ;;
        --no-permutation)
            RUN_PERMUTATION=false
            shift
            ;;
        --pca)
            RUN_PCA=true
            shift
            ;;
        --no-pca)
            RUN_PCA=false
            shift
            ;;
        --shap)
            RUN_SHAP=true
            shift
            ;;
        --no-shap)
            RUN_SHAP=false
            shift
            ;;
        --correlation)
            RUN_CORRELATION=true
            shift
            ;;
        --no-correlation)
            RUN_CORRELATION=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p logs


# Validate directories
if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model directory does not exist: $MODEL_DIR"
    exit 1
fi

if [ ! -d "$TEST_DATA_DIR" ]; then
    log_error "Test data directory does not exist: $TEST_DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "Starting feature importance analysis..."
log_info "Model directory: $MODEL_DIR"
log_info "Test data directory: $TEST_DATA_DIR"
log_info "Output directory: $OUTPUT_DIR"
log_info "Target column: $TARGET_COL"
log_info "Columns to drop: $DROP_COLS"
log_info "Model filter: $MODEL_FILTER"

# Find model and test files
if [ "$MODEL_FILTER" = "all" ]; then
    MODEL_FILES=($(find "$MODEL_DIR" -name "*.pkl" -type f))
else
    # Split comma-separated filters and build find pattern for catboost/xgboost only
    IFS=',' read -ra FILTERS <<< "$MODEL_FILTER"
    MODEL_FILES=()
    for filter in "${FILTERS[@]}"; do
        filter=$(echo "$filter" | xargs)  # trim whitespace
        # Find only catboost models for each feature type (feature benchmark, not model comparison)
        CATBOOST_FILES=($(find "$MODEL_DIR" -path "*${filter}*" -name "*catboost*.pkl" -type f))
        MODEL_FILES+=("${CATBOOST_FILES[@]}")
    done
fi

TEST_FILES=($(find "$TEST_DATA_DIR" -name "*cleaned*.csv" -type f))

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    log_error "No .pkl model files found in $MODEL_DIR"
    exit 1
fi

if [ ${#TEST_FILES[@]} -eq 0 ]; then
    log_error "No .csv test files found in $TEST_DATA_DIR"
    exit 1
fi

log_info "Found ${#MODEL_FILES[@]} model files and ${#TEST_FILES[@]} test files"

# Base analysis flags (without SHAP - will be added conditionally)
BASE_ANALYSIS_FLAGS=""
[ "$RUN_TREE" = true ] && BASE_ANALYSIS_FLAGS="$BASE_ANALYSIS_FLAGS --tree"
[ "$RUN_PERMUTATION" = true ] && BASE_ANALYSIS_FLAGS="$BASE_ANALYSIS_FLAGS --permutation"
[ "$RUN_PCA" = true ] && BASE_ANALYSIS_FLAGS="$BASE_ANALYSIS_FLAGS --pca"
[ "$RUN_CORRELATION" = true ] && BASE_ANALYSIS_FLAGS="$BASE_ANALYSIS_FLAGS --correlation"

if [ -z "$BASE_ANALYSIS_FLAGS" ] && [ "$RUN_SHAP" = false ]; then
    log_error "No analysis methods selected!"
    exit 1
fi

# Run analysis for each model-test combination
TOTAL_COMBINATIONS=$((${#MODEL_FILES[@]} * ${#TEST_FILES[@]}))
CURRENT=0

for model_file in "${MODEL_FILES[@]}"; do
    for test_file in "${TEST_FILES[@]}"; do
        CURRENT=$((CURRENT + 1))

        # Extract model and test names for output directory with full path info
        # Extract feature type and temporal scale from path: e.g., Saved_models/compression/centuries/catboost_model.pkl
        # Results in: compression_centuries_catboost_model
        MODEL_PATH_PARTS=$(echo "$model_file" | sed 's|Saved_models/||' | sed 's|/|_|g' | sed 's|\.pkl$||')
        TEST_NAME=$(basename "$test_file" .csv)

        ANALYSIS_OUTPUT_DIR="$OUTPUT_DIR/${MODEL_PATH_PARTS}_${TEST_NAME}"

        log_info "[$CURRENT/$TOTAL_COMBINATIONS] Analyzing: $MODEL_PATH_PARTS with $TEST_NAME"

        # Detect target column based on model path
        if [[ "$model_file" == *"/decades/"* ]]; then
            CURRENT_TARGET_COL="decade"
        else
            CURRENT_TARGET_COL="century"
        fi

        # Build analysis flags - add SHAP only for century models
        ANALYSIS_FLAGS="$BASE_ANALYSIS_FLAGS"
        if [[ "$model_file" == *"/centuries/"* ]] && [ "$RUN_SHAP" = true ]; then
            ANALYSIS_FLAGS="$ANALYSIS_FLAGS --shap"
            log_info "  Including SHAP analysis for century model (target: $CURRENT_TARGET_COL)"
        elif [[ "$model_file" == *"/decades/"* ]] && [ "$RUN_SHAP" = true ]; then
            log_info "  Skipping SHAP analysis for decade model (target: $CURRENT_TARGET_COL, centuries only)"
        fi

        # Run the feature importance analyzer
        python Plotting/feature_importance_analyzer.py \
            --model "$model_file" \
            --test_csv "$test_file" \
            --target "$CURRENT_TARGET_COL" \
            --drop_cols "$DROP_COLS" \
            --output_dir "$ANALYSIS_OUTPUT_DIR" \
            $ANALYSIS_FLAGS

        if [ $? -eq 0 ]; then
            log_success "Analysis completed for $MODEL_PATH_PARTS with $TEST_NAME"
        else
            log_error "Analysis failed for $MODEL_PATH_PARTS with $TEST_NAME"
        fi
    done
done

log_success "Feature importance analysis completed!"
log_info "Results saved in: $OUTPUT_DIR"
log_info "Total combinations processed: $TOTAL_COMBINATIONS"

# Generate summary
TOTAL_PLOTS=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
log_info "Total plots generated: $TOTAL_PLOTS"