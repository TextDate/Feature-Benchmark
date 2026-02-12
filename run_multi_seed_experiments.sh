#!/bin/bash
#SBATCH --job-name=multi_seed_experiments
#SBATCH --output=logs/multi_seed_experiments_%j.out
#SBATCH --error=logs/multi_seed_experiments_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240GB
#SBATCH --partition=gpu

# Multi-seed Training and Testing Pipeline for TextDate Feature-Benchmark
#
# Usage Examples:
#   sbatch run_multi_seed_experiments.sh
#   # Edit RUN_TRAINING and RUN_TESTING flags below for partial runs

set -e  # Exit on any error

# ==================== ENVIRONMENT SETUP ====================
source ../../virtual-venv/bin/activate
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Create logs directory if it doesn't exist
mkdir -p logs

# ==================== CONFIGURATION ====================

# Default values
FEATURE_SUBSETS=("compression" "lexical_structure" "distance" "neologism" "readability" "final_model")
MODELS=("catboost")
TIME_SCALES=("decades" "centuries")
N_SEEDS=10

# Control flags - set these to control what runs
RUN_TRAINING=true   # Set to true to run training phase
RUN_TESTING=true    # Set to true to run testing phase

# Data files (use the actual filenames with timestamps)
TRAIN_FILE="Extracted_features/train_features_cleaned_20251012_064527.csv"
VAL_FILE="Extracted_features/valid_features_cleaned_20251012_064527.csv"
TEST_FILE="Extracted_features/test_features_cleaned_20251012_064527.csv"
GUTENBERG_FILE="Extracted_features/gutenberg_features_cleaned_20251012_064527.csv"

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

# ==================== MAIN EXECUTION ====================

echo "======================================================================="
echo "TEXTDATE FEATURE-BENCHMARK MULTI-SEED EXPERIMENTS"
echo "======================================================================="
echo "Job ID: ${SLURM_JOB_ID:-"N/A"}"
echo "Node: ${SLURM_NODELIST:-"localhost"}"
echo "Start time: $(date)"
echo "Feature Subsets: ${FEATURE_SUBSETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Time Scales: ${TIME_SCALES[*]}"
echo "Number of Seeds: $N_SEEDS"
echo "======================================================================="

# Create output directories
mkdir -p Results/multi_seed

# Training phase
if [ "$RUN_TRAINING" = true ]; then
    log_info "=== TRAINING PHASE ==="
    for feature_subset in "${FEATURE_SUBSETS[@]}"; do
        for time_scale in "${TIME_SCALES[@]}"; do
            for model in "${MODELS[@]}"; do
                log_info "Training: $feature_subset, $time_scale, $model"

                # Determine target column and other parameters
                if [ "$time_scale" = "centuries" ]; then
                    target_col="century"
                    use_centuries_flag="--use_centuries"
                    drop_cols="file_name year decade"
                else
                    target_col="decade"
                    use_centuries_flag=""
                    drop_cols="file_name year century"
                fi

                # Run training
                python Base_model/model_trainer.py multi-seed \
                    --train_file "$TRAIN_FILE" \
                    --val_file "$VAL_FILE" \
                    --feature_subset "$feature_subset" \
                    --model "$model" \
                    --time_scale "$time_scale" \
                    --target_col "$target_col" \
                    --n_seeds $N_SEEDS \
                    $use_centuries_flag \
                    --drop_cols $drop_cols

                if [ $? -eq 0 ]; then
                    log_success "Training completed: $feature_subset, $time_scale, $model"
                else
                    log_error "Training failed: $feature_subset, $time_scale, $model"
                fi
            done
        done
    done
else
    log_info "=== SKIPPING TRAINING PHASE (RUN_TRAINING=false) ==="
fi

# Testing phase
if [ "$RUN_TESTING" = true ]; then
    log_info "=== TESTING PHASE ==="
    for feature_subset in "${FEATURE_SUBSETS[@]}"; do
        for time_scale in "${TIME_SCALES[@]}"; do
            for model in "${MODELS[@]}"; do
                log_info "Testing: $feature_subset, $time_scale, $model"

                # Check if trained models exist
                MODEL_DIR="Results/multi_seed/$feature_subset/$time_scale"
                if [ ! -d "$MODEL_DIR" ]; then
                    log_error "No model directory found: $MODEL_DIR - Skipping"
                    continue
                fi

                # Count seed models
                SEED_MODELS=$(ls "$MODEL_DIR"/${model}_seed_*_model.joblib 2>/dev/null | wc -l)
                if [ "$SEED_MODELS" -eq 0 ]; then
                    log_error "No seed models found for $model in $MODEL_DIR - Skipping"
                    continue
                fi

                log_info "Found $SEED_MODELS trained models for $model"

                # Determine target column and other parameters
                if [ "$time_scale" = "centuries" ]; then
                    target_col="century"
                    use_centuries_flag="--use_centuries"
                    drop_cols="file_name year decade"
                else
                    target_col="decade"
                    use_centuries_flag=""
                    drop_cols="file_name year century"
                fi

                # Test on main test set
                log_info "Testing on main test set..."
                python Base_model/model_tester.py multi-seed \
                    --test_file "$TEST_FILE" \
                    --feature_subset "$feature_subset" \
                    --model "$model" \
                    --time_scale "$time_scale" \
                    --target_col "$target_col" \
                    --dataset_name "test" \
                    $use_centuries_flag \
                    --drop_cols $drop_cols

                # Test on Gutenberg set
                log_info "Testing on Gutenberg set..."
                python Base_model/model_tester.py multi-seed \
                    --test_file "$GUTENBERG_FILE" \
                    --feature_subset "$feature_subset" \
                    --model "$model" \
                    --time_scale "$time_scale" \
                    --target_col "$target_col" \
                    --dataset_name "gutenberg" \
                    $use_centuries_flag \
                    --drop_cols $drop_cols

                if [ $? -eq 0 ]; then
                    log_success "Testing completed: $feature_subset, $time_scale, $model"
                else
                    log_error "Testing failed: $feature_subset, $time_scale, $model"
                fi
            done
        done
    done
else
    log_info "=== SKIPPING TESTING PHASE (RUN_TESTING=false) ==="
fi

log_info "=== AGGREGATION PHASE ==="
python Base_model/model_tester.py aggregate

log_info "=== MULTI-SEED EXPERIMENTS COMPLETED ==="
log_info "Results are stored in: Results/multi_seed/"
log_info "Summary files created for easy analysis."