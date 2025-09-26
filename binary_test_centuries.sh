#!/bin/bash
#SBATCH --job-name=binary_centuries_test   # create a short name for your job
#SBATCH --output="Logs/binary_centuries-%j.out"  # %j will be replaced by the slurm jobID
#SBATCH --error="Logs/binary_centuries-%j.err"   # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=16               # cpu-cores per task (optimized for binary testing)
#SBATCH --gres=gpu:0                      # number of gpus per node
#SBATCH --mem=64GB                        # Total amount of RAM requested (optimized for binary testing)
#SBATCH --partition=cpu                   # The queue where to submit the job

# Enable strict error handling
set -e

# Set working directory
cd /data/home/paulojnpinto02/TextDate/Feature-Benchmark

# Set environment variables
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2

echo "=========================================="
echo "Binary Model Testing - Centuries Classification"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Function to find the latest timestamped file
find_latest_file() {
    local pattern=$1
    local latest_file=$(ls -t Extracted_features/${pattern}_*.csv 2>/dev/null | head -n1)
    if [ -z "$latest_file" ]; then
        echo "ERROR: No file found matching pattern: $pattern" >&2
        exit 1
    fi
    echo "$latest_file"
}

# Function to validate file existence
validate_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: File not found: $1" >&2
        exit 1
    fi
    echo "✓ Found file: $1"
}

# Function to create directory if it doesn't exist
ensure_directory() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "✓ Created directory: $1"
    else
        echo "✓ Directory exists: $1"
    fi
}

# Find latest timestamped feature files
echo "Finding latest feature files..."
VALID_FILE=$(find_latest_file "valid_features_cleaned")
TEST_FILE=$(find_latest_file "test_features_cleaned")
GUTENBERG_FILE=$(find_latest_file "gutenberg_features_cleaned")

# Validate all input files exist
echo "Validating input files..."
validate_file "$VALID_FILE"
validate_file "$TEST_FILE"
validate_file "$GUTENBERG_FILE"

# Validate model files exist
echo "Validating model files..."
validate_file "Saved_models/centuries/random_forest_model.pkl"
validate_file "Saved_models/centuries/xgboost_model.pkl"
validate_file "Saved_models/centuries/catboost_model.pkl"

# Create output directory
OUTPUT_DIR="Saved_models_binary/centuries"
ensure_directory "$OUTPUT_DIR"

echo "=========================================="
echo "Running Binary Model Testing for Validation Data"
echo "=========================================="
python Binary_model/binary_model_tester.py \
    --test_file "$VALID_FILE" \
    --target century \
    --models Saved_models/centuries/random_forest_model.pkl Saved_models/centuries/xgboost_model.pkl Saved_models/centuries/catboost_model.pkl random \
    --drop_cols file_name year decade special_character_ratio \
    --output_file binary_results_validation.csv \
    --output_dir "$OUTPUT_DIR"

echo "✓ Validation testing completed"

echo "=========================================="
echo "Running Binary Model Testing for Test Data"
echo "=========================================="
python Binary_model/binary_model_tester.py \
    --test_file "$TEST_FILE" \
    --target century \
    --models Saved_models/centuries/random_forest_model.pkl Saved_models/centuries/xgboost_model.pkl Saved_models/centuries/catboost_model.pkl random \
    --drop_cols file_name year decade special_character_ratio \
    --output_file binary_results_test.csv \
    --output_dir "$OUTPUT_DIR"

echo "✓ Test data testing completed"

echo "=========================================="
echo "Running Binary Model Testing for Gutenberg Data"
echo "=========================================="
python Binary_model/binary_model_tester.py \
    --test_file "$GUTENBERG_FILE" \
    --target century \
    --models Saved_models/centuries/random_forest_model.pkl Saved_models/centuries/xgboost_model.pkl Saved_models/centuries/catboost_model.pkl random \
    --drop_cols file_name year decade special_character_ratio \
    --output_file binary_results_gutenberg.csv \
    --output_dir "$OUTPUT_DIR"

echo "✓ Gutenberg data testing completed"

echo "=========================================="
echo "Binary Model Testing - Centuries Classification COMPLETED"
echo "End time: $(date)"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="