#!/bin/bash
#SBATCH --job-name=model_analysis
#SBATCH --output=logs/model_analysis_%j.out
#SBATCH --error=logs/model_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

# Comprehensive Model Analysis and Plotting Script for TextDate Feature-Benchmark
#
# This script provides multiple execution modes for analyzing and visualizing
# model comparison results with SLURM resource management and comprehensive
# error handling.
#
# Author: Claude AI Assistant (SLURM Expert)
# Date: September 2025
# Version: 1.0
#
# Usage Examples:
#   sbatch run_model_analysis.sh --mode quick                    # Quick analysis mode
#   sbatch run_model_analysis.sh --mode full                     # Full visualization mode
#   sbatch run_model_analysis.sh --mode custom --plots heatmap   # Custom comparison mode
#   sbatch run_model_analysis.sh --mode validate                 # Structure validation mode
#   sbatch run_model_analysis.sh --mode all                      # Run all analysis types

set -e  # Exit on any error
set -u  # Exit on undefined variables

# ==================== SLURM ENVIRONMENT SETUP ====================
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MALLOC_TRIM_THRESHOLD_=0

# Create logs directory if it doesn't exist
mkdir -p logs

# ==================== SCRIPT CONFIGURATION ====================

# Default values
BASE_DIR="$(pwd)"
OUTPUT_DIR="model_analysis_results"
MODE="full"
PLOTS=""
METRICS=""
FEATURE_TYPES=""
TIME_SCALES=""
DATASETS=""
MODELS=""
RESOLUTION="medium"
FORMAT="png"
VERBOSE=false
DRY_RUN=false
FORCE_OVERWRITE=false
MEMORY_EFFICIENT=false

# Analysis script paths
PLOTTER_SCRIPT="model_comparison_plotter.py"
DEMO_SCRIPT="simple_demo.py"
TEST_SCRIPT="test_structure.py"

# ==================== UTILITY FUNCTIONS ====================

# Logging functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*" | tee -a "${OUTPUT_DIR}/analysis.log"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${OUTPUT_DIR}/analysis.log" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*" | tee -a "${OUTPUT_DIR}/analysis.log"
}

# Progress reporting
show_progress() {
    local current=$1
    local total=$2
    local task=$3
    local percent=$((current * 100 / total))
    echo "Progress: [$current/$total] ($percent%) - $task"
}

# Resource monitoring
check_memory_usage() {
    local mem_usage=$(ps -o pid,vsz,rss,comm -p $$ | tail -1 | awk '{print $3}')
    local mem_mb=$((mem_usage / 1024))
    log_info "Current memory usage: ${mem_mb}MB"

    # Warning if memory usage exceeds 80% of allocated
    local mem_limit_mb=$((${SLURM_MEM_PER_NODE:-32000} * 80 / 100))
    if [ "$mem_mb" -gt "$mem_limit_mb" ]; then
        log_warning "Memory usage (${mem_mb}MB) approaching limit (${SLURM_MEM_PER_NODE:-32000}MB)"
    fi
}

# Execution time tracking
start_timer() {
    echo $SECONDS
}

end_timer() {
    local start_time=$1
    local elapsed=$((SECONDS - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# ==================== VALIDATION FUNCTIONS ====================

# Check if required files exist
validate_environment() {
    log_info "Validating environment and dependencies..."

    # Check if base directory exists
    if [ ! -d "$BASE_DIR" ]; then
        log_error "Base directory does not exist: $BASE_DIR"
        exit 1
    fi

    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed or not in PATH"
        exit 1
    fi

    # Check if main plotting script exists
    if [ ! -f "$BASE_DIR/$PLOTTER_SCRIPT" ]; then
        log_error "Main plotting script not found: $BASE_DIR/$PLOTTER_SCRIPT"
        exit 1
    fi

    # Check for required Python packages (basic check)
    python3 -c "import pandas, numpy, matplotlib, seaborn" 2>/dev/null || {
        log_error "Required Python packages not installed. Please install from requirements.txt"
        exit 1
    }

    # Check for result files
    local result_dirs=("Saved_models_results" "Saved_models" "Saved_models_binary")
    local found_results=false

    for dir in "${result_dirs[@]}"; do
        if [ -d "$BASE_DIR/$dir" ] && [ "$(ls -A "$BASE_DIR/$dir" 2>/dev/null)" ]; then
            found_results=true
            log_info "Found results in: $dir"
        fi
    done

    if [ "$found_results" = false ]; then
        log_warning "No model results found. Make sure you have run the training pipeline first."
    fi

    log_info "Environment validation completed successfully"
}

# Validate command line arguments
validate_arguments() {
    # Validate mode
    case "$MODE" in
        quick|full|custom|validate|all)
            ;;
        *)
            log_error "Invalid mode: $MODE. Valid modes: quick, full, custom, validate, all"
            exit 1
            ;;
    esac

    # Validate resolution
    case "$RESOLUTION" in
        low|medium|high|ultra)
            ;;
        *)
            log_error "Invalid resolution: $RESOLUTION. Valid options: low, medium, high, ultra"
            exit 1
            ;;
    esac

    # Validate format
    case "$FORMAT" in
        png|pdf|svg|eps)
            ;;
        *)
            log_error "Invalid format: $FORMAT. Valid options: png, pdf, svg, eps"
            exit 1
            ;;
    esac
}

# ==================== ANALYSIS EXECUTION FUNCTIONS ====================

# Execute structure validation
run_structure_validation() {
    log_info "Running structure validation..."
    local start_time=$(start_timer)

    if [ -f "$BASE_DIR/$TEST_SCRIPT" ]; then
        python3 "$BASE_DIR/$TEST_SCRIPT" --base_dir "$BASE_DIR" || {
            log_error "Structure validation failed"
            return 1
        }
    else
        log_warning "Structure validation script not found: $TEST_SCRIPT"
        log_info "Creating basic structure validation..."

        # Basic validation
        local required_dirs=("Saved_models_results" "Extracted_features")
        for dir in "${required_dirs[@]}"; do
            if [ ! -d "$BASE_DIR/$dir" ]; then
                log_error "Required directory missing: $dir"
                return 1
            fi
        done
        log_info "Basic structure validation passed"
    fi

    local elapsed=$(end_timer $start_time)
    log_info "Structure validation completed in $elapsed"
}

# Execute quick analysis
run_quick_analysis() {
    log_info "Running quick analysis..."
    local start_time=$(start_timer)

    if [ -f "$BASE_DIR/$DEMO_SCRIPT" ]; then
        python3 "$BASE_DIR/$DEMO_SCRIPT" "$BASE_DIR" || {
            log_error "Quick analysis failed"
            return 1
        }
    else
        log_warning "Quick analysis script not found: $DEMO_SCRIPT"
        log_info "Running basic model summary instead..."

        # Create basic summary
        echo "=== Basic Model Results Summary ===" > "$OUTPUT_DIR/quick_summary.txt"
        echo "Generated on: $(date)" >> "$OUTPUT_DIR/quick_summary.txt"
        echo "" >> "$OUTPUT_DIR/quick_summary.txt"

        # Count available results
        for dir in Saved_models_results Saved_models Saved_models_binary; do
            if [ -d "$BASE_DIR/$dir" ]; then
                local count=$(find "$BASE_DIR/$dir" -name "*.json" 2>/dev/null | wc -l)
                echo "$dir: $count result files" >> "$OUTPUT_DIR/quick_summary.txt"
            fi
        done

        log_info "Basic summary saved to: $OUTPUT_DIR/quick_summary.txt"
    fi

    local elapsed=$(end_timer $start_time)
    log_info "Quick analysis completed in $elapsed"
}

# Execute full visualization
run_full_visualization() {
    log_info "Running full comprehensive visualization..."
    local start_time=$(start_timer)

    local cmd="python3 $BASE_DIR/$PLOTTER_SCRIPT --base_dir $BASE_DIR --output_dir $OUTPUT_DIR"

    # Add resolution settings
    case "$RESOLUTION" in
        low)
            cmd="$cmd --figsize 8 6 --dpi 150"
            ;;
        medium)
            cmd="$cmd --figsize 12 8 --dpi 300"
            ;;
        high)
            cmd="$cmd --figsize 16 12 --dpi 600"
            ;;
        ultra)
            cmd="$cmd --figsize 20 15 --dpi 1200"
            ;;
    esac

    # Add format
    cmd="$cmd --format $FORMAT"

    # Add verbose flag if requested
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd --verbose"
    fi

    # Add memory efficient flag if needed
    if [ "$MEMORY_EFFICIENT" = true ]; then
        cmd="$cmd --memory_efficient"
    fi

    # Add create_all flag for full analysis
    cmd="$cmd --create_all"

    log_info "Executing: $cmd"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would execute full visualization"
    else
        eval "$cmd" || {
            log_error "Full visualization failed"
            return 1
        }
    fi

    local elapsed=$(end_timer $start_time)
    log_info "Full visualization completed in $elapsed"
}

# Execute custom analysis
run_custom_analysis() {
    log_info "Running custom analysis..."
    local start_time=$(start_timer)

    local cmd="python3 $BASE_DIR/$PLOTTER_SCRIPT --base_dir $BASE_DIR --output_dir $OUTPUT_DIR"

    # Add custom parameters
    [ -n "$PLOTS" ] && cmd="$cmd --plot_types $PLOTS"
    [ -n "$METRICS" ] && cmd="$cmd --metrics $METRICS"
    [ -n "$FEATURE_TYPES" ] && cmd="$cmd --feature_types $FEATURE_TYPES"
    [ -n "$TIME_SCALES" ] && cmd="$cmd --time_scales $TIME_SCALES"
    [ -n "$DATASETS" ] && cmd="$cmd --datasets $DATASETS"
    [ -n "$MODELS" ] && cmd="$cmd --models $MODELS"

    # Add resolution and format
    case "$RESOLUTION" in
        low) cmd="$cmd --figsize 8 6 --dpi 150" ;;
        medium) cmd="$cmd --figsize 12 8 --dpi 300" ;;
        high) cmd="$cmd --figsize 16 12 --dpi 600" ;;
        ultra) cmd="$cmd --figsize 20 15 --dpi 1200" ;;
    esac

    cmd="$cmd --format $FORMAT"

    [ "$VERBOSE" = true ] && cmd="$cmd --verbose"
    [ "$MEMORY_EFFICIENT" = true ] && cmd="$cmd --memory_efficient"

    log_info "Executing: $cmd"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would execute custom analysis"
    else
        eval "$cmd" || {
            log_error "Custom analysis failed"
            return 1
        }
    fi

    local elapsed=$(end_timer $start_time)
    log_info "Custom analysis completed in $elapsed"
}

# ==================== MAIN EXECUTION FUNCTION ====================

run_analysis() {
    local total_steps=0
    local current_step=0

    # Count total steps based on mode
    case "$MODE" in
        quick) total_steps=1 ;;
        full) total_steps=1 ;;
        custom) total_steps=1 ;;
        validate) total_steps=1 ;;
        all) total_steps=4 ;;
    esac

    log_info "Starting analysis pipeline with mode: $MODE"
    log_info "Total steps: $total_steps"

    # Execute based on mode
    case "$MODE" in
        quick)
            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Quick Analysis"
            run_quick_analysis || exit 1
            ;;
        full)
            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Full Visualization"
            run_full_visualization || exit 1
            ;;
        custom)
            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Custom Analysis"
            run_custom_analysis || exit 1
            ;;
        validate)
            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Structure Validation"
            run_structure_validation || exit 1
            ;;
        all)
            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Structure Validation"
            run_structure_validation || exit 1

            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Quick Analysis"
            run_quick_analysis || exit 1

            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Custom Analysis (if specified)"
            if [ -n "$PLOTS" ] || [ -n "$METRICS" ] || [ -n "$FEATURE_TYPES" ]; then
                run_custom_analysis || exit 1
            else
                log_info "Skipping custom analysis (no custom parameters specified)"
            fi

            current_step=$((current_step + 1))
            show_progress $current_step $total_steps "Full Visualization"
            run_full_visualization || exit 1
            ;;
    esac

    check_memory_usage
    log_info "Analysis pipeline completed successfully!"
}

# ==================== CLEANUP AND REPORTING ====================

generate_summary_report() {
    log_info "Generating summary report..."

    local report_file="$OUTPUT_DIR/analysis_summary.txt"

    cat > "$report_file" << EOF
====================================================================
TextDate Feature-Benchmark Model Analysis Summary Report
====================================================================
Generated on: $(date)
Job ID: ${SLURM_JOB_ID:-"N/A"}
Node: ${SLURM_NODELIST:-"localhost"}
Mode: $MODE
Output Directory: $OUTPUT_DIR

Configuration:
- Base Directory: $BASE_DIR
- Resolution: $RESOLUTION
- Format: $FORMAT
- Verbose: $VERBOSE
- Memory Efficient: $MEMORY_EFFICIENT

Results:
EOF

    # Count generated files
    if [ -d "$OUTPUT_DIR" ]; then
        local plot_count=$(find "$OUTPUT_DIR" -name "*.$FORMAT" 2>/dev/null | wc -l)
        local json_count=$(find "$OUTPUT_DIR" -name "*.json" 2>/dev/null | wc -l)
        local log_count=$(find "$OUTPUT_DIR" -name "*.log" 2>/dev/null | wc -l)

        echo "- Plots generated: $plot_count" >> "$report_file"
        echo "- JSON files: $json_count" >> "$report_file"
        echo "- Log files: $log_count" >> "$report_file"
        echo "" >> "$report_file"

        echo "Generated files:" >> "$report_file"
        find "$OUTPUT_DIR" -type f | head -20 | sed 's/^/  /' >> "$report_file"

        local total_files=$(find "$OUTPUT_DIR" -type f | wc -l)
        if [ "$total_files" -gt 20 ]; then
            echo "  ... and $((total_files - 20)) more files" >> "$report_file"
        fi
    fi

    echo "" >> "$report_file"
    echo "=====================================================================" >> "$report_file"

    log_info "Summary report saved to: $report_file"
}

cleanup_on_exit() {
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        log_error "Script exited with error code: $exit_code"
    fi

    # Generate summary report regardless of exit status
    if [ -d "$OUTPUT_DIR" ]; then
        generate_summary_report
    fi

    log_info "Cleanup completed"
    exit $exit_code
}

# ==================== ARGUMENT PARSING ====================

show_help() {
    cat << EOF
TextDate Feature-Benchmark Model Analysis Script

Usage: $0 [OPTIONS]

EXECUTION MODES:
  --mode MODE             Analysis mode (quick|full|custom|validate|all) [default: full]

DIRECTORIES:
  --base-dir DIR          Base directory path [default: current directory]
  --output-dir DIR        Output directory for results [default: model_analysis_results]

ANALYSIS OPTIONS:
  --plots TYPES           Plot types for custom mode (comma-separated)
                         Options: heatmap, dataset_comparison, model_comparison, etc.
  --metrics METRICS       Metrics to analyze (comma-separated)
                         Options: accuracy, f1_macro, auc_roc, precision, recall
  --feature-types TYPES   Feature types to include (comma-separated)
                         Options: compression, lexical_structure, readability, distance, final_model
  --time-scales SCALES    Time scales to analyze (comma-separated)
                         Options: decades, centuries
  --datasets DATASETS     Datasets to include (comma-separated)
                         Options: test, gutenberg, validation
  --models MODELS         Models to analyze (comma-separated)
                         Options: random_forest, xgboost, catboost, svm, gnb, knn

OUTPUT OPTIONS:
  --resolution RES        Plot resolution (low|medium|high|ultra) [default: medium]
  --format FORMAT         Output format (png|pdf|svg|eps) [default: png]

EXECUTION OPTIONS:
  --verbose               Enable verbose logging
  --dry-run               Show what would be executed without running
  --force                 Overwrite existing output directory
  --memory-efficient      Use memory-efficient processing for large datasets

EXAMPLES:
  # Quick analysis only
  $0 --mode quick

  # Full analysis with high resolution
  $0 --mode full --resolution high --format pdf

  # Custom analysis for specific features
  $0 --mode custom --feature-types "distance,final_model" --plots "heatmap,dataset_comparison"

  # Validate structure only
  $0 --mode validate

  # Run everything
  $0 --mode all --verbose

  # Memory-efficient analysis for large datasets
  $0 --mode full --memory-efficient --resolution low

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plots)
            PLOTS="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --feature-types)
            FEATURE_TYPES="$2"
            shift 2
            ;;
        --time-scales)
            TIME_SCALES="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_OVERWRITE=true
            shift
            ;;
        --memory-efficient)
            MEMORY_EFFICIENT=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ==================== MAIN EXECUTION ====================

# Set trap for cleanup
trap cleanup_on_exit EXIT

# Print SLURM job information
echo "======================================================================="
echo "TEXTDATE FEATURE-BENCHMARK MODEL ANALYSIS PIPELINE"
echo "======================================================================="
echo "Job ID: ${SLURM_JOB_ID:-"N/A (running locally)"}"
echo "Job Name: ${SLURM_JOB_NAME:-"model_analysis"}"
echo "Node: ${SLURM_NODELIST:-"localhost"}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-"N/A"}"
echo "Memory: ${SLURM_MEM_PER_NODE:-"N/A"}MB"
echo "Start time: $(date)"
echo "Mode: $MODE"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================="

# Create output directory
if [ -d "$OUTPUT_DIR" ] && [ "$FORCE_OVERWRITE" = false ]; then
    log_error "Output directory already exists: $OUTPUT_DIR"
    log_error "Use --force to overwrite or choose a different directory"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "Analysis started at $(date)" > "$OUTPUT_DIR/analysis.log"

# Run validation
validate_arguments
validate_environment

# Execute main analysis
SCRIPT_START_TIME=$(start_timer)
run_analysis
TOTAL_ELAPSED=$(end_timer $SCRIPT_START_TIME)

log_info "==============================================="
log_info "ANALYSIS COMPLETED SUCCESSFULLY"
log_info "Total execution time: $TOTAL_ELAPSED"
log_info "Results saved to: $OUTPUT_DIR"
log_info "==============================================="