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


# Streamlined Model Analysis Script for TextDate Feature-Benchmark
#
# Usage Examples:
#   sbatch run_model_analysis.sh --mode full
#   sbatch run_model_analysis.sh --mode custom --plot_types heatmap
#   sbatch run_model_analysis.sh --mode all

set -e  # Exit on any error

# ==================== ENVIRONMENT SETUP ====================
source ../../virtual-venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

# Create logs directory if it doesn't exist
mkdir -p logs

# ==================== CONFIGURATION ====================

# Default values
BASE_DIR="$(pwd)"
OUTPUT_DIR="model_analysis_results"
MODE="all"
PLOT_TYPES=""
METRICS=""
FEATURE_TYPES=""
TIME_SCALES=""
FIGSIZE="12 8"
DPI=300
VERBOSE=false

# Script path
PLOTTER_SCRIPT="Plotting/model_comparison_plotter.py"

# ==================== UTILITY FUNCTIONS ====================

log_info() {
    echo "[$(date '+%H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2
}

# Basic validation
validate_environment() {
    log_info "Validating environment..."

    # Check Python and plotting script
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi

    if [ ! -f "$BASE_DIR/$PLOTTER_SCRIPT" ]; then
        log_error "Plotting script not found: $PLOTTER_SCRIPT"
        exit 1
    fi

    # Check required packages
    python3 -c "import pandas, numpy, matplotlib, seaborn" 2>/dev/null || {
        log_error "Required packages missing. Install: pandas numpy matplotlib seaborn"
        exit 1
    }

    log_info "Environment validation passed"
}

# ==================== EXECUTION FUNCTIONS ====================

# Execute plotting script
run_plotting() {
    log_info "Running model comparison plots..."

    local cmd="python3 $BASE_DIR/$PLOTTER_SCRIPT --base_dir $BASE_DIR"

    # Add output directory if specified
    [ -n "$OUTPUT_DIR" ] && cmd="$cmd --output_dir $OUTPUT_DIR"

    # Add custom parameters if specified
    [ -n "$PLOT_TYPES" ] && cmd="$cmd --plot_types $PLOT_TYPES"
    [ -n "$METRICS" ] && cmd="$cmd --metrics $METRICS"
    [ -n "$FEATURE_TYPES" ] && cmd="$cmd --feature_types $FEATURE_TYPES"
    [ -n "$TIME_SCALES" ] && cmd="$cmd --time_scales $TIME_SCALES"

    # Add figure parameters
    cmd="$cmd --figsize $FIGSIZE --dpi $DPI"

    # Add verbose flag if enabled
    [ "$VERBOSE" = true ] && cmd="$cmd --verbose"

    log_info "Executing: $cmd"

    eval "$cmd" || {
        log_error "Plotting script failed"
        exit 1
    }

    log_info "Plotting completed successfully"
}

# Main execution function
run_analysis() {
    case "$MODE" in
        full)
            log_info "Running full analysis (all plots)"
            run_plotting
            ;;
        custom)
            log_info "Running custom analysis"
            run_plotting
            ;;
        all)
            log_info "Running all analysis"
            run_plotting
            ;;
        *)
            log_error "Invalid mode: $MODE"
            exit 1
            ;;
    esac
}

# ==================== ARGUMENT PARSING ====================

show_help() {
    cat << EOF
Streamlined Model Analysis Script for TextDate Feature-Benchmark

Usage: $0 [OPTIONS]

OPTIONS:
  --mode MODE             Analysis mode (full|custom|all) [default: all]
  --output-dir DIR        Output directory [default: model_analysis_results]
  --plot_types TYPES      Plot types (space-separated): heatmap binary_comparison dataset_comparison time_comparison all
  --metrics METRICS       Metrics (space-separated): accuracy f1_macro auc_roc precision recall (default: all available)
  --feature_types TYPES   Feature types (space-separated): compression lexical_structure readability distance neologism final_model
  --time_scales SCALES    Time scales (space-separated): decades centuries
  --figsize WIDTH HEIGHT  Figure size [default: 12 8]
  --dpi DPI              DPI setting [default: 300]
  --verbose              Enable verbose output
  --help, -h             Show this help

EXAMPLES:
  $0 --mode full
  $0 --mode custom --plot_types heatmap dataset_comparison
  $0 --mode all --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --plot_types)
            PLOT_TYPES="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --feature_types)
            FEATURE_TYPES="$2"
            shift 2
            ;;
        --time_scales)
            TIME_SCALES="$2"
            shift 2
            ;;
        --figsize)
            FIGSIZE="$2 $3"
            shift 3
            ;;
        --dpi)
            DPI="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
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

echo "======================================================================="
echo "TEXTDATE FEATURE-BENCHMARK MODEL ANALYSIS"
echo "======================================================================="
echo "Job ID: ${SLURM_JOB_ID:-"N/A"}"
echo "Node: ${SLURM_NODELIST:-"localhost"}"
echo "Start time: $(date)"
echo "Mode: $MODE"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run validation and analysis
validate_environment
run_analysis

log_info "Analysis completed successfully - Results in: $OUTPUT_DIR"