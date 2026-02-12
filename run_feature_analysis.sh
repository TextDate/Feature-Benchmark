#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=logs/feature_analysis_%j.out
#SBATCH --error=logs/feature_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16GB
#SBATCH --partition=cpu
#SBATCH --time=02:00:00

# Feature Analysis Script for TextDate Feature-Benchmark
#
# Usage Examples:
#   sbatch run_feature_analysis.sh --target decade
#   sbatch run_feature_analysis.sh --target century --exclude_values 1600 1610
#   sbatch run_feature_analysis.sh --mode individual
#   sbatch run_feature_analysis.sh --mode combined

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
OUTPUT_DIR="Results/feature_analysis"
TARGET="decade"
MODE="both"
EXCLUDE_VALUES=""
FIGSIZE="12 8"
DPI=300
VERBOSE=false

# Feature data paths (use most recent cleaned features)
TRAIN_CSV="Extracted_features/train_features_cleaned_*.csv"
VALID_CSV="Extracted_features/valid_features_cleaned_*.csv"
TEST_CSV="Extracted_features/test_features_cleaned_*.csv"
GUTENBERG_CSV="Extracted_features/gutenberg_features_cleaned_*.csv"

# Script paths
FEATURE_PLOTTER="Plotting/feature_plotter.py"

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

# Find most recent feature files
find_latest_file() {
    local pattern="$1"
    ls -t $pattern 2>/dev/null | head -n1 || echo ""
}

# Validation
validate_environment() {
    log_info "Validating environment..."

    # Check Python and required scripts
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi

    if [ ! -f "$BASE_DIR/$FEATURE_PLOTTER" ]; then
        log_error "Feature plotting script not found: $FEATURE_PLOTTER"
        exit 1
    fi

    # Check required packages
    python3 -c "import pandas, numpy, matplotlib, seaborn, PIL" 2>/dev/null || {
        log_error "Required packages missing. Install: pandas numpy matplotlib seaborn pillow"
        exit 1
    }

    log_info "Environment validation passed"
}

# Find feature CSV files
find_feature_files() {
    log_info "Locating feature CSV files..."

    TRAIN_FILE=$(find_latest_file "$TRAIN_CSV")
    VALID_FILE=$(find_latest_file "$VALID_CSV")
    TEST_FILE=$(find_latest_file "$TEST_CSV")
    GUTENBERG_FILE=$(find_latest_file "$GUTENBERG_CSV")

    log_info "Found files:"
    [ -n "$TRAIN_FILE" ] && log_info "  Train: $TRAIN_FILE" || log_error "  Train: NOT FOUND"
    [ -n "$VALID_FILE" ] && log_info "  Valid: $VALID_FILE" || log_error "  Valid: NOT FOUND"
    [ -n "$TEST_FILE" ] && log_info "  Test: $TEST_FILE" || log_error "  Test: NOT FOUND"
    [ -n "$GUTENBERG_FILE" ] && log_info "  Gutenberg: $GUTENBERG_FILE" || log_error "  Gutenberg: NOT FOUND"

    # Check if we have minimum required files
    if [ -z "$TRAIN_FILE" ] || [ -z "$TEST_FILE" ]; then
        log_error "Missing required feature files (train and test)"
        exit 1
    fi
}

# ==================== EXECUTION FUNCTIONS ====================

# Generate individual dataset plots
run_individual_plotting() {
    log_info "Running individual feature plotting..."

    local csvs=()
    local labels=()

    # Build file and label arrays
    [ -n "$TRAIN_FILE" ] && { csvs+=("$TRAIN_FILE"); labels+=("train"); }
    [ -n "$VALID_FILE" ] && { csvs+=("$VALID_FILE"); labels+=("validation"); }
    [ -n "$TEST_FILE" ] && { csvs+=("$TEST_FILE"); labels+=("test"); }
    [ -n "$GUTENBERG_FILE" ] && { csvs+=("$GUTENBERG_FILE"); labels+=("gutenberg"); }

    local cmd="python3 $BASE_DIR/$FEATURE_PLOTTER plot"
    cmd="$cmd --csvs ${csvs[*]}"
    cmd="$cmd --labels ${labels[*]}"
    cmd="$cmd --target $TARGET"
    cmd="$cmd --output_dir $OUTPUT_DIR/individual"
    cmd="$cmd --drop_cols text"

    # Add exclude values if specified
    [ -n "$EXCLUDE_VALUES" ] && cmd="$cmd --exclude_${TARGET}s $EXCLUDE_VALUES"

    log_info "Executing: $cmd"

    eval "$cmd" || {
        log_error "Individual plotting failed"
        exit 1
    }

    log_success "Individual plotting completed"
}

# Generate combined comparison plots
run_combined_plotting() {
    log_info "Running combined feature plotting..."

    # Check if individual plots exist
    local individual_dir="$OUTPUT_DIR/individual"
    if [ ! -d "$individual_dir" ]; then
        log_error "Individual plots directory not found. Run individual plotting first."
        exit 1
    fi

    local cmd="python3 $BASE_DIR/$FEATURE_PLOTTER combine"
    cmd="$cmd --train_dir $individual_dir/train"
    cmd="$cmd --val_dir $individual_dir/validation"
    cmd="$cmd --test_dir $individual_dir/test"
    cmd="$cmd --gutenberg_dir $individual_dir/gutenberg"
    cmd="$cmd --output_dir $OUTPUT_DIR/combined"
    cmd="$cmd --target $TARGET"

    log_info "Executing: $cmd"

    eval "$cmd" || {
        log_error "Combined plotting failed"
        exit 1
    }

    log_success "Combined plotting completed"
}

# Main execution function
run_analysis() {
    case "$MODE" in
        individual)
            log_info "Running individual analysis only"
            run_individual_plotting
            ;;
        combined)
            log_info "Running combined analysis only"
            run_combined_plotting
            ;;
        both)
            log_info "Running both individual and combined analysis"
            run_individual_plotting
            run_combined_plotting
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
Feature Analysis Script for TextDate Feature-Benchmark

Usage: $0 [OPTIONS]

OPTIONS:
  --target TARGET         Target column to group by (decade|century) [default: decade]
  --mode MODE            Analysis mode (individual|combined|both) [default: both]
  --output-dir DIR       Output directory [default: Results/feature_analysis]
  --exclude-values VALS  Values to exclude (space-separated, e.g., 1600 1610)
  --verbose             Enable verbose output
  --help, -h            Show this help

MODES:
  individual    Generate individual dataset plots only
  combined      Generate combined comparison plots only
  both          Generate both individual and combined plots

EXAMPLES:
  $0 --target decade --mode both
  $0 --target century --exclude-values 1600 1610 1620
  $0 --mode individual --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --exclude-values)
            shift
            EXCLUDE_VALUES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXCLUDE_VALUES="$EXCLUDE_VALUES $1"
                shift
            done
            EXCLUDE_VALUES=$(echo $EXCLUDE_VALUES | xargs) # trim whitespace
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
echo "TEXTDATE FEATURE-BENCHMARK FEATURE ANALYSIS"
echo "======================================================================="
echo "Job ID: ${SLURM_JOB_ID:-"N/A"}"
echo "Node: ${SLURM_NODELIST:-"localhost"}"
echo "Start time: $(date)"
echo "Target: $TARGET"
echo "Mode: $MODE"
echo "Output Directory: $OUTPUT_DIR"
[ -n "$EXCLUDE_VALUES" ] && echo "Exclude Values: $EXCLUDE_VALUES"
echo "======================================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run validation and analysis
validate_environment
find_feature_files
run_analysis

log_success "Feature analysis completed - Results in: $OUTPUT_DIR"