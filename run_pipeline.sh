# ==============================================================================
# SCRIPT 1: run_pipeline.sh (The Main Orchestrator)
# ==============================================================================
# Usage: ./run_pipeline.sh <source.c> [output_directory]
# ==============================================================================

#!/bin/bash

# Complete C Code Evaluation Pipeline
# This script orchestrates the entire 3-stage evaluation process

set -e # Exit on any error

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# All Python and C scripts are assumed to be in the same directory as this bash script
TEST_GENERATOR_PY="$SCRIPT_DIR/test_generator.py"
EVALUATOR_C="$SCRIPT_DIR/enhanced_safe_eval.c"
CODE_ANALYZER_PY="$SCRIPT_DIR/code_analyzer.py"


# --- Color Codes for Output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Helper Functions ---
print_stage() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n${BLUE}  $1\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "ℹ️  $1"; }

# --- Cleanup ---
cleanup() {
  if [ -d "$TEMP_DIR" ]; then
    print_info "Cleaning up temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
  fi
}

trap cleanup EXIT

# --- Dependency Check ---
check_dependencies() {
    print_stage "CHECKING DEPENDENCIES"
    local missing_deps=0
    
    command -v gcc >/dev/null || { print_error "GCC compiler not found"; missing_deps=1; }
    command -v valgrind >/dev/null || { print_error "Valgrind not found"; missing_deps=1; }
    command -v python3 >/dev/null || { print_error "Python3 not found"; missing_deps=1; }
    command -v ollama >/dev/null || { print_error "Ollama not found"; missing_deps=1; }
    command -v jq >/dev/null || { print_error "jq (JSON processor) not found"; missing_deps=1; }
    
    python3 -c "import ollama" 2>/dev/null || { print_error "Python 'ollama' library not found (pip install ollama)"; missing_deps=1; }
    
    pkg-config --libs json-c >/dev/null 2>&1 || { 
        print_error "json-c library not found (e.g., sudo apt-get install libjson-c-dev)"; missing_deps=1; 
    }
    
    if ! ollama list | grep -q "codellama"; then
        print_warning "CodeLlama model not found. Attempting to pull 'codellama:7b'..."
        if ! ollama pull codellama:7b; then
            print_error "Failed to pull CodeLlama model. Please pull it manually."
            missing_deps=1
        fi
    fi
    
    if [ $missing_deps -eq 0 ]; then
        print_success "All dependencies satisfied."
    else
        print_error "Missing dependencies. Please install them and try again."
        exit 1
    fi
}

# --- Usage ---
usage() {
    echo "Usage: $0 <source.c> [output_directory]"
    echo ""
    echo "Arguments:"
    echo "  source.c          The C source file to evaluate."
    echo "  output_directory  Optional. Directory to save all results."
    echo "                    (Default: ./evaluation_results_<timestamp>)"
    exit 1
}

# --- Main Pipeline ---
main() {
    if [[ "$1" == "-h" || "$1" == "--help" || $# -lt 1 || $# -gt 2 ]]; then
        usage
    fi
    
    local source_file="$1"
    local output_dir="${2:-./evaluation_results_$TIMESTAMP}"
    
    if [ ! -f "$source_file" ]; then
        print_error "Source file '$source_file' not found."
        exit 1
    fi
    
    mkdir -p "$output_dir"
    local abs_source_file="$(realpath "$source_file")"
    local abs_output_dir="$(realpath "$output_dir")"
    
    # Create a single temporary directory for all transient files
    TEMP_DIR=$(mktemp -d -t code_eval_XXXXXX)

    print_stage "C CODE EVALUATION PIPELINE"
    echo -e "  ${BLUE}Source File:${NC}      $abs_source_file"
    echo -e "  ${BLUE}Output Directory:${NC} $abs_output_dir"
    echo -e "  ${BLUE}Temp Directory:${NC}   $TEMP_DIR"
    
    check_dependencies
    
    # === STAGE 1: TEST CASE GENERATION ===
    print_stage "STAGE 1: GENERATING TEST CASES"
    local test_cases_file="$TEMP_DIR/generated_tests.json"
    if python3 "$TEST_GENERATOR_PY" "$abs_source_file" "$test_cases_file"; then
        print_success "Test cases generated."
        cp "$test_cases_file" "$abs_output_dir/generated_test_cases.json"
    else
        print_error "Failed to generate test cases. Aborting."
        exit 1
    fi

    # === STAGE 2: CODE EVALUATION ===
    print_stage "STAGE 2: EXECUTING & EVALUATING CODE"
    local evaluator_exe="$TEMP_DIR/enhanced_evaluator"
    local evaluation_metrics_file="$abs_output_dir/evaluation_metrics.json"

    print_info "Compiling the C evaluator..."
    if ! gcc -o "$evaluator_exe" "$EVALUATOR_C" -ljson-c -lm; then
        print_error "Failed to compile the C evaluator. Aborting."
        exit 1
    fi
    
    print_info "Running evaluation..."
    if "$evaluator_exe" "$abs_source_file" "$test_cases_file" "$evaluation_metrics_file"; then
        print_success "Code evaluation completed."
    else
        print_error "Code evaluation failed. Check output above for errors. Aborting."
        # Even if it fails, some metrics might have been written.
        [ -f "$evaluation_metrics_file" ] && print_info "Partial metrics may be available."
        exit 1
    fi
    
    # === STAGE 3: COMPREHENSIVE ANALYSIS ===
    print_stage "STAGE 3: GENERATING FINAL REPORT"
    local report_base_name="$abs_output_dir/comprehensive_report"
    if python3 "$CODE_ANALYZER_PY" "$abs_source_file" "$evaluation_metrics_file" "$report_base_name"; then
        print_success "Comprehensive analysis report generated."
    else
        print_error "Failed to generate the final analysis report."
        exit 1
    fi

    # === SUMMARY ===
    print_stage "EVALUATION SUMMARY"
    local final_report_json="${report_base_name}.json"
    if [ -f "$evaluation_metrics_file" ] && [ -f "$final_report_json" ]; then
        # Use jq to extract multiple values in one go
        local summary
        summary=$(jq -r '"\(.tests_passed)/\(.total_tests) (\(.passrate)%)| \(.memory_score)| \(.robustness_score)"' "$evaluation_metrics_file")
        local final_grade
        final_grade=$(jq -r '"\(.overall_assessment.grade) (\(.overall_assessment.score)/100)"' "$final_report_json")
        
        IFS='|' read -r tests memory robustness <<< "$summary"

        echo -e "  ${GREEN}Final Grade:${NC}      $final_grade"
        echo -e "  ${GREEN}Correctness:${NC}      $tests Tests Passed"
        echo -e "  ${GREEN}Memory Score:${NC}     $memory/100.0"
        echo -e "  ${GREEN}Robustness Score:${NC} $robustness/100.0"
    else
        print_warning "Could not generate summary as result files were not found."
    fi

    print_stage "PIPELINE COMPLETE"
    print_success "All results have been saved to: $abs_output_dir"
}

# Run the main function with all provided arguments
main "$@"
