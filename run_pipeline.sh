#!/bin/bash

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="/tmp/code_eval_$$"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_stage() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Cleanup function
cleanup() {
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

# Check dependencies
check_dependencies() {
    print_stage "CHECKING DEPENDENCIES"
    
    local missing_deps=0
    
    # Check for required tools
    command -v gcc >/dev/null 2>&1 || { print_error "GCC compiler not found"; missing_deps=1; }
    command -v valgrind >/dev/null 2>&1 || { print_error "Valgrind not found"; missing_deps=1; }
    command -v python3 >/dev/null 2>&1 || { print_error "Python3 not found"; missing_deps=1; }
    command -v ollama >/dev/null 2>&1 || { print_error "Ollama not found"; missing_deps=1; }
    
    # Check for Python libraries
    python3 -c "import ollama" 2>/dev/null || { print_error "Python ollama library not found"; missing_deps=1; }
    python3 -c "import json" 2>/dev/null || { print_error "Python json library not found"; missing_deps=1; }
    
    # Check for C libraries
    gcc -ljson-c -x c -o /dev/null - <<<'int main(){return 0;}' 2>/dev/null || { 
        print_error "json-c library not found (install libjson-c-dev)"; missing_deps=1; 
    }
    
    # Check if CodeLlama model is available
    ollama list | grep -q "codellama" || { 
        print_warning "CodeLlama model not found. Run: ollama pull codellama:7b"
        print_info "Attempting to pull CodeLlama model..."
        ollama pull codellama:7b || { print_error "Failed to pull CodeLlama model"; missing_deps=1; }
    }
    
    if [ $missing_deps -eq 0 ]; then
        print_success "All dependencies satisfied"
    else
        print_error "Missing dependencies. Please install them before continuing."
        exit 1
    fi
}

# Usage function
usage() {
    echo "Usage: $0 <source.c> [output_directory]"
    echo ""
    echo "This script performs a comprehensive 3-stage evaluation of C code:"
    echo "  Stage 1: Generate intelligent test cases using CodeLlama"
    echo "  Stage 2: Execute tests and evaluate code quality"
    echo "  Stage 3: Provide comprehensive analysis and feedback"
    echo ""
    echo "Arguments:"
    echo "  source.c          C source file to evaluate"
    echo "  output_directory  Directory to save results (default: ./evaluation_results)"
    echo ""
    echo "Example:"
    echo "  $0 student_code.c ./results"
    exit 1
}

# Main evaluation pipeline
main() {
    # Parse arguments
    if [ $# -lt 1 ] || [ $# -gt 2 ]; then
        usage
    fi
    
    local source_file="$1"
    local output_dir="${2:-./evaluation_results_$TIMESTAMP}"
    
    # Validate input file
    if [ ! -f "$source_file" ]; then
        print_error "Source file '$source_file' not found"
        exit 1
    fi
    
    # Create directories
    mkdir -p "$output_dir"
    mkdir -p "$TEMP_DIR"
    
    local abs_source_file="$(realpath "$source_file")"
    local abs_output_dir="$(realpath "$output_dir")"
    
    print_stage "COMPREHENSIVE C CODE EVALUATION PIPELINE"
    echo -e "${BLUE}Source File:${NC} $abs_source_file"
    echo -e "${BLUE}Output Directory:${NC} $abs_output_dir"
    echo -e "${BLUE}Timestamp:${NC} $TIMESTAMP"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # ==================== STAGE 1: TEST CASE GENERATION ====================
    print_stage "STAGE 1: INTELLIGENT TEST CASE GENERATION"
    print_info "Using CodeLlama to analyze code and generate test cases..."
    
    local test_cases_file="$TEMP_DIR/generated_tests.json"
    
    if python3 "$SCRIPT_DIR/test_generator.py" "$abs_source_file" "$test_cases_file"; then
        print_success "Test cases generated successfully"
        cp "$test_cases_file" "$abs_output_dir/generated_test_cases.json"
    else
        print_error "Failed to generate test cases"
        exit 1
    fi
    
    # ==================== STAGE 2: CODE EVALUATION ====================
    print_stage "STAGE 2: CODE EXECUTION AND EVALUATION"
    print_info "Compiling and testing code with generated test cases..."
    
    local evaluation_results="$TEMP_DIR/eval_results.json"
    
    # Compile the enhanced evaluator if needed
    local evaluator_exe="$TEMP_DIR/enhanced_evaluator"
    if ! gcc -o "$evaluator_exe" "$SCRIPT_DIR/enhanced_safe_eval.c" -ljson-c; then
        print_error "Failed to compile enhanced evaluator"
        exit 1
    fi
    
    # Run evaluation
    if "$evaluator_exe" "$abs_source_file" "$test_cases_file"; then
        print_success "Code evaluation completed"
        # Copy results from /tmp/eval_results.json to our output directory
        cp "/tmp/eval_results.json" "$abs_output_dir/evaluation_metrics.json"
        cp "/tmp/eval_results.json" "$evaluation_results"
    else
        print_error "Code evaluation failed"
        exit 1
    fi
    
    # ==================== STAGE 3: COMPREHENSIVE ANALYSIS ====================
    print_stage "STAGE 3: COMPREHENSIVE CODE ANALYSIS AND FEEDBACK"
    print_info "Using CodeLlama to analyze results and provide feedback..."
    
    local final_report="$abs_output_dir/comprehensive_report.txt"
    
    if python3 "$SCRIPT_DIR/code_analyzer.py" "$abs_source_file" "$evaluation_results" "$final_report"; then
        print_success "Comprehensive analysis completed"
    else
        print_error "Failed to complete comprehensive analysis"
        exit 1
    fi
    
    # ==================== GENERATE SUMMARY ====================
    print_stage "EVALUATION SUMMARY"
    
    # Extract key metrics from results
    if [ -f "$abs_output_dir/evaluation_metrics.json" ]; then
        local passrate=$(python3 -c "import json; print(json.load(open('$abs_output_dir/evaluation_metrics.json')).get('passrate', 0))")
        local memory_score=$(python3 -c "import json; print(json.load(open('$abs_output_dir/evaluation_metrics.json')).get('memory_score', 0))")
        local robustness_score=$(python3 -c "import json; print(json.load(open('$abs_output_dir/evaluation_metrics.json')).get('robustness_score', 0))")
        local tests_passed=$(python3 -c "import json; print(json.load(open('$abs_output_dir/evaluation_metrics.json')).get('tests_passed', 0))")
        local total_tests=$(python3 -c "import json; print(json.load(open('$abs_output_dir/evaluation_metrics.json')).get('total_tests', 0))")
        
        echo -e "${GREEN}📊 EVALUATION METRICS:${NC}"
        echo -e "   Pass Rate: ${passrate}% (${tests_passed}/${total_tests} tests)"
        echo -e "   Memory Score: ${memory_score}"
        echo -e "   Robustness Score: ${robustness_score}"
    fi
    
    # Extract grade from comprehensive analysis
    if [ -f "$abs_output_dir/comprehensive_report.json" ]; then
        local grade=$(python3 -c "import json; print(json.load(open('$abs_output_dir/comprehensive_report.json')).get('overall_assessment', {}).get('grade', 'N/A'))")
        local score=$(python3 -c "import json; print(json.load(open('$abs_output_dir/comprehensive_report.json')).get('overall_assessment', {}).get('score', 0))")
        
        echo -e "${GREEN}🎓 FINAL ASSESSMENT:${NC}"
        echo -e "   Grade: ${grade}"
        echo -e "   Score: ${score}/100"
    fi
    
    print_stage "RESULTS LOCATION"
    echo -e "${GREEN}📁 All results saved to: ${abs_output_dir}${NC}"
    echo ""
    echo "Generated files:"
    echo "  📋 generated_test_cases.json    - LLM-generated test cases"
    echo "  📊 evaluation_metrics.json      - Detailed evaluation metrics"
    echo "  📄 comprehensive_report.txt     - Human-readable analysis report"
    echo "  📋 comprehensive_report.json    - Detailed analysis data"
    echo ""
    print_success "Evaluation pipeline completed successfully!"
}

# Run main function
main "$@"
