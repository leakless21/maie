#!/usr/bin/env bash
# Test script for Modular Audio Intelligence Engine (MAIE)
# Runs all tests with pytest and optional coverage reporting

set -euo pipefail

# Default environment variables
export ENVIRONMENT="${ENVIRONMENT:-test}"
export LOG_LEVEL="${LOG_LEVEL:-WARNING}"

# Setup cuDNN from venv if available (needed for GPU tests with CTranslate2)
# This ensures that faster-whisper can find cuDNN libraries when using GPU
# Note: We need to detect this BEFORE running uv, using the venv's Python
CUDNN_LIB_DIR=""
if [[ -f ".venv/bin/python" ]]; then
  CUDNN_LIB_DIR=$(.venv/bin/python -c "import site, os; sp = site.getsitepackages(); cudnn_dir = next((os.path.join(s, 'nvidia', 'cudnn', 'lib') for s in sp if os.path.isdir(os.path.join(s, 'nvidia', 'cudnn', 'lib'))), None); print(cudnn_dir if cudnn_dir else '')" 2>/dev/null || echo "")
elif command -v python &> /dev/null; then
  CUDNN_LIB_DIR=$(python -c "import site, os; sp = site.getsitepackages(); cudnn_dir = next((os.path.join(s, 'nvidia', 'cudnn', 'lib') for s in sp if os.path.isdir(os.path.join(s, 'nvidia', 'cudnn', 'lib'))), None); print(cudnn_dir if cudnn_dir else '')" 2>/dev/null || echo "")
fi

if [[ -n "$CUDNN_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  echo "Added cuDNN to LD_LIBRARY_PATH: $CUDNN_LIB_DIR"
fi

# Function to display help
show_help() {
cat << EOF
Usage: $(basename "$0") [OPTIONS]

Runs tests for MAIE using pytest with various options.

OPTIONS:
  -h, --help                    Show this help message
  --coverage                    Generate coverage report (html and console)
  --quiet                       Run with minimal output (default: true)
  --verbose                     Run with verbose output
  --unit                        Run only unit tests
  --integration                 Run only integration tests
  --pattern PATTERN             Run tests matching pattern
  --file FILE                   Run tests from specific file

EXAMPLES:
  $(basename "$0")              # Run all tests quietly
  $(basename "$0") --coverage   # Run all tests with coverage
  $(basename "$0") --unit       # Run only unit tests
  $(basename "$0") --pattern "test_api" # Run tests matching pattern
  $(basename "$0") --file tests/unit/test_config.py # Run specific test file

NOTE:
  - Only one test type (--unit, --integration) can be specified at a time
  - --pattern and --file options cannot be used with test type filters
EOF
}

# Default values
COVERAGE=false
QUIET=true
VERBOSE=false
UNIT_ONLY=false
INTEGRATION_ONLY=false
PATTERN=""
TEST_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    --coverage)
      COVERAGE=true
      shift
      ;;
    --quiet)
      QUIET=true
      VERBOSE=false
      shift
      ;;
    --verbose)
      VERBOSE=true
      QUIET=false
      shift
      ;;
    --unit)
      UNIT_ONLY=true
      shift
      ;;
    --integration)
      INTEGRATION_ONLY=true
      shift
      ;;
    --pattern)
      if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
        echo "Error: --pattern requires a pattern argument"
        exit 1
      fi
      PATTERN="$2"
      shift 2
      ;;
    --file)
      if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
        echo "Error: --file requires a file path argument"
        exit 1
      fi
      TEST_FILE="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Validate conflicting options
TEST_TYPE_COUNT=0
[[ "$UNIT_ONLY" == true ]] && TEST_TYPE_COUNT=$((TEST_TYPE_COUNT + 1))
[[ "$INTEGRATION_ONLY" == true ]] && TEST_TYPE_COUNT=$((TEST_TYPE_COUNT + 1))

if [[ $TEST_TYPE_COUNT -gt 1 ]]; then
  echo "Error: Cannot specify multiple test types (--unit, --integration) simultaneously"
  exit 1
fi

if [[ -n "$PATTERN" ]] && [[ -n "$TEST_FILE" ]]; then
  echo "Error: Cannot specify both --pattern and --file options"
  exit 1
fi

if [[ $TEST_TYPE_COUNT -gt 0 ]] && [[ -n "$PATTERN" || -n "$TEST_FILE" ]]; then
  echo "Error: Cannot combine test type filters with --pattern or --file options"
  exit 1
fi

# Validate test file exists if specified
if [[ -n "$TEST_FILE" ]] && [[ ! -f "$TEST_FILE" ]]; then
  echo "Error: Test file not found: $TEST_FILE"
  exit 1
fi

# Check if tests directory exists
if [[ ! -d "tests" ]]; then
  echo "Error: tests directory not found. Please run this script from the project root."
  exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Please install uv to manage the Python environment."
  echo "You can install it with: pip install uv"
  exit 1
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync --dev

# Build pytest command using array (safer than string concatenation)
PYTEST_CMD=(uv run pytest)

# Add verbosity flags
if [[ "$QUIET" == true ]]; then
  PYTEST_CMD+=(-q)
fi

if [[ "$VERBOSE" == true ]]; then
  PYTEST_CMD+=(-v)
fi

# Add coverage flags if requested
if [[ "$COVERAGE" == true ]]; then
  PYTEST_CMD+=(--cov=src --cov-report=html --cov-report=term)
fi

# Add test path/filter arguments
if [[ "$UNIT_ONLY" == true ]]; then
  if [[ ! -d "tests/unit" ]]; then
    echo "Error: tests/unit directory not found"
    exit 1
  fi
  PYTEST_CMD+=(tests/unit/)
elif [[ "$INTEGRATION_ONLY" == true ]]; then
  if [[ ! -d "tests/integration" ]]; then
    echo "Error: tests/integration directory not found"
    exit 1
  fi
  PYTEST_CMD+=(tests/integration/)
elif [[ -n "$TEST_FILE" ]]; then
  PYTEST_CMD+=("$TEST_FILE")
elif [[ -n "$PATTERN" ]]; then
  PYTEST_CMD+=(-k "$PATTERN")
fi

# Run tests
if [[ "$COVERAGE" == true ]]; then
  echo "Running tests with coverage report..."
else
  echo "Running tests..."
fi

# Execute pytest and capture exit code immediately
"${PYTEST_CMD[@]}"
TEST_EXIT_CODE=$?

# Display results
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
  echo "✓ All tests passed!"
  if [[ "$COVERAGE" == true ]]; then
    echo "Coverage report generated in htmlcov/ directory"
  fi
else
  echo "✗ Some tests failed (exit code: $TEST_EXIT_CODE)"
  exit $TEST_EXIT_CODE
fi