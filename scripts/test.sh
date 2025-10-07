#!/usr/bin/env bash
# Test script for Modular Audio Intelligence Engine (MAIE)
# Runs all tests with pytest and optional coverage reporting

set -euo pipefail

# Default environment variables
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
export ENVIRONMENT="${ENVIRONMENT:-test}"
export LOG_LEVEL="${LOG_LEVEL:-WARNING}"

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
  --e2e                         Run only end-to-end tests
  --pattern PATTERN             Run tests matching pattern
  --file FILE                   Run tests from specific file

EXAMPLES:
  $(basename "$0")              # Run all tests quietly
  $(basename "$0") --coverage   # Run all tests with coverage
  $(basename "$0") --unit       # Run only unit tests
  $(basename "$0") --pattern "test_api" # Run tests matching pattern
EOF
}

# Default values
COVERAGE=false
QUIET=true
VERBOSE=false
UNIT_ONLY=false
INTEGRATION_ONLY=false
E2E_ONLY=false
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
    --e2e)
      E2E_ONLY=true
      shift
      ;;
    --pattern)
      PATTERN="$2"
      shift 2
      ;;
    --file)
      TEST_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if uv is available
if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Please install uv to manage the Python environment."
  echo "You can install it with: pip install uv"
  exit 1
fi

# Create and activate virtual environment
echo "Setting up test environment with uv..."
uv venv --seed --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies and test requirements..."
uv pip install -e ".[test]"

# Build pytest command
PYTEST_CMD="uv run pytest"
if [[ "$QUIET" == true ]]; then
  PYTEST_CMD="$PYTEST_CMD -q"
fi

if [[ "$VERBOSE" == true ]]; then
  PYTEST_CMD="$PYTEST_CMD -v"
fi

if [[ "$UNIT_ONLY" == true ]]; then
  PYTEST_CMD="$PYTEST_CMD tests/unit/"
elif [[ "$INTEGRATION_ONLY" == true ]]; then
  PYTEST_CMD="$PYTEST_CMD tests/integration/"
elif [[ "$E2E_ONLY" == true ]]; then
  PYTEST_CMD="$PYTEST_CMD tests/e2e/"
elif [[ -n "$TEST_FILE" ]]; then
  PYTEST_CMD="$PYTEST_CMD $TEST_FILE"
elif [[ -n "$PATTERN" ]]; then
  PYTEST_CMD="$PYTEST_CMD -k '$PATTERN'"
fi

# Add coverage if requested
if [[ "$COVERAGE" == true ]]; then
  echo "Running tests with coverage report..."
  # Install coverage if not already installed
  uv pip install coverage[toml]
  
  # Run with coverage
  eval "$PYTEST_CMD --cov=src --cov-report=html --cov-report=term"
  echo "Coverage report generated in htmlcov/ directory"
else
  echo "Running tests..."
  eval "$PYTEST_CMD"
fi

# Check the exit code of the test run
TEST_EXIT_CODE=$?

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
  echo "All tests passed!"
else
  echo "Some tests failed with exit code: $TEST_EXIT_CODE"
  exit $TEST_EXIT_CODE
fi