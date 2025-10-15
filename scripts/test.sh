#!/usr/bin/env bash
# Test script for Modular Audio Intelligence Engine (MAIE)
# Runs all tests with pytest and optional coverage reporting

set -euo pipefail

# Default environment variables
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
  --e2e                         Run only E2E tests (requires running services)
  --pattern PATTERN             Run tests matching pattern
  --file FILE                   Run tests from specific file

EXAMPLES:
  $(basename "$0")              # Run all tests quietly
  $(basename "$0") --coverage   # Run all tests with coverage
  $(basename "$0") --unit       # Run only unit tests
  $(basename "$0") --integration # Run only integration tests
  $(basename "$0") --e2e        # Run only E2E tests
  $(basename "$0") --pattern "test_api" # Run tests matching pattern
  $(basename "$0") --file tests/unit/test_config.py # Run specific test file

NOTE:
  - Only one test type (--unit, --integration, --e2e) can be specified at a time
  - --pattern and --file options cannot be used with test type filters
  - E2E tests require docker-compose services to be running
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
[[ "$E2E_ONLY" == true ]] && TEST_TYPE_COUNT=$((TEST_TYPE_COUNT + 1))

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

# Check if pixi is available
if ! command -v pixi &> /dev/null; then
  echo "Error: pixi is not installed. Please install pixi to manage the environment."
  echo "Install: curl -fsSL https://pixi.sh/install.sh | bash"
  exit 1
fi

# Sync dependencies
echo "Installing dependencies with pixi..."
pixi install --environment dev

# Build pytest command using pixi tasks (use tasks instead of calling pytest directly)
# Choose test-debug when VERBOSE is true (pyproject defines test-debug with -vv, long TB, showlocals)
PYTEST_TASK="test"
if [[ "$VERBOSE" == true ]]; then
  PYTEST_TASK="test-debug"
fi
PYTEST_CMD=(pixi run "$PYTEST_TASK")

# Add verbosity flags only when not using the test-debug task
if [[ "$QUIET" == true ]] && [[ "$VERBOSE" != true ]]; then
  PYTEST_CMD+=(-q)
fi
# Note: when VERBOSE is true we rely on the test-debug task's verbosity settings

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
elif [[ "$E2E_ONLY" == true ]]; then
  if [[ ! -d "tests/e2e" ]]; then
    echo "Error: tests/e2e directory not found"
    exit 1
  fi
  PYTEST_CMD+=(tests/e2e/)
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
