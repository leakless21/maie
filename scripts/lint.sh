#!/usr/bin/env bash
# Lint script for Modular Audio Intelligence Engine (MAIE)
# Runs code formatters and linters to ensure code quality

set -euo pipefail

# Default environment variables
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

# Function to display help
show_help() {
cat << EOF
Usage: $(basename "$0") [OPTIONS]

Runs formatters and linters on MAIE codebase.

OPTIONS:
  -h, --help                    Show this help message
  --fix                         Automatically fix formatting issues (default: false)
  --check                       Check only mode, no fixes (default: true)
  --black                       Run Black formatter (default: true)
  --isort                       Run isort import sorter (default: true)
  --flake8                      Run flake8 linter
  --mypy                        Run mypy type checker
  --all                         Run all linters/formatters

EXAMPLES:
  $(basename "$0")              # Check formatting only
  $(basename "$0") --fix        # Fix formatting issues
  $(basename "$0") --black      # Run only Black
  $(basename "$0") --all        # Run all linters/formatters
EOF
}

# Default values
FIX=false
CHECK=true
BLACK=true
ISORT=true
FLAKE8=false
MYPY=false
RUN_ALL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    --fix)
      FIX=true
      CHECK=false
      shift
      ;;
    --check)
      CHECK=true
      FIX=false
      shift
      ;;
    --black)
      BLACK=true
      shift
      ;;
    --isort)
      ISORT=true
      shift
      ;;
    --flake8)
      FLAKE8=true
      shift
      ;;
    --mypy)
      MYPY=true
      shift
      ;;
    --all)
      RUN_ALL=true
      BLACK=true
      ISORT=true
      FLAKE8=true
      MYPY=true
      shift
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
echo "Setting up environment with uv..."
uv venv --seed --python 3.11 .venv
source .venv/bin/activate

# Install dependencies and development tools
echo "Installing dependencies and linting tools..."
uv pip install -e ".[dev]"

# Initialize exit code
EXIT_CODE=0

# Run Black formatter
if [[ "$BLACK" == true ]]; then
  echo "Running Black formatter..."
  if [[ "$FIX" == true ]]; then
    echo "  Formatting files with Black..."
    uv run black src/ tests/ scripts/ || EXIT_CODE=1
  else
    echo "  Checking Black formatting..."
    uv run black --check src/ tests/ scripts/ || EXIT_CODE=1
  fi
fi

# Run isort import sorter
if [[ "$ISORT" == true ]]; then
  echo "Running isort import sorter..."
  if [[ "$FIX" == true ]]; then
    echo "  Sorting imports with isort..."
    uv run isort src/ tests/ scripts/ || EXIT_CODE=1
  else
    echo "  Checking import sorting with isort..."
    uv run isort --check-only src/ tests/ scripts/ || EXIT_CODE=1
  fi
fi

# Run flake8 linter if requested
if [[ "$FLAKE8" == true ]]; then
  echo "Running flake8 linter..."
  uv run flake8 src/ tests/ scripts/ || EXIT_CODE=1
fi

# Run mypy type checker if requested
if [[ "$MYPY" == true ]]; then
  echo "Running mypy type checker..."
  uv run mypy src/ tests/ || EXIT_CODE=1
fi

# Summary
if [[ $EXIT_CODE -eq 0 ]]; then
  if [[ "$CHECK" == true ]]; then
    echo "All checks passed!"
  else
    echo "Formatting completed successfully!"
  fi
else
  if [[ "$CHECK" == true ]]; then
    echo "Some checks failed or formatting issues found!"
  else
    echo "Some issues found during formatting!"
  fi
  exit $EXIT_CODE
fi