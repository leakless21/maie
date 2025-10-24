#!/bin/bash
# Run linters and formatters for the MAIE project
# Uses ruff for Python linting and formatting

set -euo pipefail

echo "Running ruff linter..."
pixi run -e dev ruff check --fix

echo "Running ruff formatter..."
pixi run -e dev ruff format

echo "Linting and formatting complete."