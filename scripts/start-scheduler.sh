#!/bin/bash

# Start RQ Scheduler for MAIE application
# This script runs the scheduler as a separate process for automated cleanup tasks

set -euo pipefail

# Default environment variables
PYTHONPATH="${PYTHONPATH:-}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

echo "Starting MAIE RQ Scheduler..."
echo "Environment variables:"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  VLLM_ENABLE_V1_MULTIPROCESSING=$VLLM_ENABLE_V1_MULTIPROCESSING"
echo

# Check if we're in a virtual environment or have pixi available
if command -v pixi >/dev/null 2>&1; then
    echo "Using pixi environment"
    exec pixi run python src/scheduler/main.py
elif [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using Python virtual environment/conda environment"
    exec python src/scheduler/main.py
else
    echo "Warning: No virtual environment detected. Using system Python."
    echo "Consider using:"
    echo "  pixi run $0"
    echo "  source venv/bin/activate && $0"
    exec python src/scheduler/main.py
fi