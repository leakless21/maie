#!/bin/bash
# vLLM Server Launch Script with MAIE Configuration
# This script launches a local vLLM server using settings from src/config/model.py

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set CUDA_HOME for TVM compilation (if using pixi environment)
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime" ]; then
    export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime"
    echo -e "${GREEN}✓ CUDA_HOME set to: $CUDA_HOME${NC}"
fi

# Configuration
MODEL_TYPE="${1:-enhance}"  # Default to "enhance" model config
SHOW_CONFIG="${VLLM_SHOW_CONFIG:-true}"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${GREEN}MAIE vLLM Server Launcher${NC}"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM is not installed${NC}"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check if model type is valid
if [[ "$MODEL_TYPE" != "enhance" && "$MODEL_TYPE" != "summary" ]]; then
    echo -e "${RED}Error: Invalid model type '$MODEL_TYPE'${NC}"
    echo "Usage: $0 [enhance|summary]"
    exit 1
fi

# Get vLLM arguments from MAIE configuration
echo -e "${YELLOW}Reading configuration from src/config/model.py...${NC}"

if [[ "$SHOW_CONFIG" == "true" ]]; then
    VLLM_ARGS=$(python scripts/vllm_server_config.py --model-type "$MODEL_TYPE" --show-config)
else
    VLLM_ARGS=$(python scripts/vllm_server_config.py --model-type "$MODEL_TYPE")
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to read configuration${NC}"
    exit 1
fi

# Extract model path for validation
MODEL_PATH=$(echo "$VLLM_ARGS" | grep -oP '(?<=--model )[^ ]+' || echo "")

if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓ Model found at: $MODEL_PATH${NC}"
elif [ -n "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model path '$MODEL_PATH' is not a local directory${NC}"
    echo -e "${YELLOW}         vLLM will attempt to download from HuggingFace${NC}"
fi

echo ""
echo -e "${GREEN}Starting vLLM server...${NC}"
echo ""

# Launch vLLM server with OpenAI-compatible API
exec python -m vllm.entrypoints.openai.api_server $VLLM_ARGS
