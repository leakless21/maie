#!/usr/bin/env bash
# Model download script for Modular Audio Intelligence Engine (MAIE)
# Downloads required AI models using huggingface-cli

set -euo pipefail

# Default environment variables
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
export HF_HOME="${HF_HOME:-$(pwd)/data/models}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"

# Function to display help
show_help() {
cat << EOF
Usage: $(basename "$0") [OPTIONS]

Downloads AI models required for MAIE using huggingface-cli.

OPTIONS:
  -h, --help                    Show this help message
  --models-dir DIR              Directory to download models to (default: data/models)
  --hf-token TOKEN              Hugging Face token for authentication
  --skip-existing               Skip download if model already exists
  --era-x                       Download EraX-WoW-Turbo V1.1 model
 --chunkformer                 Download ChunkFormer Large model
  --qwen                        Download Qwen3-4B-Instruct AWQ model
  --all                         Download all models (default)

EXAMPLES:
  $(basename "$0")              # Download all models
  $(basename "$0") --era-x      # Download only EraX model
  $(basename "$0") --models-dir /tmp/models # Download to custom directory
EOF
}

# Default values
MODELS_DIR="${MODELS_DIR:-data/models}"
HF_TOKEN=""
SKIP_EXISTING=false
DOWNLOAD_ERA_X=false
DOWNLOAD_CHUNKFORMER=false
DOWNLOAD_QWEN=false
DOWNLOAD_ALL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
 case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=true
      shift
      ;;
    --era-x)
      DOWNLOAD_ERA_X=true
      DOWNLOAD_ALL=false
      shift
      ;;
    --chunkformer)
      DOWNLOAD_CHUNKFORMER=true
      DOWNLOAD_ALL=false
      shift
      ;;
    --qwen)
      DOWNLOAD_QWEN=true
      DOWNLOAD_ALL=false
      shift
      ;;
    --all)
      DOWNLOAD_ALL=true
      DOWNLOAD_ERA_X=true
      DOWNLOAD_CHUNKFORMER=true
      DOWNLOAD_QWEN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Set default model flags if --all is true
if [[ "$DOWNLOAD_ALL" == true ]]; then
  DOWNLOAD_ERA_X=true
  DOWNLOAD_CHUNKFORMER=true
  DOWNLOAD_QWEN=true
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Please install uv to manage the Python environment."
  echo "You can install it with: pip install uv"
  exit 1
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Set HF_HOME to models directory
export HF_HOME="$MODELS_DIR"

# Authenticate if token provided
if [[ -n "$HF_TOKEN" ]]; then
  echo "Authenticating with Hugging Face..."
  uv run huggingface-cli login --token "$HF_TOKEN"
fi

# Initialize exit code
EXIT_CODE=0

# Function to download a model
download_model() {
  local model_name="$1"
  local model_path="$2"
  local description="$3"
  
  echo "Checking for $description: $model_name"
  
  if [[ "$SKIP_EXISTING" == true && -d "$model_path" ]]; then
    echo "  $description already exists at $model_path, skipping..."
  else
    echo "  Downloading $description..."
    if uv run huggingface-cli download "$model_name" --local-dir "$model_path" --local-dir-use-symlinks False; then
      echo "  Successfully downloaded $description to $model_path"
    else
      echo "  Failed to download $description: $model_name"
      return 1
    fi
  fi
}

# Download EraX-WoW-Turbo V1.1 model if requested
if [[ "$DOWNLOAD_ERA_X" == true ]]; then
 echo "Downloading EraX-WoW-Turbo V1.1 model..."
  # Note: Using a placeholder model name since the exact model identifier may vary
  # This would need to be updated with the actual model identifier
  if ! download_model "microsoft/EraX-WoW-Turbo-V1.1" "$MODELS_DIR/era-x-wow-turbo-v1.1" "EraX-WoW-Turbo V1.1"; then
    EXIT_CODE=1
 fi
fi

# Download ChunkFormer Large model if requested
if [[ "$DOWNLOAD_CHUNKFORMER" == true ]]; then
 echo "Downloading ChunkFormer Large model..."
  # Note: Using a placeholder model name since the exact model identifier may vary
  # This would need to be updated with the actual model identifier
  if ! download_model "facebook/chunkformer-large-16khz" "$MODELS_DIR/chunkformer-large" "ChunkFormer Large"; then
    EXIT_CODE=1
  fi
fi

# Download Qwen3-4B-Instruct AWQ model if requested
if [[ "$DOWNLOAD_QWEN" == true ]]; then
  echo "Downloading Qwen3-4B-Instruct AWQ model..."
  # Note: Using a placeholder model name since the exact model identifier may vary
  # This would need to be updated with the actual model identifier
  if ! download_model "Qwen/Qwen3-4B-Instruct-AWQ" "$MODELS_DIR/qwen3-4b-instruct-awq" "Qwen3-4B-Instruct AWQ"; then
    EXIT_CODE=1
  fi
fi

# Verify downloads
echo "Verifying model downloads..."
for model_dir in "$MODELS_DIR"/*; do
  if [[ -d "$model_dir" ]]; then
    model_name=$(basename "$model_dir")
    echo "  Found model: $model_name"
    # Check if model directory contains expected files
    if [[ -f "$model_dir/config.json" ]] || [[ -f "$model_dir/tokenizer.json" ]] || [[ -f "$model_dir/model.safetensors" ]] || [[ -f "$model_dir/pytorch_model.bin" ]]; then
      echo "    ✓ $model_name appears to be complete"
    else
      echo "    ⚠ $model_name may be incomplete"
    fi
  fi
done

# Summary
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "All model downloads completed successfully!"
  echo "Models are stored in: $MODELS_DIR"
else
  echo "Some model downloads failed!"
  exit $EXIT_CODE
fi