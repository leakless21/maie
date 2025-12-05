#!/usr/bin/env bash
# vLLM build script for Jetson Orin Nano Super (ARM64)
# Based on: https://github.com/dusty-nv/jetson-containers/blob/master/packages/llm/vllm/build.sh
# Uses pixi for package management (https://pixi.sh)
set -ex

# Configuration (can be overridden via environment)
VLLM_VERSION="${VLLM_VERSION:-}"
VLLM_BRANCH="${VLLM_BRANCH:-}"

echo "Building vLLM ${VLLM_VERSION:-latest} (${VLLM_BRANCH:-main})"

# Find pixi project directory (where pyproject.toml with [tool.pixi] is located)
# This script should be run from the maie project root or PIXI_PROJECT_DIR must be set
if [[ -z "${PIXI_PROJECT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PIXI_PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

if [[ ! -f "${PIXI_PROJECT_DIR}/pyproject.toml" ]]; then
  echo "❌ ERROR: pyproject.toml not found in ${PIXI_PROJECT_DIR}"
  echo "Set PIXI_PROJECT_DIR to the maie project root"
  exit 1
fi
echo "Using pixi project: ${PIXI_PROJECT_DIR}"

# Find pixi binary (may not be in root's PATH when using sudo)
if command -v pixi >/dev/null 2>&1; then
  PIXI_BIN="pixi"
elif [[ -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
elif [[ -n "${SUDO_USER:-}" ]] && [[ -x "/home/${SUDO_USER}/.pixi/bin/pixi" ]]; then
  PIXI_BIN="/home/${SUDO_USER}/.pixi/bin/pixi"
else
  echo "❌ ERROR: pixi not found. Please install pixi first"
  echo "Visit: https://pixi.sh/latest/"
  exit 1
fi
echo "Using pixi binary: ${PIXI_BIN}"

# Helper function to run pixi commands from the project directory
pixi_run() {
  (cd "${PIXI_PROJECT_DIR}" && ${PIXI_BIN} run "$@")
}

# Install pre-commit and nanobind as per official script
pixi_run pip install pre-commit nanobind==2.5.0

# Clone the repository (or reuse existing)
if [[ -d "/opt/vllm/.git" ]]; then\n  echo "vLLM repository already exists, using existing clone"
  cd /opt/vllm
  git fetch origin "${VLLM_BRANCH}" 2>/dev/null || true
  git checkout "${VLLM_BRANCH}" 2>/dev/null || true
else
  echo "Cloning vLLM repository..."
  git clone --branch="${VLLM_BRANCH}" --recursive --depth=1 https://github.com/vllm-project/vllm.git /opt/vllm || \
  git clone --recursive --depth=1 https://github.com/vllm-project/vllm.git /opt/vllm
fi

cd /opt/vllm
env

# Official requirement file optimizations
# (loosen versions for Jetson compatibility)
sed -i \
  -e 's|^gguf.*|gguf|g' \
  -e 's|^opencv-python-headless.*||g' \
  -e 's|^mistral_common.*|mistral_common|g' \
  -e 's|^compressed-tensors.*||g' \
  -e 's|^xgrammar.*||g' \
  requirements/common.txt

# Loosen flashinfer-python requirement to allow latest version
sed -i \
  -e 's|^flashinfer-python.*|flashinfer-python|g' \
  requirements/cuda.txt

grep gguf requirements/common.txt

# Official vLLM environment variables
export USE_CUDNN=1
export VERBOSE=1
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
export DG_JIT_USE_NVRTC=1  # DeepGEMM now supports NVRTC with up to 10x compilation speedup

# Set version for setuptools_scm if specified
if [[ -n "${VLLM_VERSION}" ]]; then
  export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}"
fi

pixi_run python /opt/vllm/use_existing_torch.py || echo "skipping vllm/use_existing_torch.py"

pixi_run pip install -r /opt/vllm/requirements/build.txt -v
pixi_run python -m setuptools_scm

# Jetson ARM64-specific optimizations
ARCH=$(uname -i)
if [[ "${ARCH}" = "aarch64" ]]; then
  export NVCC_THREADS=1
  export CUDA_NVCC_FLAGS="-Xcudafe --threads=1"
  export MAKEFLAGS='-j2'
  export CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-6}
  export NINJAFLAGS='-j2'
fi

cd /opt/vllm
pixi_run python -m build --wheel --no-build-isolation -v --out-dir /opt/vllm/wheels /opt/vllm
pixi_run pip install /opt/vllm/wheels/vllm*.whl

pixi_run pip install compressed-tensors

# Optional: upload to PyPI repository (if configured)
pixi_run twine upload --verbose /opt/vllm/wheels/vllm*.whl || echo "Failed to upload wheel to ${TWINE_REPOSITORY_URL:-not-set}"

# Verify installation
echo ""
echo "🧪 Verifying vLLM installation..."
pixi_run python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed successfully')"

echo ""
echo "🎉 vLLM build completed!"
echo "Wheel location: /opt/vllm/wheels/vllm*.whl"