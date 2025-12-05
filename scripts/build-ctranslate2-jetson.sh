#!/usr/bin/env bash
# CTranslate2 build script for Jetson Orin Nano Super (ARM64)
# Based on: https://github.com/dusty-nv/jetson-containers/blob/master/packages/ml/ctranslate2/build.sh
# Uses pixi for package management (https://pixi.sh)
set -ex

# Configuration (can be overridden via environment)
CTRANSLATE_VERSION="${CTRANSLATE_VERSION:-4.6.1}"
CTRANSLATE_BRANCH="${CTRANSLATE_BRANCH:-v${CTRANSLATE_VERSION}}"
CTRANSLATE_SOURCE="${CTRANSLATE_SOURCE:-/opt/CTranslate2}"

echo "Building CTranslate2 ${CTRANSLATE_VERSION} (${CTRANSLATE_BRANCH})"

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

# Clone sources (or reuse existing directory)
if [[ -d "${CTRANSLATE_SOURCE}/.git" ]]; then
  echo "CTranslate2 repository already exists, using existing clone"
  cd "${CTRANSLATE_SOURCE}"
  git fetch origin "${CTRANSLATE_BRANCH}" 2>/dev/null || true
  git checkout "${CTRANSLATE_BRANCH}" 2>/dev/null || git switch -c "${CTRANSLATE_BRANCH}" origin/"${CTRANSLATE_BRANCH}" 2>/dev/null || true
else
  echo "Cloning CTranslate2 repository..."
  git clone --branch="${CTRANSLATE_BRANCH}" --recursive https://github.com/OpenNMT/CTranslate2.git "${CTRANSLATE_SOURCE}" || \
  git clone --recursive https://github.com/OpenNMT/CTranslate2.git "${CTRANSLATE_SOURCE}"
fi

mkdir -p "${CTRANSLATE_SOURCE}/build"
cd "${CTRANSLATE_SOURCE}/build"

install_dir="${CTRANSLATE_SOURCE}/build/install"

# Build C++ libraries (matches official dusty-nv script exactly)
cmake .. \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_INSTALL_PREFIX="${install_dir}"

make -j$(nproc)
make install

# Install to system
cp -r "${install_dir}"/* /usr/local/
ldconfig

# Build Python packages using pixi
cd "${CTRANSLATE_SOURCE}/python"
# Install requirements in pixi environment (pybind11 etc.)
pixi_run pip install --no-build-isolation -r "${CTRANSLATE_SOURCE}/python/install_requirements.txt"
# Build wheel from pixi environment where pybind11 is available
pixi_run python "${CTRANSLATE_SOURCE}/python/setup.py" --verbose bdist_wheel --dist-dir /opt

# Install wheel using pixi
pixi_run pip install --force-reinstall /opt/ctranslate2*.whl

# Optional: upload to PyPI repository (if configured)
pixi_run twine upload --verbose /opt/ctranslate2*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL:-not-set}"

# Optional: create tarball (if tarpack is available)
if command -v tarpack >/dev/null 2>&1; then
  tarpack upload "ctranslate2-${CTRANSLATE_VERSION}" "${install_dir}" || echo "failed to upload tarball"
fi

# Verify installation
echo ""
echo "🧪 Verifying CTranslate2 installation..."
pixi_run python -c "import ctranslate2; print(f'✅ CTranslate2 {ctranslate2.__version__} installed successfully')"

echo ""
echo "🎉 CTranslate2 build completed!"
echo "Version: ${CTRANSLATE_VERSION}"
echo "Wheel location: /opt/ctranslate2*.whl"