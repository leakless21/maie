#!/usr/bin/env bash
# CTranslate2 isolated build using uv (separate from pixi project)
# Builds in /tmp/ctranslate2-build, copies wheel back to /opt
set -ex

CTRANSLATE_VERSION="${CTRANSLATE_VERSION:-4.6.1}"
CTRANSLATE_BRANCH="${CTRANSLATE_BRANCH:-v${CTRANSLATE_VERSION}}"
CTRANSLATE_SOURCE="${CTRANSLATE_SOURCE:-/opt/CTranslate2}"
BUILD_DIR="/tmp/ctranslate2-build"

echo "🔨 Building CTranslate2 ${CTRANSLATE_VERSION} in isolated uv environment"

# Clone or reuse existing CTranslate2 repository (C++ part)
if [[ -d "${CTRANSLATE_SOURCE}/.git" ]]; then
  echo "✓ CTranslate2 repository already exists at ${CTRANSLATE_SOURCE}"
  cd "${CTRANSLATE_SOURCE}"
  git fetch origin "${CTRANSLATE_BRANCH}" 2>/dev/null || true
  git checkout "${CTRANSLATE_BRANCH}" 2>/dev/null || git switch -c "${CTRANSLATE_BRANCH}" origin/"${CTRANSLATE_BRANCH}" 2>/dev/null || true
else
  echo "📦 Cloning CTranslate2 repository..."
  git clone --branch="${CTRANSLATE_BRANCH}" --recursive https://github.com/OpenNMT/CTranslate2.git "${CTRANSLATE_SOURCE}" || \
  git clone --recursive https://github.com/OpenNMT/CTranslate2.git "${CTRANSLATE_SOURCE}"
fi

# Build C++ libraries
echo "🔧 Building C++ libraries..."
mkdir -p "${CTRANSLATE_SOURCE}/build"
cd "${CTRANSLATE_SOURCE}/build"

install_dir="${CTRANSLATE_SOURCE}/build/install"

cmake .. \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_INSTALL_PREFIX="${install_dir}"

make -j$(nproc)
make install

# Install to system
echo "📍 Installing to /usr/local/..."
cp -r "${install_dir}"/* /usr/local/
ldconfig

# Build Python wheel in isolated uv environment
echo "🐍 Building Python wheel in isolated uv environment..."
mkdir -p "${BUILD_DIR}"

# Create minimal pyproject.toml for uv build
cat > "${BUILD_DIR}/pyproject.toml" << 'EOF'
[project]
name = "ctranslate2-builder"
version = "1.0.0"

[build-system]
requires = ["setuptools", "wheel", "pybind11"]
build-backend = "setuptools.build_meta"
EOF

cd "${BUILD_DIR}"

# Initialize uv virtual environment
echo "📦 Setting up uv environment..."
if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  source .venv/bin/activate
else
  # Fallback to python venv if uv not available
  python3 -m venv .venv
  source .venv/bin/activate
fi

# Install build dependencies in isolated environment
echo "📥 Installing build dependencies..."
pip install -q setuptools wheel pybind11

# Build wheel
echo "🏗️  Building wheel..."
cd "${CTRANSLATE_SOURCE}/python"
python setup.py --verbose bdist_wheel --dist-dir "${BUILD_DIR}/wheels"

# Deactivate venv
deactivate

# Copy wheel to /opt for pixi installation
echo "📋 Copying wheel to /opt..."
mkdir -p /opt
cp "${BUILD_DIR}"/wheels/ctranslate2*.whl /opt/

# Clean up build directory (optional)
echo "🧹 Cleaning up temporary build directory..."
rm -rf "${BUILD_DIR}"

# Install wheel using pixi
echo "📦 Installing wheel via pixi..."
PIXI_PROJECT_DIR="${PIXI_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if command -v pixi >/dev/null 2>&1; then
  PIXI_BIN="pixi"
elif [[ -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
elif [[ -n "${SUDO_USER:-}" ]] && [[ -x "/home/${SUDO_USER}/.pixi/bin/pixi" ]]; then
  PIXI_BIN="/home/${SUDO_USER}/.pixi/bin/pixi"
else
  echo "⚠️  pixi not found, skipping pixi installation"
  echo "To install manually: pixi run pip install --force-reinstall /opt/ctranslate2*.whl"
  exit 0
fi

(cd "${PIXI_PROJECT_DIR}" && ${PIXI_BIN} run pip install --force-reinstall --no-deps /opt/ctranslate2*.whl)

# Verify installation
echo ""
echo "🧪 Verifying CTranslate2 installation..."
(cd "${PIXI_PROJECT_DIR}" && ${PIXI_BIN} run python -c "import ctranslate2; print(f'✅ CTranslate2 {ctranslate2.__version__} installed successfully')")

echo ""
echo "🎉 CTranslate2 build completed!"
echo "Version: ${CTRANSLATE_VERSION}"
echo "Wheel location: /opt/ctranslate2*.whl"
