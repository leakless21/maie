#!/usr/bin/env bash
set -euo pipefail

# Docker Build Script for MAIE
# Builds optimized production images with proper metadata

# Configuration
VERSION="${VERSION:-1.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
IMAGE_NAME="${IMAGE_NAME:-maie}"
REGISTRY="${REGISTRY:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
  log_info "Checking prerequisites..."
  
  if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
  fi
  
  if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    exit 1
  fi
  
  log_info "Prerequisites check passed"
}

# Build image
build_image() {
  local tag="${IMAGE_NAME}:${VERSION}"
  local latest_tag="${IMAGE_NAME}:latest"
  
  log_info "Building Docker image: ${tag}"
  log_info "Build date: ${BUILD_DATE}"
  log_info "VCS ref: ${VCS_REF}"
  
  docker build \
    --target production \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg VERSION="${VERSION}" \
    --tag "${tag}" \
    --tag "${latest_tag}" \
    .
  
  log_info "Build completed successfully"
  
  # Show image details
  log_info "Image details:"
  docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

# Tag for registry
tag_for_registry() {
  if [[ -n "${REGISTRY}" ]]; then
    local local_tag="${IMAGE_NAME}:${VERSION}"
    local registry_tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    local registry_latest="${REGISTRY}/${IMAGE_NAME}:latest"
    
    log_info "Tagging for registry: ${REGISTRY}"
    docker tag "${local_tag}" "${registry_tag}"
    docker tag "${local_tag}" "${registry_latest}"
    
    log_info "Registry tags created"
  fi
}

# Push to registry
push_image() {
  if [[ -n "${REGISTRY}" ]]; then
    local registry_tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    local registry_latest="${REGISTRY}/${IMAGE_NAME}:latest"
    
    log_info "Pushing to registry: ${REGISTRY}"
    docker push "${registry_tag}"
    docker push "${registry_latest}"
    
    log_info "Push completed successfully"
  else
    log_warn "No registry specified, skipping push"
  fi
}

# Main execution
main() {
  log_info "Starting MAIE Docker build"
  log_info "Version: ${VERSION}"
  
  check_prerequisites
  build_image
  tag_for_registry
  
  if [[ "${PUSH:-false}" == "true" ]]; then
    push_image
  fi
  
  log_info "Build process completed successfully!"
}

# Help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  cat <<EOF
MAIE Docker Build Script

Usage: $0 [OPTIONS]

Environment Variables:
  VERSION         Version tag for the image (default: 1.0.0)
  IMAGE_NAME      Name of the Docker image (default: maie)
  REGISTRY        Container registry URL (optional)
  PUSH            Push to registry if set to 'true' (default: false)

Examples:
  # Build with default version
  $0

  # Build specific version
  VERSION=1.2.3 $0

  # Build and push to registry
  VERSION=1.2.3 REGISTRY=registry.example.com PUSH=true $0

EOF
  exit 0
fi

main
