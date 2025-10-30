# Multi-stage Dockerfile for Modular Audio Intelligence Engine (MAIE)
# 
# This Dockerfile creates an optimized image for both API server and worker processes
# with CUDA support for GPU-accelerated audio processing and AI inference.
# 
# Reference: TDD Section 3.6 - Deployment Architecture
# Build: docker build -t maie:latest .
# Run: docker-compose up -d

# ============================================================================
# BASE STAGE - Common dependencies and environment
# ============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

# Set build arguments for versioning
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Labels for metadata (OCI standard)
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="MAIE" \
      org.opencontainers.image.description="Modular Audio Intelligence Engine" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="MAIE Team" \
      maintainer="maie@example.com"

# Set environment variables for non-interactive installation and Python optimization
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies in a single layer with version pinning where critical
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    ca-certificates \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symbolic link for python3 (use Python 3.12 as specified in pyproject.toml)
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install Pixi package manager with retry logic
RUN curl -fsSL https://pixi.sh/install.sh | bash || \
    (sleep 5 && curl -fsSL https://pixi.sh/install.sh | bash)
ENV PATH="/root/.pixi/bin:${PATH}"

# Create application directory
WORKDIR /app

# ============================================================================
# DEPENDENCIES STAGE - Install Python dependencies
# ============================================================================
FROM base AS dependencies

# Copy only dependency files first for better layer caching
COPY pyproject.toml ./
COPY pixi.toml pixi.lock* ./

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.pixi/cache \
    pixi install --locked || pixi install

# Copy source code
COPY src/ ./src/
COPY templates/ ./templates/
COPY main.py ./

# ============================================================================
# PRODUCTION STAGE - Final image with minimal footprint
# ============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS production

# Copy labels from base
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="MAIE" \
      org.opencontainers.image.description="Modular Audio Intelligence Engine" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/home/maie/.pixi/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID for better security
RUN groupadd -r -g 1000 maie && \
    useradd -r -u 1000 -g maie -m -s /bin/bash maie

# Create application directory and set ownership
WORKDIR /app
RUN chown -R maie:maie /app

# Copy Pixi environment from dependencies stage
COPY --from=dependencies --chown=maie:maie /root/.pixi /home/maie/.pixi

# Copy application code from dependencies stage
COPY --from=dependencies --chown=maie:maie /app/src ./src
COPY --from=dependencies --chown=maie:maie /app/templates ./templates
COPY --from=dependencies --chown=maie:maie /app/main.py ./
COPY --from=dependencies --chown=maie:maie /app/pyproject.toml ./
COPY --from=dependencies --chown=maie:maie /app/pixi.toml ./

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/audio \
    /app/data/models \
    /app/data/redis \
    /app/logs \
    /app/templates && \
    chown -R maie:maie /app

# Switch to non-root user
USER maie

# Expose API port
EXPOSE 8000

# Add healthcheck (will be customized by docker-compose for each service)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (will be overridden by docker-compose)
CMD ["pixi", "run", "python", "-m", "src.api.main"]
