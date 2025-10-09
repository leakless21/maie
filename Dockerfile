# Multi-stage Dockerfile for Modular Audio Intelligence Engine (MAIE)
# 
# This Dockerfile creates an optimized image for both API server and worker processes
# with CUDA support for GPU-accelerated audio processing and AI inference.
# 
# Reference: TDD Section 3.6 - Deployment Architecture

# ============================================================================
# BASE STAGE - Common dependencies and environment
# ============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    curl \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3
RUN ln -sf python3.13 /usr/bin/python3

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.local/bin:${PATH}"

# Create application directory
WORKDIR /app

# Copy project files
COPY . .

# ============================================================================
# DEPENDENCIES STAGE - Install Python dependencies
# ============================================================================
FROM base AS dependencies

# Create virtual environment and install dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pyproject.toml and lock file
COPY pyproject.toml uv.lock ./

# Install dependencies with UV
RUN uv sync --locked

# ============================================================================
# PRODUCTION STAGE - Final image with minimal footprint
# ============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS production

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    python3.13 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r maie && useradd -r -g maie maie

# Create application directory and set ownership
WORKDIR /app
RUN chown -R maie:maie /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies --chown=maie:maie /opt/venv /opt/venv

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Copy application code
COPY --chown=maie:maie . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/audio /app/data/models /app/data/redis /app/templates && \
    chown -R maie:maie /app

# Switch to non-root user
USER maie

# Expose API port
EXPOSE 8000

# Default command (will be overridden by docker-compose)
CMD ["python3", "-m", "src.api.main"]