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

# Install Pixi package manager
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Create application directory
WORKDIR /app

# Copy project files
COPY . .

# ============================================================================
# DEPENDENCIES STAGE - Install Python dependencies
# ============================================================================
FROM base AS dependencies

# Copy Pixi manifest and install dependencies
COPY pixi.toml ./
RUN pixi install

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

# Copy Pixi environment and binary from dependencies stage
COPY --from=dependencies --chown=maie:maie /root/.pixi /home/maie/.pixi
# Ensure Pixi is on PATH for the non-root user
ENV PATH="/home/maie/.pixi/bin:$PATH"

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
