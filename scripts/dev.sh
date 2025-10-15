#!/usr/bin/env bash
# Development script for Modular Audio Intelligence Engine (MAIE)
# Starts local development environment with API server and worker

set -euo pipefail

# Default environment variables
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
export ENVIRONMENT="${ENVIRONMENT:-development}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# Load environment file if it exists
if [[ -f .env ]]; then
  echo "Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# Function to display help
show_help() {
cat << EOF
Usage: $(basename "$0") [OPTIONS]

Starts the development environment for MAIE with API server and worker processes.

OPTIONS:
  -h, --help                    Show this help message
  --api-only                    Start only the API server
  --worker-only                 Start only the worker process
  --reload                      Enable auto-reload for development (default: true)
  --port PORT                   API server port (default: 8000)
  --host HOST                   API server host (default: localhost)

EXAMPLES:
  $(basename "$0")              # Start both API and worker
  $(basename "$0") --api-only   # Start only API server
  $(basename "$0") --worker-only # Start only worker
EOF
}

# Default values
API_ONLY=false
WORKER_ONLY=false
RELOAD=true
PORT="${PORT:-8000}"
HOST="${HOST:-localhost}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    --api-only)
      API_ONLY=true
      shift
      ;;
    --worker-only)
      WORKER_ONLY=true
      shift
      ;;
    --no-reload)
      RELOAD=false
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if pixi is available
if ! command -v pixi &> /dev/null; then
  echo "Error: pixi is not installed. Please install pixi to manage the environment."
  echo "Install: curl -fsSL https://pixi.sh/install.sh | bash"
  exit 1
fi

# Sync dependencies
echo "Installing dependencies with pixi..."
pixi install

# Start services based on options
if [[ "$API_ONLY" == true && "$WORKER_ONLY" == true ]]; then
  echo "Error: Cannot specify both --api-only and --worker-only"
  exit 1
elif [[ "$API_ONLY" == true ]]; then
  echo "Starting API server on $HOST:$PORT..."
  if [[ "$RELOAD" == true ]]; then
    exec pixi run api --host "$HOST" --port "$PORT" --reload
  else
    exec pixi run api --host "$HOST" --port "$PORT"
  fi
elif [[ "$WORKER_ONLY" == true ]]; then
  echo "Starting worker process..."
  exec pixi run worker
else
  echo "Starting API server on $HOST:$PORT and worker process..."
  
  # Start API server in background
  if [[ "$RELOAD" == true ]]; then
    pixi run api --host "$HOST" --port "$PORT" --reload &
  else
    pixi run api --host "$HOST" --port "$PORT" &
  fi
  API_PID=$!
  
  # Give API server a moment to start
  sleep 2
  
  # Start worker in foreground
  pixi run worker &
  WORKER_PID=$!
  
  # Function to handle script termination
  cleanup() {
    echo "Shutting down services..."
    kill $API_PID $WORKER_PID 2>/dev/null || true
    wait $API_PID $WORKER_PID 2>/dev/null || true
    exit 0
 }
  
  # Set up signal traps
  trap cleanup SIGINT SIGTERM
  
  # Wait for both processes
  wait $API_PID $WORKER_PID
fi
