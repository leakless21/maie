#!/usr/bin/env bash
# Development script for Modular Audio Intelligence Engine (MAIE)
# Starts local development environment with API server and worker

set -euo pipefail

# Default environment variables
export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"
export ENVIRONMENT="${ENVIRONMENT:-development}"
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"

# Default to first GPU if not explicitly set (dev expects GPU usage)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# Fix PyTorch CUDA memory fragmentation (NFR-10: Memory efficiency)
# This allows PyTorch to use expandable memory segments, preventing OOM
# errors caused by fragmentation when running sequential ASR->LLM pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  --scheduler-only              Start only the scheduler process
  --no-scheduler                Start API and worker without scheduler (no cleanup)
  --reload                      Enable auto-reload for development (default: true)
  --port PORT                   API server port (default: 8000)
  --host HOST                   API server host (default: localhost)

EXAMPLES:
  $(basename "$0")              # Start API, worker, and scheduler
  $(basename "$0") --api-only   # Start only API server
  $(basename "$0") --worker-only # Start only worker
  $(basename "$0") --scheduler-only # Start only scheduler
  $(basename "$0") --no-scheduler # Start API and worker without cleanup
EOF
}

# Default values
API_ONLY=false
WORKER_ONLY=false
SCHEDULER_ONLY=false
NO_SCHEDULER=false
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
    --scheduler-only)
      SCHEDULER_ONLY=true
      shift
      ;;
    --no-scheduler)
      NO_SCHEDULER=true
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

# Validate conflicting options
if [[ "$API_ONLY" == true && "$WORKER_ONLY" == true ]]; then
  echo "Error: Cannot specify both --api-only and --worker-only"
  exit 1
elif [[ "$API_ONLY" == true && "$SCHEDULER_ONLY" == true ]]; then
  echo "Error: Cannot specify both --api-only and --scheduler-only"
  exit 1
elif [[ "$WORKER_ONLY" == true && "$SCHEDULER_ONLY" == true ]]; then
  echo "Error: Cannot specify both --worker-only and --scheduler-only"
  exit 1
elif [[ "$NO_SCHEDULER" == true && "$SCHEDULER_ONLY" == true ]]; then
  echo "Error: Cannot specify both --no-scheduler and --scheduler-only"
  exit 1
fi

# Sync dependencies
echo "Installing dependencies with pixi..."
pixi install

# Start services based on options
if [[ "$SCHEDULER_ONLY" == true ]]; then
  echo "Starting scheduler process..."
  
  # Function to handle script termination
  cleanup() {
    echo "Shutting down scheduler process..."
    echo "Cleaning up any remaining RQ scheduler processes..."
    pkill -f "rq:scheduler" 2>/dev/null || true
    pkill -f "pixi run scheduler" 2>/dev/null || true
    echo "Cleanup completed."
    exit 0
  }
  
  # Set up signal traps
  trap cleanup SIGINT SIGTERM
  
  exec pixi run scheduler
elif [[ "$API_ONLY" == true ]]; then
  echo "Starting API server on $HOST:$PORT..."
  
  # Function to handle script termination
  cleanup() {
    echo "Shutting down API server..."
    echo "Cleaning up any remaining RQ worker processes..."
    pkill -f "rq:worker" 2>/dev/null || true
    pkill -f "pixi run worker" 2>/dev/null || true
    echo "Cleanup completed."
    exit 0
  }
  
  # Set up signal traps
  trap cleanup SIGINT SIGTERM
  
  if [[ "$RELOAD" == true ]]; then
    exec pixi run api --host "$HOST" --port "$PORT" --reload 
  else
    exec pixi run api --host "$HOST" --port "$PORT" 
  fi
elif [[ "$WORKER_ONLY" == true ]]; then
  echo "Starting worker process..."
  
  # Start worker in background to capture PID
  pixi run worker &
  WORKER_PID=$!
  
  # Function to handle script termination
  cleanup() {
    echo "Shutting down worker process..."
    
    # Kill worker process if it was started
    if [[ -n "${WORKER_PID:-}" ]]; then
      echo "Stopping worker process (PID: $WORKER_PID)..."
      kill $WORKER_PID 2>/dev/null || true
    fi
    
    # Also kill any remaining RQ worker processes
    echo "Cleaning up any remaining RQ worker processes..."
    pkill -f "rq:worker" 2>/dev/null || true
    pkill -f "pixi run worker" 2>/dev/null || true
    
    # Wait for process to terminate
    if [[ -n "${WORKER_PID:-}" ]]; then
      wait $WORKER_PID 2>/dev/null || true
    fi
    
    echo "Cleanup completed."
    exit 0
  }
  
  # Set up signal traps
  trap cleanup SIGINT SIGTERM
  
  # Wait for worker process
  wait $WORKER_PID
else
  echo "Starting API server on $HOST:$PORT, worker process, and scheduler..."
  
  # Start API server in background
  if [[ "$RELOAD" == true ]]; then
    pixi run api --host "$HOST" --port "$PORT" --reload --log-level warning &
  else
    pixi run api --host "$HOST" --port "$PORT" --log-level warning &
  fi
  API_PID=$!
  
  # Give API server a moment to start
  sleep 2
  
  # Start worker in background
  pixi run worker &
  WORKER_PID=$!
  
  # Start scheduler in background (unless disabled)
  if [[ "$NO_SCHEDULER" == false ]]; then
    pixi run scheduler &
    SCHEDULER_PID=$!
    echo "Started scheduler (PID: $SCHEDULER_PID) - using environment configuration"
  else
    echo "Scheduler disabled - no automatic cleanup will occur"
  fi
  
# Function to handle script termination
cleanup() {
  echo "Shutting down services..."
  
  # Kill API server if it was started
  if [[ -n "${API_PID:-}" ]]; then
    echo "Stopping API server (PID: $API_PID)..."
    kill $API_PID 2>/dev/null || true
  fi
  
  # Kill worker process if it was started
  if [[ -n "${WORKER_PID:-}" ]]; then
    echo "Stopping worker process (PID: $WORKER_PID)..."
    kill $WORKER_PID 2>/dev/null || true
  fi
  
  # Kill scheduler process if it was started
  if [[ -n "${SCHEDULER_PID:-}" ]]; then
    echo "Stopping scheduler process (PID: $SCHEDULER_PID)..."
    kill $SCHEDULER_PID 2>/dev/null || true
  fi
  
  # Also kill any remaining RQ processes
  echo "Cleaning up any remaining RQ processes..."
  pkill -f "rq:worker" 2>/dev/null || true
  pkill -f "rq:scheduler" 2>/dev/null || true
  pkill -f "pixi run worker" 2>/dev/null || true
  pkill -f "pixi run scheduler" 2>/dev/null || true
  
  # Wait for processes to terminate
  if [[ -n "${API_PID:-}" ]]; then
    wait $API_PID 2>/dev/null || true
  fi
  if [[ -n "${WORKER_PID:-}" ]]; then
    wait $WORKER_PID 2>/dev/null || true
  fi
  if [[ -n "${SCHEDULER_PID:-}" ]]; then
    wait $SCHEDULER_PID 2>/dev/null || true
  fi
  
  echo "Cleanup completed."
  exit 0
}
  
  # Set up signal traps
  trap cleanup SIGINT SIGTERM
  
  # Wait for all processes
  if [[ -n "${SCHEDULER_PID:-}" ]]; then
    wait $API_PID $WORKER_PID $SCHEDULER_PID
  else
    wait $API_PID $WORKER_PID
  fi
fi
