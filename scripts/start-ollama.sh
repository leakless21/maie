#!/bin/bash
# Ollama Server Launch Script with MAIE Configuration
# This script ensures Ollama is running and configured for MAIE.
#
# Usage:
#   ./scripts/start-ollama.sh              # Start/check Ollama and pull models
#   ./scripts/start-ollama.sh --status     # Show status only
#   ./scripts/start-ollama.sh --systemd    # Generate systemd service file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default model for MAIE
DEFAULT_MODEL="${OLLAMA_MODEL:-ministral-3:3b}"

# Ollama settings
OLLAMA_HOST="${OLLAMA_HOST:-localhost}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  MAIE Ollama Launcher${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if Ollama is installed
check_ollama_installed() {
    if command -v ollama &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check if Ollama server is running
check_ollama_running() {
    local host="${1:-localhost}"
    local port="${2:-11434}"
    
    if curl -sf "http://${host}:${port}/api/version" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Wait for Ollama to be ready
wait_for_ollama() {
    local host="${1:-localhost}"
    local port="${2:-11434}"
    local max_wait=30
    local count=0
    
    echo -n "Waiting for Ollama to be ready"
    while ! check_ollama_running "$host" "$port"; do
        if [ $count -ge $max_wait ]; then
            echo ""
            print_error "Timeout waiting for Ollama"
            return 1
        fi
        echo -n "."
        sleep 1
        ((count++))
    done
    echo ""
    return 0
}

# Get list of available models
list_models() {
    local host="${1:-localhost}"
    local port="${2:-11434}"
    
    curl -sf "http://${host}:${port}/api/tags" 2>/dev/null | \
        python3 -c "import sys,json; data=json.load(sys.stdin); print('\n'.join(m['name'] for m in data.get('models',[])))" 2>/dev/null || echo ""
}

# Check if a model exists
model_exists() {
    local model="$1"
    local host="${2:-localhost}"
    local port="${3:-11434}"
    
    local models=$(list_models "$host" "$port")
    if echo "$models" | grep -q "^${model}$"; then
        return 0
    fi
    # Also check without tag
    local base_model="${model%%:*}"
    if echo "$models" | grep -q "^${base_model}"; then
        return 0
    fi
    return 1
}

# Pull a model
pull_model() {
    local model="$1"
    
    echo -e "${YELLOW}Pulling model: ${model}${NC}"
    ollama pull "$model"
}

# Generate systemd service file
generate_systemd() {
    local user="${USER:-$(whoami)}"
    local ollama_path=$(command -v ollama)
    
    cat << EOF
# Ollama systemd service for MAIE
# Save to: /etc/systemd/system/ollama.service
# Then run:
#   sudo systemctl daemon-reload
#   sudo systemctl enable ollama
#   sudo systemctl start ollama

[Unit]
Description=Ollama Server for MAIE
After=network.target

[Service]
Type=simple
User=${user}
ExecStart=${ollama_path} serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0"

[Install]
WantedBy=multi-user.target
EOF
}

# Show status
show_status() {
    cd "$PROJECT_ROOT"
    
    echo ""
    echo -e "${BLUE}Ollama Status:${NC}"
    echo "─────────────────────────────────────────"
    
    if check_ollama_installed; then
        print_success "Ollama installed: $(ollama --version 2>/dev/null | head -1)"
    else
        print_error "Ollama not installed"
        echo ""
        echo "Install from: https://ollama.ai/download"
        return 1
    fi
    
    if check_ollama_running "$OLLAMA_HOST" "$OLLAMA_PORT"; then
        print_success "Ollama server running at http://${OLLAMA_HOST}:${OLLAMA_PORT}"
    else
        print_warning "Ollama server not running"
        echo ""
        echo "Start with: ollama serve"
        return 1
    fi
    
    echo ""
    echo -e "${BLUE}Available Models:${NC}"
    echo "─────────────────────────────────────────"
    
    local models=$(list_models "$OLLAMA_HOST" "$OLLAMA_PORT")
    if [ -n "$models" ]; then
        echo "$models" | while read -r m; do
            echo "  - $m"
        done
    else
        echo "  (none)"
    fi
    
    echo ""
    echo -e "${BLUE}MAIE Configuration:${NC}"
    echo "─────────────────────────────────────────"
    echo "  Default model: $DEFAULT_MODEL"
    
    if model_exists "$DEFAULT_MODEL" "$OLLAMA_HOST" "$OLLAMA_PORT"; then
        print_success "Model '$DEFAULT_MODEL' is available"
    else
        print_warning "Model '$DEFAULT_MODEL' needs to be pulled"
    fi
    
    echo ""
}

# Main function
main() {
    print_header
    
    # Parse arguments
    case "${1:-}" in
        --status|-s)
            show_status
            exit $?
            ;;
        --systemd)
            generate_systemd
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --status, -s     Show Ollama status"
            echo "  --systemd        Generate systemd service file"
            echo "  --help, -h       Show this help"
            echo ""
            echo "Environment variables:"
            echo "  OLLAMA_HOST      Ollama server host (default: localhost)"
            echo "  OLLAMA_PORT      Ollama server port (default: 11434)"
            echo "  OLLAMA_MODEL     Default model (default: ministral-3b:3b)"
            echo ""
            exit 0
            ;;
    esac
    
    cd "$PROJECT_ROOT"
    
    # Step 1: Check installation
    echo -e "${BLUE}Step 1: Checking Ollama installation${NC}"
    echo "─────────────────────────────────────────"
    
    if ! check_ollama_installed; then
        print_error "Ollama is not installed"
        echo ""
        echo "Install from: https://ollama.ai/download"
        echo ""
        echo "For Linux/Jetson:"
        echo "  curl -fsSL https://ollama.ai/install.sh | sh"
        echo ""
        exit 1
    fi
    print_success "Ollama installed: $(ollama --version 2>/dev/null | head -1)"
    echo ""
    
    # Step 2: Check/start server
    echo -e "${BLUE}Step 2: Checking Ollama server${NC}"
    echo "─────────────────────────────────────────"
    
    if check_ollama_running "$OLLAMA_HOST" "$OLLAMA_PORT"; then
        print_success "Ollama server is running"
    else
        print_warning "Ollama server not running"
        echo ""
        echo "Starting Ollama server in background..."
        
        # Start ollama serve in background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!
        
        if wait_for_ollama "$OLLAMA_HOST" "$OLLAMA_PORT"; then
            print_success "Ollama server started (PID: $OLLAMA_PID)"
        else
            print_error "Failed to start Ollama server"
            echo "Check logs: /tmp/ollama.log"
            exit 1
        fi
    fi
    echo ""
    
    # Step 3: Check/pull models
    echo -e "${BLUE}Step 3: Checking models${NC}"
    echo "─────────────────────────────────────────"
    
    # Priority for model selection:
    # 1. OLLAMA_MODEL environment variable (explicit user choice)
    # 2. MAIE config (if model exists in Ollama)
    # 3. DEFAULT_MODEL fallback
    
    ENHANCE_MODEL=""
    SUMMARY_MODEL=""
    
    # If OLLAMA_MODEL is set, use it for both
    if [ -n "${OLLAMA_MODEL:-}" ]; then
        ENHANCE_MODEL="$OLLAMA_MODEL"
        SUMMARY_MODEL="$OLLAMA_MODEL"
        echo "Using OLLAMA_MODEL environment variable: $OLLAMA_MODEL"
    else
        # Try to read from MAIE config
        if command -v pixi &> /dev/null && [ -f "pyproject.toml" ]; then
            MAIE_ENHANCE=$(pixi run python -c "
import sys; sys.path.insert(0, '.')
try:
    from src.config import settings
    print(settings.llm_server.enhance_model_name or '')
except:
    print('')
" 2>/dev/null) || MAIE_ENHANCE=""

            MAIE_SUMMARY=$(pixi run python -c "
import sys; sys.path.insert(0, '.')
try:
    from src.config import settings
    print(settings.llm_server.summary_model_name or '')
except:
    print('')
" 2>/dev/null) || MAIE_SUMMARY=""
            
            # Check if MAIE config models exist in Ollama
            if [ -n "$MAIE_ENHANCE" ] && model_exists "$MAIE_ENHANCE" "$OLLAMA_HOST" "$OLLAMA_PORT"; then
                ENHANCE_MODEL="$MAIE_ENHANCE"
            fi
            if [ -n "$MAIE_SUMMARY" ] && model_exists "$MAIE_SUMMARY" "$OLLAMA_HOST" "$OLLAMA_PORT"; then
                SUMMARY_MODEL="$MAIE_SUMMARY"
            fi
        fi
    fi
    
    # Fall back to default if not set
    [ -z "$ENHANCE_MODEL" ] && ENHANCE_MODEL="$DEFAULT_MODEL"
    [ -z "$SUMMARY_MODEL" ] && SUMMARY_MODEL="$DEFAULT_MODEL"
    
    echo ""
    echo "Required models:"
    echo "  - Enhance: $ENHANCE_MODEL"
    echo "  - Summary: $SUMMARY_MODEL"
    echo ""
    
    # Build unique list of models to check
    declare -A models_to_check
    models_to_check["$ENHANCE_MODEL"]=1
    models_to_check["$SUMMARY_MODEL"]=1
    
    for model in "${!models_to_check[@]}"; do
        if model_exists "$model" "$OLLAMA_HOST" "$OLLAMA_PORT"; then
            print_success "Model '$model' is available"
        else
            print_warning "Model '$model' not found, pulling..."
            pull_model "$model"
            if model_exists "$model" "$OLLAMA_HOST" "$OLLAMA_PORT"; then
                print_success "Model '$model' pulled successfully"
            else
                print_error "Failed to pull model '$model'"
                exit 1
            fi
        fi
    done
    echo ""
    
    # Step 4: Show configuration
    echo -e "${BLUE}Step 4: MAIE Configuration${NC}"
    echo "─────────────────────────────────────────"
    echo ""
    echo "Add these to your .env file:"
    echo ""
    echo "  APP_LLM_BACKEND=vllm_server"
    echo "  APP_LLM_SERVER__ENHANCE_BASE_URL=http://${OLLAMA_HOST}:${OLLAMA_PORT}/v1"
    echo "  APP_LLM_SERVER__SUMMARY_BASE_URL=http://${OLLAMA_HOST}:${OLLAMA_PORT}/v1"
    echo "  APP_LLM_SERVER__ENHANCE_MODEL_NAME=${ENHANCE_MODEL}"
    echo "  APP_LLM_SERVER__SUMMARY_MODEL_NAME=${SUMMARY_MODEL}"
    echo "  APP_LLM_SUM__STRUCTURED_OUTPUTS_ENABLED=false"
    echo ""
    
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Ollama is ready for MAIE!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "To run MAIE with Ollama, start:"
    echo "  ./scripts/dev.sh"
    echo ""
    echo "For production, create a systemd service:"
    echo "  $0 --systemd | sudo tee /etc/systemd/system/ollama.service"
    echo "  sudo systemctl enable --now ollama"
    echo ""
}

main "$@"
