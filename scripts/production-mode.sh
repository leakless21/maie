#!/usr/bin/env bash
# Production Mode Script
# Minimizes development overhead for edge deployment
# Usage: ./production-mode.sh [start|stop|status]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Production configuration
PRODUCTION_PID_FILE="/tmp/maie-production.pid"
PRODUCTION_LOG_FILE="/tmp/maie-production.log"
MINIMAL_SERVICES=()

# Check if running in production mode
is_production_mode() {
    [[ -f "$PRODUCTION_PID_FILE" ]] && kill -0 "$(cat "$PRODUCTION_PID_FILE" 2>/dev/null)" 2>/dev/null
}

# Start production mode
start_production_mode() {
    echo -e "${BLUE}🚀 Starting Production Mode${NC}"
    echo "=============================="
    
    if is_production_mode; then
        echo -e "${YELLOW}Production mode already running (PID: $(cat $PRODUCTION_PID_FILE))${NC}"
        return 0
    fi
    
    # Create minimal uvicorn startup
    cat > /tmp/maie-prod-server.py << 'EOF'
import os
import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "maie" if Path(__file__).parent.parent.name == "scripts" else Path(__file__).parent))

# Minimal production configuration
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("WORKERS", "1")

# Disable development features
os.environ.setdefault("DEV_MODE", "false")
os.environ.setdefault("DEBUG", "false")

# Import and run main application
try:
    from main import app
    import uvicorn
    
    logging.basicConfig(level=logging.WARNING)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        access_log=False,
        log_level="warning"
    )
except ImportError as e:
    print(f"Error importing application: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)
EOF

    # Kill development processes first
    echo "Stopping development processes..."
    pkill -f "uvicorn.*main" 2>/dev/null || true
    pkill -f "python.*main" 2>/dev/null || true
    pkill -f "vscode" 2>/dev/null || true
    pkill -f "node" 2>/dev/null || true
    
    # Start minimal production server
    echo "Starting minimal production server..."
    cd "$PROJECT_ROOT"
    python /tmp/maie-prod-server.py > "$PRODUCTION_LOG_FILE" 2>&1 &
    local server_pid=$!
    echo $server_pid > "$PRODUCTION_PID_FILE"
    
    # Wait a moment and check if started successfully
    sleep 2
    if is_production_mode; then
        echo -e "${GREEN}✓ Production mode started successfully (PID: $server_pid)${NC}"
        echo "Log file: $PRODUCTION_LOG_FILE"
        echo "Health check: curl http://localhost:8000/health"
    else
        echo -e "${RED}✗ Failed to start production mode${NC}"
        echo "Check log: $PRODUCTION_LOG_FILE"
        return 1
    fi
}

# Stop production mode
stop_production_mode() {
    echo -e "${BLUE}🛑 Stopping Production Mode${NC}"
    echo "=========================="
    
    if ! is_production_mode; then
        echo -e "${YELLOW}Production mode not running${NC}"
        return 0
    fi
    
    local pid=$(cat "$PRODUCTION_PID_FILE")
    echo "Stopping production server (PID: $pid)..."
    
    # Graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true
    sleep 3
    
    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
    
    # Clean up PID file
    rm -f "$PRODUCTION_PID_FILE"
    
    echo -e "${GREEN}✓ Production mode stopped${NC}"
}

# Show production mode status
show_status() {
    echo -e "${BLUE}📊 Production Mode Status${NC}"
    echo "========================="
    
    if is_production_mode; then
        local pid=$(cat "$PRODUCTION_PID_FILE")
        echo -e "${GREEN}✓ Production mode is running${NC}"
        echo "PID: $pid"
        echo "Log: $PRODUCTION_LOG_FILE"
        
        # Show resource usage
        if command -v ps >/dev/null 2>&1; then
            local memory=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{print int($1/1024)" MB"}' || echo "unknown")
            local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | awk '{print $1"%"}' || echo "unknown")
            echo "Memory: $memory"
            echo "CPU: $cpu"
        fi
        
        # Show health check
        if command -v curl >/dev/null 2>&1; then
            echo "Health check:"
            curl -s --connect-timeout 5 http://localhost:8000/health 2>/dev/null || echo "Server not responding"
        fi
    else
        echo -e "${YELLOW}Production mode is not running${NC}"
        echo "PID file: ${PRODUCTION_PID_FILE:-not set}"
    fi
}

# Optimize system for production
optimize_for_production() {
    echo -e "${BLUE}⚡ Optimizing System for Production${NC}"
    echo "==================================="
    
    # Set production environment variables
    export ENVIRONMENT=production
    export PYTHONUNBUFFERED=1
    export UVICORN_WORKERS=1
    export DEBUG=false
    export DEV_MODE=false
    export LOG_LEVEL=WARNING
    
    # Disable development features
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        echo "Backing up .env file..."
        cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.backup.$(date +%s)" 2>/dev/null || true
        
        # Set production variables
        cat >> "$PROJECT_ROOT/.env" << EOF

# Production optimization
ENVIRONMENT=production
DEBUG=false
DEV_MODE=false
LOG_LEVEL=WARNING
UVICORN_WORKERS=1
PYTHONUNBUFFERED=1
EOF
    fi
    
    echo -e "${GREEN}✓ System optimized for production${NC}"
}

# Cleanup production artifacts
cleanup_production() {
    echo -e "${BLUE}🧹 Cleaning Production Artifacts${NC}"
    echo "================================="
    
    # Remove temporary files
    rm -f /tmp/maie-prod-server.py
    rm -f "$PRODUCTION_LOG_FILE"
    
    # Clean up any remaining development processes
    pkill -f "uvicorn.*main" 2>/dev/null || true
    pkill -f "python.*main" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Production artifacts cleaned${NC}"
}

# Health check for production
health_check() {
    echo -e "${BLUE}🔍 Production Health Check${NC}"
    echo "==========================="
    
    if ! is_production_mode; then
        echo -e "${RED}✗ Production mode not running${NC}"
        return 1
    fi
    
    local pid=$(cat "$PRODUCTION_PID_FILE")
    
    # Check if process is responsive
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${RED}✗ Production process not responding${NC}"
        rm -f "$PRODUCTION_PID_FILE"
        return 1
    fi
    
    # Check memory usage
    local memory=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{print $1}' || echo "0")
    local memory_mb=$((memory / 1024))
    
    if [[ $memory_mb -gt 1024 ]]; then
        echo -e "${YELLOW}⚠ High memory usage: ${memory_mb} MB${NC}"
    else
        echo -e "${GREEN}✓ Memory usage normal: ${memory_mb} MB${NC}"
    fi
    
    # Check HTTP endpoint
    if command -v curl >/dev/null 2>&1; then
        local response=$(curl -s --connect-timeout 5 http://localhost:8000/health 2>/dev/null || echo "timeout")
        if [[ "$response" == "timeout" ]]; then
            echo -e "${RED}✗ HTTP endpoint not responding${NC}"
        else
            echo -e "${GREEN}✓ HTTP endpoint responding${NC}"
        fi
    fi
    
    echo -e "${GREEN}✓ Health check completed${NC}"
}

# Help function
show_help() {
    cat << EOF
Production Mode Manager
Minimizes development overhead for edge deployment

Usage: $0 [COMMAND]

Commands:
    start       Start production mode
    stop        Stop production mode
    status      Show production mode status
    health      Run health check
    optimize    Optimize system for production
    cleanup     Clean up production artifacts
    restart     Stop and start production mode

Examples:
    $0 start              # Start production mode
    $0 status             # Check status
    $0 health             # Run health check
    $0 restart            # Restart production mode

EOF
}

# Main function
main() {
    local command="${1:-help}"
    
    case "$command" in
        start)
            start_production_mode
            ;;
        stop)
            stop_production_mode
            ;;
        status)
            show_status
            ;;
        health)
            health_check
            ;;
        optimize)
            optimize_for_production
            ;;
        cleanup)
            cleanup_production
            ;;
        restart)
            stop_production_mode
            sleep 2
            start_production_mode
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $command${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"