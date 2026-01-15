#!/usr/bin/env bash
# Development Environment Cleanup Script
# Selective cleanup of development tools and services to reduce memory usage
# Usage: ./cleanup-dev-env.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SAFE_SERVICES=(
    "ssh"
    "systemd"
    "NetworkManager"
    "dbus"
    "avahi-daemon"
    "cron"
)

# Memory monitoring functions
measure_memory_usage() {
    echo -e "\n${BLUE}=== Memory Usage Before Cleanup ===${NC}"
    ps aux | head -1
    ps aux | grep -E "(python|vscode|docker|uvicorn)" | grep -v grep || echo "No target processes found"
    echo -e "\nMemory Summary:"
    free -h
    echo -e "\nDisk Usage:"
    df -h | grep -E "(Filesystem|/home|/)"
}

compare_memory_usage() {
    echo -e "\n${BLUE}=== Memory Usage After Cleanup ===${NC}"
    ps aux | head -1
    ps aux | grep -E "(python|vscode|docker|uvicorn)" | grep -v grep || echo "No target processes found"
    echo -e "\nMemory Summary:"
    free -h
    echo -e "\nDisk Usage:"
    df -h | grep -E "(Filesystem|/home|/)"
}

# Safety checks
is_safe_service() {
    local service="$1"
    for safe_service in "${SAFE_SERVICES[@]}"; do
        if [[ "$service" == *"$safe_service"* ]]; then
            return 0
        fi
    done
    return 1
}

confirm_action() {
    local action="$1"
    local target="$2"
    echo -e "${YELLOW}⚠️  About to $action: $target${NC}"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipped: $target${NC}"
        return 1
    fi
    return 0
}

# VS Code cleanup functions
cleanup_vscode() {
    echo -e "\n${BLUE}=== VS Code Cleanup ===${NC}"
    
    # Find VS Code processes
    local vscode_pids=$(ps aux | grep -E "(vscode|code)" | grep -v grep | awk '{print $2}' || true)
    
    if [[ -n "$vscode_pids" ]]; then
        echo "Found VS Code processes: $vscode_pids"
        if confirm_action "stop VS Code processes" "vscode"; then
            echo "$vscode_pids" | xargs -r kill -TERM
            sleep 3
            # Force kill if still running
            echo "$vscode_pids" | xargs -r kill -KILL 2>/dev/null || true
            echo -e "${GREEN}✓ VS Code processes stopped${NC}"
        fi
    else
        echo "No VS Code processes found"
    fi
    
    # Clean VS Code extensions cache
    local vscode_cache_dirs=(
        "$HOME/.vscode-server"
        "$HOME/.vscode"
        "$HOME/.config/Code"
        "$HOME/Library/Application Support/Code"
    )
    
    for cache_dir in "${vscode_cache_dirs[@]}"; do
        if [[ -d "$cache_dir" ]]; then
            local size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo "VS Code cache found at $cache_dir ($size)"
            if confirm_action "clean VS Code cache" "$cache_dir"; then
                rm -rf "$cache_dir"/*
                echo -e "${GREEN}✓ Cleaned $cache_dir${NC}"
            fi
        fi
    done
}

# Python cleanup functions
cleanup_python() {
    echo -e "\n${BLUE}=== Python Cache Cleanup ===${NC}"
    
    # Find Python cache directories in project
    local cache_dirs=$(find "$PROJECT_ROOT" -type d -name "__pycache__" 2>/dev/null || true)
    local pyc_files=$(find "$PROJECT_ROOT" -name "*.pyc" 2>/dev/null || true)
    local pyo_files=$(find "$PROJECT_ROOT" -name "*.pyo" 2>/dev/null || true)
    
    if [[ -n "$cache_dirs" ]]; then
        echo "Found Python cache directories:"
        echo "$cache_dirs" | head -10
        if confirm_action "remove Python cache directories" "__pycache__"; then
            find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
            echo -e "${GREEN}✓ Python cache directories removed${NC}"
        fi
    fi
    
    if [[ -n "$pyc_files" ]]; then
        echo "Found .pyc files"
        if confirm_action "remove .pyc files" "*.pyc"; then
            find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
            echo -e "${GREEN}✓ .pyc files removed${NC}"
        fi
    fi
    
    if [[ -n "$pyo_files" ]]; then
        echo "Found .pyo files"
        if confirm_action "remove .pyo files" "*.pyo"; then
            find "$PROJECT_ROOT" -name "*.pyo" -delete 2>/dev/null || true
            echo -e "${GREEN}✓ .pyo files removed${NC}"
        fi
    fi
    
    # Clean pip cache
    if command -v pip >/dev/null 2>&1; then
        echo "Cleaning pip cache..."
        pip cache purge 2>/dev/null || true
        echo -e "${GREEN}✓ Pip cache cleaned${NC}"
    fi
    
    # Clean uv cache if available
    if command -v uv >/dev/null 2>&1; then
        echo "Cleaning uv cache..."
        uv cache clean 2>/dev/null || true
        echo -e "${GREEN}✓ UV cache cleaned${NC}"
    fi
}

# Docker cleanup functions
cleanup_docker() {
    echo -e "\n${BLUE}=== Docker Cleanup ===${NC}"
    
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker not found, skipping Docker cleanup"
        return
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        echo "Docker not running, skipping Docker cleanup"
        return
    fi
    
    # List running containers
    local running_containers=$(docker ps --format "{{.Names}} ({{.Image}})" 2>/dev/null || true)
    if [[ -n "$running_containers" ]]; then
        echo "Running containers:"
        echo "$running_containers"
        if confirm_action "stop Docker containers" "containers"; then
            docker stop $(docker ps -q) 2>/dev/null || true
            echo -e "${GREEN}✓ Docker containers stopped${NC}"
        fi
    else
        echo "No running Docker containers found"
    fi
    
    # Clean Docker system (dangling images, unused volumes, etc.)
    if confirm_action "clean Docker system" "unused images and volumes"; then
        docker system prune -f --volumes 2>/dev/null || true
        echo -e "${GREEN}✓ Docker system cleaned${NC}"
    fi
}

# Package manager cleanup
cleanup_package_managers() {
    echo -e "\n${BLUE}=== Package Manager Cleanup ===${NC}"
    
    # APT cleanup (Debian/Ubuntu)
    if command -v apt >/dev/null 2>&1; then
        echo "Cleaning APT cache..."
        sudo apt autoremove -y 2>/dev/null || true
        sudo apt autoclean 2>/dev/null || true
        echo -e "${GREEN}✓ APT cache cleaned${NC}"
    fi
    
    # NPM cleanup
    if command -v npm >/dev/null 2>&1; then
        echo "Cleaning NPM cache..."
        npm cache clean --force 2>/dev/null || true
        echo -e "${GREEN}✓ NPM cache cleaned${NC}"
    fi
    
    # Yarn cleanup
    if command -v yarn >/dev/null 2>&1; then
        echo "Cleaning Yarn cache..."
        yarn cache clean 2>/dev/null || true
        echo -e "${GREEN}✓ Yarn cache cleaned${NC}"
    fi
    
    # PNPM cleanup
    if command -v pnpm >/dev/null 2>&1; then
        echo "Cleaning PNPM store..."
        pnpm store prune 2>/dev/null || true
        echo -e "${GREEN}✓ PNPM store cleaned${NC}"
    fi
}

# Application-specific cleanup
cleanup_application() {
    echo -e "\n${BLUE}=== Application Cleanup ===${NC}"
    
    # Stop uvicorn/ASGI servers
    local uvicorn_pids=$(ps aux | grep uvicorn | grep -v grep | awk '{print $2}' || true)
    if [[ -n "$uvicorn_pids" ]]; then
        echo "Found uvicorn processes: $uvicorn_pids"
        if confirm_action "stop uvicorn servers" "uvicorn"; then
            echo "$uvicorn_pids" | xargs -r kill -TERM
            sleep 3
            echo "$uvicorn_pids" | xargs -r kill -KILL 2>/dev/null || true
            echo -e "${GREEN}✓ Uvicorn processes stopped${NC}"
        fi
    else
        echo "No uvicorn processes found"
    fi
    
    # Stop any background processes related to the project
    local project_processes=$(ps aux | grep -E "(python.*main|maie|workers)" | grep -v grep | awk '{print $2}' || true)
    if [[ -n "$project_processes" ]]; then
        echo "Found project processes: $project_processes"
        if confirm_action "stop project processes" "maie-related processes"; then
            echo "$project_processes" | xargs -r kill -TERM
            sleep 3
            echo "$project_processes" | xargs -r kill -KILL 2>/dev/null || true
            echo -e "${GREEN}✓ Project processes stopped${NC}"
        fi
    else
        echo "No project processes found"
    fi
}

# Log cleanup
cleanup_logs() {
    echo -e "\n${BLUE}=== Log Cleanup ===${NC}"
    
    # Find log files in project
    local log_files=$(find "$PROJECT_ROOT" -name "*.log" -o -name "logs" -type d 2>/dev/null || true)
    
    if [[ -n "$log_files" ]]; then
        echo "Found log files/directories"
        if confirm_action "clean log files" "project logs"; then
            find "$PROJECT_ROOT" -name "*.log" -delete 2>/dev/null || true
            find "$PROJECT_ROOT" -name "logs" -type d -exec rm -rf {} + 2>/dev/null || true
            echo -e "${GREEN}✓ Log files cleaned${NC}"
        fi
    else
        echo "No project log files found"
    fi
    
    # Clean system logs if user confirms
    if confirm_action "clean system logs" "/var/log/*"; then
        sudo journalctl --vacuum-time=1d 2>/dev/null || true
        echo -e "${GREEN}✓ System logs cleaned${NC}"
    fi
}

# Help function
show_help() {
    cat << EOF
Development Environment Cleanup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -v, --vscode        Clean VS Code only
    -p, --python        Clean Python cache only
    -d, --docker        Clean Docker only
    -a, --apps          Clean application processes only
    -l, --logs          Clean log files only
    -m, --packages      Clean package manager caches only
    -f, --full          Full cleanup (all of the above)
    -s, --safe          Interactive mode with safety confirmations
    --measure           Show memory usage before and after
    --no-confirm        Skip all confirmations (use with caution)

Examples:
    $0 --vscode                    # Clean VS Code only
    $0 --python --logs             # Clean Python cache and logs
    $0 --full --measure            # Full cleanup with memory measurement
    $0 --safe                      # Interactive mode with confirmations

EOF
}

# Main function
main() {
    local vscode=false
    local python=false
    local docker=false
    local apps=false
    local logs=false
    local packages=false
    local full=false
    local safe=false
    local measure=false
    local no_confirm=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--vscode)
                vscode=true
                shift
                ;;
            -p|--python)
                python=true
                shift
                ;;
            -d|--docker)
                docker=true
                shift
                ;;
            -a|--apps)
                apps=true
                shift
                ;;
            -l|--logs)
                logs=true
                shift
                ;;
            -m|--packages)
                packages=true
                shift
                ;;
            -f|--full)
                full=true
                shift
                ;;
            -s|--safe)
                safe=true
                shift
                ;;
            --measure)
                measure=true
                shift
                ;;
            --no-confirm)
                no_confirm=true
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Default to safe mode if no options specified
    if [[ "$vscode" == false && "$python" == false && "$docker" == false && 
          "$apps" == false && "$logs" == false && "$packages" == false && 
          "$full" == false && "$safe" == false ]]; then
        safe=true
    fi
    
    echo -e "${BLUE}🧹 Development Environment Cleanup${NC}"
    echo "====================================="
    
    # Show memory usage before
    if [[ "$measure" == true ]]; then
        measure_memory_usage
    fi
    
    # Set up signal handlers for cleanup on interrupt
    trap 'echo -e "\n${YELLOW}Cleanup interrupted by user${NC}"; exit 130' INT
    
    # Execute cleanup based on options
    if [[ "$full" == true || "$safe" == true ]]; then
        cleanup_vscode
        cleanup_python
        cleanup_docker
        cleanup_application
        cleanup_logs
        cleanup_package_managers
    else
        [[ "$vscode" == true ]] && cleanup_vscode
        [[ "$python" == true ]] && cleanup_python
        [[ "$docker" == true ]] && cleanup_docker
        [[ "$apps" == true ]] && cleanup_application
        [[ "$logs" == true ]] && cleanup_logs
        [[ "$packages" == true ]] && cleanup_package_managers
    fi
    
    # Show memory usage after cleanup
    if [[ "$measure" == true ]]; then
        compare_memory_usage
    fi
    
    echo -e "\n${GREEN}✓ Cleanup completed!${NC}"
    echo -e "\n${BLUE}Tips:${NC}"
    echo "• Run with --measure to see memory usage before/after"
    echo "• Use --safe for interactive mode with confirmations"
    echo "• Check system health: systemctl status"
    echo "• Monitor memory: watch -n 2 'free -h'"
}

# Run main function
main "$@"