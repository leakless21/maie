#!/usr/bin/env bash
# Memory Monitoring Utilities for Development Environment
# Provides real-time monitoring and memory usage analysis
# Usage: ./memory-monitor.sh [options]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default refresh interval
REFRESH_INTERVAL=2

# Memory analysis functions
get_memory_stats() {
    local mem_info=$(free -b)
    local total=$(echo "$mem_info" | grep Mem: | awk '{print $2}')
    local used=$(echo "$mem_info" | grep Mem: | awk '{print $3}')
    local available=$(echo "$mem_info" | grep Mem: | awk '{print $7}')
    local total_gb=$(echo "scale=2; $total / 1024 / 1024 / 1024" | bc)
    local used_gb=$(echo "scale=2; $used / 1024 / 1024 / 1024" | bc)
    local available_gb=$(echo "scale=2; $available / 1024 / 1024 / 1024" | bc)
    local used_percent=$(echo "scale=1; $used * 100 / $total" | bc)
    
    echo "$total:$used:$available:$total_gb:$used_gb:$available_gb:$used_percent"
}

get_top_memory_processes() {
    ps aux --sort=-%mem | head -10
}

get_project_memory_usage() {
    echo -e "\n${BLUE}=== Project-Specific Memory Usage ===${NC}"
    
    # Python processes
    local python_procs=$(ps aux | grep -E "(python|uvicorn)" | grep -v grep || true)
    if [[ -n "$python_procs" ]]; then
        echo -e "${YELLOW}Python/UVicorn Processes:${NC}"
        echo "$python_procs"
        echo "Total Python memory: $(echo "$python_procs" | awk '{sum+=$6} END {print int(sum/1024)" MB"}')"
    else
        echo -e "${GREEN}No Python processes found${NC}"
    fi
    
    # VS Code processes
    local vscode_procs=$(ps aux | grep -E "(vscode|code)" | grep -v grep || true)
    if [[ -n "$vscode_procs" ]]; then
        echo -e "\n${YELLOW}VS Code Processes:${NC}"
        echo "$vscode_procs"
        echo "Total VS Code memory: $(echo "$vscode_procs" | awk '{sum+=$6} END {print int(sum/1024)" MB"}')"
    else
        echo -e "${GREEN}No VS Code processes found${NC}"
    fi
    
    # Docker processes
    local docker_procs=$(ps aux | grep -E "(docker|containerd)" | grep -v grep || true)
    if [[ -n "$docker_procs" ]]; then
        echo -e "\n${YELLOW}Docker/Containerd Processes:${NC}"
        echo "$docker_procs"
        echo "Total Docker memory: $(echo "$docker_procs" | awk '{sum+=$6} END {print int(sum/1024)" MB"}')"
    else
        echo -e "${GREEN}No Docker processes found${NC}"
    fi
}

get_disk_usage() {
    echo -e "\n${BLUE}=== Disk Usage Analysis ===${NC}"
    df -h | grep -E "(Filesystem|/home|/tmp|/)"
    
    echo -e "\n${YELLOW}Large Directories in Project:${NC}"
    if command -v du >/dev/null 2>&1; then
        find "$PROJECT_ROOT" -maxdepth 2 -type d -exec du -sh {} \; 2>/dev/null | sort -hr | head -10
    fi
    
    echo -e "\n${YELLOW}Cache and Temporary Files:${NC}"
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec du -sh {} \; 2>/dev/null | wc -l | xargs echo "Python cache directories found:"
    find "$PROJECT_ROOT" -name "*.pyc" 2>/dev/null | wc -l | xargs echo "Python .pyc files found:"
    find "$PROJECT_ROOT" -name "*.log" 2>/dev/null | wc -l | xargs echo "Log files found:"
}

real_time_monitor() {
    local interval=${1:-$REFRESH_INTERVAL}
    
    echo -e "${BLUE}🔍 Real-Time Memory Monitor${NC}"
    echo "Press Ctrl+C to stop monitoring"
    echo "Refresh interval: ${interval}s"
    echo "================================="
    
    # Initial stats
    local prev_stats=$(get_memory_stats)
    IFS=':' read -r prev_total prev_used prev_available prev_total_gb prev_used_gb prev_available_gb prev_used_percent <<< "$prev_stats"
    
    trap 'echo -e "\n${YELLOW}Monitoring stopped${NC}"; exit 0' INT
    
    while true; do
        clear
        echo -e "${BLUE}🔍 Real-Time Memory Monitor - $(date)${NC}"
        echo "================================================"
        
        # Current memory stats
        local current_stats=$(get_memory_stats)
        IFS=':' read -r total used available total_gb used_gb available_gb used_percent <<< "$current_stats"
        
        # Calculate changes
        local used_change=$(echo "scale=2; $used - $prev_used" | bc)
        local percent_change=$(echo "scale=1; $used_percent - $prev_used_percent" | bc)
        
        # Display current memory
        echo -e "${CYAN}Memory Usage:${NC}"
        echo "  Total: ${total_gb} GB"
        echo "  Used:  ${used_gb} GB (${used_percent}%)"
        echo "  Free:  ${available_gb} GB"
        
        if (( $(echo "$used_change > 0" | bc -l) )); then
            echo -e "  ${RED}Change: +${used_change} bytes (+${percent_change}%)${NC}"
        elif (( $(echo "$used_change < 0" | bc -l) )); then
            echo -e "  ${GREEN}Change: ${used_change} bytes (${percent_change}%)${NC}"
        else
            echo -e "  ${YELLOW}Change: No change${NC}"
        fi
        
        # Top memory processes
        echo -e "\n${CYAN}Top Memory Processes:${NC}"
        ps aux --sort=-%mem | head -5 | while read line; do
            echo "  $line"
        done
        
        # Warnings for high usage
        if (( $(echo "$used_percent > 80" | bc -l) )); then
            echo -e "\n${RED}⚠️  HIGH MEMORY USAGE WARNING: ${used_percent}% used!${NC}"
        elif (( $(echo "$used_percent > 60" | bc -l) )); then
            echo -e "\n${YELLOW}⚠️  Memory usage elevated: ${used_percent}% used${NC}"
        fi
        
        sleep "$interval"
        prev_stats=$current_stats
        IFS=':' read -r prev_total prev_used prev_available prev_total_gb prev_used_gb prev_available_gb prev_used_percent <<< "$prev_stats"
    done
}

memory_report() {
    echo -e "${BLUE}📊 Memory Usage Report${NC}"
    echo "======================="
    
    # System memory
    echo -e "${CYAN}System Memory:${NC}"
    free -h
    echo -e "\nSystem memory info:"
    cat /proc/meminfo | grep -E "(MemTotal|MemFree|MemAvailable|Cached)" | head -4
    
    # Process memory breakdown
    echo -e "\n${CYAN}Process Memory Breakdown:${NC}"
    get_project_memory_usage
    
    # Disk usage
    get_disk_usage
    
    # Memory optimization suggestions
    echo -e "\n${YELLOW}💡 Optimization Suggestions:${NC}"
    
    local memory_stats=$(get_memory_stats)
    IFS=':' read -r total used available total_gb used_gb available_gb used_percent <<< "$memory_stats"
    
    if (( $(echo "$used_percent > 70" | bc -l) )); then
        echo -e "${RED}• High memory usage detected (${used_percent}%)${NC}"
        echo "  - Run: ./cleanup-dev-env.sh --full"
        echo "  - Consider stopping development services"
        echo "  - Check for memory leaks in long-running processes"
    fi
    
    # Check for specific services
    if pgrep -f "vscode" >/dev/null; then
        echo -e "${YELLOW}• VS Code running: Consider closing unused extensions${NC}"
    fi
    
    if pgrep -f "docker" >/dev/null; then
        echo -e "${YELLOW}• Docker running: Stop unused containers${NC}"
    fi
    
    if pgrep -f "uvicorn" >/dev/null; then
        echo -e "${YELLOW}• Uvicorn running: Check if development server is needed${NC}"
    fi
    
    echo -e "${GREEN}• Run: ./production-mode.sh start - for minimal resource usage${NC}"
}

benchmark_memory() {
    echo -e "${BLUE}🧪 Memory Benchmark${NC}"
    echo "==================="
    
    # Baseline memory
    echo "Measuring baseline memory..."
    local baseline_stats=$(get_memory_stats)
    IFS=':' read -r baseline_total baseline_used baseline_available baseline_total_gb baseline_used_gb baseline_available_gb baseline_used_percent <<< "$baseline_stats"
    
    echo "Baseline: ${baseline_used_gb} GB used (${baseline_used_percent}%)"
    
    # Memory stress test (lightweight)
    echo -e "\n${YELLOW}Running memory stress test...${NC}"
    
    # Create a temporary file to simulate memory usage
    local test_size=${1:-100}  # MB
    echo "Creating ${test_size}MB test file..."
    
    dd if=/dev/zero of=/tmp/memory_test_$$ bs=1M count=$test_size 2>/dev/null
    
    # Measure after stress
    local stress_stats=$(get_memory_stats)
    IFS=':' read -r stress_total stress_used stress_available stress_total_gb stress_used_gb stress_available_gb stress_used_percent <<< "$stress_stats"
    
    echo "After stress: ${stress_used_gb} GB used (${stress_used_percent}%)"
    
    # Cleanup
    rm -f /tmp/memory_test_$$
    
    # Memory efficiency
    local memory_efficiency=$(echo "scale=1; ($stress_used - $baseline_used) / 1024 / 1024" | bc)
    echo -e "\n${GREEN}Memory efficiency test completed${NC}"
    echo "Memory delta: ${memory_efficiency} MB"
    
    if (( $(echo "$memory_efficiency < 5" | bc -l) )); then
        echo -e "${GREEN}✓ Good memory efficiency${NC}"
    else
        echo -e "${YELLOW}⚠ Potential memory inefficiency detected${NC}"
    fi
}

# Help function
show_help() {
    cat << EOF
Memory Monitoring Utilities

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    monitor     Real-time memory monitoring (default)
    report      Detailed memory usage report
    benchmark   Run memory benchmark test
    top         Show top memory consuming processes
    analyze     Analyze project-specific memory usage
    help        Show this help message

Options:
    -i, --interval SECONDS   Set refresh interval for monitor (default: 2s)
    -s, --size MB           Set test file size for benchmark (default: 100MB)

Examples:
    $0 monitor               # Real-time monitoring
    $0 monitor -i 5          # Monitor with 5-second refresh
    $0 report                # Detailed memory report
    $0 benchmark             # Run memory benchmark
    $0 benchmark -s 200      # Benchmark with 200MB test file
    $0 analyze               # Analyze project-specific usage

EOF
}

# Main function
main() {
    local command="${1:-monitor}"
    local interval=$REFRESH_INTERVAL
    local test_size=100
    
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interval)
                interval="$2"
                shift 2
                ;;
            -s|--size)
                test_size="$2"
                shift 2
                ;;
            monitor|report|benchmark|top|analyze|help)
                command="$1"
                shift
                ;;
            *)
                if [[ "$1" == "--help" || "$1" == "-h" ]]; then
                    show_help
                    exit 0
                fi
                echo -e "${RED}Unknown option: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    case "$command" in
        monitor)
            real_time_monitor "$interval"
            ;;
        report)
            memory_report
            ;;
        benchmark)
            benchmark_memory "$test_size"
            ;;
        top)
            echo -e "${BLUE}🔝 Top Memory Consuming Processes${NC}"
            get_top_memory_processes
            ;;
        analyze)
            get_project_memory_usage
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