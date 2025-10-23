#!/bin/bash

# Disk space monitoring script for MAIE project
# Monitors disk usage and can trigger emergency cleanup if needed

set -euo pipefail

DISK_THRESHOLD="${DISK_THRESHOLD:-80}"
CHECK_DIR="${CHECK_DIR:-.}"
EMERGENCY_CLEANUP="${EMERGENCY_CLEANUP:-false}"

echo "Disk monitor script for MAIE project"
echo "Monitoring directory: $CHECK_DIR"
echo "Alert threshold: $DISK_THRESHOLD%"
echo "Emergency cleanup: $EMERGENCY_CLEANUP"
echo

# Function to get disk usage percentage for a directory
get_disk_usage() {
    local dir="$1"
    # Use df to get filesystem usage for the directory
    df -P "$dir" | tail -1 | awk '{print $5}' | sed 's/%//'
}

# Function to get human-readable disk usage
get_disk_info() {
    local dir="$1"
    echo "Disk usage for $dir:"
    df -h "$dir" | tail -1
}

# Function to trigger emergency cleanup
emergency_cleanup() {
    echo "WARNING: Disk usage above threshold! Triggering emergency cleanup..."
    echo

    # Get the directory where this script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Run emergency cleanup scripts
    local scripts=("clean-logs.sh" "clean-audio.sh")

    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        if [ -x "$script_path" ]; then
            echo "Running emergency cleanup: $script"
            "$script_path" || echo "Warning: $script failed during emergency cleanup"
        fi
    done

    echo
    echo "Emergency cleanup completed."
}

# Main monitoring logic
main() {
    # Get current disk usage
    local usage
    usage=$(get_disk_usage "$CHECK_DIR")

    # Show current disk status
    get_disk_info "$CHECK_DIR"
    echo
    echo "Current disk usage: ${usage}%"
    echo "Threshold: ${DISK_THRESHOLD}%"
    echo

    # Check if usage exceeds threshold
    if [ "$usage" -gt "$DISK_THRESHOLD" ]; then
        echo "ALERT: Disk usage (${usage}%) exceeds threshold (${DISK_THRESHOLD}%)!"

        if [ "$EMERGENCY_CLEANUP" = "true" ]; then
            emergency_cleanup

            # Check usage after cleanup
            local new_usage
            new_usage=$(get_disk_usage "$CHECK_DIR")
            echo
            echo "Disk usage after emergency cleanup: ${new_usage}%"

            if [ "$new_usage" -gt "$DISK_THRESHOLD" ]; then
                echo "WARNING: Disk usage still above threshold after emergency cleanup."
                return 1
            else
                echo "✓ Disk usage reduced to acceptable levels."
            fi
        else
            echo "Emergency cleanup disabled. Manual intervention required."
            return 1
        fi
    else
        echo "✓ Disk usage is within acceptable limits."
    fi
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Monitor disk space usage for MAIE project directories."
    echo
    echo "Environment variables:"
    echo "  DISK_THRESHOLD     Alert threshold percentage (default: 80)"
    echo "  CHECK_DIR          Directory to monitor (default: .)"
    echo "  EMERGENCY_CLEANUP  Enable automatic cleanup when threshold exceeded (default: false)"
    echo
    echo "Exit codes:"
    echo "  0  Success - disk usage within limits"
    echo "  1  Alert - disk usage exceeded threshold"
    echo
    echo "Examples:"
    echo "  $0                                    # Check current usage"
    echo "  DISK_THRESHOLD=90 $0                 # Custom threshold"
    echo "  CHECK_DIR=/data $0                   # Monitor specific directory"
    echo "  EMERGENCY_CLEANUP=true $0           # Enable auto-cleanup on alert"
}

# Handle help flag
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    show_usage
    exit 0
fi

# Run main monitoring
main