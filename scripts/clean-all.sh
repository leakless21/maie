#!/bin/bash

# Master cleanup script for MAIE project
# Runs all individual cleanup scripts in sequence

set -euo pipefail

DRY_RUN="${DRY_RUN:-false}"
SKIP_CACHE="${SKIP_CACHE:-false}"

echo "=== MAIE Master Cleanup Script ==="
echo "Runs all cleanup scripts in sequence"
echo "Dry run mode: $DRY_RUN"
echo "Skip cache cleanup: $SKIP_CACHE"
echo

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to run individual cleanup scripts
run_cleanup() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/$script_name"

    echo "--- Running $script_name ---"

    if [ -x "$script_path" ]; then
        "$script_path"
        echo "✓ $script_name completed successfully"
        echo
    else
        echo "✗ Error: $script_path not found or not executable"
        return 1
    fi
}

# Function to clear system cache (cautiously)
clear_system_cache() {
    echo "--- Clearing system cache ---"

    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN: Would clear system cache"
        echo "DRY RUN: Would clean package manager cache"
        return 0
    fi

    # Clear Python cache files
    echo "Clearing Python cache files..."
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    # Clear system package cache (with error handling)
    if command -v apt-get >/dev/null 2>&1; then
        echo "Clearing APT cache..."
        apt-get clean >/dev/null 2>&1 || echo "Warning: Failed to clean APT cache"
    elif command -v dnf >/dev/null 2>&1; then
        echo "Clearing DNF cache..."
        dnf clean all >/dev/null 2>&1 || echo "Warning: Failed to clean DNF cache"
    elif command -v yum >/dev/null 2>&1; then
        echo "Clearing YUM cache..."
        yum clean all >/dev/null 2>&1 || echo "Warning: Failed to clean YUM cache"
    fi

    echo "✓ System cache cleared"
    echo
}

# Show initial disk usage
echo "Initial disk usage:"
df -h . | tail -1
echo

# Run individual cleanup scripts
echo "Running individual cleanup scripts..."
echo

run_cleanup "clean-logs.sh"
run_cleanup "clean-audio.sh"

if [ "$SKIP_CACHE" != "true" ]; then
    run_cleanup "clean-cache.sh"
else
    echo "--- Skipping cache cleanup (SKIP_CACHE=true) ---"
    echo
fi

# Clear system cache
clear_system_cache

# Show final disk usage
echo "Final disk usage:"
df -h . | tail -1
echo

if [ "$DRY_RUN" = "true" ]; then
    echo "=== DRY RUN COMPLETED ==="
    echo "No files were actually deleted."
else
    echo "=== MASTER CLEANUP COMPLETED SUCCESSFULLY ==="
fi

echo
echo "Cleanup Summary:"
echo "- Logs: Cleaned old log files"
echo "- Audio: Cleaned old preprocessed.wav files (status checked)"
echo "- Cache: Redis TTL-based cleanup"
echo "- System: Cleared Python bytecode and package cache"