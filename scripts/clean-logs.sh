#!/bin/bash

# Clean up old log files for MAIE project
# Removes log files older than specified days

set -euo pipefail

LOG_DIR="${LOG_DIR:-logs}"
DAYS_TO_KEEP="${DAYS_TO_KEEP:-7}"
DRY_RUN="${DRY_RUN:-false}"

echo "Log cleanup script for MAIE project"
echo "Log directory: $LOG_DIR"
echo "Days to keep: $DAYS_TO_KEEP"
echo "Dry run mode: $DRY_RUN"
echo

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory $LOG_DIR does not exist. Nothing to clean."
    exit 0
fi

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN: The following files would be deleted:"
    find "$LOG_DIR" -name "*.log*" -type f -mtime +$DAYS_TO_KEEP -print
    echo "DRY RUN: Completed. No files were actually deleted."
else
    echo "Deleting log files older than $DAYS_TO_KEEP days..."
    find "$LOG_DIR" -name "*.log*" -type f -mtime +$DAYS_TO_KEEP -delete
    echo "Log cleanup completed."
fi

# Show disk space used by logs
echo
echo "Current log directory size:"
du -sh "$LOG_DIR" 2>/dev/null || echo "Unable to determine log directory size"