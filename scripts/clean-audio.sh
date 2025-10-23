#!/bin/bash

# Clean up old audio files (preprocessed.wav) for MAIE project
# Removes preprocessed audio files older than retention period,
# but only for tasks that are COMPLETE or FAILED

set -euo pipefail

AUDIO_DIR="${AUDIO_DIR:-data/audio}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379/1}"
DRY_RUN="${DRY_RUN:-false}"

echo "Audio cleanup script for MAIE project"
echo "Audio directory: $AUDIO_DIR"
echo "Retention days: $RETENTION_DAYS"
echo "Redis URL: $REDIS_URL"
echo "Dry run mode: $DRY_RUN"
echo

if [ ! -d "$AUDIO_DIR" ]; then
    echo "Audio directory $AUDIO_DIR does not exist. Nothing to clean."
    exit 0
fi

# Function to check task status in Redis
get_task_status() {
    local task_id="$1"
    # Extract host, port, and db from Redis URL
    # Format: redis://host:port/db
    local redis_host
    local redis_port
    local redis_db

    # Parse Redis URL
    local url_no_proto="${REDIS_URL#redis://}"
    redis_host="${url_no_proto%%:*}"
    local rest="${url_no_proto#*:}"
    redis_port="${rest%%/*}"
    redis_db="${rest#*/}"

    # Query Redis for task status
    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli -h "$redis_host" -p "$redis_port" -n "$redis_db" HGET "task:$task_id" "status" 2>/dev/null || echo "UNKNOWN"
    else
        echo "redis-cli not found, skipping status check"
        echo "UNKNOWN"
    fi
}

# Find preprocessed.wav files older than retention period
find "$AUDIO_DIR" -name "preprocessed.wav" -type f -mtime +$RETENTION_DAYS | while read -r file_path; do
    # Extract task_id from path (format: data/audio/{task-id}/preprocessed.wav)
    task_id=$(basename "$(dirname "$file_path")")

    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN: Would check task $task_id"
    else
        echo "Checking task $task_id..."
    fi

    # Check task status in Redis
    task_status=$(get_task_status "$task_id")

    echo "  Task $task_id status: $task_status"

    # Only delete files for complete or failed tasks
    if [ "$task_status" = "COMPLETE" ] || [ "$task_status" = "FAILED" ]; then
        if [ "$DRY_RUN" = "true" ]; then
            echo "  DRY RUN: Would delete $file_path"
        else
            echo "  Deleting $file_path"
            rm -f "$file_path"
        fi
    else
        echo "  Skipping $file_path (task status: $task_status)"
    fi
    echo
done

# Show disk space used by audio files
echo
echo "Current audio directory size:"
du -sh "$AUDIO_DIR" 2>/dev/null || echo "Unable to determine audio directory size"

if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN: Completed. No files were actually deleted."
else
    echo "Audio cleanup completed."
fi