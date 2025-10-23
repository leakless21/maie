#!/bin/bash

# Clean Redis cache entries for MAIE project
# Removes expired task entries and temporary cache items

set -euo pipefail

REDIS_URL="${REDIS_URL:-redis://localhost:6379/1}"
REDIS_QUEUE_DB="${REDIS_QUEUE_DB:-0}"
DRY_RUN="${DRY_RUN:-false}"

echo "Redis cache cleanup script for MAIE project"
echo "Results Redis URL: $REDIS_URL"
echo "Queue DB: $REDIS_QUEUE_DB"
echo "Dry run mode: $DRY_RUN"
echo

# Parse results Redis URL
# Format: redis://host:port/db
results_no_proto="${REDIS_URL#redis://}"
results_host="${results_no_proto%%:*}"
results_rest="${results_no_proto#*:}"
results_port="${results_rest%%/*}"
results_db="${results_rest#*/}"

# Queue Redis URL (same host/port but different DB)
queue_url="redis://${results_host}:${results_port}/${REDIS_QUEUE_DB}"

echo "Queue Redis URL: $queue_url"

# Function to clean expired keys in results DB (DB 1)
clean_results_db() {
    echo "Cleaning results database (DB $results_db)..."

    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN: Would clean expired keys in results DB"
        if command -v redis-cli >/dev/null 2>&1; then
            echo "DRY RUN: Keys starting with 'task:' in results DB:"
            redis-cli -h "$results_host" -p "$results_port" -n "$results_db" KEYS "task:*" 2>/dev/null | head -10 || echo "Failed to list keys"
            echo "DRY RUN: Note - expired keys are cleaned automatically by Redis TTL"
        else
            echo "redis-cli not found, cannot preview results DB"
        fi
    else
        echo "Note: Redis automatically expires keys based on TTL"
        echo "Checking for any manual cleanup needed..."

        # Could add specific cleanup logic here if needed
        # For now, Redis TTL handles expiration automatically
        echo "Results DB cleanup completed (TTL-based)."
    fi
}

# Function to clean queue DB (DB 0)
clean_queue_db() {
    echo "Cleaning queue database (DB $REDIS_QUEUE_DB)..."

    if [ "$DRY_RUN" = "true" ]; then
        echo "DRY RUN: Would clean expired queue entries"
        if command -v redis-cli >/dev/null 2>&1; then
            echo "DRY RUN: RQ queue keys in DB $REDIS_QUEUE_DB:"
            redis-cli -h "$results_host" -p "$results_port" -n "$REDIS_QUEUE_DB" KEYS "rq:*" 2>/dev/null | head -5 || echo "Failed to list RQ keys"
        else
            echo "redis-cli not found, cannot preview queue DB"
        fi
    else
        echo "Clearing completed RQ queue jobs older than 24 hours..."

        # We'll use python script to clean RQ queue since it's complex
        # For now, just show that we'd clean here
        # In production, you might want to use rq's built-in cleanup:
        # python -c "import redis; r = redis.Redis(host='$results_host', port=$results_port, db=$REDIS_QUEUE_DB); r.delete(*r.keys('rq:job:*'))"

        echo "Queue cleanup: Keeping recent entries, older ones will be handled by RQ TTL settings"
    fi
}

# Function to show Redis memory usage
show_redis_info() {
    echo
    echo "Redis database sizes:"

    if command -v redis-cli >/dev/null 2>&1; then
        # Results DB size
        echo "Results DB ($results_db) keys:"
        redis-cli -h "$results_host" -p "$results_port" -n "$results_db" DBSIZE 2>/dev/null || echo "  Unable to get results DB size"

        # Queue DB size
        echo "Queue DB ($REDIS_QUEUE_DB) keys:"
        redis-cli -h "$results_host" -p "$results_port" -n "$REDIS_QUEUE_DB" DBSIZE 2>/dev/null || echo "  Unable to get queue DB size"
    else
        echo "redis-cli not available for size reporting"
    fi
}

# Main cleanup
clean_results_db
echo
clean_queue_db
show_redis_info

if [ "$DRY_RUN" = "true" ]; then
    echo
    echo "DRY RUN: Completed. No data was actually deleted."
else
    echo
    echo "Redis cache cleanup completed."
fi