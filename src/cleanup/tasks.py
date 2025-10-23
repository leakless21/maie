"""
Cleanup tasks for MAIE application using RQ Scheduler.

This module provides Python implementations of cleanup functionality,
replacing shell scripts with proper error handling and integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import redis
from rq import get_current_job

from ..config.logging import get_module_logger


logger = get_module_logger(__name__)


def cleanup_audio_files(dry_run: bool = False) -> Dict[str, Any]:
    """
    Cleanup audio files similar to scripts/clean-audio.sh

    Removes preprocessed.wav files older than retention period for COMPLETE or FAILED tasks.

    :param dry_run: If True, only log what would be deleted without actually deleting
    :return: Summary of cleanup results
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"

    logger.info("Starting audio files cleanup", extra={
        "job_id": job_id,
        "dry_run": dry_run
    })

    from ..config.loader import settings

    audio_dir = settings.paths.audio_dir
    deleted_count = 0
    checked_count = 0

    if not audio_dir.exists():
        logger.info("Audio directory does not exist, nothing to clean", extra={
            "audio_dir": str(audio_dir)
        })
        return {"checked": 0, "deleted": 0, "skipped": 0}

    try:
        # Check Redis connection
        redis_conn = redis.from_url(settings.redis.url, db=settings.redis.results_db)
        redis_conn.ping()  # Test connection

        # Find audio directories (UUID-named) containing potential preprocessed.wav files
        for task_dir in audio_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            checked_count += 1
            
            # Look for preprocessed.wav in each task directory
            preprocessed_file = task_dir / "preprocessed.wav"
            if not preprocessed_file.exists() or not preprocessed_file.is_file():
                continue

            # Extract task_id from directory name (which should be a UUID)
            task_id = task_dir.name

            # Check task status in Redis
            try:
                task_status_bytes = redis_conn.hget(f"task:{task_id}", "status")
                if task_status_bytes:
                    # Handle both bytes and string responses
                    if isinstance(task_status_bytes, bytes):
                        task_status = task_status_bytes.decode('utf-8')
                    else:
                        task_status = str(task_status_bytes)
                else:
                    task_status = "UNKNOWN"
            except Exception as e:
                logger.warning(f"Failed to check Redis status for task {task_id}", extra={
                    "task_id": task_id,
                    "error": str(e)
                })
                task_status = "UNKNOWN"

            if task_status in ["COMPLETE", "FAILED"]:
                if dry_run:
                    logger.info("Would delete audio file", extra={
                        "task_id": task_id,
                        "file": str(preprocessed_file),
                        "status": task_status,
                        "dry_run": True
                    })
                else:
                    try:
                        preprocessed_file.unlink()
                        logger.info("Deleted audio file", extra={
                            "task_id": task_id,
                            "file": str(preprocessed_file),
                            "status": task_status
                        })
                        deleted_count += 1
                    except Exception as e:
                        logger.error("Failed to delete audio file", extra={
                            "task_id": task_id,
                            "file": str(preprocessed_file),
                            "error": str(e)
                        })
            else:
                logger.debug("Skipping audio file", extra={
                    "task_id": task_id,
                    "file": str(preprocessed_file),
                    "status": task_status
                })

    except redis.ConnectionError as e:
        logger.error("Redis connection failed, cannot check task statuses", extra={
            "error": str(e)
        })
        return {"checked": checked_count, "deleted": 0, "skipped": checked_count, "error": "redis_unavailable"}

    # Get directory size
    try:
        def get_dir_size(path: Path) -> float:
            total_size = 0.0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # MB

        dir_size_mb = get_dir_size(audio_dir)
        result = {
            "checked": checked_count,
            "deleted": deleted_count,
            "skipped": checked_count - deleted_count,
            "directory_size_mb": round(dir_size_mb, 2)
        }

    except Exception as e:
        logger.warning("Failed to calculate directory size", extra={"error": str(e)})
        result = {
            "checked": checked_count,
            "deleted": deleted_count,
            "skipped": checked_count - deleted_count
        }

    logger.info("Audio files cleanup completed", extra=result)
    return result



def cleanup_logs(dry_run: bool = False) -> Dict[str, Any]:
    """
    Cleanup log files similar to scripts/clean-logs.sh

    Removes log files older than specified retention period.

    :param dry_run: If True, only log what would be deleted without actually deleting
    :return: Summary of cleanup results
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"

    logger.info("Starting log files cleanup", extra={
        "job_id": job_id,
        "dry_run": dry_run
    })

    from ..config.loader import settings

    log_dir = Path(settings.logging.log_dir)
    retention_days = settings.cleanup.logs_retention_days
    deleted_count = 0
    total_size_mb = 0.0

    if not log_dir.exists():
        logger.info("Log directory does not exist, nothing to clean", extra={
            "log_dir": str(log_dir)
        })
        return {"checked": 0, "deleted": 0}

    # Find log files older than retention period
    try:
        import time

        cutoff_time = time.time() - (retention_days * 24 * 3600)

        for file_path in log_dir.glob("*.log*"):
            if not file_path.is_file():
                continue

            try:
                file_stat = file_path.stat()
                if file_stat.st_mtime < cutoff_time:
                    if dry_run:
                        logger.info("Would delete log file", extra={
                            "file": str(file_path),
                            "age_days": round((time.time() - file_stat.st_mtime) / (24 * 3600), 1),
                            "dry_run": True
                        })
                    else:
                        size_mb = file_stat.st_size / (1024 * 1024)
                        total_size_mb += size_mb
                        file_path.unlink()
                        logger.info("Deleted log file", extra={
                            "file": str(file_path),
                            "size_mb": round(size_mb, 2)
                        })
                        deleted_count += 1
            except Exception as e:
                logger.warning("Failed to process log file", extra={
                    "file": str(file_path),
                    "error": str(e)
                })

    except Exception as e:
        logger.error("Log cleanup failed", extra={"error": str(e)})
        return {"checked": 0, "deleted": 0, "error": str(e)}

    result = {
        "deleted": deleted_count,
        "space_freed_mb": round(total_size_mb, 2)
    }

    logger.info("Log files cleanup completed", extra=result)
    return result


def cleanup_cache(dry_run: bool = False) -> Dict[str, Any]:
    """
    Cleanup Redis cache entries similar to scripts/clean-cache.sh

    Removes expired task entries and temporary cache items.

    :param dry_run: If True, only log what would be deleted without actually deleting
    :return: Summary of cleanup results
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"

    logger.info("Starting Redis cache cleanup", extra={
        "job_id": job_id,
        "dry_run": dry_run
    })

    from ..config.loader import settings
    import time

    try:
        # Results DB connection (store results)
        results_conn = redis.from_url(settings.redis.url, db=settings.redis.results_db)
        results_conn.ping()

        # Queue DB connection (for RQ jobs)
        queue_db = 0 if settings.redis.results_db == 1 else settings.redis.results_db
        queue_conn = redis.from_url(settings.redis.url, db=queue_db)

        results_keys_before = results_conn.dbsize()
        queue_keys_before = queue_conn.dbsize()

        # Redis TTL handles automatic expiration of task entries
        # We mainly clean up RQ completed jobs older than 24 hours
        cleaned_count = 0
        cutoff_timestamp = time.time() - (24 * 3600)  # 24 hours ago

        try:
            # Look for RQ finished job keys
            pattern = "rq:job:*"
            for key in queue_conn.scan_iter(pattern):
                try:
                    # Get job data to check when it was created
                    job_data = queue_conn.hgetall(key)
                    if not job_data:
                        continue

                    # Check created_at timestamp if available
                    created_at_bytes = job_data.get(b"created_at")
                    if created_at_bytes:
                        try:
                            if isinstance(created_at_bytes, bytes):
                                created_at = float(created_at_bytes.decode())
                            else:
                                created_at = float(created_at_bytes)
                            # Only delete jobs older than 24 hours
                            if created_at < cutoff_timestamp:
                                if dry_run:
                                    logger.info("Would clean old RQ finished job", extra={
                                        "key": key.decode() if isinstance(key, bytes) else key,
                                        "age_hours": round((time.time() - created_at) / 3600, 1),
                                        "dry_run": True
                                    })
                                else:
                                    queue_conn.delete(key)
                                    cleaned_count += 1
                        except (ValueError, TypeError):
                            logger.debug("Could not parse created_at for RQ job", extra={
                                "key": key.decode() if isinstance(key, bytes) else key
                            })
                    else:
                        # Fallback: use TTL as indicator
                        ttl = queue_conn.ttl(key)
                        # If TTL is set and low (< 1 hour), it's likely a finishing/finished job
                        if isinstance(ttl, int) and 0 < ttl < 3600:
                            if not dry_run:
                                queue_conn.delete(key)
                                cleaned_count += 1

                except Exception as e:
                    logger.debug("Failed to check RQ job key", extra={
                        "key": key.decode() if isinstance(key, bytes) else key,
                        "error": str(e)
                    })

        except Exception as e:
            logger.warning("Failed to clean RQ jobs", extra={"error": str(e)})

        results_keys_after = results_conn.dbsize()
        queue_keys_after = queue_conn.dbsize()

        result = {
            "results_db_keys_before": results_keys_before,
            "results_db_keys_after": results_keys_after,
            "queue_db_keys_before": queue_keys_before,
            "queue_db_keys_after": queue_keys_after,
            "rq_jobs_cleaned": cleaned_count,
            "ttl_handling": "Redis TTL handles task entry expiration; RQ jobs > 24h removed"
        }

        logger.info("Redis cache cleanup completed", extra=result)
        return result

    except redis.ConnectionError as e:
        logger.error("Redis connection failed", extra={"error": str(e)})
        return {"error": "redis_unavailable", "details": str(e)}
    except Exception as e:
        logger.error("Cache cleanup failed", extra={"error": str(e)})
        return {"error": str(e)}


def disk_monitor() -> Dict[str, Any]:
    """
    Monitor disk usage and potentially trigger emergency cleanup.

    Similar to scripts/disk-monitor.sh but uses Python for better integration.

    :return: Disk usage information
    """
    job = get_current_job()
    job_id = job.id if job else "unknown"

    logger.info("Starting disk monitor", extra={"job_id": job_id})

    from ..config.loader import settings

    check_dir = Path(settings.cleanup.check_dir)
    threshold_pct = settings.cleanup.disk_threshold_pct

    try:
        import shutil
        disk_usage = shutil.disk_usage(check_dir)
        usage_pct = round((disk_usage.used / disk_usage.total) * 100, 2)
        usage_gb = round(disk_usage.used / (1024 ** 3), 2)  # Convert to GB
        total_gb = round(disk_usage.total / (1024 ** 3), 2)

        result = {
            "usage_percent": usage_pct,
            "usage_gb": usage_gb,
            "total_gb": total_gb,
            "check_dir": str(check_dir),
            "threshold_percent": threshold_pct
        }

        if usage_pct >= threshold_pct:
            logger.warning("Disk usage above threshold!", extra={
                **result,
                "action": "Consider emergency cleanup"
            })

            # Trigger emergency cleanup if emergency mode enabled
            if settings.cleanup.emergency_cleanup:
                logger.info("Emergency cleanup triggered", extra=result)

                # Run cleanup tasks synchronously
                audio_result = cleanup_audio_files(dry_run=False)
                logs_result = cleanup_logs(dry_run=False)

                result["emergency_cleanup"] = {
                    "audio": audio_result,
                    "logs": logs_result
                }

                # Re-check usage after cleanup
                disk_usage_after = shutil.disk_usage(check_dir)
                new_usage_pct = round((disk_usage_after.used / disk_usage_after.total) * 100, 2)

                result["cleanup_effective"] = new_usage_pct < threshold_pct
                result["usage_percent_after"] = new_usage_pct
                result["usage_gb_after"] = round(disk_usage_after.used / (1024 ** 3), 2)

            result["alert"] = True
        else:
            logger.info("Disk usage within acceptable limits", extra=result)

    except Exception as e:
        logger.error("Disk monitor failed", extra={"error": str(e)})
        return {"error": str(e)}

    logger.info("Disk monitor completed", extra=result)
    return result


__all__ = [
    "cleanup_audio_files",
    "cleanup_logs",
    "cleanup_cache",
    "disk_monitor",
]

