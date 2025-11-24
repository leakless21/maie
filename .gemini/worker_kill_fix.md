# Worker Kill Issue - Root Cause and Fix

## Problem
The `kill_maie.sh` script was unable to permanently kill worker processes. After killing the worker, it would immediately restart.

## Root Cause
The `dev.sh` script runs workers in a **continuous restart loop**:

```bash
# From scripts/dev.sh lines 196-201 and 218-222
while true; do
    echo "Starting worker process..."
    pixi run worker || true
    echo "Worker exited or crashed, restarting in 2 seconds..."
    sleep 2
done
```

This creates a process hierarchy:
1. **Parent**: `bash ./scripts/dev.sh` (PID 610616)
2. **Subshell**: `bash ./scripts/dev.sh` (PID 610731) - runs the while loop
3. **Worker**: `pixi run worker` (PID 630495)
4. **RQ Worker**: `rq:worker:maie-worker-dev:` (PID 630531)

When `kill_maie.sh` killed only the child processes (PIDs 630495 and 630531), the parent shell scripts (PIDs 610616 and 610731) remained alive and immediately restarted new worker processes.

## Solution
Updated `scripts/process_manager.py` to detect and kill **parent `dev.sh` processes** in addition to worker processes:

### Changes Made

1. **Updated `find_worker_processes()`** (lines 61-88):
   - Added detection for `bash ./scripts/dev.sh` processes
   - Now catches all processes in the hierarchy

2. **Updated `find_api_processes()`** (lines 40-74):
   - Added detection for `dev.sh` processes running API server
   - Checks for uvicorn child processes to confirm it's an API-running script

3. **Updated documentation**:
   - Updated module docstring to mention parent script detection
   - Updated `kill_maie.sh` header to clarify Redis worker support

## Verification
Before fix:
```
⚙️  Worker Processes (2 found):
  PID: 630495 |         pixi | pixi run worker...
  PID: 630531 | rq:worker:maie-worker-dev: | ...
```

After fix:
```
⚙️  Worker Processes (4 found):
  PID: 610616 |         bash | bash ./scripts/dev.sh...
  PID: 610731 |         bash | bash ./scripts/dev.sh...
  PID: 630495 |         pixi | pixi run worker...
  PID: 630531 | rq:worker:maie-worker-dev: | ...
```

Now when killing workers, all 4 processes are terminated, preventing automatic restarts.

## Testing
To test the fix:
1. Start worker: `./scripts/dev.sh --worker-only`
2. Run kill script: `./scripts/kill_maie.sh`
3. Choose option 1 (SIGTERM) or 2 (SIGKILL)
4. Verify no worker processes remain: `./scripts/kill_maie.sh` (option 4)

The worker should now be completely terminated without restarting.
