# MAIE Process Management Scripts

This directory contains scripts to help manage MAIE development processes.

## Scripts

### `process_manager.py`
A comprehensive Python script to identify and kill running MAIE server and worker processes.

**Usage:**
```bash
# Show running processes
pixi run python scripts/process_manager.py

# Kill all processes
pixi run python scripts/process_manager.py --kill

# Force kill all processes
pixi run python scripts/process_manager.py --kill --force

# Interactive mode (ask before killing)
pixi run python scripts/process_manager.py --interactive

# JSON output
pixi run python scripts/process_manager.py --json
```

**Features:**
- Detects API server processes (uvicorn, pixi run api)
- Detects worker processes (python src/worker/main.py, pixi run worker, rq workers)
- Checks Redis for active RQ workers
- Checks port 8000 for listening processes
- Interactive and non-interactive kill modes
- JSON output option
- Force kill option (SIGKILL vs SIGTERM)

### `kill_maie.sh`
A simple bash wrapper that provides an interactive menu for process management.

**Usage:**
```bash
./scripts/kill_maie.sh
```

**Features:**
- Interactive menu with options
- Colorized output
- Multiple kill modes
- Process status display

## Common Use Cases

### 1. Check what's running
```bash
pixi run python scripts/process_manager.py
```

### 2. Kill everything and start fresh
```bash
pixi run python scripts/process_manager.py --kill
./scripts/dev.sh
```

### 3. Force kill stuck processes
```bash
pixi run python scripts/process_manager.py --kill --force
```

### 4. Interactive cleanup
```bash
./scripts/kill_maie.sh
```

## Troubleshooting

### "Address already in use" error
This usually means there's already a server running on port 8000. Use the process manager to kill existing processes:

```bash
pixi run python scripts/process_manager.py --kill
```

### "There exists an active worker named 'maie-worker-dev' already"
This means there's already an RQ worker running. Kill all processes and restart:

```bash
pixi run python scripts/process_manager.py --kill --force
./scripts/dev.sh
```

### Processes won't die
Use force kill:

```bash
pixi run python scripts/process_manager.py --kill --force
```

## Dependencies

The scripts require:
- `psutil` - for process management
- `redis` - for Redis worker detection

These are already included in the MAIE development environment.

