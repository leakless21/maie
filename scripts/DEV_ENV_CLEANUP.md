# Development Environment Cleanup Guide

A comprehensive solution for managing and optimizing development environment memory usage, particularly useful for edge deployment scenarios with limited resources.

## 🎯 Overview

This cleanup solution addresses the major memory consumers in typical development environments:

- **VS Code Server**: 869 MB
- **Pylance Extension**: 422 MB  
- **Python/uvicorn API**: 425 MB (application)
- **Docker + Containerd**: 135 MB
- **Cloudflared**: 41 MB

## 📁 Scripts Overview

### 1. `cleanup-dev-env.sh` - Main Cleanup Script
Modular cleanup script with selective and comprehensive cleanup options.

### 2. `production-mode.sh` - Production Mode Manager
Minimizes development overhead for edge deployment scenarios.

### 3. `memory-monitor.sh` - Real-time Memory Monitoring
Provides monitoring and analysis tools for tracking memory usage.

## 🚀 Quick Start

### Basic Cleanup
```bash
# Interactive mode with safety confirmations
./scripts/cleanup-dev-env.sh

# Clean VS Code only
./scripts/cleanup-dev-env.sh --vscode

# Full cleanup with memory measurement
./scripts/cleanup-dev-env.sh --full --measure

# Start production mode (minimal resource usage)
./scripts/production-mode.sh start
```

### Memory Monitoring
```bash
# Real-time memory monitoring
./scripts/memory-monitor.sh monitor

# Detailed memory report
./scripts/memory-monitor.sh report

# Memory benchmark test
./scripts/memory-monitor.sh benchmark
```

## 📋 Detailed Usage Guide

### cleanup-dev-env.sh Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-v, --vscode` | Clean VS Code processes and cache |
| `-p, --python` | Clean Python cache (__pycache__, *.pyc, pip/uv cache) |
| `-d, --docker` | Stop Docker containers and clean system |
| `-a, --apps` | Stop application processes (uvicorn, workers) |
| `-l, --logs` | Clean log files (project and system) |
| `-m, --packages` | Clean package manager caches (apt, npm, yarn) |
| `-f, --full` | Execute all cleanup operations |
| `-s, --safe` | Interactive mode with safety confirmations |
| `--measure` | Show memory usage before and after cleanup |
| `--no-confirm` | Skip all confirmations (use with caution) |

### production-mode.sh Commands

| Command | Description |
|---------|-------------|
| `start` | Start production mode with minimal overhead |
| `stop` | Stop production mode |
| `status` | Show production mode status and health |
| `health` | Run production health check |
| `optimize` | Optimize system for production |
| `cleanup` | Clean up production artifacts |
| `restart` | Stop and start production mode |

### memory-monitor.sh Commands

| Command | Description |
|---------|-------------|
| `monitor` | Real-time memory monitoring (default) |
| `report` | Detailed memory usage report |
| `benchmark` | Run memory benchmark test |
| `top` | Show top memory consuming processes |
| `analyze` | Analyze project-specific memory usage |

## 🛡️ Safety Features

### Protected Services
The cleanup scripts automatically protect critical system services including:
- SSH (sshd)
- Systemd (system management)
- NetworkManager (network connectivity)
- D-Bus (system message bus)
- Avahi (service discovery)
- Cron (scheduled tasks)

### Safety Checks
- **Process Validation**: Before killing processes, scripts verify they're not protected
- **User Confirmation**: Interactive mode requires explicit consent for destructive operations
- **Graceful Shutdown**: Processes are terminated with SIGTERM before SIGKILL
- **Backup Creation**: Important files are backed up before modification
- **Rollback Capability**: All scripts can be run multiple times without adverse effects

## 💡 Memory Optimization Strategies

### Development Mode
1. **Monitor Memory Usage**: 
   ```bash
   ./scripts/memory-monitor.sh monitor
   ```

2. **Selective Cleanup**:
   ```bash
   # Clean VS Code if it's consuming too much memory
   ./scripts/cleanup-dev-env.sh --vscode
   
   # Clean Python cache
   ./scripts/cleanup-dev-env.sh --python
   ```

3. **Package Manager Cleanup**:
   ```bash
   ./scripts/cleanup-dev-env.sh --packages
   ```

### Production Mode
1. **Start Minimal Production**:
   ```bash
   ./scripts/production-mode.sh start
   ```

2. **Monitor Production Health**:
   ```bash
   ./scripts/production-mode.sh health
   ```

3. **Full Production Optimization**:
   ```bash
   ./scripts/production-mode.sh optimize
   ```

### Edge Deployment
1. **Production-Ready Setup**:
   ```bash
   ./scripts/production-mode.sh optimize
   ./scripts/production-mode.sh start
   ```

2. **Resource Monitoring**:
   ```bash
   ./scripts/memory-monitor.sh report
   ```

3. **Emergency Cleanup**:
   ```bash
   ./scripts/cleanup-dev-env.sh --full --no-confirm
   ```

## 📊 Monitoring and Metrics

### Memory Usage Analysis
```bash
# Get detailed memory report
./scripts/memory-monitor.sh report

# Monitor specific processes
./scripts/memory-monitor.sh analyze

# Benchmark memory efficiency
./scripts/memory-monitor.sh benchmark
```

### Expected Memory Savings

| Component | Memory Usage | Cleanup Action | Expected Savings |
|-----------|--------------|----------------|------------------|
| VS Code Server | ~869 MB | `--vscode` | 800-900 MB |
| Pylance Extension | ~422 MB | `--vscode` | 400-450 MB |
| Python/uvicorn | ~425 MB | `--apps` | 400-450 MB |
| Docker + Containerd | ~135 MB | `--docker` | 100-150 MB |
| Cloudflared | ~41 MB | `--apps` | 30-50 MB |

**Total Potential Savings: ~1.7-2.0 GB**

## 🔧 Configuration

### Environment Variables
The scripts respect these environment variables:
- `ENVIRONMENT`: Set to 'production' for production mode
- `LOG_LEVEL`: Controls verbosity (WARNING, INFO, DEBUG)
- `DEBUG`: Set to 'false' in production mode
- `DEV_MODE`: Set to 'false' in production mode

### Custom Safe Services
Modify the `SAFE_SERVICES` array in `cleanup-dev-env.sh` to add your own protected services:

```bash
SAFE_SERVICES=(
    "ssh"
    "systemd"
    "NetworkManager"
    "dbus"
    "avahi-daemon"
    "cron"
    "your-custom-service"
)
```

## 🐛 Troubleshooting

### Common Issues

#### "Permission Denied" Errors
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Use sudo for system-level cleanup
sudo ./scripts/cleanup-dev-env.sh --packages
```

#### "Command Not Found" Errors
```bash
# Check if required tools are available
which docker python3 curl bc

# Install missing dependencies
sudo apt update && sudo apt install bc curl
```

#### Production Mode Won't Start
```bash
# Check if port 8000 is available
netstat -tulpn | grep :8000

# Kill existing processes
./scripts/cleanup-dev-env.sh --apps

# Check logs
tail -f /tmp/maie-production.log
```

#### High Memory Usage Persists
```bash
# Run comprehensive analysis
./scripts/memory-monitor.sh report

# Check for memory leaks
./scripts/memory-monitor.sh benchmark

# Force complete cleanup
./scripts/cleanup-dev-env.sh --full --no-confirm
```

### Log Files and Debugging

All scripts provide detailed logging:
- Production mode logs: `/tmp/maie-production.log`
- Cleanup operations are logged to console with timestamps
- Memory monitoring shows real-time changes

### Recovery Procedures

If cleanup causes issues:

1. **Restart Services**:
   ```bash
   ./scripts/dev.sh  # Restart development environment
   ```

2. **System Recovery**:
   ```bash
   # Restart all services
   sudo systemctl restart systemd
   ```

3. **Rollback Changes**:
   ```bash
   # Find and restore backed up files
   ls -la .env.backup.*
   ```

## 📈 Performance Tips

### For Development
- Use `--safe` mode for interactive cleanup
- Monitor memory regularly with `./scripts/memory-monitor.sh monitor`
- Clean Python cache frequently with `--python` option
- Stop Docker containers when not needed with `--docker`

### For Production
- Always start with `./scripts/production-mode.sh optimize`
- Use production mode for minimal resource usage
- Monitor health with `./scripts/production-mode.sh health`
- Set up automated monitoring for resource usage

### For Edge Deployment
- Use production mode exclusively
- Implement automated cleanup in cron jobs
- Monitor memory usage thresholds
- Use `--measure` flag to track improvements

## 🔐 Security Considerations

### Safe Operations
- All scripts avoid modifying system-critical files
- Process killing requires explicit pattern matching
- User confirmation prevents accidental data loss
- Backup creation for important configuration files

### Sensitive Data
The scripts are designed to avoid:
- Credential files (`*.json`, `.env*`)
- User data directories
- Application databases
- Configuration files unless explicitly requested

### Recommendations
- Run cleanup scripts during low-activity periods
- Test scripts in non-production environments first
- Keep backups of critical configuration files
- Monitor system health after cleanup operations

## 📚 Examples

### Daily Development Workflow
```bash
# Morning: Start development
./scripts/dev.sh

# Check memory usage
./scripts/memory-monitor.sh report

# During development: Monitor memory
./scripts/memory-monitor.sh monitor

# End of day: Cleanup before leaving
./scripts/cleanup-dev-env.sh --safe --measure
```

### Edge Deployment Setup
```bash
# Initial optimization
./scripts/production-mode.sh optimize

# Start production
./scripts/production-mode.sh start

# Monitor health
watch -n 30 './scripts/production-mode.sh health'

# Weekly cleanup
./scripts/cleanup-dev-env.sh --packages --logs
```

### Emergency Memory Recovery
```bash
# Quick analysis
./scripts/memory-monitor.sh top

# Aggressive cleanup
./scripts/cleanup-dev-env.sh --full --no-confirm

# Restart minimal services
./scripts/production-mode.sh start
```

## 🤝 Contributing

To extend the cleanup solution:

1. **Add New Cleanup Modules**: Follow the modular pattern in `cleanup-dev-env.sh`
2. **Add Monitoring Features**: Extend `memory-monitor.sh` with new metrics
3. **Add Safety Checks**: Enhance the `is_safe_service()` function
4. **Add Documentation**: Update this guide with new features

## 📞 Support

For issues or improvements:
1. Check the troubleshooting section above
2. Review script help output with `--help`
3. Examine log files for error details
4. Test scripts in safe mode first (`--safe`)

---

**Note**: Always test cleanup operations in a non-production environment first. The scripts are designed to be safe but system configurations may vary.