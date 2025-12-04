# Quick Start: MAIE on Jetson Orin Nano Super

**Time to Deploy**: 2-3 hours (after initial setup)  
**Prerequisites**: Jetson Orin Nano Super with JetPack SDK 6.2

---

## 📦 What You'll Need

### Hardware

- ✅ NVIDIA Jetson Orin Nano Super Developer Kit ($249)
- ✅ 64GB+ microSD or 128GB+ NVMe SSD
- ✅ 65W USB-C power supply
- ✅ Internet connection (for model downloads)

### Software (Pre-installed with JetPack 6.2)

- ✅ Ubuntu 22.04
- ✅ CUDA 12.6
- ✅ Python 3.10+ (Python 3.12 recommended for MAIE; see notes on Python version matching)
- ✅ Pixi (package manager; see below)
- ✅ Docker (optional - we recommend `Pixi`-based deployment for Jetson)

---

## 🚀 Quick Deploy (3 Steps)

### Step 1: Prepare Jetson (10 minutes)

```bash
# Enable maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Install monitoring tools
sudo pip3 install -U jetson-stats

# Verify GPU
nvidia-smi

# Increase swap (if using microSD)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Step 2: Clone & Configure (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd maie

# Create Jetson branch
git checkout -b jetson-deployment

# Copy environment template
cp .env.template .env.jetson

# Edit configuration (use your API key)
nano .env.jetson
```

**Key Settings** (`.env.jetson`):

```bash
SECRET_API_KEY=your-secure-key-here
GPU_MEMORY_UTILIZATION=0.85
LLM_MAX_MODEL_LEN=8192
MAX_NUM_SEQS=1
WHISPER_COMPUTE_TYPE=int8_float16
```

### Step 3: Install & Run (30-60 minutes)

```bash
# Install Pixi if not already installed
# See: https://pixi.sh/latest/

# Install project dependencies with Pixi
# Configure Jetson PyPI index in pyproject.toml [tool.pixi.pypi-options]
# extra-index-url = "https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"
pixi install

# Pre-download models
pixi run download-models

# Start API (in foreground)
pixi run api

# In a new terminal, start worker
pixi run worker

# Use logs to watch startup logs; ensure to wait for "Application startup complete"
```

---

## ✅ Verify Deployment

### Quick Health Check

```bash
# Check Python processes are running (API + Worker)
ps aux | rg 'src.api.main|src.worker.main'

# Test API health
curl http://localhost:8000/health

# Expected output:
# {"status": "healthy", "version": "1.0.0-jetson"}
```

### Process Test Audio

```bash
# Test transcription
curl -X POST "http://localhost:8000/v1/process" \
  -H "X-API-Key: your-key-here" \
  -F "file=@test_audio.wav" \
  -F "features=clean_transcript"

# Note the task_id in response
# {"task_id": "abc-123", "status": "PENDING"}

# Check status
curl "http://localhost:8000/v1/status/abc-123" \
  -H "X-API-Key: your-key-here"
```

---

## 📊 Monitor Performance

### Real-time Monitoring

```bash
# Start jtop (Jetson monitoring tool)
jtop

# Key metrics to watch:
# - GPU: Should be 70-85% utilized
# - RAM: Should stay under 6.5GB
# - Power: 15-25W (Super Mode)
# - Temp: Should stay under 70°C
```

### Check Processing Speed

```bash
# Run quick benchmark locally
pixi run python scripts/benchmark_jetson.sh

# Expected results:
# - ASR RTF: 0.15-0.18 (faster than real-time)
# - LLM: 30-35 tokens/second
# - E2E: 22-28 seconds for 2-minute audio
```

---

## 🐛 Troubleshooting

### Issue: Build fails with "out of memory"

**Solution**: Reduce parallelism during build and disable parallel NVCC jobs when compiling native extensions.

```bash
export MAX_JOBS=2
export NVCC_THREADS=1
```

### Issue: Application won't detect GPU

**Solution**: Verify Jetson runtime and CUDA stack

```bash
# Confirm CUDA is present
nvcc --version || echo "NVCC not found"
nvidia-smi || echo "nvidia-smi not available on Jetson (use tegrastats/jtop)"

# Test CUDA in Python
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### Issue: vLLM fails to load model

**Solution**: Check memory settings

```bash
# Edit docker-compose.jetson.yml
environment:
  - GPU_MEMORY_UTILIZATION=0.80  # More conservative
  - LLM_MAX_MODEL_LEN=6144       # Smaller context
```

### Issue: Slow performance

**Solution**: Enable Super Mode

```bash
# On Jetson
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks  # Lock clocks

# Verify mode
sudo nvpmodel -q
# Should show "Mode: 15W+67T Super"
```

---

## 🔧 Configuration Tips

### For Maximum Performance

```yaml
# docker-compose.jetson.yml
environment:
  - GPU_MEMORY_UTILIZATION=0.90 # Aggressive
  - WHISPER_COMPUTE_TYPE=float16 # Less quantization
  - WHISPER_BEAM_SIZE=5 # Better accuracy
```

### For Maximum Stability

```yaml
# docker-compose.jetson.yml
environment:
  - GPU_MEMORY_UTILIZATION=0.75 # Very conservative
  - WHISPER_COMPUTE_TYPE=int8_float16 # More quantization
  - WHISPER_BEAM_SIZE=3 # Faster decoding
  - MAX_QUEUE_DEPTH=3 # Limit backlog
```

### For Low Power Mode

```bash
# On Jetson
sudo nvpmodel -m 1  # 15W mode instead of 25W

# Adjust config
environment:
  - GPU_MEMORY_UTILIZATION=0.70
  - MAX_NUM_BATCHED_TOKENS=2048
```

---

## 📈 Performance Expectations

### Typical Processing Times

| Audio Length | Processing Time | Real-time Factor |
| ------------ | --------------- | ---------------- |
| 30 seconds   | 5-7 seconds     | 0.16-0.23        |
| 2 minutes    | 22-28 seconds   | 0.18-0.23        |
| 5 minutes    | 60-75 seconds   | 0.20-0.25        |
| 10 minutes   | 2.5-3 minutes   | 0.25-0.30        |

### Resource Usage

| Metric     | Idle | Processing | Notes            |
| ---------- | ---- | ---------- | ---------------- |
| GPU Memory | ~2GB | 6-7GB      | Peaks during LLM |
| CPU Memory | ~1GB | 4-5GB      | Stable           |
| GPU Util   | 0%   | 80-95%     | Normal           |
| Power      | 8W   | 20-24W     | Super Mode       |

---

## 🎯 Optimization Checklist

### ✅ Before First Use

- [ ] Enable Super Mode (`sudo nvpmodel -m 0`)
- [ ] Lock clocks (`sudo jetson_clocks`)
- [ ] Configure swap space (if using microSD)
- [ ] Pre-download models to avoid timeout
- [ ] Set appropriate API key
- [ ] Configure memory settings for your workload

### ✅ For Production

- [ ] Monitor temperature (keep < 70°C)
- [ ] Set up log rotation
- [ ] Configure auto-restart policies
- [ ] Implement health check monitoring
- [ ] Set up backup strategy
- [ ] Document configuration changes

### ✅ For Best Performance

- [ ] Use NVMe SSD (not microSD) for storage
- [ ] Add active cooling fan
- [ ] Pre-load models in worker on startup
- [ ] Tune vLLM parameters for your models
- [ ] Monitor and adjust based on workload

---

## 📚 Need Help?

### Documentation

- **Full Guide**: [JETSON_MIGRATION_PLAN.md](./JETSON_MIGRATION_PLAN.md)
- **Compatibility**: [JETSON_COMPATIBILITY_MATRIX.md](./JETSON_COMPATIBILITY_MATRIX.md)
- **Overview**: [JETSON_EXECUTIVE_SUMMARY.md](./JETSON_EXECUTIVE_SUMMARY.md)

### Resources

- [Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [NVIDIA Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)
- [JetPack Documentation](https://developer.nvidia.com/embedded/jetpack)

### Common Commands

```bash
# System info
cat /etc/nv_tegra_release
nvidia-smi

# Running services (non-container)
ps aux | rg 'src.api.main|src.worker.main'
pixi run api &> logs/api.log &
pixi run worker &> logs/worker.log &
tail -f logs/api.log -n 100

# Monitoring
jtop
htop

# Restart services (non-container run)
pkill -f src.api.main || true; pixi run api &> logs/api.log &
pkill -f src.worker.main || true; pixi run worker &> logs/worker.log &

# Clean (manual cleanup)
rm -rf logs/*
rm -rf data/audio/*
```

---

## 🎉 Success Indicators

You're ready for production when:

✅ All services start without errors  
✅ Health check returns 200 OK  
✅ Test transcription completes successfully  
✅ GPU memory stays under 7GB  
✅ Processing is faster than real-time (RTF < 0.3)  
✅ Temperature stays under 70°C  
✅ No OOM errors in logs  
✅ Queue processes smoothly

---

## 💡 Pro Tips

1. **Use NVMe over microSD** - 5x faster I/O
2. **Pre-download models** - Avoid first-run delays
3. **Monitor temperature** - Add cooling if needed
4. **Start conservative** - Tune up, not down
5. **Log everything** - Helps with debugging
6. **Backup config** - Save working configurations
7. **Test incrementally** - Don't change everything at once
8. **Use jtop** - Essential monitoring tool

---

**Quick Start Version**: 1.0  
**Last Updated**: November 25, 2025  
**Estimated Setup Time**: 2-3 hours (including build)  
**Support**: See full documentation for detailed help
