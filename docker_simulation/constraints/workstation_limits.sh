#!/bin/bash

# Medical Workstation Resource Constraints
echo "🖥️  Applying Medical Workstation constraints..."

# Memory constraints - generous for workstation
echo "Memory limit: $MEMORY_LIMIT"
ulimit -v 8388608  # 8GB virtual memory limit in KB

# CPU optimization for workstation performance
echo "Configuring workstation performance profile..."

# High performance mode
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true

# Allow more threads for workstation
export TORCH_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Workstation-specific optimizations
export PYTORCH_JIT=1
export TORCH_TENSORRT_ENABLED=1

# Simulate workstation environment
echo "Workstation environment configured:"
echo "   - High-performance CPU scaling"
echo "   - Multi-threading enabled (4 threads)"
echo "   - JIT compilation enabled"

# Check for GPU availability (workstations often have dedicated GPUs)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "   - GPU acceleration: Available"
    export CUDA_VISIBLE_DEVICES=0
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "   - GPU acceleration: CPU-only mode"
fi

# Network - workstations typically have stable, fast network
echo "Network: Wired Gigabit Ethernet"
# Minimal network simulation (stable connection)
tc qdisc add dev eth0 root netem delay 2ms 1ms 2>/dev/null || true

# Professional workstation reliability simulation
echo "System reliability: Medical-grade (99.9% uptime)"

# Set higher process priority (workstations get priority)
renice -5 $$ >/dev/null 2>&1 || true

echo "✅ Workstation constraints applied"
echo "   - Memory: $MEMORY_LIMIT available"
echo "   - CPU: High-performance mode (4 cores)"
echo "   - Network: Stable gigabit connection"
echo "   - Priority: Medical workstation priority"

# Start resource monitoring in background
/usr/local/bin/monitor_resources.sh &

# Execute the main command
exec "$@"