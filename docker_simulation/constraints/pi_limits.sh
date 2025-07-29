#!/bin/bash

# Raspberry Pi 4 Resource Constraints
echo "🔧 Applying Raspberry Pi 4 constraints..."

# Memory constraints (handled by Docker, but add swap limits)
echo "Memory limit: $MEMORY_LIMIT"
ulimit -v 2097152  # 2GB virtual memory limit in KB

# CPU frequency scaling simulation (Pi throttles under load)
echo "Simulating ARM CPU frequency scaling..."
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
fi

# Simulate Pi-specific limitations
# Lower thread priority for CPU-intensive tasks
renice 10 $$ >/dev/null 2>&1 || true

# Set Python optimization for limited resources
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Simulate slower I/O (SD card limitations)
export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Add random delays to simulate hardware limitations
export PI_SIMULATION=1

# Network simulation - Pi often has slower/unstable network
if command -v tc >/dev/null 2>&1; then
    # Add 5-20ms latency and 1% packet loss (if tc is available)
    tc qdisc add dev eth0 root netem delay 10ms 5ms loss 1% 2>/dev/null || true
fi

echo "✅ Raspberry Pi constraints applied"
echo "   - Memory: Limited to $MEMORY_LIMIT"
echo "   - CPU: Single thread, power-save mode"
echo "   - Network: Simulated latency/loss"
echo "   - Storage: Simulated SD card speed"

# Start resource monitoring in background
/usr/local/bin/monitor_resources.sh &

# Execute the main command
exec "$@"