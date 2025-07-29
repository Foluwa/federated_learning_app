#!/bin/bash

# Mobile Device Resource Constraints
echo "📱 Applying Mobile Device constraints..."

# Memory constraints
echo "Memory limit: $MEMORY_LIMIT"
ulimit -v 4194304  # 4GB virtual memory limit in KB

# CPU constraints with mobile-specific behavior
echo "Simulating mobile CPU management..."

# Simulate battery-based performance scaling
BATTERY_LEVEL=$(shuf -i 20-100 -n 1)
echo "Battery level: $BATTERY_LEVEL%"

if [ $BATTERY_LEVEL -lt 30 ]; then
    echo "🔋 Low battery - enabling power saving mode"
    export TORCH_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    # Reduce CPU frequency
    cpulimit -l 25 -i &
elif [ $BATTERY_LEVEL -lt 60 ]; then
    echo "🔋 Medium battery - balanced performance"
    export TORCH_NUM_THREADS=2
    export OMP_NUM_THREADS=2
else
    echo "🔋 Good battery - normal performance"
    export TORCH_NUM_THREADS=2
    export OMP_NUM_THREADS=2
fi

# Simulate thermal throttling (mobile devices heat up)
TEMPERATURE=$(shuf -i 35-85 -n 1)
echo "Device temperature: ${TEMPERATURE}°C"

if [ $TEMPERATURE -gt 70 ]; then
    echo "🌡️  Thermal throttling activated"
    cpulimit -l 50 -i &
fi

# Network simulation - mobile networks are variable
NETWORK_TYPE=$(shuf -e "4G" "5G" "WiFi" -n 1)
echo "Network: $NETWORK_TYPE"

case $NETWORK_TYPE in
    "4G")
        # Higher latency, moderate bandwidth
        tc qdisc add dev eth0 root netem delay 50ms 20ms loss 2% 2>/dev/null || true
        ;;
    "5G")
        # Lower latency, high bandwidth
        tc qdisc add dev eth0 root netem delay 15ms 5ms loss 0.5% 2>/dev/null || true
        ;;
    "WiFi")
        # Variable latency
        tc qdisc add dev eth0 root netem delay 20ms 10ms loss 1% 2>/dev/null || true
        ;;
esac

# Simulate app backgrounding (mobile apps can be suspended)
if [ $(shuf -i 1-10 -n 1) -le 2 ]; then
    echo "📱 Simulating app backgrounding (reduced priority)"
    renice 15 $$ >/dev/null 2>&1 || true
fi

# Mobile optimization flags
export PYTHONOPTIMIZE=1
export MOBILE_SIMULATION=1

echo "✅ Mobile constraints applied"
echo "   - Memory: Limited to $MEMORY_LIMIT"
echo "   - CPU: Battery-aware scaling"
echo "   - Network: $NETWORK_TYPE simulation"
echo "   - Battery: $BATTERY_LEVEL%"
echo "   - Temperature: ${TEMPERATURE}°C"

# Start resource monitoring in background
/usr/local/bin/monitor_resources.sh &

# Execute the main command
exec "$@"