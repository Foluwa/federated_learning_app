#!/bin/bash

# Federated Learning Docker Simulation Quick Start
# Usage: ./start_federation.sh [preset] [pi_count] [mobile_count] [workstation_count] [server_count]

set -e

echo "🚀 Federated Learning Docker Simulation"
echo "========================================"

# Default values
PI_COUNT=2
MOBILE_COUNT=2
WORKSTATION_COUNT=1
SERVER_COUNT=0
MODE="dev"

# Preset configurations
case "${1:-dev}" in
    "dev"|"development")
        PI_COUNT=3
        MOBILE_COUNT=2
        WORKSTATION_COUNT=1
        SERVER_COUNT=0
        MODE="dev"
        echo "📱 Using DEVELOPMENT preset: 3 Pi + 2 Mobile + 1 Workstation"
        ;;
    "test"|"testing")
        PI_COUNT=4
        MOBILE_COUNT=3
        WORKSTATION_COUNT=2
        SERVER_COUNT=1
        MODE="test"
        echo "🧪 Using TESTING preset: 4 Pi + 3 Mobile + 2 Workstation + 1 Server"
        ;;
    "prod"|"production")
        PI_COUNT=6
        MOBILE_COUNT=4
        WORKSTATION_COUNT=2
        SERVER_COUNT=1
        MODE="prod"
        echo "🏭 Using PRODUCTION preset: 6 Pi + 4 Mobile + 2 Workstation + 1 Server"
        ;;
    "custom")
        PI_COUNT=${2:-2}
        MOBILE_COUNT=${3:-2}
        WORKSTATION_COUNT=${4:-1}
        SERVER_COUNT=${5:-0}
        MODE=${6:-dev}
        echo "⚙️  Using CUSTOM configuration: $PI_COUNT Pi + $MOBILE_COUNT Mobile + $WORKSTATION_COUNT Workstation + $SERVER_COUNT Server"
        ;;
    "demo")
        PI_COUNT=2
        MOBILE_COUNT=1
        WORKSTATION_COUNT=1
        SERVER_COUNT=0
        MODE="dev"
        echo "🎬 Using DEMO preset: 2 Pi + 1 Mobile + 1 Workstation (quick demo)"
        ;;
    *)
        echo "❓ Unknown preset. Available presets:"
        echo "   dev      - Development (3 Pi + 2 Mobile + 1 Workstation)"
        echo "   test     - Testing (4 Pi + 3 Mobile + 2 Workstation + 1 Server)"  
        echo "   prod     - Production (6 Pi + 4 Mobile + 2 Workstation + 1 Server)"
        echo "   demo     - Demo (2 Pi + 1 Mobile + 1 Workstation)"
        echo "   custom   - Custom (specify counts: ./start_federation.sh custom 3 2 1 0 dev)"
        exit 1
        ;;
esac

# Calculate total devices
TOTAL_DEVICES=$((PI_COUNT + MOBILE_COUNT + WORKSTATION_COUNT + SERVER_COUNT))

echo ""
echo "🔧 Configuration Summary:"
echo "   📱 Raspberry Pi devices: $PI_COUNT"
echo "   📱 Mobile devices: $MOBILE_COUNT"
echo "   🖥️  Workstation devices: $WORKSTATION_COUNT"
echo "   🏥 Server devices: $SERVER_COUNT"
echo "   📊 Total client devices: $TOTAL_DEVICES"
echo "   🎯 Training mode: $MODE"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Estimate resource requirements
# TOTAL_MEMORY=$((PI_COUNT * 2 + MOBILE_COUNT * 4 + WORKSTATION_COUNT * 8 + SERVER_COUNT * 16 + 16))  # +16 for FL server

# Read memory limits from .env or use defaults
PI_MEM=$(grep PI_MEMORY .env 2>/dev/null | cut -d'=' -f2 | sed 's/m//' | head -1)
MOBILE_MEM=$(grep MOBILE_MEMORY .env 2>/dev/null | cut -d'=' -f2 | sed 's/m//' | head -1)
WORKSTATION_MEM=$(grep WORKSTATION_MEMORY .env 2>/dev/null | cut -d'=' -f2 | sed 's/m//' | head -1)
SERVER_MEM=$(grep SERVER_MEMORY .env 2>/dev/null | cut -d'=' -f2 | sed 's/m//' | head -1)

# Use defaults if not found
PI_MEM=${PI_MEM:-1024}
MOBILE_MEM=${MOBILE_MEM:-2048}
WORKSTATION_MEM=${WORKSTATION_MEM:-4096}
SERVER_MEM=${SERVER_MEM:-8192}

# Calculate actual memory requirements in GB
TOTAL_MEMORY=$(echo "scale=0; ($PI_COUNT * $PI_MEM + $MOBILE_COUNT * $MOBILE_MEM + $WORKSTATION_COUNT * $WORKSTATION_MEM + $SERVER_COUNT * $SERVER_MEM + $SERVER_MEM) / 1024" | bc)

# TOTAL_CPU=$(echo "$PI_COUNT * 1 + $MOBILE_COUNT * 2 + $WORKSTATION_COUNT * 4 + $SERVER_COUNT * 8 + 8" | bc)
TOTAL_CPU=$((PI_COUNT * 1 + MOBILE_COUNT * 2 + WORKSTATION_COUNT * 4 + SERVER_COUNT * 8 + 8))

echo ""
echo "💻 Estimated Resource Requirements:"
echo "   Memory: ${TOTAL_MEMORY}GB"
echo "   CPU cores: ${TOTAL_CPU}"

# Check available resources
# AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
# AVAILABLE_CPU=$(nproc)


# macOS compatible resource checking
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    AVAILABLE_MEMORY=$(echo "$(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc)
    AVAILABLE_CPU=$(sysctl -n hw.ncpu)
else
    # Linux
    AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
    AVAILABLE_CPU=$(nproc)
fi


if [ "$TOTAL_MEMORY" -gt "$AVAILABLE_MEMORY" ]; then
    echo "⚠️  Warning: Estimated memory usage (${TOTAL_MEMORY}GB) exceeds available memory (${AVAILABLE_MEMORY}GB)"
    echo "   Consider reducing device counts or using a more powerful machine"
fi

if [ "$TOTAL_CPU" -gt "$AVAILABLE_CPU" ]; then
    echo "⚠️  Warning: Estimated CPU usage (${TOTAL_CPU} cores) exceeds available CPU (${AVAILABLE_CPU} cores)"
    echo "   Performance may be degraded"
fi

echo ""
read -p "🚀 Do you want to start the simulation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "👋 Simulation cancelled"
    exit 0
fi

# Change to the docker simulation directory
cd "$(dirname "$0")/.."

# Start the simulation using device manager
echo "🏃 Starting federated learning simulation..."
python3 scripts/device_manager.py start \
    --pi $PI_COUNT \
    --mobile $MOBILE_COUNT \
    --workstation $WORKSTATION_COUNT \
    --server $SERVER_COUNT \
    --mode $MODE

echo ""
echo "✅ Federated Learning simulation started!"
echo ""
echo "🌐 Access Points:"
echo "   📊 FL Server Dashboard: http://localhost:8080"
echo "   📈 Resource Monitor: http://localhost:9100"
echo "   📋 Container Status: docker ps"
echo ""
echo "📝 Useful Commands:"
echo "   View logs: python3 scripts/device_manager.py logs [service-name]"
echo "   Monitor resources: python3 scripts/device_manager.py monitor"
echo "   Scale devices: python3 scripts/device_manager.py scale pi 5"
echo "   Stop simulation: python3 scripts/device_manager.py stop"
echo ""
echo "🎯 The federated learning will start automatically."
echo "   Monitor progress with: python3 scripts/device_manager.py logs fl-server -f"