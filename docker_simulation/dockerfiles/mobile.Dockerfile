FROM python:3.11-slim-bullseye

# Set device identification
ENV DEVICE_TYPE=mobile
ENV DEVICE_NAME="Mobile Device"
ENV MEMORY_LIMIT=4096m
ENV CPU_LIMIT=2.0
ENV TARGET_INFERENCE_MS=50

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    htop \
    stress-ng \
    cpulimit \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY ../requirements.txt .

# Install Python dependencies with mobile optimizations
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy federated learning code
COPY ../federated_learning_app ./federated_learning_app
COPY ../pyproject.toml .

# Copy device constraints
COPY constraints/mobile_limits.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/mobile_limits.sh

# Create mobile-specific monitoring
RUN echo '#!/bin/bash\n\
echo "📱 Mobile Device Resource Monitor"\n\
echo "Memory limit: $MEMORY_LIMIT"\n\
echo "CPU limit: $CPU_LIMIT cores"\n\
echo "Target inference: $TARGET_INFERENCE_MS ms"\n\
echo "Battery simulation: $(shuf -i 20-100 -n 1)%"\n\
echo "Network: $(shuf -e "4G" "5G" "WiFi" -n 1)"\n\
free -h | head -n 2\n\
' > /usr/local/bin/monitor_resources.sh && chmod +x /usr/local/bin/monitor_resources.sh

# Simulate mobile power management (CPU throttling)
RUN echo '#!/bin/bash\n\
# Simulate mobile CPU throttling\n\
if [ $(shuf -i 1-10 -n 1) -le 3 ]; then\n\
    echo "📱 Simulating battery-saving mode (CPU throttled)"\n\
    cpulimit -l 50 -i &\n\
fi\n\
exec "$@"\n\
' > /usr/local/bin/mobile_power_mgmt.sh && chmod +x /usr/local/bin/mobile_power_mgmt.sh

# Set resource limits at runtime
ENTRYPOINT ["/usr/local/bin/mobile_limits.sh"]

# Default command
CMD ["python", "-m", "flwr.client", "--server", "server:8080"]

# Resource limits
LABEL device.memory="4096m"
LABEL device.cpu="2.0"
LABEL device.type="mobile"