FROM balenalib/raspberry-pi-python:3.11-bullseye

# Set device identification
ENV DEVICE_TYPE=raspberry_pi
ENV DEVICE_NAME="Raspberry Pi 4"
ENV MEMORY_LIMIT=2048m
ENV CPU_LIMIT=1.0
ENV TARGET_INFERENCE_MS=100

# Install system dependencies for ARM simulation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    stress-ng \
    cgroup-tools \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Python requirements
COPY ../requirements.txt .

# Install Python dependencies with ARM optimizations
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy federated learning code
COPY ../federated_learning_app ./federated_learning_app
COPY ../pyproject.toml .

# Copy device constraints
COPY constraints/pi_limits.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/pi_limits.sh

# Create resource monitoring script
RUN echo '#!/bin/bash\n\
echo "🔧 Raspberry Pi 4 Resource Monitor"\n\
echo "Memory limit: $MEMORY_LIMIT"\n\
echo "CPU limit: $CPU_LIMIT cores"\n\
echo "Target inference: $TARGET_INFERENCE_MS ms"\n\
echo "Current memory usage:"\n\
free -h\n\
echo "Current CPU usage:"\n\
top -bn1 | grep "Cpu(s)"\n\
' > /usr/local/bin/monitor_resources.sh && chmod +x /usr/local/bin/monitor_resources.sh

# Set resource limits at runtime
ENTRYPOINT ["/usr/local/bin/pi_limits.sh"]

# Default command
CMD ["python", "-m", "flwr.client", "--server", "server:8080"]

# Resource limits (enforced by Docker Compose)
LABEL device.memory="2048m"
LABEL device.cpu="1.0"
LABEL device.type="raspberry_pi"