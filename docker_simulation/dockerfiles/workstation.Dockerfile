# Medical Workstation Simulation
FROM python:3.11-bullseye

# Set device identification
ENV DEVICE_TYPE=workstation
ENV DEVICE_NAME="Medical Workstation"
ENV MEMORY_LIMIT=8192m
ENV CPU_LIMIT=4.0
ENV TARGET_INFERENCE_MS=20

# Install system dependencies with development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    stress-ng \
    nvidia-ml-py3 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY ../requirements.txt .

# Install Python dependencies with workstation optimizations
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    && pip install --no-cache-dir -r requirements.txt

# Copy federated learning code
COPY ../federated_learning_app ./federated_learning_app
COPY ../pyproject.toml .

# Copy device constraints
COPY constraints/workstation_limits.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/workstation_limits.sh

# Create workstation-specific monitoring
RUN echo '#!/bin/bash\n\
echo "🖥️  Medical Workstation Resource Monitor"\n\
echo "Memory limit: $MEMORY_LIMIT"\n\
echo "CPU limit: $CPU_LIMIT cores"\n\
echo "Target inference: $TARGET_INFERENCE_MS ms"\n\
echo "Workstation mode: High Performance"\n\
echo "GPU acceleration: $(if command -v nvidia-smi >/dev/null; then echo "Available"; else echo "CPU-only"; fi)"\n\
free -h | head -n 2\n\
echo "CPU cores available: $(nproc)"\n\
' > /usr/local/bin/monitor_resources.sh && chmod +x /usr/local/bin/monitor_resources.sh

# Set resource limits at runtime
ENTRYPOINT ["/usr/local/bin/workstation_limits.sh"]

# Default command
CMD ["python", "-m", "flwr.client", "--server", "server:8080"]

# Resource limits
LABEL device.memory="8192m"
LABEL device.cpu="4.0"
LABEL device.type="workstation"