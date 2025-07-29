# Federated Learning Server
FROM python:3.11-bullseye

# Set server identification
ENV DEVICE_TYPE=server
ENV DEVICE_NAME="FL Server"
ENV MEMORY_LIMIT=16384m
ENV CPU_LIMIT=8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    stress-ng \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    && pip install --no-cache-dir -r requirements.txt

# Copy federated learning code
COPY ../federated_learning_app ./federated_learning_app
COPY ../pyproject.toml .

# Create server monitoring
RUN echo '#!/bin/bash\n\
echo "🏥 Federated Learning Server Monitor"\n\
echo "Memory limit: $MEMORY_LIMIT"\n\
echo "CPU limit: $CPU_LIMIT cores"\n\
echo "Server role: Coordinator & Knowledge Distillation"\n\
echo "Connected devices: $(netstat -an | grep :8080 | grep ESTABLISHED | wc -l)"\n\
free -h | head -n 2\n\
echo "Load average: $(uptime | awk -F"load average:" "{print $2}")"\n\
' > /usr/local/bin/monitor_resources.sh && chmod +x /usr/local/bin/monitor_resources.sh

# Expose Flower server port
EXPOSE 8080

# Default command - run Flower server
CMD ["python", "-m", "flwr.server", "--config", "num_rounds=3"]

# Resource limits
LABEL device.memory="16384m"
LABEL device.cpu="8.0"
LABEL device.type="server"