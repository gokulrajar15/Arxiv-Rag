# Use CPU Triton base image
FROM nvcr.io/nvidia/tritonserver:25.08-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies and Python for CPU
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies directly with pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Python scripts
COPY model_loading.py .
COPY onnx_conversion.py .

# Create logs directory
RUN mkdir -p /workspace/logs

# Wrapper script to run Python scripts first, then Triton
COPY run_all.sh .
RUN chmod +x run_all.sh

# Default command
CMD ["./run_all.sh"]
