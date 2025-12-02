# Use dustynv's Ollama image for Jetson with GPU support
FROM dustynv/ollama:r36.4.0

# Set NVIDIA library paths for Jetson GPU support
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/cuda/compat:${LD_LIBRARY_PATH}

# Install additional system dependencies
# libgl1 and libglib2.0-0 are required for opencv-python
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    socat \
    && rm -rf /var/lib/apt/lists/*

# Update Ollama to the latest version (dustynv image has 0.6.5, we need newer for qwen3-vl)
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# Remove custom pip config and use standard PyPI
# Use --ignore-installed to handle distutils conflicts
RUN rm -f /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf && \
    pip3 install --upgrade pip -i https://pypi.org/simple/ && \
    pip3 install --no-cache-dir --ignore-installed -r requirements.txt -i https://pypi.org/simple/

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p detected_images processed_frames

# Expose ports
# 8080 for Web App
# 11434 for Ollama (optional, if you want to access it externally)
EXPOSE 8080
EXPOSE 11434
# Entrypoint script
COPY start.sh .
RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]
