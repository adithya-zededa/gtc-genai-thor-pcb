# Slim Python image for vLLM backend mode
# This image does NOT include Ollama - inference is handled by external vLLM server
FROM python:3.11-slim

# Install system dependencies for OpenCV, PDF generation (WeasyPrint), and audio playback
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    # WeasyPrint PDF generation dependencies
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    # Audio playback tools
    alsa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p detected_images processed_frames

# Expose web app port
EXPOSE 8080

# Entrypoint script
COPY start.sh .
RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]
