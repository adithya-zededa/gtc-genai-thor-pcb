# Slim Python image for vLLM backend mode
# This image does NOT include Ollama - inference is handled by external vLLM server
FROM python:3.11-slim

# System dependencies for OpenCV, plus curl for the start.sh readiness wait.
# The WeasyPrint (Pango/Cairo) and ALSA/ffmpeg packages that used to live here
# were dropped along with their Python packages: reports are generated as
# JSON, not PDF, and the audio playback module is archived.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
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

# Create directories for data and move the video simulator to an organized
# location. IMG_1043.MOV is a large local-only asset (gitignored), so it is
# absent from a clean clone's build context — the move is best-effort rather
# than a hard build dependency. When it is missing, core/config.py drops the
# CAMERA_VIDEO_SOURCE below and the agent uses the live camera instead.
RUN mkdir -p detected_images processed_frames video && \
    if [ -f /app/IMG_1043.MOV ]; then mv /app/IMG_1043.MOV /app/video/IMG_1043.MOV; \
    else echo "IMG_1043.MOV not in build context — video simulator unavailable"; fi

# Video feed simulator — overridden at runtime by setting CAMERA_VIDEO_SOURCE=""
ENV CAMERA_VIDEO_SOURCE=/app/video/IMG_1043.MOV

# Expose web app port
EXPOSE 8080

# Entrypoint script
COPY start.sh .
RUN chmod +x start.sh

# Run as a non-root user. UID/GID 1000 matches the fsGroup the Helm chart
# sets on the persistent data volume, so writes to /app/data still work
# under Kubernetes without the image needing to run as root.
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -M -s /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["./start.sh"]
