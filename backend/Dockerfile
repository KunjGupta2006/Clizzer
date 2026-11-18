# ==============================================================================
# Clizzer Backend Dockerfile
# Python 3.11 Slim with FFmpeg, OpenCV, and faster-whisper support
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    WHISPER_MODEL=tiny \
    WHISPER_MODEL_SIZE=tiny \
    DEVICE=cpu \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8

# Set working directory
WORKDIR /app

# Install system dependencies (ffmpeg is required for audio/video processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . /app/

# Create runtime scratch and cache directories
RUN mkdir -p /app/temp /app/output /app/models_cache

# Expose backend application port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the Clizzer backend server
CMD ["python", "app.py"]
