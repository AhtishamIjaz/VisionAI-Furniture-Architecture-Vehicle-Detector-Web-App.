# Use lightweight Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5050

# Install system dependencies needed for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements first (layer caching)
COPY requirements.txt .

# Install dependencies cleanly
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, templates, static assets and preloaded models
COPY app.py .
COPY yolo11m.onnx .
COPY templates ./templates
COPY static ./static
COPY sample_interior.jpg .
COPY sample_bedroom.jpg .
COPY sample_dining.jpg .

# Create writable upload/result directories
RUN mkdir -p static/uploads static/results

# Expose port 5050
EXPOSE 5050

# Health check so Docker knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5050/ || exit 1

# Run with Gunicorn production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "2", "--timeout", "120", "app:app"]
