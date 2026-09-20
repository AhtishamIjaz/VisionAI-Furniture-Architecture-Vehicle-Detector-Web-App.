# Use lightweight official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies required by OpenCV and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU first (keeps image compact and fast)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Ensure opencv headless is present
RUN pip install --no-cache-dir opencv-python-headless
# Install clip for YOLO-World support
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# Copy project files
COPY . .

# Create uploads directory
RUN mkdir -p uploads samples

# Expose port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/api/samples || exit 1

# Start Flask application
CMD ["python", "app.py"]
