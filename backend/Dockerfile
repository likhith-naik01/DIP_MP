FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy ML package and install it in editable mode
COPY ml /app/ml
RUN pip install --no-cache-dir -e /app/ml

# Copy frontend assets for direct web studio serving
COPY frontend /app/frontend

# Copy backend application
COPY backend /app/backend
WORKDIR /app

# Create storage directories
RUN mkdir -p /app/storage/uploads /app/storage/results /app/storage/sample_images

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
