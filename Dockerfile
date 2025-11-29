# --- STAGE 1: LFS Fetcher ---
FROM alpine/git:latest AS lfs-fetcher

# REPLACE THIS with your actual LSTM repo URL
ARG REPO_URL="https://github.com/PrajwalShetty-114/LSTM-Model.git"
ARG BRANCH="master"

WORKDIR /repo

# Clone and pull LFS files
RUN git clone --branch ${BRANCH} ${REPO_URL} .
RUN git lfs install
RUN git lfs pull

# --- STAGE 2: Application ---
# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies
# libgomp1 is often needed for math libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Install Python requirements
COPY requirements.txt .
# Install tensorflow-cpu specifically to save space/RAM
RUN pip install --no-cache-dir tensorflow-cpu && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy code and LFS files from Stage 1
COPY --from=lfs-fetcher /repo/data/ /app/data/
COPY --from=lfs-fetcher /repo/main.py .

# 4. Run the application
# Optimized for Free Tier: 1 worker, 300s timeout
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "main:app", "--bind", "0.0.0.0:$PORT"]