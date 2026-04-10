FROM python:3.10-slim

WORKDIR /app

# System dependencies (slim image ships without gcc needed by some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only must be installed before requirements.txt so that pip
# does not pull the CUDA wheel from PyPI.
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# src/ is the root package directory
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/main.py"]
