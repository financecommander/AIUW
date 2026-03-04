FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements.txt requirements-dev.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/

# Default entry point: run Phase 1 pipeline
ENTRYPOINT ["python", "-m", "src.training.pipeline"]
CMD ["--config", "configs/phase1.yaml"]
