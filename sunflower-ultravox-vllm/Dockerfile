# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages system-wide (not --user)
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy the app code
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV PORT=8080

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
