# Start from the official lightweight Python 3.10 image.
FROM python:3.10-slim

# Set /app as the working directory inside the container.
WORKDIR /app

# Copy the service dependency list into the container.
COPY apps/inference-router/requirements.txt .

# Install the Python dependencies without keeping pip's download cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container.
COPY apps/inference-router/main.py .
COPY cosmosai ./cosmosai

# Document the local default. Cloud Run assigns a distinct sidecar port.
EXPOSE 8000

# Keep Compose on 8000 while allowing a Cloud Run sidecar port override.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --limit-concurrency 8"]
