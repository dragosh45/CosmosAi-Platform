# Start from the official lightweight Python 3.10 image.
FROM python:3.10-slim

# Set /app as the working directory inside the container.
WORKDIR /app

# Copy the service dependency list into the container.
COPY apps/api-gateway/requirements.txt .

# Install the Python dependencies without keeping pip's download cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container.
COPY apps/api-gateway/main.py .
COPY cosmosai ./cosmosai
COPY apps/api-gateway/static ./static
COPY apps/learning-replay ./learning-replay
COPY apps/learning-content/site ./learning-content
COPY data/samples/images/gz2_pilot_preview ./demo-images
ENV COSMOSAI_PREVIEW_ROOT=/app/demo-images
ENV COSMOSAI_REPLAY_ROOT=/app/learning-replay
ENV COSMOSAI_LEARNING_CONTENT_ROOT=/app/learning-content

# Document the local default. Cloud Run injects PORT=8080 for ingress.
EXPOSE 8000

# Keep Compose on 8000 while allowing the same image to run as Cloud Run ingress.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --limit-concurrency 16"]
