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

# Document that the container listens on port 8000.
EXPOSE 8000

# Start the FastAPI app with Uvicorn when the container runs.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
