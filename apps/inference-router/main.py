# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI


# Create the FastAPI application object.
# The title appears in generated API docs such as /docs.
app = FastAPI(title="CosmosAI Inference Router")


# Register a GET endpoint at /health for service health checks.
@app.get("/health")
# Define the function FastAPI runs when /health is requested.
def health() -> dict[str, str]:
    # Return a small JSON response confirming the service is running.
    return {"status": "ok", "service": "inference-router"}
