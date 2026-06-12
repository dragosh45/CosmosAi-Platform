# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel


# Define the JSON body expected by POST /classify.
class ClassifyRequest(BaseModel):
    # Optional local/demo identifier for the galaxy image.
    image_id: str | None = None
    # Optional URI/path for the galaxy image.
    image_uri: str | None = None


# Define the JSON body returned by POST /classify.
class ClassifyResponse(BaseModel):
    # Echo the image identifier when provided.
    image_id: str | None
    # Echo the image URI/path when provided.
    image_uri: str | None
    # Stub galaxy morphology label; this will come from a model later.
    label: str
    # Stub confidence score; real confidence will come from a model later.
    confidence: float
    # "stub" means this endpoint shape is ready, but no ML is running yet.
    status: str


# Create the FastAPI application object.
# The title appears in generated API docs such as /docs.
app = FastAPI(title="CosmosAI Galaxy Classifier Service")


# Register a GET endpoint at /health for service health checks.
@app.get("/health")
# Define the function FastAPI runs when /health is requested.
def health() -> dict[str, str]:
    # Return a small JSON response confirming the service is running.
    return {"status": "ok", "service": "galaxy-classifier-service"}


# Register a POST endpoint for galaxy image classification.
@app.post("/classify", response_model=ClassifyResponse)
# Define the function FastAPI runs when /classify receives JSON input.
def classify(request: ClassifyRequest) -> ClassifyResponse:
    # Return a deterministic placeholder classification without loading an ML model.
    return ClassifyResponse(
        image_id=request.image_id,
        image_uri=request.image_uri,
        label="spiral",
        confidence=0.0,
        status="stub",
    )
