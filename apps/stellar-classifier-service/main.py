# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel


# Define the JSON body expected by POST /classify.
class ClassifyRequest(BaseModel):
    # Optional local/demo identifier for the stellar spectrum.
    spectrum_id: str | None = None
    # Optional URI/path for the stellar spectrum file.
    spectrum_uri: str | None = None


# Define the JSON body returned by POST /classify.
class ClassifyResponse(BaseModel):
    # Echo the spectrum identifier when provided.
    spectrum_id: str | None
    # Echo the spectrum URI/path when provided.
    spectrum_uri: str | None
    # Stub Harvard spectral type; this will come from a model later.
    spectral_type: str
    # Stub confidence score; real confidence will come from a model later.
    confidence: float
    # "stub" means this endpoint shape is ready, but no ML is running yet.
    status: str


# Create the FastAPI application object.
# The title appears in generated API docs such as /docs.
app = FastAPI(title="CosmosAI Stellar Classifier Service")


# Register a GET endpoint at /health for service health checks.
@app.get("/health")
# Define the function FastAPI runs when /health is requested.
def health() -> dict[str, str]:
    # Return a small JSON response confirming the service is running.
    return {"status": "ok", "service": "stellar-classifier-service"}


# Register a POST endpoint for stellar spectrum classification.
@app.post("/classify", response_model=ClassifyResponse)
# Define the function FastAPI runs when /classify receives JSON input.
def classify(request: ClassifyRequest) -> ClassifyResponse:
    # Return a deterministic placeholder classification without loading an ML model.
    return ClassifyResponse(
        spectrum_id=request.spectrum_id,
        spectrum_uri=request.spectrum_uri,
        spectral_type="G",
        confidence=0.0,
        status="stub",
    )
