# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel

# Import requests so the gateway can call other internal services over HTTP.
import requests


# Internal Docker Compose URL for the inference-router service.
INFERENCE_ROUTER_URL = "http://inference-router:8000/route"


# Define the JSON body expected by POST /route.
class RouteRequest(BaseModel):
    # The kind of input the platform needs to route.
    input_type: str
    # Optional local/demo identifier for image-based requests.
    image_id: str | None = None
    # Optional URI/path for image-based requests.
    image_uri: str | None = None
    # Optional local/demo identifier for spectrum-based requests.
    spectrum_id: str | None = None
    # Optional URI/path for spectrum-based requests.
    spectrum_uri: str | None = None


# Define the JSON body returned by POST /route.
class RouteResponse(BaseModel):
    # Echo the input type from the request.
    input_type: str
    # Name of the downstream service selected by inference-router, if supported.
    selected_service: str | None
    # "stub" means the input is recognized; "unsupported" means no route exists yet.
    status: str
    # Optional result returned by a downstream classifier service.
    classification: dict[str, object] | None = None


# Create the FastAPI application object.
# The title appears in generated API docs such as /docs.
app = FastAPI(title="CosmosAI API Gateway")


# Register a GET endpoint at /health for service health checks.
@app.get("/health")
# Define the function FastAPI runs when /health is requested.
def health() -> dict[str, str]:
    # Return a small JSON response confirming the service is running.
    return {"status": "ok", "service": "api-gateway"}


# Register a POST endpoint that forwards routing requests to inference-router.
@app.post("/route", response_model=RouteResponse)
# Define the function FastAPI runs when /route receives JSON input.
def route(request: RouteRequest) -> RouteResponse:
    # Send the request body to inference-router using the internal Compose URL.
    response = requests.post(
        INFERENCE_ROUTER_URL,
        json=request.model_dump(),
        timeout=5,
    )

    # Raise an error if inference-router returns a non-success HTTP status.
    response.raise_for_status()

    # Validate and return the inference-router JSON response.
    return RouteResponse(**response.json())
