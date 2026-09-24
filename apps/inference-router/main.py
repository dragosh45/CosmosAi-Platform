# Docs: docs/architecture.md Step 0 explains this service's routing flow and links to this code.

# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI, HTTPException, Request
import os

# Direct runs from the service directory and Docker both find the shared package.
import sys
from pathlib import Path

parents = Path(__file__).resolve().parents
package_root = parents[2] if len(parents) > 2 else parents[0]
if (package_root / "cosmosai").is_dir() and str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from cosmosai.upload_proxy import forward_image, upstream_request

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel

# Import requests so the router can call internal model services over HTTP.
import requests


# Internal Docker Compose URL for the galaxy-classifier-service endpoint.
GALAXY_BASE_URL = os.getenv("COSMOSAI_GALAXY_BASE_URL", "http://galaxy-classifier-service:8000").rstrip("/")
GALAXY_CLASSIFIER_URL = f"{GALAXY_BASE_URL}/classify"

# Internal Docker Compose URL for the stellar-classifier-service endpoint.
STELLAR_CLASSIFIER_URL = "http://stellar-classifier-service:8000/classify"


# Map known input types to the service that will handle them later.
# Galaxy and stellar routing call service stubs; no real ML runs yet.
SERVICE_BY_INPUT_TYPE = {
    "galaxy_image": "galaxy-classifier-service",
    "stellar_spectrum": "stellar-classifier-service",
}


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
    # Name of the downstream service selected by the router, if supported.
    selected_service: str | None
    # "stub" means the input is recognized; "unsupported" means no route exists yet.
    status: str
    # Optional result returned by a downstream classifier service.
    classification: dict[str, object] | None = None


# Create the FastAPI application object.
# The title appears in generated API docs such as /docs.
app = FastAPI(title="CosmosAI Inference Router")


# Register a GET endpoint at /health for service health checks.
@app.get("/health")
# Define the function FastAPI runs when /health is requested.
def health() -> dict[str, str]:
    # Return a small JSON response confirming the service is running.
    return {"status": "ok", "service": "inference-router"}


# Register a POST endpoint that chooses a downstream service by input type.
@app.post("/route", response_model=RouteResponse)
# Define the function FastAPI runs when /route receives JSON input.
def route(request: RouteRequest) -> RouteResponse:
    # Look up the service that should handle this input type.
    selected_service = SERVICE_BY_INPUT_TYPE.get(request.input_type)

    # Return an unsupported response when no service mapping exists.
    if selected_service is None:
        return RouteResponse(
            input_type=request.input_type,
            selected_service=None,
            status="unsupported",
        )

    # Call the galaxy classifier stub when the request is for a galaxy image.
    if request.input_type == "galaxy_image":
        classifier_response = requests.post(
            GALAXY_CLASSIFIER_URL,
            json={
                "image_id": request.image_id,
                "image_uri": request.image_uri,
            },
            timeout=5,
        )

        # Raise an error if galaxy-classifier-service returns a non-success status.
        classifier_response.raise_for_status()

        # A checkpoint result must remain identifiable through the router.
        classification = classifier_response.json()
        return RouteResponse(
            input_type=request.input_type,
            selected_service=selected_service,
            status=classification.get("status", "stub"),
            classification=classification,
        )

    # Call the stellar classifier stub when the request is for a stellar spectrum.
    if request.input_type == "stellar_spectrum":
        classifier_response = requests.post(
            STELLAR_CLASSIFIER_URL,
            json={
                "spectrum_id": request.spectrum_id,
                "spectrum_uri": request.spectrum_uri,
            },
            timeout=5,
        )

        # Raise an error if stellar-classifier-service returns a non-success status.
        classifier_response.raise_for_status()

        # A checkpoint result must remain identifiable through the router.
        classification = classifier_response.json()
        return RouteResponse(
            input_type=request.input_type,
            selected_service=selected_service,
            status=classification.get("status", "stub"),
            classification=classification,
        )

    # Return the stub route for future supported services that are not wired yet.
    return RouteResponse(
        input_type=request.input_type,
        selected_service=selected_service,
        status="stub",
    )



@app.post("/classify/image", response_model=RouteResponse)
async def classify_image(request: Request) -> RouteResponse:
    """Route an uploaded image to real galaxy checkpoint inference."""
    classification = await forward_image(request, f"{GALAXY_BASE_URL}/classify/image", timeout=45)
    if (classification.get("status") != "checkpoint_inference"
            or not isinstance(classification.get("label"), str)
            or not isinstance(classification.get("probabilities"), dict)):
        raise HTTPException(502, "The classifier did not return a checkpoint prediction")
    return RouteResponse(
        input_type="galaxy_image", selected_service="galaxy-classifier-service",
        status="checkpoint_inference", classification=classification,
    )


@app.get("/model")
def model_information() -> dict:
    return upstream_request("GET", f"{GALAXY_BASE_URL}/model", timeout=25)
