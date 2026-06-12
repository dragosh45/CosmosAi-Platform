# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel

# Import requests so the router can call internal model services over HTTP.
import requests


# Internal Docker Compose URL for the galaxy-classifier-service endpoint.
GALAXY_CLASSIFIER_URL = "http://galaxy-classifier-service:8000/classify"

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

        # Return the selected service plus the classifier stub result.
        return RouteResponse(
            input_type=request.input_type,
            selected_service=selected_service,
            status="stub",
            classification=classifier_response.json(),
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

        # Return the selected service plus the classifier stub result.
        return RouteResponse(
            input_type=request.input_type,
            selected_service=selected_service,
            status="stub",
            classification=classifier_response.json(),
        )

    # Return the stub route for future supported services that are not wired yet.
    return RouteResponse(
        input_type=request.input_type,
        selected_service=selected_service,
        status="stub",
    )
