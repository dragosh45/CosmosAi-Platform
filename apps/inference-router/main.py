# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel


# Map known input types to the service that will handle them later.
# This is only a stub for now; it does not call any model service yet.
SERVICE_BY_INPUT_TYPE = {
    "galaxy_image": "galaxy-classifier-service",
    "stellar_spectrum": "stellar-classifier-service",
}


# Define the JSON body expected by POST /route.
class RouteRequest(BaseModel):
    # The kind of input the platform needs to route.
    input_type: str


# Define the JSON body returned by POST /route.
class RouteResponse(BaseModel):
    # Echo the input type from the request.
    input_type: str
    # Name of the downstream service selected by the router, if supported.
    selected_service: str | None
    # "stub" means the input is recognized; "unsupported" means no route exists yet.
    status: str


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

    # Return the stub route without calling any real ML service yet.
    return RouteResponse(
        input_type=request.input_type,
        selected_service=selected_service,
        status="stub",
    )
