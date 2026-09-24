# Docs: docs/architecture.md Step 0 explains this service's API flow and links to this code.
#
# Concepts in this file:
# - FastAPI app: the HTTP server object that exposes endpoints.
# - Pydantic models: typed request/response schemas for JSON bodies.
# - API Gateway: the public entry point that hides internal service details.
# - Docker Compose hostname: "inference-router" resolves to another container.
# - HTTP forwarding: the gateway sends the request to inference-router.
# - Response validation: gateway JSON is converted back into RouteResponse.

# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Import requests so the gateway can call other internal services over HTTP.
import requests


# Internal Docker Compose URL for the inference-router service.
ROUTER_BASE_URL = os.getenv("COSMOSAI_ROUTER_BASE_URL", "http://inference-router:8000").rstrip("/")
INFERENCE_ROUTER_URL = f"{ROUTER_BASE_URL}/route"
SERVICE_DIR = Path(__file__).resolve().parent
PREVIEW_ROOT = Path(os.getenv(
    "COSMOSAI_PREVIEW_ROOT",
    str(SERVICE_DIR.parent.parent / "data/samples/images/gz2_pilot_preview"),
))
REPLAY_ROOT = Path(os.getenv("COSMOSAI_REPLAY_ROOT", str(SERVICE_DIR.parent / "learning-replay")))
LEARNING_CONTENT_ROOT = Path(os.getenv(
    "COSMOSAI_LEARNING_CONTENT_ROOT", str(SERVICE_DIR.parent / "learning-content/site"),
))


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



# Docs: docs/local_classifier_demo.md, M53b browser -> gateway -> model flow.
@app.post("/classify/image", response_model=RouteResponse)
async def classify_image(request: Request) -> RouteResponse:
    """Forward bounded image bytes and return the router's actual inference status."""
    payload = await forward_image(request, f"{ROUTER_BASE_URL}/classify/image", timeout=60)
    try:
        result = RouteResponse(**payload)
    except ValueError as error:
        raise HTTPException(502, "The router returned an invalid prediction") from error
    if result.status != "checkpoint_inference" or not result.classification:
        raise HTTPException(502, "The router did not return a checkpoint prediction")
    return result


@app.get("/model")
def model_information() -> dict:
    """Expose the active checkpoint and any evaluation bound to its checksum."""
    return upstream_request("GET", f"{ROUTER_BASE_URL}/model", timeout=30)


@app.get("/", include_in_schema=False)
def classify_page() -> FileResponse:
    return FileResponse(SERVICE_DIR / "static/index.html", headers={"Cache-Control": "no-store"})


@app.get("/learn/", include_in_schema=False)
def learning_replay_page() -> FileResponse:
    """Serve the recorded teaching trace without calling model services."""
    return FileResponse(REPLAY_ROOT / "index.html", headers={"Cache-Control": "no-store"})


# These specific mounts must precede /learn, which owns the replay's static files.
app.mount("/learn/concepts", StaticFiles(directory=LEARNING_CONTENT_ROOT / "concepts", html=True, check_dir=False), name="learning-concepts")
app.mount("/learn/diagrams", StaticFiles(directory=LEARNING_CONTENT_ROOT / "diagrams", html=True, check_dir=False), name="learning-diagrams")
app.mount("/learn", StaticFiles(directory=REPLAY_ROOT, html=True, check_dir=False), name="learning-replay")
app.mount("/assets", StaticFiles(directory=SERVICE_DIR / "static", check_dir=False), name="assets")
app.mount("/samples", StaticFiles(directory=PREVIEW_ROOT, check_dir=False), name="samples")
