# Docs: docs/architecture.md Step 0 explains this service stub and links to this code.
# Docs: docs/architecture.md Step 10 explains the optional checkpoint inference helper.

# Import os so the service can read optional local checkpoint configuration.
import os

# Import Path so environment paths can become filesystem paths.
from pathlib import Path

# Import sys so the local scripts/ folder can be added only when needed.
import sys

# Import dataclass so checkpoint configuration can be passed around clearly.
from dataclasses import dataclass

# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel


# Name of the environment variable that points to a saved PyTorch checkpoint.
CHECKPOINT_PATH_ENV = "COSMOSAI_GALAXY_CHECKPOINT_PATH"

# Name of the environment variable that points to the manifest CSV used locally.
MANIFEST_PATH_ENV = "COSMOSAI_GALAXY_MANIFEST_PATH"

# Name of the environment variable that points to the image data root used locally.
DATA_ROOT_ENV = "COSMOSAI_GALAXY_DATA_ROOT"

# Name of the optional environment variable that points to shared helper scripts.
# In Docker checkpoint mode, the service image is separate from the repo checkout,
# so this tells Python where copied helper files such as predict_galaxy_checkpoint.py live.
SCRIPTS_PATH_ENV = "COSMOSAI_SCRIPTS_PATH"


# Store optional checkpoint paths after reading the environment.
@dataclass(frozen=True)
class CheckpointInferenceConfig:
    # Saved PyTorch weights created by the training script.
    checkpoint_path: Path
    # CSV manifest that maps image_id values to image files and labels.
    manifest_path: Path
    # Folder used to resolve image_path values from the manifest.
    galaxy_data_root: Path


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


# Read optional local checkpoint configuration from environment variables.
def get_checkpoint_inference_config() -> CheckpointInferenceConfig | None:
    # All three paths are required because inference needs weights, metadata, and pixels.
    checkpoint_path = os.getenv(CHECKPOINT_PATH_ENV)
    manifest_path = os.getenv(MANIFEST_PATH_ENV)
    galaxy_data_root = os.getenv(DATA_ROOT_ENV)

    # Missing config means "stay in stub mode"; this keeps Docker/local service safe.
    if not checkpoint_path or not manifest_path or not galaxy_data_root:
        return None

    # Convert string environment values into Path objects for the prediction helper.
    return CheckpointInferenceConfig(
        checkpoint_path=Path(checkpoint_path),
        manifest_path=Path(manifest_path),
        galaxy_data_root=Path(galaxy_data_root),
    )


# Add the repository scripts folder to sys.path for local optional inference.
def ensure_scripts_import_path() -> None:
    # Allow Docker or local shells to provide an explicit scripts path.
    configured_scripts_path = os.getenv(SCRIPTS_PATH_ENV)

    # Store candidate script folders from most explicit to most automatic.
    candidate_paths: list[Path] = []

    # In optional Docker checkpoint mode, this usually points to /app/scripts.
    if configured_scripts_path:
        candidate_paths.append(Path(configured_scripts_path))

    # In a copied container image, scripts may live next to main.py as /app/scripts.
    candidate_paths.append(Path(__file__).resolve().parent / "scripts")

    # In the source tree, main.py lives under apps/galaxy-classifier-service/.
    # A minimal container path like /app/main.py has fewer parents, so guard it.
    main_file_parents = Path(__file__).resolve().parents
    if len(main_file_parents) > 2:
        source_tree_root = main_file_parents[2]
        candidate_paths.append(source_tree_root / "scripts")

    # Add the first existing path so lazy imports can find prediction helpers.
    for scripts_dir in candidate_paths:
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
            return

    # If no candidate exists, add the most explicit value anyway so import errors show it.
    if candidate_paths and str(candidate_paths[0]) not in sys.path:
        sys.path.insert(0, str(candidate_paths[0]))


# Try to classify with a saved checkpoint; return None when stub mode should stay active.
def classify_with_optional_checkpoint(
    request: ClassifyRequest,
) -> ClassifyResponse | None:
    # Read config at request time so tests and local runs can set env vars easily.
    config = get_checkpoint_inference_config()
    if config is None:
        return None

    # The current tiny checkpoint proof predicts by image_id, not by arbitrary upload URI.
    if request.image_id is None:
        return None

    # Make scripts/predict_galaxy_checkpoint.py importable for this local helper.
    ensure_scripts_import_path()

    # Import lazily so normal stub mode does not require torch or the scripts folder.
    from predict_galaxy_checkpoint import predict_from_checkpoint

    # Reuse the existing manifest -> preprocessing -> checkpoint prediction path.
    prediction = predict_from_checkpoint(
        config.manifest_path,
        config.galaxy_data_root,
        request.image_id,
        config.checkpoint_path,
    )

    # Confidence is the predicted class probability from softmax.
    confidence = prediction.probabilities[prediction.predicted_label_id]

    # Return the same API response shape, but mark that a checkpoint was used.
    return ClassifyResponse(
        image_id=request.image_id,
        image_uri=request.image_uri,
        label=prediction.predicted_label,
        confidence=confidence,
        status="checkpoint_inference",
    )


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
    # Try optional checkpoint inference first.
    # If config is missing, the helper returns None and the old stub remains active.
    checkpoint_response = classify_with_optional_checkpoint(request)
    if checkpoint_response is not None:
        return checkpoint_response

    # Return a deterministic placeholder classification without loading an ML model.
    return ClassifyResponse(
        image_id=request.image_id,
        image_uri=request.image_uri,
        label="spiral",
        confidence=0.0,
        status="stub",
    )
