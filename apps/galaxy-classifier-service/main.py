# Docs: docs/architecture.md Step 0 explains this service stub and links to this code.
# Docs: docs/local_classifier_demo.md explains the label-free /classify/image endpoint.
# Docs: docs/architecture.md Step 10 explains the optional checkpoint inference helper.
# Docs: docs/architecture.md Step 12 explains the shared package inference helper.
# Docs: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw explains the lazy import path.

# Import os so the service can read optional local checkpoint configuration.
import os
import hashlib
import json

# Import Path so environment paths can become filesystem paths.
from pathlib import Path

# Import sys so local direct service runs can find the shared cosmosai package.
import sys

# Import dataclass so checkpoint configuration can be passed around clearly.
from dataclasses import dataclass

# Import FastAPI, the web framework used to create HTTP API endpoints.
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from io import BytesIO
from pickle import UnpicklingError
from tempfile import TemporaryDirectory
from threading import Lock

# Import BaseModel to define structured request and response bodies.
from pydantic import BaseModel


# Name of the environment variable that points to a saved PyTorch checkpoint.
CHECKPOINT_PATH_ENV = "COSMOSAI_GALAXY_CHECKPOINT_PATH"

# Name of the environment variable that points to the manifest CSV used locally.
MANIFEST_PATH_ENV = "COSMOSAI_GALAXY_MANIFEST_PATH"

# Name of the environment variable that points to the image data root used locally.
DATA_ROOT_ENV = "COSMOSAI_GALAXY_DATA_ROOT"


# Make the shared cosmosai package importable for local direct service runs.
def ensure_shared_package_import_path() -> None:
    # Docker checkpoint mode copies cosmosai/ beside main.py under /app.
    service_dir = Path(__file__).resolve().parent

    # Source-tree mode has apps/galaxy-classifier-service/main.py under the repo root.
    candidate_roots = [service_dir]
    main_file_parents = Path(__file__).resolve().parents
    if len(main_file_parents) > 2:
        candidate_roots.append(main_file_parents[2])

    # Add the first folder that contains cosmosai/ so lazy imports can find it.
    for candidate_root in candidate_roots:
        if (candidate_root / "cosmosai").exists() and str(candidate_root) not in sys.path:
            sys.path.insert(0, str(candidate_root))
            return


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

    # Make the shared package importable for both Docker and local direct service runs.
    ensure_shared_package_import_path()

    # Import lazily so normal stub mode does not require torch or shared ML helpers.
    from cosmosai.galaxy.checkpoint_inference import predict_from_checkpoint
    from cosmosai.galaxy.preprocessing import CheckpointContractError

    # The checkpoint supplies its resize policy; the API must not guess image size.
    try:
        prediction = predict_from_checkpoint(
            config.manifest_path,
            config.galaxy_data_root,
            request.image_id,
            config.checkpoint_path,
        )
    except CheckpointContractError as error:
        # Invalid server model configuration is not a successful stub prediction.
        raise HTTPException(status_code=503, detail=str(error)) from error

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


# Bound uploads for the local CPU demo, whose preprocessing uses Python lists.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_PIXELS = 1024 * 1024
UPLOAD_INFERENCE_LOCK = Lock()


class ImagePredictionResponse(BaseModel):
    """Real checkpoint output for an image with no known target label."""

    label: str
    probabilities: dict[str, float]
    logits: dict[str, float]
    input_shape: tuple[int, int, int, int]
    checkpoint_name: str
    checkpoint_sha256: str
    preprocessing: dict[str, object]
    status: str = "checkpoint_inference"


def predict_uploaded_image(
    content: bytes, content_type: str, checkpoint_path: Path,
) -> ImagePredictionResponse:
    """Validate image bytes, reuse shared inference, and remove temporary pixels."""
    ensure_shared_package_import_path()
    try:
        from PIL import Image
        from cosmosai.galaxy.checkpoint_inference import predict_image_from_checkpoint
        from cosmosai.galaxy.labels import ID_TO_LABEL
        from cosmosai.galaxy.preprocessing import CheckpointContractError
    except ImportError as error:
        raise HTTPException(503, "Checkpoint inference dependencies are not installed") from error

    # Inspect dimensions before decoding; never trust a filename or MIME alone.
    try:
        with Image.open(BytesIO(content)) as opened:
            if opened.format not in {"JPEG", "PNG"}:
                raise HTTPException(415, "Only JPEG and PNG images are supported")
            expected_type = "image/jpeg" if opened.format == "JPEG" else "image/png"
            if content_type != expected_type:
                raise HTTPException(415, "Content-Type does not match the image format")
            if opened.width * opened.height > MAX_UPLOAD_PIXELS:
                raise HTTPException(413, "Image exceeds the 1,048,576-pixel local demo limit")
            opened.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, "Image dimensions are too large") from error
    except (OSError, ValueError, SyntaxError) as error:
        raise HTTPException(422, "The uploaded image cannot be decoded") from error

    # Internal names cannot contain an uploaded path; files live only for this call.
    with TemporaryDirectory(prefix="cosmosai-upload-") as directory:
        image_path = Path(directory) / ("image.jpg" if content_type == "image/jpeg" else "image.png")
        image_path.write_bytes(content)
        try:
            prediction = predict_image_from_checkpoint(image_path, checkpoint_path)
        except (CheckpointContractError, OSError, RuntimeError, EOFError, UnpicklingError) as error:
            raise HTTPException(503, "The configured checkpoint is unavailable or incompatible") from error
        except ValueError as error:
            raise HTTPException(422, "The uploaded image cannot be prepared") from error

    return ImagePredictionResponse(
        label=prediction.predicted_label,
        probabilities={ID_TO_LABEL[i]: value for i, value in enumerate(prediction.probabilities)},
        logits={ID_TO_LABEL[i]: value for i, value in enumerate(prediction.logits)},
        input_shape=prediction.input_shape,
        checkpoint_name=checkpoint_path.name,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        preprocessing=prediction.preprocessing.to_metadata(),
    )


@app.post(
    "/classify/image",
    response_model=ImagePredictionResponse,
    openapi_extra={"requestBody": {
        "required": True,
        "content": {kind: {"schema": {"type": "string", "format": "binary"}}
                    for kind in ("image/jpeg", "image/png")},
    }},
)
async def classify_image(request: Request) -> ImagePredictionResponse:
    """Predict raw JPG/PNG bytes with saved weights; no training or manifest lookup.

    Send the file as the request body with its image Content-Type. The local
    limit is 5 MiB and 1,048,576 pixels. Probabilities are not calibrated accuracy.
    """
    checkpoint_path = os.getenv(CHECKPOINT_PATH_ENV)
    if not checkpoint_path:
        raise HTTPException(503, "Configure COSMOSAI_GALAXY_CHECKPOINT_PATH before predicting")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Send JPEG or PNG bytes with image/jpeg or image/png Content-Type")
    # Apply the limit even when a client omits or misreports Content-Length.
    if not UPLOAD_INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(503, "Another image is being processed; retry shortly")
    try:
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Image exceeds the 5 MiB upload limit")
            content.extend(chunk)
        if not content:
            raise HTTPException(422, "The image is empty")
        # CPU work runs off the event loop, so health checks stay responsive.
        return await run_in_threadpool(
            predict_uploaded_image, bytes(content), content_type, Path(checkpoint_path),
        )
    finally:
        UPLOAD_INFERENCE_LOCK.release()



@app.get("/model")
def model_information() -> dict:
    """Identify active weights; show metrics only when the model-card hash matches."""
    configured = os.getenv(CHECKPOINT_PATH_ENV)
    if not configured:
        raise HTTPException(503, "Configure COSMOSAI_GALAXY_CHECKPOINT_PATH before predicting")
    ensure_shared_package_import_path()
    try:
        from cosmosai.galaxy.model import load_torch_checkpoint_bundle
        from cosmosai.galaxy.labels import ID_TO_LABEL
        checkpoint_path = Path(configured)
        bundle = load_torch_checkpoint_bundle(checkpoint_path)
        checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    except (ImportError, OSError, RuntimeError, ValueError, EOFError, UnpicklingError) as error:
        raise HTTPException(503, "The configured checkpoint is unavailable or incompatible") from error
    evaluation = None
    card_path = os.getenv("COSMOSAI_GALAXY_MODEL_CARD")
    if card_path:
        try:
            card = json.loads(Path(card_path).read_text())
            if isinstance(card, dict) and card.get("checkpoint_sha256") == checksum:
                evaluation = card
        except (OSError, ValueError):
            pass  # Missing evaluation does not prevent prediction with valid weights.
    return {
        "status": "ready", "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": checksum, "labels": list(ID_TO_LABEL.values()),
        "parameter_count": sum(p.numel() for p in bundle.model.parameters()),
        "preprocessing": bundle.preprocessing.to_metadata(), "evaluation": evaluation,
    }
