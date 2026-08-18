# Import Path so tests can pass manifest, image-root, and checkpoint paths.
from pathlib import Path


# Test the API Gateway service contract without starting Uvicorn or Docker.
def test_api_gateway_health(load_service_module):
    # Load the API Gateway main.py file as an isolated test module.
    api_gateway = load_service_module(
        "api_gateway_main",
        "apps/api-gateway/main.py",
    )

    # Call the endpoint function directly and verify the public health response.
    assert api_gateway.health() == {"status": "ok", "service": "api-gateway"}


# Test that unsupported router inputs return the current contract response.
def test_inference_router_unsupported_route(load_service_module):
    # Load the inference-router main.py file as an isolated test module.
    inference_router = load_service_module(
        "inference_router_main",
        "apps/inference-router/main.py",
    )

    # Build the same request model FastAPI would create from JSON input.
    request = inference_router.RouteRequest(input_type="unknown")

    # Call the route function directly so no downstream service or Docker network is needed.
    response = inference_router.route(request)

    # Verify the unsupported route contract stays stable.
    assert response.model_dump() == {
        "input_type": "unknown",
        "selected_service": None,
        "status": "unsupported",
        "classification": None,
    }


# Build one tiny checkpoint for tests that exercise optional checkpoint inference.
def _create_tiny_galaxy_checkpoint(load_service_module, tmp_path):
    # Load the training helper so the test checkpoint uses the same code as the CLI proof.
    training_script = load_service_module(
        "service_contract_galaxy_checkpoint_training",
        "scripts/train_galaxy_cnn_baseline.py",
    )

    # Load the current tiny sample dataset.
    dataset_splits = training_script.load_dataset_splits_from_manifest(
        Path("data/samples/galaxy_manifest_sample.csv"),
        Path("data/samples/images"),
    )

    # Train the tiny model for two epochs so the checkpoint stores updated weights.
    model = training_script.create_tiny_galaxy_cnn()
    training_script.run_torch_training_loop(
        dataset_splits,
        epochs=2,
        model=model,
    )

    # Save into pytest's temporary folder, not the repo models/ artifact folder.
    checkpoint_path = tmp_path / "tiny_galaxy_cnn.pt"
    training_script.save_torch_checkpoint(model, checkpoint_path)
    return checkpoint_path


# Test the galaxy classifier stub contract before optional checkpoint config is set.
def test_galaxy_classifier_stub(load_service_module, monkeypatch):
    # Remove optional checkpoint config so this test proves the safe default path.
    monkeypatch.delenv("COSMOSAI_GALAXY_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("COSMOSAI_GALAXY_MANIFEST_PATH", raising=False)
    monkeypatch.delenv("COSMOSAI_GALAXY_DATA_ROOT", raising=False)

    # Load the galaxy classifier main.py file as an isolated test module.
    galaxy_classifier = load_service_module(
        "galaxy_classifier_main",
        "apps/galaxy-classifier-service/main.py",
    )

    # Build the request model used by the /classify endpoint.
    request = galaxy_classifier.ClassifyRequest(image_id="demo-galaxy-001")

    # Call the classify function directly and capture the stub response.
    response = galaxy_classifier.classify(request)

    # Verify the stub response shape and placeholder values.
    assert response.model_dump() == {
        "image_id": "demo-galaxy-001",
        "image_uri": None,
        "label": "spiral",
        "confidence": 0.0,
        "status": "stub",
    }


# Test that galaxy classifier can optionally use the local checkpoint helper.
def test_galaxy_classifier_optional_checkpoint_inference(
    load_service_module,
    monkeypatch,
    tmp_path,
):
    # Create a small checkpoint artifact using the current training proof path.
    checkpoint_path = _create_tiny_galaxy_checkpoint(load_service_module, tmp_path)

    # Set all required paths. Without these, the service intentionally stays stub-only.
    monkeypatch.setenv(
        "COSMOSAI_GALAXY_CHECKPOINT_PATH",
        str(checkpoint_path),
    )
    monkeypatch.setenv(
        "COSMOSAI_GALAXY_MANIFEST_PATH",
        "data/samples/galaxy_manifest_sample.csv",
    )
    monkeypatch.setenv(
        "COSMOSAI_GALAXY_DATA_ROOT",
        "data/samples/images",
    )

    # Load the service after env setup; the code still reads env at request time.
    galaxy_classifier = load_service_module(
        "galaxy_classifier_checkpoint_main",
        "apps/galaxy-classifier-service/main.py",
    )

    # The tiny checkpoint proof predicts by image_id, not arbitrary upload URI yet.
    request = galaxy_classifier.ClassifyRequest(image_id="gz2-000001")
    response = galaxy_classifier.classify(request)

    # Verify the service response now comes from the checkpoint helper.
    assert response.image_id == "gz2-000001"
    assert response.image_uri is None
    assert response.label == "spiral"
    assert response.confidence > 0.0
    assert response.status == "checkpoint_inference"


# Test that checkpoint config does not break image_uri-only stub requests.
def test_galaxy_classifier_checkpoint_config_keeps_uri_requests_stubbed(
    load_service_module,
    monkeypatch,
    tmp_path,
):
    # Create and configure a checkpoint, proving the image_uri fallback is intentional.
    checkpoint_path = _create_tiny_galaxy_checkpoint(load_service_module, tmp_path)
    monkeypatch.setenv("COSMOSAI_GALAXY_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv(
        "COSMOSAI_GALAXY_MANIFEST_PATH",
        "data/samples/galaxy_manifest_sample.csv",
    )
    monkeypatch.setenv("COSMOSAI_GALAXY_DATA_ROOT", "data/samples/images")

    # Load the service and call it with only an image URI.
    galaxy_classifier = load_service_module(
        "galaxy_classifier_checkpoint_uri_main",
        "apps/galaxy-classifier-service/main.py",
    )
    request = galaxy_classifier.ClassifyRequest(
        image_uri="file:///tmp/demo-galaxy.jpg"
    )
    response = galaxy_classifier.classify(request)

    # The current checkpoint proof cannot infer arbitrary URI uploads yet.
    assert response.model_dump() == {
        "image_id": None,
        "image_uri": "file:///tmp/demo-galaxy.jpg",
        "label": "spiral",
        "confidence": 0.0,
        "status": "stub",
    }


# Test the stellar classifier stub contract before real ML is added.
def test_stellar_classifier_stub(load_service_module):
    # Load the stellar classifier main.py file as an isolated test module.
    stellar_classifier = load_service_module(
        "stellar_classifier_main",
        "apps/stellar-classifier-service/main.py",
    )

    # Build the request model used by the /classify endpoint.
    request = stellar_classifier.ClassifyRequest(spectrum_id="demo-star-001")

    # Call the classify function directly and capture the stub response.
    response = stellar_classifier.classify(request)

    # Verify the stub response shape and placeholder values.
    assert response.model_dump() == {
        "spectrum_id": "demo-star-001",
        "spectrum_uri": None,
        "spectral_type": "G",
        "confidence": 0.0,
        "status": "stub",
    }
