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


# Test the galaxy classifier stub contract before real ML is added.
def test_galaxy_classifier_stub(load_service_module):
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
