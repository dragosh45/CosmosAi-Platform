#!/usr/bin/env bash

# Stop the script as soon as a command fails, an unset variable is used, or a pipeline fails.
set -euo pipefail

# Store the repository root so the script can be run from any directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Move to the repository root because docker-compose.yml lives there.
cd "$REPO_ROOT"

# Start all local services and rebuild images so the test uses the current code.
docker compose up --build -d

# Always stop the Compose services when this script exits.
trap 'docker compose down' EXIT

# Check that an HTTP endpoint eventually returns the expected JSON response.
assert_json() {
  # Store a short label for clear test output.
  local name="$1"

  # Store the URL that should be called.
  local url="$2"

  # Store the exact JSON response expected from the service.
  local expected="$3"

  # Try several times because containers can need a few seconds to become ready.
  for attempt in {1..30}; do
    # Capture the HTTP response body; keep retrying if the service is not ready yet.
    actual="$(curl -fsS "$url" 2>/dev/null || true)"

    # Pass the check when the response body matches the expected JSON exactly.
    if [[ "$actual" == "$expected" ]]; then
      echo "PASS: $name"
      return 0
    fi

    # Wait briefly before trying again.
    sleep 1
  done

  # Print the mismatch to make failures easy to understand.
  echo "FAIL: $name"
  echo "Expected: $expected"
  echo "Actual:   $actual"
  return 1
}

# Send a POST request with JSON and compare the response to the expected JSON.
assert_post_json() {
  # Store a short label for clear test output.
  local name="$1"

  # Store the URL that should receive the POST request.
  local url="$2"

  # Store the JSON request body.
  local payload="$3"

  # Store the exact JSON response expected from the service.
  local expected="$4"

  # Send the JSON request to the running service.
  actual="$(curl -fsS -X POST "$url" -H 'Content-Type: application/json' -d "$payload")"

  # Pass the check when the response body matches the expected JSON exactly.
  if [[ "$actual" == "$expected" ]]; then
    echo "PASS: $name"
    return 0
  fi

  # Print the mismatch to make failures easy to understand.
  echo "FAIL: $name"
  echo "Expected: $expected"
  echo "Actual:   $actual"
  return 1
}

# Verify every service responds on its host-mapped health endpoint.
assert_json "api-gateway health" "http://127.0.0.1:8000/health" '{"status":"ok","service":"api-gateway"}'
assert_json "inference-router health" "http://127.0.0.1:8001/health" '{"status":"ok","service":"inference-router"}'
assert_json "galaxy-classifier-service health" "http://127.0.0.1:8002/health" '{"status":"ok","service":"galaxy-classifier-service"}'
assert_json "stellar-classifier-service health" "http://127.0.0.1:8003/health" '{"status":"ok","service":"stellar-classifier-service"}'

# Verify the real container path for a galaxy request: api-gateway -> router -> galaxy classifier.
assert_post_json \
  "api-gateway galaxy route" \
  "http://127.0.0.1:8000/route" \
  '{"input_type":"galaxy_image","image_id":"demo-galaxy-001"}' \
  '{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub","classification":{"image_id":"demo-galaxy-001","image_uri":null,"label":"spiral","confidence":0.0,"status":"stub"}}'

# Verify the real container path for a stellar request: api-gateway -> router -> stellar classifier.
assert_post_json \
  "api-gateway stellar route" \
  "http://127.0.0.1:8000/route" \
  '{"input_type":"stellar_spectrum","spectrum_id":"demo-star-001"}' \
  '{"input_type":"stellar_spectrum","selected_service":"stellar-classifier-service","status":"stub","classification":{"spectrum_id":"demo-star-001","spectrum_uri":null,"spectral_type":"G","confidence":0.0,"status":"stub"}}'

# Verify unsupported inputs still return the known unsupported contract.
assert_post_json \
  "api-gateway unsupported route" \
  "http://127.0.0.1:8000/route" \
  '{"input_type":"unknown"}' \
  '{"input_type":"unknown","selected_service":null,"status":"unsupported","classification":null}'

# Print the final success line after all checks pass.
echo "All Docker Compose smoke tests passed."
