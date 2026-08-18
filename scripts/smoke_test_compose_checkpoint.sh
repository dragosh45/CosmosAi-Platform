#!/usr/bin/env bash

# Stop the script on errors, unset variables, or failed pipeline commands.
set -euo pipefail

# Store the repository root so this script works from any current directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Move to the repository root because Compose files and data paths live there.
cd "$REPO_ROOT"

# Store the tiny local checkpoint path used by the optional Compose override.
CHECKPOINT_PATH="models/tiny_galaxy_cnn_baseline.pt"

# Create the tiny checkpoint if it is missing.
# This keeps the optional Docker demo reproducible on a fresh local checkout.
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  .venv/bin/python scripts/train_galaxy_cnn_baseline.py \
    data/samples/galaxy_manifest_sample.csv \
    --galaxy-data-root data/samples/images \
    --epochs 2 \
    --checkpoint-path "$CHECKPOINT_PATH"
fi

# Store the Compose command with the checkpoint override.
COMPOSE_CMD=(docker compose -f docker-compose.yml -f docker-compose.checkpoint.yml)

# Start all local services and rebuild images so the test uses current code.
"${COMPOSE_CMD[@]}" up --build -d

# Always stop the Compose services when this script exits.
trap '"${COMPOSE_CMD[@]}" down' EXIT

# Check that an HTTP endpoint eventually returns the expected JSON response.
assert_json() {
  # Store a short label for readable output.
  local name="$1"

  # Store the URL that should be called.
  local url="$2"

  # Store the exact JSON response expected from the service.
  local expected="$3"

  # Retry because containers can need a few seconds to become ready.
  for attempt in {1..30}; do
    # Capture the HTTP response body, or keep retrying if the service is not ready.
    actual="$(curl -fsS "$url" 2>/dev/null || true)"

    # Pass when the exact response matches.
    if [[ "$actual" == "$expected" ]]; then
      echo "PASS: $name"
      return 0
    fi

    # Wait briefly before trying again.
    sleep 1
  done

  # Print the mismatch to make failures easy to inspect.
  echo "FAIL: $name"
  echo "Expected: $expected"
  echo "Actual:   $actual"
  return 1
}

# Send a POST request and verify important response fragments are present.
assert_post_contains() {
  # Store a short label for readable output.
  local name="$1"

  # Store the URL that should receive the POST request.
  local url="$2"

  # Store the JSON request body.
  local payload="$3"

  # Shift away the named arguments so the rest are required substrings.
  shift 3

  # Send the JSON request to the running service.
  actual="$(curl -fsS -X POST "$url" -H 'Content-Type: application/json' -d "$payload")"

  # Verify every expected fragment exists in the response body.
  for expected_fragment in "$@"; do
    if [[ "$actual" != *"$expected_fragment"* ]]; then
      echo "FAIL: $name"
      echo "Missing fragment: $expected_fragment"
      echo "Actual:           $actual"
      return 1
    fi
  done

  # Print the actual response so the checkpoint confidence can be observed.
  echo "PASS: $name"
  echo "$actual"
}

# Verify every service responds on its host-mapped health endpoint.
assert_json "api-gateway health" "http://127.0.0.1:8000/health" '{"status":"ok","service":"api-gateway"}'
assert_json "inference-router health" "http://127.0.0.1:8001/health" '{"status":"ok","service":"inference-router"}'
assert_json "galaxy-classifier-service health" "http://127.0.0.1:8002/health" '{"status":"ok","service":"galaxy-classifier-service"}'
assert_json "stellar-classifier-service health" "http://127.0.0.1:8003/health" '{"status":"ok","service":"stellar-classifier-service"}'

# Verify the optional checkpoint path through the full service chain.
assert_post_contains \
  "api-gateway galaxy checkpoint route" \
  "http://127.0.0.1:8000/route" \
  '{"input_type":"galaxy_image","image_id":"gz2-000001"}' \
  '"selected_service":"galaxy-classifier-service"' \
  '"image_id":"gz2-000001"' \
  '"label":"spiral"' \
  '"status":"checkpoint_inference"'

# Verify stellar routing still stays on its stub path.
assert_post_contains \
  "api-gateway stellar stub route" \
  "http://127.0.0.1:8000/route" \
  '{"input_type":"stellar_spectrum","spectrum_id":"demo-star-001"}' \
  '"selected_service":"stellar-classifier-service"' \
  '"spectral_type":"G"' \
  '"status":"stub"'

# Print the final success line after all checks pass.
echo "All optional checkpoint Docker Compose smoke tests passed."
