#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-cosmosai-demo}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-cosmosai-demo}"

gcloud run services remove-iam-policy-binding "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member allUsers \
  --role roles/run.invoker
