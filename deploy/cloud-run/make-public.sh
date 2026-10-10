#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-cosmosai-demo}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-cosmosai-demo}"

read -r -p "Type PUBLIC to allow anyone to invoke ${SERVICE}: " confirmation
[[ "$confirmation" == "PUBLIC" ]] || {
  echo "No IAM policy was changed."
  exit 1
}

gcloud run services add-iam-policy-binding "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member allUsers \
  --role roles/run.invoker

gcloud run services describe "$SERVICE" --project "$PROJECT_ID" \
  --region "$REGION" --format='value(status.url)'
