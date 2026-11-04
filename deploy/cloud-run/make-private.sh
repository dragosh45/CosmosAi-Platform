#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-cosmosai-demo}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-cosmosai-demo}"

gcloud run services update "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --invoker-iam-check

# Older releases may also have made the service public by granting the Invoker
# role to allUsers. Remove that legacy binding when it is present.
if gcloud run services get-iam-policy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --flatten='bindings[].members' \
  --filter='bindings.role:roles/run.invoker AND bindings.members:allUsers' \
  --format='value(bindings.members)' | grep -qx 'allUsers'; then
  gcloud run services remove-iam-policy-binding "$SERVICE" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --member allUsers \
    --role roles/run.invoker
fi

gcloud run services describe "$SERVICE" --project "$PROJECT_ID" \
  --region "$REGION" --format='value(status.url)'
