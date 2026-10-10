#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ID="${PROJECT_ID:-cosmosai-demo}"
REGION="${REGION:-europe-west1}"
REPOSITORY="${REPOSITORY:-cosmosai}"
SERVICE="${SERVICE:-cosmosai-demo}"
TAG="${TAG:-$(git -C "$ROOT" rev-parse --short HEAD)}"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}"
RENDERED="${TMPDIR:-/tmp}/cosmosai-cloud-run-${TAG}.yaml"

command -v gcloud >/dev/null || {
  echo "gcloud is required. Run this script from Google Cloud Shell." >&2
  exit 1
}
command -v envsubst >/dev/null || {
  echo "envsubst is required to render the Cloud Run service template." >&2
  exit 1
}

billing_enabled="$(gcloud billing projects describe "$PROJECT_ID" \
  --format='value(billingEnabled)')"
if [[ "$billing_enabled" != "True" && "$billing_enabled" != "true" ]]; then
  echo "Billing is not enabled for ${PROJECT_ID}. Refusing to create resources." >&2
  exit 1
fi

if [[ "${COSMOSAI_BUDGET_CONFIRMED:-}" != "YES" ]]; then
  echo "Before deployment, create a USD 5 budget alert for ${PROJECT_ID}."
  read -r -p "Type BUDGET-READY after verifying it in Billing: " confirmation
  [[ "$confirmation" == "BUDGET-READY" ]] || {
    echo "Deployment cancelled before creating resources." >&2
    exit 1
  }
fi

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com

if ! gcloud artifacts repositories describe "$REPOSITORY" \
  --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format docker \
    --location "$REGION" \
    --description "CosmosAI release containers"
fi

gcloud builds submit "$ROOT" \
  --config "$ROOT/deploy/cloud-run/cloudbuild.yaml" \
  --substitutions "_REGION=${REGION},_REPOSITORY=${REPOSITORY},_TAG=${TAG}"

export REGION REGISTRY SERVICE TAG
envsubst '${REGION} ${REGISTRY} ${SERVICE} ${TAG}' \
  < "$ROOT/deploy/cloud-run/service.template.yaml" > "$RENDERED"

# A new Cloud Run service is private until its Invoker role is granted to allUsers.
gcloud run services replace "$RENDERED" \
  --project "$PROJECT_ID" \
  --region "$REGION"
gcloud run services update "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --min 0 \
  --max 1

url="$(gcloud run services describe "$SERVICE" --project "$PROJECT_ID" \
  --region "$REGION" --format='value(status.url)')"
cat <<EOF

Private deployment ready: ${url}

Verify it before making it public:
  TOKEN=\$(gcloud auth print-identity-token)
  python3 scripts/check_release_demo.py --base-url '${url}' --identity-token "\$TOKEN"

After verification, publish deliberately:
  deploy/cloud-run/make-public.sh
EOF
