# Portable Demo and Cloud Run Release

The local bundle and the Google Cloud Run deployment package use the same three
service images. The model is still the experimental 25%-accuracy pilot, so the
public UI must continue to present its evaluation warning.

The public release was verified on 10 October 2026 at
**https://cosmosai-demo-bbrzmrrfeq-ew.a.run.app**. The authenticated and public
smoke checks both passed against checkpoint tag `f7f9de5`.

## Start Locally

From the repository root, with Docker Engine and Compose available:

```bash
docker compose -p cosmosai-release -f docker-compose.release.yml build
docker compose -p cosmosai-release -f docker-compose.release.yml up -d --wait --wait-timeout 180
.venv/bin/python scripts/check_release_demo.py
```

Open **http://127.0.0.1:8090**. This does not conflict with the Python demo on 8080.
To use another port, set `COSMOSAI_RELEASE_PORT=8095` for Compose and pass
`--base-url http://127.0.0.1:8095` to the checker. Dependencies for the checker
are in the ordinary project requirements; it needs `requests`, not PyTorch.

The three images contain the gateway UI, learning content, replay, four previews,
trusted checkpoint and checksum-matched model card. No bind mounts, Kingston
drive, training dataset, authoring notes or `.env` file are needed at runtime.
Generated `apps/learning-content/site/` must accompany the code. To rebuild it,
see [the learning build](../apps/learning-content/README.md).

The first build downloads CPU PyTorch and can take several minutes. No training
runs. For a constrained machine, build each service separately:

```bash
docker compose -p cosmosai-release -f docker-compose.release.yml build galaxy-classifier-service
docker compose -p cosmosai-release -f docker-compose.release.yml build inference-router
docker compose -p cosmosai-release -f docker-compose.release.yml build api-gateway
```

## Boundaries

- Only the gateway is published, bound to localhost; router/classifier have no host ports.
- Non-root UID 10001, read-only root filesystems, dropped capabilities and bounded temporary storage.
- Classifier limit: 1 GiB, two CPUs, two PyTorch threads; gateway/router: 256 MiB and 0.5 CPU each.
- Health-checked startup; the classifier checks that its model can load.
- Upload limits remain 5 MiB and 1,048,576 pixels. Temporary uploads are removed after prediction.
- Concurrency limits and one active classifier inference bound local work. These are not per-client abuse prevention or a spending cap.

Runtime limits and startup dependencies use the official
[Compose service options](https://docs.docker.com/reference/compose-file/services/)
and [health-based startup](https://docs.docker.com/compose/how-tos/startup-order/).

## Inspect, Restart, Stop

```bash
docker compose -p cosmosai-release -f docker-compose.release.yml ps
docker compose -p cosmosai-release -f docker-compose.release.yml logs --tail 50
docker compose -p cosmosai-release -f docker-compose.release.yml restart
.venv/bin/python scripts/check_release_demo.py
docker compose -p cosmosai-release -f docker-compose.release.yml down
```

Stopping removes this project's containers/network, not the checkpoint, source,
Docker images or unrelated services. Wait for healthy services after a restart
before running the check. Do not restart the PC to update this application.

## Deploy Privately to Cloud Run

The checked-in deployment targets project `cosmosai-demo`, region
`europe-west1`, and an Artifact Registry repository named `cosmosai`. Override
`PROJECT_ID`, `REGION`, `REPOSITORY`, `SERVICE`, or `TAG` when needed.

Create a USD 5 billing budget alert first. Then clone this repository in Google
Cloud Shell and run:

```bash
deploy/cloud-run/deploy.sh
```

The script verifies billing, asks for explicit budget confirmation, enables only
Artifact Registry, Cloud Build, and Cloud Run, builds versioned images remotely,
and deploys one private multi-container service. It sets minimum instances to 0,
maximum instances to 1, request-based CPU throttling, concurrency 1, and startup
ordering from classifier to router to gateway.

Use the authenticated smoke command printed by the script. Only after it passes:

```bash
deploy/cloud-run/make-public.sh
```

Return to private access at any time with:

```bash
deploy/cloud-run/make-private.sh
```

The public action is intentionally separate and asks for confirmation. A billing
budget is an alert, not a hard spending cap. The first request after scale-to-zero
will be slower because it starts three containers and loads PyTorch.

## Public Hosting Checklist

1. [x] Confirm the USD 5 budget alert and email notifications.
2. [x] Preserve the experimental-model warning in the public UI.
3. [x] Run the authenticated release checker against the private URL.
4. [x] Make the service public and repeat the upload/error smoke check without a token.
5. [x] Add the verified HTTPS URL to the README.
6. [ ] Inspect ongoing Cloud Run cost/logs and test a cold start from another device.

Cloud Run maps the gateway to port 8080. The router and classifier listen only
inside the same instance on ports 8001 and 8002 via localhost. The service can be
deleted with `gcloud run services delete cosmosai-demo --region europe-west1`;
Artifact Registry images are retained separately until explicitly removed.
