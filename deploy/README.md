# Portable Demo and Cloud Run Release

The local bundle and the Google Cloud Run deployment package use the same three
service images. The model is still the experimental 25%-accuracy pilot, so the
public UI must continue to present its evaluation warning.

The public release was verified on 10 October 2026 at
**https://cosmosai-demo-bbrzmrrfeq-ew.a.run.app**. The authenticated and public
smoke checks both passed against checkpoint tag `f7f9de5`.

## How The Cloud Deployment Works

```text
GitHub commit
  -> manually cloned/pulled in Google Cloud Shell
  -> deploy.sh submits that checkout to Cloud Build
  -> Cloud Build creates three versioned Docker images
  -> Artifact Registry stores the images
  -> Cloud Run pulls them into one managed multi-container service
```

GitHub is the source history, but it is not currently connected to an automatic
deployment trigger. Cloud Shell starts the release; Cloud Build performs the
remote Docker builds; Artifact Registry is durable image storage; Cloud Run is
the managed runtime. No dedicated VM is created or administered by this project.
Cloud Run starts an ephemeral instance when needed and can remove it while idle.

At runtime:

```text
browser -> HTTPS -> api-gateway:8080
                         |
                         +-> inference-router:8001
                                  |
                                  +-> galaxy-classifier:8002 -> saved checkpoint
```

All three containers share one Cloud Run instance and communicate on localhost.
Only the gateway declares the ingress port. The classifier image contains the
small checkpoint and model card; the full training dataset and external dataset drive
are not deployed. Upload requests perform inference only and do not train or
persist a new model.

## Configuration Concepts

| Setting | Current value | Why it exists |
| --- | --- | --- |
| Service | One multi-container Cloud Run service | Preserves gateway/router/classifier boundaries while allowing private localhost calls. |
| Ports | Gateway `8080`, router `8001`, classifier `8002` | Cloud Run sends public traffic only to the ingress container; sidecars need distinct local ports. |
| Startup order | Classifier -> router -> gateway | Prevents the public gateway becoming ready before its dependencies can answer. |
| Startup probes | `/model` for classifier; `/health` for router/gateway | Confirms model loading and service readiness before traffic is routed. Probe timeout must be shorter than its period. |
| Scaling | Minimum `0`, maximum `1` | Scales to zero while idle and bounds demo cost/exposure. The first request can be slower. |
| Concurrency | `1` request per instance | Bounds simultaneous work for the small CPU model; it is not per-client rate limiting. |
| CPU | Request-based throttling with startup boost | CPU is primarily allocated while requests run; startup boost helps load PyTorch. |
| IAM | Private first, then `allUsers` gets `roles/run.invoker` | Separates deployment verification from the deliberate decision to publish. |
| Budget | USD 5 monthly alert | Sends warnings; it is not a hard spending cap. |
| Image tag | Git commit SHA | Connects a deployed image to a reviewable source version. |

The service template is [`service.template.yaml`](cloud-run/service.template.yaml).
The build recipe is [`cloudbuild.yaml`](cloud-run/cloudbuild.yaml).

## What The Deployment Script Does

[`deploy.sh`](cloud-run/deploy.sh) performs these operations in order:

1. Read project/region/repository/service settings and choose the Git SHA as the image tag.
2. Verify `gcloud`, template rendering and enabled project billing.
3. Require explicit confirmation that the budget alert exists.
4. Enable only Artifact Registry, Cloud Build and Cloud Run APIs.
5. Create the private Docker repository if it does not exist.
6. Submit the checkout to Cloud Build and push three tagged images.
7. Render the Cloud Run YAML with concrete service, region, registry and tag values.
8. Replace the Cloud Run service privately and reassert min/max scaling limits.
9. Print the private URL and authenticated smoke-test command.

`SKIP_BUILD=YES` is a recovery option for reusing an existing image tag when
only the service configuration failed. It must be paired with the known existing
`TAG`; otherwise a template could reference images that were never published.
Public/private IAM remains in separate, explicit scripts:
[`make-public.sh`](cloud-run/make-public.sh) and
[`make-private.sh`](cloud-run/make-private.sh).

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
trusted checkpoint and checksum-matched model card. No bind mounts, external-drive
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
