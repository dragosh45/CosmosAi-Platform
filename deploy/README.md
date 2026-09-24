# Portable Demo Rehearsal

This package prepares M55 without choosing a hosting account, enabling billing,
publishing images or creating cloud resources. Public HTTPS hosting remains a
separate approved step. The model is still the experimental 25%-accuracy pilot.

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

## Before Public Hosting

1. Choose account/provider, region and budget. Approve any billable resources.
2. Review the files/data licenses for publication; publish only the intended repository and build context.
3. Push versioned images to the selected private registry. Record image digests, checkpoint hash and dependency versions.
4. Map the host's ingress port and private service URLs; add HTTPS, low instance/concurrency limits, alerts and per-client rate limiting or reviewer access control.
5. Run the same smoke/browser checks through the public URL, including cold start, restart and a separate device. Test public-host-specific upload and timeout limits.
6. Record the approved URL, deployment/rollback commands and provider-specific deletion steps.

Google Cloud Run remains a proposed target, not a selected account or completed
deployment. Its platform configuration, cost estimate and authentication depend
on the hosting decision. Do not expose port 8090 on all interfaces as a substitute
for that work. Local memory/latency observations are not cloud measurements.
