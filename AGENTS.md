# CosmosAI Agent Guide

This repository is a learning-focused astronomy AI platform. Preserve the
distinction between implemented evidence, experimental results and future plans.

## Start Here

Read these files before making a substantial change:

1. `docs/current_status.md`
2. `docs/README.md`
3. `docs/workflow.md`
4. `docs/build_log.md` (current block first; older entries are history)
5. The active milestone document linked from current status

The public overview is `README.md`. Architecture is in `docs/architecture.md`.
Cloud release mechanics and the pending-release queue are in `deploy/README.md`.

## Current Position

- M55 public Cloud Run preview is complete.
- The live revision is still image tag `f7f9de5`.
- The current model is an experimental 80-image pilot: 25% validation/test,
  predicting elliptical for every evaluated image.
- M54 model/data quality is the active implementation milestone.
- A 68-topic Concepts build is committed but intentionally waits for the next
  substantial application/model release; do not deploy it alone.

## Runtime Shape

One Cloud Run multi-container service runs:

```text
browser -> gateway:8080 -> router:8001 -> classifier:8002
```

Only the gateway is public. Router and classifier are localhost sidecars. The
service scales to zero, is limited to one instance and has a USD 5 budget alert.
The alert is not a hard cap.

## Working Rules

- Prefer existing project patterns and keep each milestone bounded.
- Do not claim useful scientific accuracy from the current checkpoint.
- Keep test isolated while selecting model candidates; use validation for choices.
- Update current status, build log and TODO when a milestone changes.
- Keep historical build-log entries truthful; add a new current correction rather
  than rewriting history.
- Do not serve `docs/` directly. The public Concepts/Diagrams pages are generated
  into `apps/learning-content/site/` from an explicit allowlist.
- Batch documentation-only web changes into the next meaningful cloud release.
- Never commit credentials, `.env` files, full datasets, caches or arbitrary run
  directories. Only the exact small model artifacts allowlisted in `.gitignore`
  belong in normal Git.
- Do not modify or commit the unrelated `assignment-property-dashboard/` folder.

## Common Checks

```bash
PYTHONPATH=. .venv/bin/pytest -q
git diff --check
```

Focused deployment/content checks:

```bash
PYTHONPATH=. .venv/bin/pytest -q \
  tests/test_cloud_run_deployment.py \
  tests/test_learning_content.py \
  tests/test_service_contracts.py
```

Local demo:

```bash
.venv/bin/python scripts/run_local_demo.py \
  --checkpoint data/demo/gz2_pilot_80.pt
```

Before the next Cloud Run deployment, follow `deploy/README.md`, inspect commits
since `f7f9de5`, run the release smoke check and verify the queued Concepts topic.
