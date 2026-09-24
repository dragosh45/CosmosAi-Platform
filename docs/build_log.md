
# Build Log

This file tracks project progress and next steps.

## Current status

Updated 24 September 2026: **Milestone 51 is completed locally; Phase 1A / R1 galaxy foundation.** M50-L1 measured trace and M50-L2 browser replay are complete local learning work. M51 moved model-ready image loading into `Dataset.__getitem__()` while preserving the M50 quality policy. Current capabilities, verification and limitations live in [current_status.md](current_status.md). No usable real-data model or public release is claimed yet.

**Active next implementation:** [bounded real-data subset](#active-next-milestone-bounded-real-data-subset). Future learning-web pages and hosting are recorded in the [24 September planning entry](#planning-update-learning-website-and-concept-navigation), after the historical entries below.

## Historical Baseline Through Milestone 26

The following early snapshot is preserved as history, not the current project state. Later milestone entries supersede it.

Completed:

- Ubuntu development environment prepared
- VS Code installed using deb/apt version, not Snap
- Codex CLI installed using npm/nvm
- Project created at `/opt/projects/cosmosai-platform`
- Git initialized on branch `main`
- Private project plan added to `.gitignore`
- Minimal Python virtual environment prepared
- Basic project documentation planned
- FastAPI api-gateway service created in `apps/api-gateway/`
- `GET /health` endpoint added
- Service-level `requirements.txt` added
- Service Dockerfile added
- api-gateway local Uvicorn run verified
- api-gateway Docker image built locally
- api-gateway Docker container run locally on port 8000
- Dockerized `GET /health` verified from host machine
- Minimal project-level `docker-compose.yml` added for api-gateway
- api-gateway Docker Compose run verified
- Compose `GET /health` verified from host machine
- FastAPI inference-router service created in `apps/inference-router/`
- inference-router `GET /health` endpoint added
- inference-router service-level `requirements.txt` added
- inference-router Dockerfile added
- docker-compose.yml updated to run inference-router on host port 8001
- Two-service Docker Compose run verified
- api-gateway and inference-router health endpoints verified from host machine
- inference-router Pydantic request/response contract added
- inference-router placeholder `POST /route` endpoint added
- `POST /route` verified for galaxy, stellar, and unsupported inputs
- api-gateway Pydantic request/response contract added for routing
- api-gateway `POST /route` endpoint added
- api-gateway forwards routing requests to inference-router inside Docker Compose
- End-to-end route proxy verified through api-gateway on host port 8000
- FastAPI galaxy-classifier-service created in `apps/galaxy-classifier-service/`
- galaxy-classifier-service `GET /health` endpoint added
- galaxy-classifier-service service-level `requirements.txt` added
- galaxy-classifier-service Dockerfile added
- docker-compose.yml updated to run galaxy-classifier-service on host port 8002
- Three-service Docker Compose run verified
- api-gateway, inference-router, and galaxy-classifier-service health endpoints verified from host machine
- galaxy-classifier-service Pydantic request/response contract added
- galaxy-classifier-service placeholder `POST /classify` endpoint added
- `POST /classify` verified for image_id and image_uri inputs
- inference-router now calls galaxy-classifier-service for `galaxy_image`
- api-gateway route response supports optional classifier result payload
- End-to-end api-gateway → inference-router → galaxy-classifier-service chain verified
- FastAPI stellar-classifier-service created in `apps/stellar-classifier-service/`
- stellar-classifier-service `GET /health` endpoint added
- stellar-classifier-service service-level `requirements.txt` added
- stellar-classifier-service Dockerfile added
- docker-compose.yml updated to run stellar-classifier-service on host port 8003
- Four-service Docker Compose run verified
- api-gateway, inference-router, galaxy-classifier-service, and stellar-classifier-service health endpoints verified from host machine
- stellar-classifier-service Pydantic request/response contract added
- stellar-classifier-service placeholder `POST /classify` endpoint added
- `POST /classify` verified for spectrum_id and spectrum_uri inputs
- inference-router now calls stellar-classifier-service for `stellar_spectrum`
- api-gateway route request supports optional spectrum_id and spectrum_uri
- End-to-end api-gateway → inference-router → stellar-classifier-service chain verified
- Architecture documentation updated with current service ports and stub contracts
- Minimal pytest development requirements added
- Basic service contract tests added for current stubs
- api-gateway health contract tested locally without Docker
- inference-router unsupported route contract tested locally without Docker
- galaxy-classifier-service classify stub contract tested locally without Docker
- stellar-classifier-service classify stub contract tested locally without Docker
- Local pytest run verified
- Docker Compose smoke test script added
- Real container smoke test verified for all health endpoints
- Real container smoke test verified for galaxy, stellar, and unsupported routes through api-gateway
- Galaxy classifier data contract documented
- Galaxy labels, manifest columns, external data path, and split rules defined
- No dataset download, ML training, RAG, or cloud deployment added
- Tiny sample galaxy manifest added
- Galaxy manifest validation script added
- Manifest validator checks required columns, allowed labels, allowed splits, duplicate image IDs, and empty required values
- Manifest validator pytest coverage added
- Local pytest run verified with service contract tests and manifest validator tests
- Tiny galaxy manifest loader script added
- Manifest loader reads a validated CSV into structured records
- Manifest loader groups records by train, val, and test split
- Manifest loader pytest coverage added
- Local pytest run verified with service contract tests, validator tests, and loader tests
- Galaxy image path resolution helper added
- Manifest-relative image paths can resolve against a configurable galaxy data root
- Absolute image paths pass through unchanged
- Image ID to resolved path mapping added for future training code
- Local pytest run verified with service contract tests, validator tests, loader tests, and path resolution tests
- Optional image existence helper added
- Loader CLI can report missing resolved image files with `--check-images`
- Default loader behavior still works without real image files
- Local pytest run verified with service contract tests, validator tests, loader tests, path resolution tests, and image existence tests
- Galaxy manifest tooling committed without docs
- Tiny sample PPM image added for image-loading proof
- Galaxy image loader script added
- Image loader can load one image from a manifest record
- Image loader reports width, height, channel count, and pixel value count
- Local pytest run verified with service contract tests, manifest tooling tests, and image loader tests
- Architecture documentation updated with Step 0 and Step 1 named snapshots and data-flow diagrams
- Galaxy image preprocessing script added
- Preprocessing converts loaded 0-255 RGB pixel values into normalized 0.0-1.0 values
- Preprocessing returns a simple tensor-like object with shape `(height, width, channels)`
- Preprocessing pytest coverage added for sample image, bad pixel count, and out-of-range pixels
- Local pytest run verified with service contract tests, manifest tooling tests, image loader tests, and preprocessing tests
- Galaxy training sample script added
- Training sample proof combines manifest metadata, normalized image tensor, label, and label_id
- Stable label mapping added for elliptical, spiral, lenticular, and irregular classes
- Training sample pytest coverage added for sample creation, label mapping, and unknown-label failure
- Local pytest run verified with service contract tests, manifest tooling tests, image loader tests, preprocessing tests, and training sample tests

## Important local paths

Project:

```bash
/opt/projects/cosmosai-platform
```

Private internal plan:

```bash
docs/portfolio_ai_project_plan_ultimate.md
```

External data storage:

```bash
/media/h1dr0/KINGSTON/cosmosai-data
```

## Historical Milestone 26 Snapshot

Build Milestone 26 completed:

```text
FastAPI api-gateway
GET /health endpoint
service requirements.txt
Dockerfile
local run instructions added to README.md
local Uvicorn run verified
Docker build verified
Docker run verified
Docker Compose run verified
inference-router skeleton added
two-service Compose run verified
inference-router stub route endpoint added
route contract verified through Docker Compose
api-gateway route proxy added
api-gateway to inference-router service call verified
galaxy-classifier-service skeleton added
three-service Compose run verified
galaxy-classifier-service stub classify endpoint added
galaxy classify contract verified through Docker Compose
inference-router calls galaxy-classifier-service for galaxy_image
end-to-end galaxy route/classify chain verified
stellar-classifier-service skeleton added
four-service Compose run verified
stellar-classifier-service stub classify endpoint added
stellar classify contract verified through Docker Compose
inference-router calls stellar-classifier-service for stellar_spectrum
end-to-end stellar route/classify chain verified
architecture docs updated for current contracts
basic pytest setup added
service contract tests added
local pytest run verified
Docker Compose smoke test script added
real container wiring verified through api-gateway
galaxy classifier data contract documented
sample galaxy manifest added
galaxy manifest validator added
validator tests added
galaxy manifest loader added
loader tests added
image path resolution helper added
path resolution tests added
optional image existence checks added
image existence tests added
manifest tooling committed
tiny sample image added
galaxy image loader added
image loader tests added
architecture docs updated for Step 1
galaxy image preprocessing script added
normalized tensor-like output added
preprocessing tests added
galaxy training sample script added
label to label_id mapping added
training sample tests added
```

Tested training sample command:

```bash
cd /opt/projects/cosmosai-platform
scripts/create_galaxy_training_sample.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001
```

Result:

```text
image_id: gz2-000001
label: spiral
label_id: 1
split: train
shape: (3, 3, 3)
normalized_values: 27
first_values: [0.0, 0.0, 0.0, 0.251, 0.251, 0.251]
```

Tested preprocessing command:

```bash
cd /opt/projects/cosmosai-platform
scripts/preprocess_galaxy_image.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001
```

Result:

```text
Preprocessed image: data/samples/images/processed/images_224/gz2-000001.ppm
shape: (3, 3, 3)
normalized_values: 27
min_value: 0.0000
max_value: 1.0000
```

Tested local pytest run:

```bash
.venv/bin/python -m pytest
```

Result:

```text
23 passed
```

Tested local Uvicorn run:

```bash
cd /opt/projects/cosmosai-platform/apps/api-gateway
../../.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health
```

Result:

```json
{"status":"ok","service":"api-gateway"}
```

Tested Docker run:

```bash
cd /opt/projects/cosmosai-platform/apps/api-gateway
docker build -t cosmosai-api-gateway .
docker run --rm -p 8000:8000 cosmosai-api-gateway
curl http://127.0.0.1:8000/health
```

Result:

```json
{"status":"ok","service":"api-gateway"}
```

Tested Docker Compose run:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl http://127.0.0.1:8000/health
docker compose down
```

Result:

```json
{"status":"ok","service":"api-gateway"}
```

Tested two-service Docker Compose run:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8001/health
docker compose down
```

Results:

```json
{"status":"ok","service":"api-gateway"}
{"status":"ok","service":"inference-router"}
```

Tested inference-router route endpoint:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s -X POST http://127.0.0.1:8001/route -H 'Content-Type: application/json' -d '{"input_type":"galaxy_image"}'
curl -s -X POST http://127.0.0.1:8001/route -H 'Content-Type: application/json' -d '{"input_type":"stellar_spectrum"}'
curl -s -X POST http://127.0.0.1:8001/route -H 'Content-Type: application/json' -d '{"input_type":"unknown"}'
docker compose down
```

Results:

```json
{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub"}
{"input_type":"stellar_spectrum","selected_service":"stellar-classifier-service","status":"stub"}
{"input_type":"unknown","selected_service":null,"status":"unsupported"}
```

Tested three-service Docker Compose run:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8002/health
docker compose down
```

Results:

```json
{"status":"ok","service":"api-gateway"}
{"status":"ok","service":"inference-router"}
{"status":"ok","service":"galaxy-classifier-service"}
```

Tested galaxy-classifier-service classify endpoint:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s -X POST http://127.0.0.1:8002/classify -H 'Content-Type: application/json' -d '{"image_id":"demo-galaxy-001"}'
curl -s -X POST http://127.0.0.1:8002/classify -H 'Content-Type: application/json' -d '{"image_uri":"file:///tmp/demo-galaxy.jpg"}'
docker compose down
```

Results:

```json
{"image_id":"demo-galaxy-001","image_uri":null,"label":"spiral","confidence":0.0,"status":"stub"}
{"image_id":null,"image_uri":"file:///tmp/demo-galaxy.jpg","label":"spiral","confidence":0.0,"status":"stub"}
```

Tested end-to-end galaxy routing chain:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"galaxy_image","image_id":"demo-galaxy-001"}'
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"galaxy_image","image_uri":"file:///tmp/demo-galaxy.jpg"}'
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"stellar_spectrum"}'
docker compose down
```

Results:

```json
{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub","classification":{"image_id":"demo-galaxy-001","image_uri":null,"label":"spiral","confidence":0.0,"status":"stub"}}
{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub","classification":{"image_id":null,"image_uri":"file:///tmp/demo-galaxy.jpg","label":"spiral","confidence":0.0,"status":"stub"}}
{"input_type":"stellar_spectrum","selected_service":"stellar-classifier-service","status":"stub","classification":null}
```

Tested four-service Docker Compose run:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8003/health
docker compose down
```

Results:

```json
{"status":"ok","service":"api-gateway"}
{"status":"ok","service":"inference-router"}
{"status":"ok","service":"galaxy-classifier-service"}
{"status":"ok","service":"stellar-classifier-service"}
```

Tested stellar-classifier-service classify endpoint:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s -X POST http://127.0.0.1:8003/classify -H 'Content-Type: application/json' -d '{"spectrum_id":"demo-star-001"}'
curl -s -X POST http://127.0.0.1:8003/classify -H 'Content-Type: application/json' -d '{"spectrum_uri":"file:///tmp/demo-spectrum.csv"}'
docker compose down
```

Results:

```json
{"spectrum_id":"demo-star-001","spectrum_uri":null,"spectral_type":"G","confidence":0.0,"status":"stub"}
{"spectrum_id":null,"spectrum_uri":"file:///tmp/demo-spectrum.csv","spectral_type":"G","confidence":0.0,"status":"stub"}
```

Tested end-to-end stellar routing chain:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"stellar_spectrum","spectrum_id":"demo-star-001"}'
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"stellar_spectrum","spectrum_uri":"file:///tmp/demo-spectrum.csv"}'
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"galaxy_image","image_id":"demo-galaxy-001"}'
docker compose down
```

Results:

```json
{"input_type":"stellar_spectrum","selected_service":"stellar-classifier-service","status":"stub","classification":{"spectrum_id":"demo-star-001","spectrum_uri":null,"spectral_type":"G","confidence":0.0,"status":"stub"}}
{"input_type":"stellar_spectrum","selected_service":"stellar-classifier-service","status":"stub","classification":{"spectrum_id":null,"spectrum_uri":"file:///tmp/demo-spectrum.csv","spectral_type":"G","confidence":0.0,"status":"stub"}}
{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub","classification":{"image_id":"demo-galaxy-001","image_uri":null,"label":"spiral","confidence":0.0,"status":"stub"}}
```

Updated architecture documentation:

```text
docs/architecture.md
```

Tested api-gateway route proxy:

```bash
cd /opt/projects/cosmosai-platform
docker compose up --build
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"galaxy_image"}'
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"stellar_spectrum"}'
curl -s -X POST http://127.0.0.1:8000/route -H 'Content-Type: application/json' -d '{"input_type":"unknown"}'
docker compose down
```

Results:

```json
{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub"}
{"input_type":"stellar_spectrum","selected_service":"stellar-classifier-service","status":"stub"}
{"input_type":"unknown","selected_service":null,"status":"unsupported"}
```

Tested local pytest service contracts:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Results:

```text
tests/test_service_contracts.py .... [100%]
4 passed in 0.34s
```

Tested Docker Compose smoke test:

```bash
cd /opt/projects/cosmosai-platform
scripts/smoke_test_compose.sh
```

Results:

```text
PASS: api-gateway health
PASS: inference-router health
PASS: galaxy-classifier-service health
PASS: stellar-classifier-service health
PASS: api-gateway galaxy route
PASS: api-gateway stellar route
PASS: api-gateway unsupported route
All Docker Compose smoke tests passed.
```

Documented galaxy data contract:

```text
docs/galaxy_data_contract.md
```

Defined:

```text
external data root: /media/h1dr0/KINGSTON/cosmosai-data/galaxy/
allowed labels: elliptical, spiral, lenticular, irregular
manifest file: manifests/galaxy_manifest.csv
required columns: image_id, image_path, label, split, source
splits: train, val, test
```

No dataset was downloaded and no model code was added.

Tested galaxy manifest validator:

```bash
cd /opt/projects/cosmosai-platform
scripts/validate_galaxy_manifest.py data/samples/galaxy_manifest_sample.csv
```

Result:

```text
Galaxy manifest is valid: data/samples/galaxy_manifest_sample.csv
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
7 passed in 0.32s
```

Tested galaxy manifest loader:

```bash
cd /opt/projects/cosmosai-platform
scripts/load_galaxy_manifest.py data/samples/galaxy_manifest_sample.csv
```

Result:

```text
Loaded 4 galaxy manifest records from data/samples/galaxy_manifest_sample.csv
train: 2
val: 1
test: 1
```

Tested full pytest suite after adding loader:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
10 passed in 0.35s
```

Tested galaxy image path resolution:

```bash
cd /opt/projects/cosmosai-platform
scripts/load_galaxy_manifest.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root /tmp/cosmosai-data/galaxy
```

Result:

```text
Loaded 4 galaxy manifest records from data/samples/galaxy_manifest_sample.csv
train: 2
val: 1
test: 1
first resolved image path: /tmp/cosmosai-data/galaxy/processed/images_224/gz2-000001.jpg
```

Tested full pytest suite after adding image path resolution:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
13 passed in 0.44s
```

Tested optional image existence checks:

```bash
cd /opt/projects/cosmosai-platform
scripts/load_galaxy_manifest.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root /tmp/cosmosai-data/galaxy --check-images
```

Result:

```text
Loaded 4 galaxy manifest records from data/samples/galaxy_manifest_sample.csv
train: 2
val: 1
test: 1
first resolved image path: /tmp/cosmosai-data/galaxy/processed/images_224/gz2-000001.jpg
missing image files: 4
```

Tested full pytest suite after adding optional image existence checks:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
15 passed in 0.39s
```

Committed galaxy manifest tooling:

```bash
cd /opt/projects/cosmosai-platform
git commit -m "Add galaxy manifest tooling"
```

Result:

```text
c6e3802 Add galaxy manifest tooling
```

Tested tiny galaxy image loader:

```bash
cd /opt/projects/cosmosai-platform
scripts/load_galaxy_image.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001
```

Result:

```text
Loaded image: data/samples/images/processed/images_224/gz2-000001.ppm
width: 3
height: 3
channels: 3
pixel_values: 27
```

Tested full pytest suite after adding image loader:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
17 passed in 0.37s
```

## Completed milestone 24

Added tiny image preprocessing for future CNN input:

```text
scripts/preprocess_galaxy_image.py
tests/test_galaxy_image_preprocessing.py
```

What this proved:

```text
load one tiny image from the manifest
normalize RGB pixel values from 0-255 to 0.0-1.0
return a tensor-like object with shape (height, width, channels)
```

Tested preprocessing:

```bash
cd /opt/projects/cosmosai-platform
scripts/preprocess_galaxy_image.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001
```

Result:

```text
Preprocessed image: data/samples/images/processed/images_224/gz2-000001.ppm
shape: (3, 3, 3)
normalized_values: 27
min_value: 0.0000
max_value: 1.0000
```

Committed preprocessing:

```bash
cd /opt/projects/cosmosai-platform
git commit -m "Add galaxy image preprocessing proof"
```

Result:

```text
481d33e Add galaxy image preprocessing proof
```

## Completed milestone 26

Added one tiny galaxy training sample builder:

```text
scripts/create_galaxy_training_sample.py
tests/test_galaxy_training_sample.py
```

What this proved:

```text
manifest row + normalized image tensor + known label = one training sample
the script does not predict spiral
it reads spiral from the manifest label
the label becomes a numeric class id for future training
```

Tested training sample creation:

```bash
cd /opt/projects/cosmosai-platform
scripts/create_galaxy_training_sample.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001
```

Result:

```text
image_id: gz2-000001
label: spiral
label_id: 1
split: train
shape: (3, 3, 3)
normalized_values: 27
first_values: [0.0, 0.0, 0.0, 0.251, 0.251, 0.251]
```

## Completed milestone 27

Added tiny galaxy dataset split loader:

```text
scripts/load_galaxy_dataset_splits.py
tests/test_galaxy_dataset_splits.py
```

What this proved:

```text
load validated manifest rows
create model-ready training samples
group samples by train, val, and test split
skip sample manifest rows whose tiny local image file does not exist yet
support strict mode for future real datasets
```

Tested dataset split loading:

```bash
cd /opt/projects/cosmosai-platform
scripts/load_galaxy_dataset_splits.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images
```

Result:

```text
train samples: 1
val samples: 0
test samples: 0
skipped train: 1
skipped val: 1
skipped test: 1
first train sample:
  image_id: gz2-000001
  label: spiral
  label_id: 1
  shape: (3, 3, 3)
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
25 passed in 0.71s
```

## Historical Milestone 28 Checkpoint

Build Milestone 28 was completed:

```text
committed the tiny training sample and dataset split loader implementation
commit: 0ab8e55 Add galaxy training sample split loader
```

## Completed milestone 29

Added Step 2 architecture and code-flow diagrams:

```text
docs/architecture.md
docs/excalidraw/galaxy_training_sample_flow.excalidraw
```

What this documented:

```text
Step 2: Tiny Model-Ready Dataset Proof
module dependency diagram
object/class diagram
function call graph
sequence diagram
per-script mini boxes
clickable links to code files, functions, and objects
```

This is a documentation and understanding checkpoint. It does not change runtime code.

## Completed milestone 30

Added the first minimal CNN-shaped training loop skeleton:

```text
scripts/train_galaxy_cnn_baseline.py
tests/test_galaxy_cnn_baseline_skeleton.py
```

What this proved:

```text
reuse the existing dataset split loader
load tiny train/val/test sample buckets
loop over available train samples
read tensor values and label_id
produce placeholder logits and probabilities
calculate placeholder loss
print a training-style summary
do not update weights yet
```

Tested training skeleton:

```bash
cd /opt/projects/cosmosai-platform
scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images
```

Result:

```text
CNN baseline skeleton
epochs: 1
train samples: 1
val samples: 0
test samples: 0
skipped rows: 3
training steps: 1
average placeholder loss: 1.3360
first training step:
  image_id: gz2-000001
  true_label: spiral
  true_label_id: 1
  predicted_label_id: 1
  placeholder_probabilities: [0.2564, 0.2629, 0.2415, 0.2392]
status: skeleton only - no weights updated
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
28 passed in 0.40s
```

## Completed milestone 31

Committed the CNN-shaped training loop skeleton implementation:

```text
commit: fdf8fbe Add galaxy CNN training skeleton
```

## Completed milestone 32

Refactored the training skeleton to use a model-like placeholder interface:

```text
PlaceholderGalaxyModel.forward(sample) -> logits
```

What this proved:

```text
the training loop can use one model object for the run
the fake forward pass is isolated behind a model-like interface
the future PyTorch model can replace the placeholder more cleanly
the command output stayed stable
```

Tested training skeleton:

```bash
cd /opt/projects/cosmosai-platform
scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images
```

Result:

```text
CNN baseline skeleton
epochs: 1
train samples: 1
val samples: 0
test samples: 0
skipped rows: 3
training steps: 1
average placeholder loss: 1.3360
first training step:
  image_id: gz2-000001
  true_label: spiral
  true_label_id: 1
  predicted_label_id: 1
  placeholder_probabilities: [0.2564, 0.2629, 0.2415, 0.2392]
status: skeleton only - no weights updated
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
29 passed in 0.45s
```

## Documentation refresh after returning to project

Added missing Step 3 documentation for the training skeleton:

```text
docs/architecture.md
docs/excalidraw/galaxy_cnn_training_skeleton_flow.excalidraw
docs/excalidraw/galaxy_training_concepts_in_code.excalidraw
docs/run_me_observe_results.md
```

What this documented:

```text
Step 3: CNN Baseline Training Skeleton
module dependency diagram
object/class diagram
function call graph
sequence diagram
per-script mini boxes
concept flow diagram with real current numbers
run commands with observable outputs and concept meanings
```

Purpose:

```text
make train_galaxy_cnn_baseline.py easier to understand after a break
show how it depends on the existing manifest, image, sample, and split scripts
make clear what is real, what is placeholder, and why we have not trained more data yet
```

## Completed milestone 33

Added the first real PyTorch CNN forward-pass proof:

```text
requirements.txt
scripts/train_galaxy_cnn_baseline.py
tests/test_galaxy_cnn_baseline_skeleton.py
```

What this proved:

```text
PyTorch CPU is available in the local virtual environment
one GalaxyTrainingSample can become a torch.Tensor
the tensor can be reshaped from HWC to NCHW
TinyGalaxyCNN can run one real forward pass
the real PyTorch model returns one logit per galaxy label
torch.softmax can turn those logits into probabilities
no weights are updated yet
```

Tested training script:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2
```

Result:

```text
CNN baseline skeleton
epochs: 2
train samples: 1
val samples: 0
test samples: 0
skipped rows: 3
training steps: 2
average placeholder loss: 1.3360
first training step:
  image_id: gz2-000001
  true_label: spiral
  true_label_id: 1
  predicted_label_id: 1
  placeholder_probabilities: [0.2564, 0.2629, 0.2415, 0.2392]
status: skeleton only - no weights updated
PyTorch forward proof
  image_id: gz2-000001
  input_shape: (1, 3, 3, 3)
  logits: [0.1853, 0.1107, -0.3626, -0.2596]
  probabilities: [0.3177, 0.2949, 0.1837, 0.2036]
  predicted_label_id: 0
  status: real PyTorch forward pass only - no weights updated
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
31 passed in 1.88s
```

## Completed milestone 34

Added the first tiny real PyTorch training step:

```text
scripts/train_galaxy_cnn_baseline.py
tests/test_galaxy_cnn_baseline_skeleton.py
docs/architecture.md
docs/concepts_explanations.md
docs/run_me_observe_results.md
```

What this proved:

```text
TinyGalaxyCNN can compute real PyTorch CrossEntropyLoss
loss.backward() can calculate gradients
torch.optim.SGD can update model weights
the script can prove at least one classifier weight changed
the correct class probability improved on the one tiny local sample
the loss decreased on the one tiny local sample
```

Tested training script:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2
```

Result:

```text
CNN baseline skeleton
epochs: 2
train samples: 1
val samples: 0
test samples: 0
skipped rows: 3
training steps: 2
average placeholder loss: 1.3360
first training step:
  image_id: gz2-000001
  true_label: spiral
  true_label_id: 1
  predicted_label_id: 1
  placeholder_probabilities: [0.2564, 0.2629, 0.2415, 0.2392]
status: skeleton only - no weights updated
PyTorch forward proof
  image_id: gz2-000001
  input_shape: (1, 3, 3, 3)
  logits: [0.1853, 0.1107, -0.3626, -0.2596]
  probabilities: [0.3177, 0.2949, 0.1837, 0.2036]
  predicted_label_id: 0
  status: real PyTorch forward pass only - no weights updated
PyTorch training step proof
  image_id: gz2-000001
  true_label: spiral
  true_label_id: 1
  input_shape: (1, 3, 3, 3)
  loss_before: 1.2211
  loss_after: 1.1285
  probabilities_before: [0.3177, 0.2949, 0.1837, 0.2036]
  probabilities_after: [0.3006, 0.3235, 0.1766, 0.1992]
  predicted_label_id_before: 0
  predicted_label_id_after: 1
  weight_changed: True
  status: one real PyTorch optimizer step completed
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
32 passed in 2.67s
```

## Completed milestone 35

Added a tiny real PyTorch training loop:

```text
scripts/train_galaxy_cnn_baseline.py
tests/test_galaxy_cnn_baseline_skeleton.py
docs/architecture.md
docs/concepts_explanations.md
docs/run_me_observe_results.md
docs/excalidraw/galaxy_pytorch_training_loop_flow.excalidraw
```

What this proved:

```text
TinyGalaxyCNN can train through a real loop shape
one model and one optimizer persist across epochs
the loop trains only on available train samples
validation and test samples remain read-only
real PyTorch loss is tracked per epoch and overall
the correct class probability improved after repeated tiny updates
the model weight changed during the loop
```

Tested training script:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2
```

Result:

```text
PyTorch training loop proof
  epochs: 2
  train samples: 1
  val samples: 0
  test samples: 0
  skipped rows: 3
  training steps: 2
  average real loss: 1.1748
  epoch 1: steps=1, average_loss=1.2211
  epoch 2: steps=1, average_loss=1.1285
  first_sample_loss_before_loop: 1.2211
  first_sample_loss_after_loop: 1.0440
  first_sample_correct_probability_before_loop: 0.2949
  first_sample_correct_probability_after_loop: 0.3521
  final_predicted_label_id: 1
  weight_changed: True
  status: tiny real PyTorch training loop completed
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
34 passed in 3.46s
```

## Completed milestone

Build Milestone 36 completed:

```text
add a tiny validation/evaluation pass for the PyTorch model
run evaluation without gradient tracking
report validation/test sample counts even when no usable samples exist
calculate accuracy only when evaluation samples exist
keep using tiny local sample data
do not download the full dataset yet
do not save checkpoints yet
```

Tested training/evaluation command:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2
```

Result:

```text
PyTorch evaluation proof
  train: samples=1, correct=1, accuracy=1.0000, average_loss=1.0440
    first_true_label_id=1, first_predicted_label_id=1, first_correct_probability=0.3521
  val: samples=0, correct=0, accuracy=N/A, average_loss=0.0000
  test: samples=0, correct=0, accuracy=N/A, average_loss=0.0000
  status: read-only evaluation completed - no weights updated
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
35 passed in 4.30s
```

## Completed milestone

Build Milestone 37 completed:

```text
add a tiny PyTorch checkpoint save/load proof
save the trained TinyGalaxyCNN state_dict to a local ignored artifacts path
load the checkpoint into a fresh TinyGalaxyCNN
prove loaded model predictions match the saved trained model
keep using tiny local sample data
do not connect the classifier service yet
do not download the full dataset yet
```

Tested checkpoint command:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
```

Result:

```text
PyTorch checkpoint proof
  checkpoint_path: models/tiny_galaxy_cnn_baseline.pt
  image_id: gz2-000001
  saved_model_predicted_label_id: 1
  loaded_model_predicted_label_id: 1
  logits_match: True
  probabilities_match: True
  status: checkpoint saved and loaded into a fresh model
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
36 passed in 4.73s
```

## Completed milestone

Build Milestone 38 completed:

```text
add a tiny checkpoint inference command
load a saved TinyGalaxyCNN checkpoint from disk
load and preprocess one sample image through the existing data pipeline
return predicted label_id, label name, logits, and probabilities
keep this as a local script/helper first
do not connect the classifier service yet
do not download the full dataset yet
```

Tested checkpoint inference command:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python scripts/predict_galaxy_checkpoint.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
```

Result:

```text
Galaxy checkpoint inference
  checkpoint_path: models/tiny_galaxy_cnn_baseline.pt
  image_id: gz2-000001
  manifest_label_for_learning: spiral
  input_shape: (1, 3, 3, 3)
  predicted_label_id: 1
  predicted_label: spiral
  logits: [0.0834, 0.2975, -0.4331, -0.2974]
  probabilities: [0.2842, 0.3521, 0.1696, 0.1942]
  status: loaded checkpoint inference completed - no training
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
38 passed in 3.51s
```

## Completed planning milestone

Master's integration roadmap alignment completed:

```text
updated the canonical portfolio roadmap status from planning to active development
linked the new master's integration companion document from the canonical roadmap
created docs/master_project_integration.md
documented which master's topics directly integrate, which are optional, which are syllabus-dependent, and which should stay separate
updated Phase 1 so it does not pretend the PyTorch/CNN path starts from zero
updated Phase 4 to match the implemented api-gateway -> inference-router -> classifier-services topology
added the CPU -> CUDA -> profiling -> ONNX/TensorRT -> Triton learning progression
added optional Edge / Embedded AI and Multimodal Astronomy / Data Fusion tracks
updated README current status and test count
```

Release scope:

```text
documentation and roadmap alignment only
no runtime behavior changed in this planning milestone
no cloud, GPU, dataset download, RAG, Kubernetes, or edge code added
```

Master's planning note:

```text
the master's mapping is provisional until official syllabi/project briefs are available
not every master's course necessarily requires a project
when a course does require a project, compare the official requirements before deciding whether CosmosAI is the correct base
```

## Completed milestone

Build Milestone 39 completed:

```text
prepared galaxy-classifier-service for optional local checkpoint inference
kept current stub behavior when no checkpoint config is set
added checkpoint, manifest, and data-root environment configuration
added a small internal helper that reuses the checkpoint prediction path
added tests for stub fallback and optional checkpoint inference
kept Docker Compose checkpoint integration out of scope
did not download the full dataset
```

Implementation files:

```text
apps/galaxy-classifier-service/main.py
tests/test_service_contracts.py
```

Documentation updated:

```text
docs/architecture.md
docs/concepts_explanations.md
docs/run_me_observe_results.md
```

Tested direct service helper:

```bash
cd /opt/projects/cosmosai-platform
COSMOSAI_GALAXY_CHECKPOINT_PATH=models/tiny_galaxy_cnn_baseline.pt COSMOSAI_GALAXY_MANIFEST_PATH=data/samples/galaxy_manifest_sample.csv COSMOSAI_GALAXY_DATA_ROOT=data/samples/images .venv/bin/python -c "import importlib.util; from pathlib import Path; p=Path('apps/galaxy-classifier-service/main.py'); s=importlib.util.spec_from_file_location('galaxy_service_demo', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); r=m.classify(m.ClassifyRequest(image_id='gz2-000001')); print(r.model_dump())"
```

Result:

```text
{'image_id': 'gz2-000001', 'image_uri': None, 'label': 'spiral', 'confidence': 0.352059543132782, 'status': 'checkpoint_inference'}
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
40 passed in 3.11s
```

## Completed milestone

Build Milestone 40 completed:

```text
added optional Docker Compose checkpoint inference wiring
kept default docker-compose.yml stub-safe
added docker-compose.checkpoint.yml override for local checkpoint demos
added Dockerfile.checkpoint for the galaxy classifier service
added checkpoint-specific service requirements with CPU PyTorch
copied shared scripts into the optional checkpoint image
added COSMOSAI_SCRIPTS_PATH so the container can import helper scripts
mounted local models/ and data/samples/ read-only in the optional override
added an optional checkpoint Docker Compose smoke test
verified api-gateway -> inference-router -> galaxy-classifier-service with checkpoint_inference enabled
kept stellar route on stub behavior
kept service stub fallback tests
did not download the full dataset
did not add cloud, Kubernetes, Triton, or edge deployment
```

Implementation files:

```text
apps/galaxy-classifier-service/main.py
apps/galaxy-classifier-service/Dockerfile.checkpoint
apps/galaxy-classifier-service/requirements-checkpoint.txt
docker-compose.checkpoint.yml
scripts/smoke_test_compose_checkpoint.sh
```

Documentation updated:

```text
docs/architecture.md
docs/run_me_observe_results.md
```

Tested optional checkpoint Docker Compose smoke test:

```bash
cd /opt/projects/cosmosai-platform
scripts/smoke_test_compose_checkpoint.sh
```

Result:

```text
PASS: api-gateway health
PASS: inference-router health
PASS: galaxy-classifier-service health
PASS: stellar-classifier-service health
PASS: api-gateway galaxy checkpoint route
PASS: api-gateway stellar stub route
All optional checkpoint Docker Compose smoke tests passed.
```

Observed checkpoint route response:

```json
{"input_type":"galaxy_image","selected_service":"galaxy-classifier-service","status":"stub","classification":{"image_id":"gz2-000001","image_uri":null,"label":"spiral","confidence":0.352059543132782,"status":"checkpoint_inference"}}
```

## Completed milestone

Build Milestone 41 completed:

```text
added cosmosai/ shared package root
added cosmosai.galaxy modules for labels, manifest loading, image loading, preprocessing, training samples, tiny model helpers, and checkpoint inference
refactored scripts/predict_galaxy_checkpoint.py into a thin CLI wrapper around cosmosai.galaxy.checkpoint_inference
refactored galaxy-classifier-service to call cosmosai.galaxy.checkpoint_inference directly
removed the service helper that added scripts/ to sys.path
updated Dockerfile.checkpoint to copy cosmosai/ instead of scripts/
removed COSMOSAI_SCRIPTS_PATH from docker-compose.checkpoint.yml
preserved default docker-compose.yml stub-safe behavior
preserved optional Docker checkpoint override behavior
kept the current training script working for checkpoint creation
did not download the full dataset yet
did not add cloud, Kubernetes, Triton, or edge deployment yet
```

Implementation files:

```text
cosmosai/__init__.py
cosmosai/galaxy/__init__.py
cosmosai/galaxy/labels.py
cosmosai/galaxy/manifest.py
cosmosai/galaxy/image_loader.py
cosmosai/galaxy/preprocessing.py
cosmosai/galaxy/training_sample.py
cosmosai/galaxy/model.py
cosmosai/galaxy/checkpoint_inference.py
scripts/predict_galaxy_checkpoint.py
apps/galaxy-classifier-service/main.py
apps/galaxy-classifier-service/Dockerfile.checkpoint
docker-compose.checkpoint.yml
```

Documentation updated:

```text
docs/architecture.md
```

Tested focused checkpoint/service path:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest tests/test_predict_galaxy_checkpoint.py tests/test_service_contracts.py
```

Result:

```text
8 passed in 4.07s
```

Tested full pytest suite:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
```

Result:

```text
40 passed in 3.45s
```

Tested optional checkpoint Docker Compose smoke test:

```bash
cd /opt/projects/cosmosai-platform
scripts/smoke_test_compose_checkpoint.sh
```

Result:

```text
PASS: api-gateway health
PASS: inference-router health
PASS: galaxy-classifier-service health
PASS: stellar-classifier-service health
PASS: api-gateway galaxy checkpoint route
PASS: api-gateway stellar stub route
All optional checkpoint Docker Compose smoke tests passed.
```

Boundary note:

```text
checkpoint inference is now shared package code
the prediction CLI and FastAPI service use the same package helper
the large training script still has its own training-loop implementation
full deduplication of training code is a future cleanup step
```

## Build Milestone 42 completed

Goal:

```text
deduplicate the training script by importing shared model/data helpers from cosmosai.galaxy
keep current training CLI output stable
keep checkpoint creation compatible with the shared inference package
```

What changed:

```text
added cosmosai/galaxy/dataset_splits.py
training script now imports shared LABEL_TO_ID, GalaxyTrainingSample, GalaxyDatasetSplits, TinyGalaxyCNN, tensor conversion, forward-pass, and checkpoint helpers
load_galaxy_dataset_splits.py is now a command-line wrapper that re-exports shared split helpers
predict CLI, training CLI, and FastAPI checkpoint path now depend on the same package-owned model/checkpoint helpers
```

Main implementation files:

```text
cosmosai/galaxy/dataset_splits.py
cosmosai/galaxy/model.py
cosmosai/galaxy/checkpoint_inference.py
scripts/train_galaxy_cnn_baseline.py
scripts/load_galaxy_dataset_splits.py
scripts/predict_galaxy_checkpoint.py
apps/galaxy-classifier-service/main.py
```

Verification:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
.venv/bin/python scripts/predict_galaxy_checkpoint.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
scripts/smoke_test_compose_checkpoint.sh
```

Result:

```text
40 passed
training command completed with same learning output shape
checkpoint prediction returned predicted_label=spiral for gz2-000001
optional checkpoint Docker Compose smoke test passed
```

Boundary note:

```text
this changed code ownership and imports
this did not improve model accuracy
the training loop still lives in the training script
the dataset is still the tiny local sample, not real Galaxy Zoo data
```

## Build Milestone 43 completed

Goal:

```text
document the post-refactor shared package / OOP structure
show module dependencies, object models, function call graph, sequence flow, and lazy import behavior
keep architecture.md high-level and put detailed code-flow explanation in Excalidraw
```

What changed:

```text
architecture.md now has Step 12: Shared Galaxy Runtime Package
architecture.md now has Step 13: Shared Galaxy Package Review Checkpoint
concepts_explanations.md has notes for package vs script, dataclasses/OOP, TinyGalaxyCNN state, and lazy import
run_me_observe_results.md has Milestones 42-43 observe commands
new Excalidraw diagram: docs/excalidraw/shared_galaxy_package_oop_flow.excalidraw
```

Follow-up clarity update:

```text
added focused sequence diagram: docs/excalidraw/galaxy_checkpoint_api_sequence_flow.excalidraw
the sequence diagram separates "CLI training creates checkpoint" from "API loads checkpoint for prediction"
added clickable arrows/boxes for route(), classify(), predict_from_checkpoint(), load_torch_checkpoint(), and run_torch_forward_pass()
improved shared_galaxy_package_oop_flow.excalidraw module dependency section with clearer boxes around the scripts and shared package
architecture.md links to both diagrams
```

Diagram focus:

```text
scripts are command-line entrypoints
cosmosai.galaxy owns reusable data/model/checkpoint logic
dataclasses carry structured objects through the flow
TinyGalaxyCNN owns model behavior and weights
galaxy-classifier-service lazily imports checkpoint inference only in optional checkpoint mode
```

Boundary note:

```text
Milestone 43 is documentation and understanding work
no new runtime behavior was added in this milestone
```

## Build Milestone 44 completed

Goal:

```text
add a slightly larger tiny local galaxy sample set
make train, val, and test splits each have at least one loadable sample
keep the images tiny/local so tests stay fast
update tests so evaluation is no longer only train=1, val=0, test=0
run pytest and the current training/prediction commands
do not download the full real dataset yet
```

What changed:

```text
data/samples/galaxy_manifest_sample.csv now points all four rows to tiny .ppm files
added gz2-000002.ppm as a tiny validation sample
added gz2-000003.ppm as a tiny test sample
added gz2-000004.ppm as a second tiny training sample
dataset split tests now expect train=2, val=1, test=1, skipped=0
training/evaluation tests now verify non-empty validation and test splits
strict split loading now passes for the sample manifest because all sample files exist
architecture.md and run_me_observe_results.md now describe the new split coverage
```

Verification:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
.venv/bin/python scripts/validate_galaxy_manifest.py data/samples/galaxy_manifest_sample.csv
.venv/bin/python scripts/load_galaxy_dataset_splits.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --strict
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
.venv/bin/python scripts/predict_galaxy_checkpoint.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
```

Result:

```text
40 passed
manifest is valid
strict split loading: train=2, val=1, test=1, skipped=0
training loop: 2 train samples * 2 epochs = 4 optimizer steps
evaluation now reports metrics for train, val, and test
checkpoint prediction still returns predicted_label=spiral for gz2-000001
```

Boundary note:

```text
the sample data is still artificial 3x3 PPM data
this improves pipeline coverage, not model intelligence
validation/test accuracy numbers are now calculated, but not meaningful yet
no full Galaxy Zoo data was downloaded
no cloud, Kubernetes, Triton, or edge deployment was added
```

## Build Milestone 45: Tiny PyTorch batch training proof

Goal:

```text
add a tiny batch/DataLoader-shape proof for the galaxy CNN training path
stack multiple GalaxyTrainingSample objects into one PyTorch batch tensor
create a matching label tensor for the batch
update or add tests that prove batch shape is NCHW and labels are correct
optionally let the training loop process tiny batches instead of one sample at a time
keep the current tiny local dataset and fast tests
do not download the full real dataset yet
do not add GPU/cloud training yet
```

What changed:

```text
added TorchGalaxyBatch as the shared object for image and label tensors
added galaxy_samples_to_torch_batch() to stack multiple samples into NCHW shape
added create_sample_batches() to group train samples by --batch-size
added run_torch_batch_shape_proof() so the CLI prints the batch shape before training
updated run_torch_training_loop() so one optimizer step can train on a batch
added --batch-size to the training CLI
added focused tests for batch shapes, labels, and invalid batch sizes
updated architecture.md, concepts_explanations.md, and run_me_observe_results.md
added a batch-training Excalidraw diagram for the learning flow
```

Verification:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest
.venv/bin/python scripts/load_galaxy_dataset_splits.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --strict
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --batch-size 2 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
.venv/bin/python scripts/predict_galaxy_checkpoint.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
```

Result:

```text
44 passed
strict split loading: train=2, val=1, test=1, skipped=0
batch proof: requested_batch_size=2, actual_batch_size=2
image_tensor_shape: (2, 3, 3, 3)
label_tensor_shape: (2,)
training loop: 2 epochs x 1 train batch = 2 optimizer steps
checkpoint prediction still returns predicted_label=spiral for gz2-000001
```

Boundary note:

```text
this is manual batch creation, not a full PyTorch Dataset/DataLoader yet
the sample images are still artificial 3x3 PPM files
this improves training-loop shape, not model intelligence
no real dataset download happened
no GPU/cloud training happened
```

## Build Milestone 46: Pillow real-image loading proof

Goal:

```text
add a real image format loading proof for the galaxy pipeline
support a normal image format such as PNG or JPG with Pillow
resize one local sample image into model-ready RGB tensor shape
keep the current tiny PPM loader and tests as fast proof data
add tests that prove real image loading, resizing, and normalization work
update run notes, architecture, concepts, and Excalidraw if the flow changes
do not download the full Galaxy Zoo dataset yet
do not add GPU/cloud training yet
```

What changed:

```text
added Pillow to the root requirements
added Pillow to the optional checkpoint service requirements
added load_pillow_image() for PNG/JPG/JPEG files
added load_image_file() to choose PPM or Pillow by extension
kept the tiny PPM loader for hand-readable sample fixtures
threaded optional target_size through preprocessing, training samples, dataset splits, training CLI, and prediction CLI
added --image-width and --image-height CLI options where the image path is loaded
added a clear batch-shape guard explaining when target_size is needed
added tests that generate tiny PNG files at runtime
updated README tech stack, concepts, architecture, run notes, and Excalidraw
```

Verification:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pip install "Pillow>=10.0.0,<13.0.0"
.venv/bin/python -m pytest tests/test_galaxy_image_loader.py tests/test_galaxy_image_preprocessing.py tests/test_galaxy_dataset_splits.py
.venv/bin/python -m pytest
.venv/bin/python scripts/load_galaxy_dataset_splits.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --strict --image-width 3 --image-height 3
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --batch-size 2 --image-width 3 --image-height 3 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
.venv/bin/python scripts/predict_galaxy_checkpoint.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --image-id gz2-000001 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt --image-width 3 --image-height 3
```

Result:

```text
Pillow installed in .venv as 12.3.0
12 focused image/split tests passed
51 passed
PNG loader tests passed
PNG resize tests passed
PNG preprocessing tests passed
PNG dataset split test passed
existing PPM sample tests still passed
strict split loading still reports train=2, val=1, test=1, skipped=0
training with --image-width 3 --image-height 3 still works for current PPM samples
checkpoint prediction still returns predicted_label=spiral for gz2-000001
```

Boundary note:

```text
pytest generates temporary PNG files for this proof
no binary image file was committed for the sample data
the real dataset was not downloaded
image augmentation was not added
model quality did not improve yet
```

## Build Milestone 47 completed

Goal:

```text
add a tiny real PyTorch Dataset/DataLoader proof
wrap GalaxyTrainingSample objects in a Dataset-style class
return image tensors and label tensors from __getitem__
use torch.utils.data.DataLoader to create batches
compare DataLoader batch shape with the manual batch proof from Milestone 45
keep tiny local sample data and fast tests
update concepts, architecture, run notes, and Excalidraw if the code flow changes
do not download the full Galaxy Zoo dataset yet
do not add GPU/cloud training yet
```

What changed:

```text
added cosmosai/galaxy/torch_dataset.py
added GalaxyTorchDataset
added create_galaxy_dataloader()
added preview_first_dataloader_batch()
training CLI now prints a PyTorch DataLoader proof
tests now verify one Dataset item and one DataLoader batch
concept notes, architecture, run notes, and Excalidraw were updated
```

Why this was done:

```text
Milestone 45 proved manual batches
Milestone 46 proved normal image loading and resizing
real PyTorch training normally uses Dataset/DataLoader to feed batches into the model
this gives us the normal PyTorch data-feeding shape before larger real data work
```

Verification:

```bash
cd /opt/projects/cosmosai-platform
.venv/bin/python -m pytest tests/test_galaxy_torch_dataset.py
.venv/bin/python -m pytest tests/test_galaxy_cnn_baseline_skeleton.py
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --batch-size 2 --image-width 3 --image-height 3 --checkpoint-path models/tiny_galaxy_cnn_baseline.pt
```

Result:

```text
5 DataLoader tests passed
17 training skeleton tests passed
training command prints matching manual batch and DataLoader batch shapes
DataLoader batch image_tensor_shape is (2, 3, 3, 3)
DataLoader batch label_tensor_shape is (2,)
```

Boundary note:

```text
the existing training loop still uses the current manual batch helper
this milestone proves DataLoader wiring separately before replacing the loop path
no full dataset was downloaded
no data augmentation was added
no GPU/cloud training was added
```

## Build Milestone 48 Completed

Started and verified locally on 9 September 2026. Release: R1 galaxy classifier foundation; Phase 1A. Learning question: can the same training math consume Dataset/DataLoader batches instead of manually constructed ones?

Scope agreed at start:

```text
use the PyTorch DataLoader as the main batch source inside the real training loop
keep the old manual batch helper available for comparison/tests
preserve the same tiny local sample data
verify loss, evaluation, and checkpoint behavior still work
update concepts, architecture, run notes, and Excalidraw if the runtime flow changes
do not download the full Galaxy Zoo dataset yet
do not add GPU/cloud training yet
```

What changed:

- `run_torch_training_loop()` imports and uses `create_galaxy_dataloader()`; each epoch starts a fresh iteration over train-only samples. Ordered batches retain the last partial batch.
- Per-epoch and total loss reports count actual images, fixing mean-of-batch-means bias for unequal batches. This does not change SGD updates.
- Manual batch/one-step proofs remain comparisons, with explanatory comments and explicit DataLoader output.
- Five regression cases verify Dataset access every epoch, no manual batching in the loop, matching manual SGD weights, short-batch loss accounting and empty/small splits.
- Added current-status and cited AAIMR research pages; refreshed architecture, concepts, run notes, job/master alignment, workflow and roadmap. Added the code/concept-linked `galaxy_dataloader_training_loop_flow.excalidraw`.

Verification:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --batch-size 2 --image-width 3 --image-height 3 --checkpoint-path /tmp/cosmosai-m48-review/tiny_galaxy_cnn.pt
```

Observed: **61 tests passed**. Two train images, batch size 2, two epochs -> two real optimizer updates; `weight_changed=True`; checkpoint logits/probabilities match after reload. Still four artificial fixtures, no useful accuracy claim, no full dataset, no GPU/cloud or Docker rerun. DataLoader still wraps eagerly loaded images.

Documentation checks: 19 Excalidraw JSON files validated; existing diagram Obsidian URL targets/counts preserved while function links were refreshed. New diagram code targets and concept block IDs resolve; research/planning heading links resolve. The new diagram had a static geometry preview check, not an interactive test in the user's Obsidian/VS Code session.

Planning decision: delivery is R1 galaxy classifier, R2 astronomy assistant, R3 serving/performance. Stellar is later, not an R1 gate. Replace the historical ten-week/skill-count promises with evidence gates. The preferred master's proposal is the bounded two-model study in [the research report](craiova_master_course_research.md); official assignment briefs and cohort applicability remain unconfirmed.

Commit status: implementation and local checks complete; no commit/push performed for this request. Private ignored docs are not automatically published.

## Milestone 49: Checkpoint Preprocessing Contract

**Completed locally, 9 September 2026.** R1 correctness milestone, not another new model or infrastructure layer.

Started 9 September 2026. The checkpoint will own the inference input recipe; training math and the existing DataLoader loop stay unchanged. Legacy files without that recipe will be rejected with regeneration guidance, not assigned an assumed resize policy.

Acceptance criteria:

- Save the RGB/resize/normalization policy required for inference alongside model weights and labels.
- Apply the same policy in checkpoint prediction CLI and galaxy API; reject incompatible settings clearly.
- Define and test how old checkpoints without metadata are handled, without silently inventing a resize policy.
- Test CLI/API parity with a non-square PNG resized to a fixed training size. Cover invalid metadata and unchanged default stub behavior.
- Update concept/code links, run observations and the checkpoint-flow diagram where behavior changes.

Why: identical weights are insufficient if training/CLI and API preprocess the image differently. Before M49 the optional resize setting was not stored in checkpoints.

Implemented:

- `GalaxyPreprocessingPolicy` serializes the actual RGB/bilinear resize/divide-by-255 recipe with version and optional width/height. Metadata is validated before inference.
- `LoadedGalaxyCheckpoint` keeps a loaded CNN and recipe together. The model-only loader remains available but validates the same contract. Only one checkpoint read per prediction.
- Training forwards its actual resize setting into save/round-trip. Both CLIs print the recipe. Prediction flags become optional assertions, not overrides.
- Shared inference loads metadata before pixels, so the CLI and API use identical preparation without a separate API resize setting.
- Legacy metadata-free files fail with regeneration guidance; explicit no-resize is distinct from missing metadata. API contract errors become HTTP 503; unconfigured stub and URI-only stub behavior remain.
- Added `tests/test_checkpoint_preprocessing.py`; updated architecture Step 19, concepts, run observations, status and the existing planning pointers. Added `galaxy_checkpoint_preprocessing_flow.excalidraw` with the confirmed native heading links; refreshed affected code targets without changing existing concept targets.

Verification:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train_galaxy_cnn_baseline.py data/samples/galaxy_manifest_sample.csv --galaxy-data-root data/samples/images --epochs 2 --batch-size 2 --checkpoint-path /tmp/cosmosai-m49-review/native_size.pt
```

Observed **81 tests passed**. The new suite covers metadata round-trip, a real training CLI run on a patterned RGBA `13x9` PNG resized to RGB `5x3`, exact prepared pixel/logit/probability parity, matching/conflicting flags, one checkpoint read per request, invalid/legacy metadata, explicit no-resize, prepared-sample shape checking, and a fresh-interpreter stub without torch imports. Existing tests continue covering PPM, training, Dataset/DataLoader and service contracts.

A separate real localhost HTTP smoke test used the same PNG checkpoint: CLI and `POST /classify` agreed on `spiral`, confidence `0.3493151068687439`. The temporary server was stopped. PPM run: two optimizer updates, losses `1.4572 -> 1.4210`, saved/reloaded predictions match. No new packages installed, no dataset downloads, no GPU/cloud or Compose rerun. The new diagram passed a static preview/target check; native app click behavior was not driven from this session.

Changed runtime/test files: `cosmosai/galaxy/{preprocessing,model,checkpoint_inference}.py`, `scripts/{train_galaxy_cnn_baseline,predict_galaxy_checkpoint}.py`, `apps/galaxy-classifier-service/main.py`, and new `tests/test_checkpoint_preprocessing.py`. M48 and earlier uncommitted changes were preserved. No commit/push or forced publication of ignored docs.

## M49 Learning Follow-Up: Exact Weight Observations

Completed locally, 10 September 2026. The owner asked to see actual neuron connections, matrix multiplication and individual weight changes between backward and optimizer steps, following the 3Blue1Brown lessons. This is a learning follow-up, not M50 or a new model architecture.

- Added `scripts/observe_galaxy_weight_updates.py`: temporary forward/optimizer probes around the real `run_torch_training_loop()`. Captures all parameters and gradients, pooled features, logits and same-batch losses over successive updates. CPU, ordered batches, plain SGD, at most 20 updates; hooks are removed even on error.
- Added `tests/test_galaxy_weight_observation.py`: trace versus plain training produces identical final parameters/losses; classifier matrix and analytic cross-entropy gradient agree with PyTorch; exports include unchanged entries; probes are cleaned up on failure.
- The current `model.py` lacked `TorchGalaxyBatch`, `galaxy_samples_to_torch_batch` and the M49 checkpoint-policy block while callers still required them. The owner explicitly approved restoring both missing parts. Other worktree edits were preserved.
- Created `docs/observations/galaxy_weight_updates_seed7.json`, its full scalar CSV, and an explanatory README. Updated concepts, run observations section 25, current status and architecture. Added `galaxy_cnn_exact_weight_updates.excalidraw` with measured neuron connections, matrices, a convolution slice and two optimizer updates.

Verification: `.venv/bin/python -m pytest -q` -> **88 passed**, including 7 observation cases. `.venv/bin/python scripts/observe_galaxy_weight_updates.py` traced 132 trainable scalars: each update changed 124 (116 weights + 8 biases); backward itself changed none. Same-batch mean loss: `1.457183 -> 1.420960 -> 1.387335`. Full float32 observations are saved, while documentation rounds displayed values. These new PPM-fixture runs explain mechanics, not useful classification accuracy.

Review entry points: [[run_me_observe_results#25. Observe Exact CNN Weight Updates|commands and observed values]], [[concepts_explanations#Our CNN Matrices And Exact Weight Updates|matrix/backward/SGD explanation]], [[excalidraw/galaxy_cnn_exact_weight_updates.excalidraw|scrollable weight diagram]]. The drawing has a checked static preview and heading/code target checks; native Obsidian/VS Code clicks still need the owner's UI review. No dataset download, new dependencies, checkpoint overwrite, commit or push.

### Activation Concept Follow-Up

10 September 2026: added [[concepts_explanations#Sigma, Sigmoid, And ReLU|sigma/sum notation, sigmoid, ReLU, nonlinear layers and backward derivatives]], including calculations reconstructed from the saved input and pre-update parameters. Added a runnable inspection example to run observations and section 7 to the exact-weight diagram. New concept links use native heading targets; all 255 pre-existing drawing elements and their links were preserved. The displayed sigmoid map is a hypothetical comparison, not a replacement model or a new training result.

Verification: the documented command reproduces saved features/logits, the derivative check reproduces recorded gradients, and all 7 observation tests pass. Static appendix preview and unique-heading checks pass; native UI clicks still require user review. Runtime code, checkpoints, previous observed artifacts and M50 scope are unchanged.

### Pixels-To-Classes Reading Guide

10 September 2026: clarified the exact-weight diagram after the owner confused internal feature values with galaxy classes and spatial matrices with output probabilities. Added a leading guide with the actual RGB fixture, four measured post-ReLU maps, separate pooling, a labeled classifier weight table and the four outputs. Relabeled existing feature/class nodes, decoded each gradient-example number, and marked the later kernel section as a zoom back into an earlier operation. Existing concept/code link targets were preserved; sections moved to make room.

Added [[concepts_explanations#Reading The CNN Diagram From Pixels To Classes|the matching concept guide]] with kernel slices, center convolution, row-dot-feature arithmetic, target indicators and per-weight chain-rule gradients. Expanded the documented read-only inspection command to print all four maps and check the 27-product center calculation. No runtime model/trainer, checkpoint or saved observation data changed. Verified maps, pooling, logits, center convolution and the individual gradient against the saved trace; checked static layout and native heading targets. At this point in the history, M50 was still next.

### RGB Versus Feature Multiplication

10 September 2026: added `galaxy_rgb_to_features_to_logits.excalidraw` as a separate neuron-style explanation. The first panel shows all 27 normalized RGB channel inputs and their filter-0 weights for the center window, followed by ReLU/map/pooling. The second shows four pooled features feeding four class logits. One actual multiplication is highlighted in each; SGD examples show subtraction of the learning-rate-scaled gradient for both weight tensors.

Added [[concepts_explanations#RGB Is Multiplied First, Features Are Multiplied Later|the matching concepts]], including the equivalent `(4x27) @ RGB_patch` and `(4x4) @ features` views. Checked both equations against the saved trace, checked static previews and heading links, and reran the 7 observation tests. Existing diagrams were not modified. No runtime changes, checkpoint changes, commit/push or new implementation milestone.

Clarification on 10 September 2026: relabeled the highlighted feature-to-spiral calculation so `-0.065661` is explicitly one partial addend, not another input or the complete logit. The diagram now says the classifier weight was a reproducible random initial parameter and prints all four partial products plus the spiral bias before the complete `0.110709` logit. Expanded the matching concept text. JSON and layout spacing were checked; runtime code, saved observations and M50 behavior are unchanged.

## Milestone 50 Agreed Scope (Completed)

**Build Milestone 50: explicit manifest and dataset-quality errors.** Keep the current model, checkpoint contract and DataLoader learning equations unchanged.

Acceptance criteria:

- Validate malformed/short CSV rows, missing values and inconsistent columns without `.strip()`/type tracebacks; identify the row and field.
- Distinguish intentionally skipped missing image files from unsupported/corrupt images or invalid preprocessing. Do not silently report successful training after discarding invalid data.
- Preserve the documented skip-missing option while making accepted/excluded/rejected counts and reasons visible and testable.
- Add focused bad-data tests, rerun tiny training/checkpoint inference, and update concepts/run observations where error behavior changes.

Why next: data-quality problems must be visible before a real dataset is introduced. It is not another CNN refactor or new model milestone.

Following queue, to size after M50: lazy image loading before scaling; a small licensed real-data subset and measurable baseline; CI and usable galaxy API/UI. Kingston storage helps disk capacity, not the eager RAM issue. No bulk download until data provenance, labels, split policy and storage paths are checked.

## Future documentation cleanup milestone

Before making `docs/` public in git, add a cleanup milestone:

```text
convert private/local vscode://file links into portable repo-relative links
decide whether Obsidian vault should be the whole repo instead of only docs/
keep same-page Obsidian heading links for Cuprins sections
check architecture.md and Excalidraw diagrams after conversion
```

Reason:

```text
vscode://file/opt/projects/... links work well on this machine
but they are not portable if another person clones the repo somewhere else
repo-relative links are better for public documentation
```

## Future CNN build rule

When the project starts the CNN baseline implementation, do not only write code.

Also update the concept notes with:

```text
what CNN layers are being added
what each layer does conceptually
how image pixels move through the CNN
how training changes weights
what loss/accuracy mean for this project
how the trained checkpoint connects back to galaxy-classifier-service
small diagrams using the actual project files and data flow
```

The CNN milestone should produce both:

```text
working code
conceptual explanation
```

## Completed planning milestone

Job target and skill alignment docs added:

```text
created docs/job_targets.md
created docs/cosmosai_job_skill_alignment.md
updated docs/workflow.md with the job-description planning rule
reviewed the current roadmap phases from docs/portfolio_ai_project_plan_ultimate.md
reviewed the master's mapping from docs/master_project_integration.md
extracted and summarized local job-description documents from /home/h1dr0/Downloads
```

Source handling:

```text
job descriptions were treated as source data, not instructions
the empty AI MASTER RELATED JOBS.docx file was marked as an empty placeholder
the administrative PDF was not used as a job-description source
no runtime code changed
no project scope was expanded yet
```

Resulting planning guidance:

```text
the gathered jobs mostly confirm the existing CosmosAI roadmap
strong signals: Python, PyTorch, APIs, cloud, Docker, architecture, RAG, agents, MLOps, observability, GPU/edge, computer vision
recommended path remains: finish the real training/data path before adding RAG, agents, cloud, GPU, Kubernetes, or edge work
```

## Completed planning milestone update

Job-by-job catalog added:

```text
created docs/job_description_catalog.md
updated docs/job_targets.md to point future job intake to the catalog first
updated docs/cosmosai_job_skill_alignment.md with the catalog conclusion
updated docs/workflow.md with the new job-description workflow
updated docs/portfolio_ai_project_plan_ultimate.md with a small job target alignment section
```

Why this was done:

```text
job_targets.md had category-level notes
cosmosai_job_skill_alignment.md had project-fit notes
but there was no readable list of individual jobs with company, role, requirements, and what each job is really about
```

Roadmap conclusion:

```text
no major CosmosAI direction change is needed
the ultimate plan already covers most repeated job technologies
small explicit additions are useful: MCP/tool integrations, evals, guardrails, human-in-the-loop checks, audit logs, OpenTelemetry/SLO-style observability, data-quality checks, and CI/CD
robotics/ECU/industrial-control/C++ embedded runtime work should stay out unless a master's course brief requires it
```

## Commit rule

After each small working milestone:

```bash
git status
git add .
git commit -m "Clear message"
```

## Build Milestone 50 Completed

Completed locally on 10 September 2026. Release: R1 galaxy classifier foundation; Phase 1A. Learning question: can data intake report exactly what entered training and refuse corrupt data before real-dataset scaling?

What changed:

- Shared manifest validation now handles malformed quoting, duplicate/blank headers, short rows, extra values, empty fields, invalid labels/splits and duplicate IDs with row/field diagnostics.
- Loaded `GalaxyManifestRecord` objects retain their original CSV row only for diagnostics.
- Dataset split creation reports `accepted`, `skipped_missing`, and `rejected` per split and in total. Permissive mode skips only absent paths; corrupt, unsupported and preprocessing-invalid files always reject the run.
- `GalaxyDatasetQualityError` retains the partial result so counts and rejected rows remain testable.
- Old manifest loader/validator scripts became wrappers around the shared package contract.
- Added the concept explanation, architecture Step 20, observed-run section and `galaxy_dataset_quality_gate.excalidraw`.

Verification:

```text
19 focused manifest/loader/split tests passed
93 full tests passed
valid fixture report: train 2, val 1, test 1 accepted; zero skipped/rejected
unchanged training: two optimizer updates, losses 1.4572 -> 1.4210
checkpoint reload logits/probabilities match; prediction CLI returns spiral
```

Boundary: no CNN, logits, loss, SGD, DataLoader iteration, evaluation, checkpoint, API, Docker, GPU, or cloud behavior changed. No real data were downloaded. Accepted samples are still eagerly decoded into RAM.

Next: **M51 lazy image loading.** Keep manifest records/paths in the PyTorch Dataset and decode/preprocess in `__getitem__()` so dataset size is bounded by disk rather than eager decoded RAM. Preserve M50's explicit quality policy and keep real-data download as a separate reviewable step. No commit/push was performed.

## M49 Learning Follow-Up: Whole CNN Training Cycle

Added a separate [[excalidraw/galaxy_cnn_training_cycle_overview.excalidraw|zoomed-out training-cycle diagram]] and [[concepts_explanations#Whole Tiny CNN Training Cycle|matching concept]]. They connect initialization, the unchanged RGB batch, convolution, ReLU, pooling, classifier logits, loss, backward, both parameter-group updates and the next epoch using the saved seed-7 observations. Existing diagrams and concept headings were not changed. This is documentation only; M51 did not start and runtime behavior did not change.

Expanded the convolution learning follow-up with [[excalidraw/galaxy_center_convolution_calculation.excalidraw|a dedicated center-calculation diagram]]. It shows the actual normalized RGB planes, all three filter-0 kernel slices, all 27 matching products, channel subtotals, bias, the `0.037116` center response, border padding, pre/post-ReLU maps and the `/9` pooled feature calculation. The existing detailed weight diagram now links to this zoom; its working concept link remains intact. No runtime or milestone behavior changed.

## Planning Update: CNN Learning Replay

16 September 2026. Owner requested a replayable visualization of the actual CNN computation and its place in the service architecture. **Planning only; implementation pending.** This decision supersedes the historical M50 entry's immediate-next-M51 queue, without renumbering M51 or changing its lazy-loading scope.

Reviewed the shared galaxy model/data/inference code, trainer, observation probes/tests, four API services, Compose/checkpoint packaging, roadmap/workflow, and job/master alignment. The current observer already records all parameter updates; convolution maps and intermediate backward derivatives need expansion. No CosmosAI frontend exists. The unrelated dashboard directory is outside this work.

Decisions:

- Extend the observer around the real trainer; no second CNN/training loop or live web-training service.
- Produce a bounded, versioned trace and local HTML/JavaScript report. Keep JSON/CSV export and checkpoint/inference behavior compatible.
- Reuse this frontend for Learn now, Classify in R1 and Ask in R2. The viewer reads artifacts; the gateway/router continue prediction routing. Stellar training stays later.
- Complete concrete calculus exercises alongside implementation. Preserve existing concept/Excalidraw links and update diagrams when implemented flow changes.
- Optional Manim video consumes the same trace later and is not required before M51. No LLM/video API is needed for the numerical replay.
- Keep actual batch metadata in the trace contract so lazy loading does not force the observer to retain an entire dataset. Correct the router's status contract in later R1 API/UI work, not as part of this docs update.

Created [cnn_learning_replay_plan.md](cnn_learning_replay_plan.md); synchronized the ultimate roadmap, architecture, current status, workflow, learning TODOs and relevant job/master notes. The Documents TODO copy was synchronized too. The plan contains proposed paths/options, not runnable new commands. Runtime code, dependencies, service topology and existing diagrams are unchanged.

Verification for this planning update: 102 local Markdown links checked across 11 files with Obsidian heading/block conventions, balanced code fences, and `git diff --check`. Existing regression command `.venv/bin/python -m pytest -q tests/test_galaxy_weight_observation.py tests/test_service_contracts.py tests/test_checkpoint_preprocessing.py` passed **33 tests**. This checks the current observer/checkpoint/API baseline, not an implemented replay UI; no new browser test or full-suite run is claimed. The historical 93-test result remains the M50 record. New private planning docs remain ignored by Git under the existing policy; no publication change was made.

## Next Milestone: M50-L1 Measured Learning Trace

Release: R1 foundation, Phase 1A learning. Question: can we follow an actual convolution/classifier weight through its forward contribution, chain-rule gradient and SGD update without changing the training result?

Scope: extend `scripts/observe_galaxy_weight_updates.py`, using the [L1 contract and gates](cnn_learning_replay_plan.md#completion-gates). Observable result: real RGB/kernel planes, 27 products plus bias, pre/post-ReLU maps, pooling/classifier calculations, one shared weight's 18-use gradient, and all 132 before/gradient/after values for the current seed-7 run.

Verification entry point: `.venv/bin/python -m pytest -q tests/test_galaxy_weight_observation.py`, expanded with intermediate/gradient equations, batch-boundary cases, parity and cleanup checks. Then run the existing training/checkpoint/service regressions and publish the actual observation command/results. A passing teaching example does not establish real-data accuracy.

Following queue: **M50-L2 browser replay -> M51 lazy image loading -> bounded real data -> useful baseline/API/interface/CI**. Study the [To-Do ChatGPT exercises](To-Do%20ChatGPT.md) while building L1/L2, not as an additional prerequisite milestone. No commit/push requested or performed.

## M50-L1 Completed: Measured CNN Learning Trace

Completed 22 September 2026. Learning question: can one real convolution weight be followed from its forward response, through the chain-rule gradient, to the exact SGD update while preserving the real training loop?

What changed:

- Added an optional `on_training_batch` callback to `run_torch_training_loop()`. Normal callers pass no callback, so the training equation and existing behavior stay unchanged. The observer now receives actual DataLoader IDs, labels and tensors, including partial batches.
- Extended `observe_galaxy_weight_updates.py` to record pre-ReLU convolution maps, post-ReLU maps, `dL/dlogits`, `dL/d(pooled features)`, `dL/dz`, and selected-weight contribution rows.
- Added the selected equation `dL/dw = sum(dL/dz * aligned_input)` for `features.0.weight[0,0,1,1]`, with 18 terms across two images and nine spatial positions per image. The contribution sum is checked against PyTorch autograd.
- Versioned the JSON trace as format 2. Existing parameter CSV columns and the observer command remain available; no checkpoint is saved or overwritten.
- Added tests for maps/shapes, exact center/pooling values, gradient expansion, actual batch metadata and partial-batch behavior.

Measured command:

```bash
.venv/bin/python scripts/observe_galaxy_weight_updates.py \
  --output /tmp/cosmosai-m50l1/trace.json
```

Observed values from the fresh run:

```text
format_version: 2
trainable scalars: 132
updates: 2
loss: 1.457183 -> 1.420960 -> 1.387335
filter-0 image-A center response: 0.037116095
filter-0 image-A pooled feature: 0.020664100
selected conv gradient: 0.014876489
gradient terms: 18; image subtotals +0.016883992, -0.002007503
sum matches autograd: True
```

Verification: `tests/test_galaxy_weight_observation.py` passes **9 tests**. `git diff --check` passes. The focused implementation/checkpoint/service command passed 33 tests before this change. A repository-wide pytest invocation is currently blocked during collection by the unrelated `assignment-property-dashboard` tests missing `httpx2`, `redis`, and `sqlalchemy`; no CosmosAI failure was reported.

Boundary: this is still a bounded CPU teaching trace, not production telemetry. It does not add the browser viewer, video export, live training HTTP, real-data accuracy or a classifier UI. Those remain M50-L2/R1 work. No commit/push was performed.

## M50-L2 Implemented: HTML/JavaScript Learning Replay

Implemented 22 September 2026. The page is a static, read-only consumer of the M50-L1 version-2 trace. It does not import FastAPI, Docker, the classifier services or a second CNN implementation.

Added:

- `apps/learning-replay/index.html`: Learn page structure and accessible controls.
- `apps/learning-replay/viewer.js`: trace validation, timeline playback, RGB/kernel/product calculations, feature/class diagram, gradient/update inspection and metric rendering. User-supplied metadata is rendered with text/validated JSON rather than injected as HTML.
- `apps/learning-replay/viewer.css`: responsive desktop/mobile layout with stable matrix dimensions and readable numeric tables.
- `apps/learning-replay/sample_trace.js`: bundled 112 KB measured seed-7 example so `index.html` works from `file://` without `fetch()` or a server.
- `scripts/build_learning_replay.py`: copies the viewer and embeds any accepted version-2 trace into a movable report directory.
- `tests/test_learning_replay_static.py`: static page, bundled-trace and report-builder checks.

Run it:

```bash
.venv/bin/python scripts/observe_galaxy_weight_updates.py \
  --output /tmp/cosmosai-m50l1/trace.json
.venv/bin/python scripts/build_learning_replay.py \
  --trace /tmp/cosmosai-m50l1/trace.json \
  --output /tmp/cosmosai-m50l1/replay
```

Open `/tmp/cosmosai-m50l1/replay/index.html`. The page can also load another version-2 JSON through its file picker, with a 2 MB teaching-trace limit. It never retrains while scrubbing; the after-step values are recorded read-only measurements.

Verification: 12 focused replay/observer tests passed, JavaScript syntax checks passed, and `git diff --check` passed. Firefox visual verification could not run in this environment because the installed Snap binary cannot create its runtime directory. Record desktop/mobile screenshots and keyboard interaction as the final M50-L2 acceptance task before M51. No commit/push was performed.

## M51 Completed: Lazy Image Loading

Completed 22 September 2026. Learning question: can the Dataset keep manifest metadata and load one model-ready image only when a batch requests it, without changing the CNN or training equations?

What changed:

- Added `GalaxyLazyTrainingSample`, which keeps the manifest record, data root and preprocessing target but does not retain a decoded tensor.
- `create_dataset_splits()` validates each source file one at a time, preserves accepted/skipped/rejected reporting, and creates lazy samples by default.
- `GalaxyTorchDataset.__getitem__()` now triggers RGB loading, optional resize and normalization for the requested item. Lazy tensors are not cached in the dataset, so completed batches can be released.
- Preserved the eager `create_training_sample_for_record()` path for single-image checkpoint/inference helpers.
- Updated the DataLoader training diagram and current-status/architecture/concept notes to distinguish source-file validation from on-demand model preprocessing.
- Added a regression test proving split creation does not call model preprocessing and one Dataset item loads exactly once.

Measured verification:

```text
focused M51/data tests: 29 passed
dataset/training/checkpoint regression tests: 55 passed before the final lazy-load regression was added
real CLI: 2 epochs, batch size 2, losses 1.4572 -> 1.4210
checkpoint logits/probabilities after reload: match
observer: 132 parameters, 2 updates, losses 1.457183 -> 1.420960 -> 1.387335
```

Boundary: the repository still uses four artificial 3x3 fixtures. No data was downloaded, no model architecture or checkpoint schema changed, and no useful galaxy accuracy claim is made. Next: prepare a small provenance-documented real-data subset on the Kingston drive, with leak-free splits and class counts before scaling training.

## Planning Update: Learning Website And Concept Navigation

24 September 2026. The owner wants to continue project implementation while retaining a future learning-work queue. The current replay has worked pooling, classifier/softmax, cross-entropy, shared-weight gradient and SGD calculations. Longer explanations still live in Markdown and Excalidraw.

Decisions recorded in [the detailed plan](cnn_learning_replay_plan.md#future-learning-website):

- Separate **Learn / Replay** (measured examples and calculations) from **Learn / Concepts** (definitions, formulas and existing conceptual illustrations).
- Link Backward and other replay topics to stable concept routes and return to the same selection. Use canonical Markdown sections and preserve editable Excalidraw sources.
- Export reviewed SVG/PNG diagrams for web pages; translate local-only links in generated output while preserving the originals.
- Plan a shared static web build and local HTTP serving through the API Gateway. Keep current API paths, offline reports and service responsibilities intact; account for gateway Docker packaging.
- Future Classify/Galaxy calls the gateway/router; Stellar and Ask pages depend on their later implementations. Larger real-data learning traces need explicit shape and capture limits.

Future packages: W1 hosting/navigation, W2 concept pilot, W3 diagram publishing, W4 bounded real-data examples. Their checklists remain open. Reconciled the ultimate plan, architecture, current status, workflow and learning TODOs; the Documents resume note mirrors the canonical checklist. No app, service, model or diagram source was changed, and no data was acquired by this planning task.

Planning verification: checked 64 local links/anchors in the updated planning sections and learning notes, the four pilot concept IDs and balanced code fences across the eight edited documents; `git diff --check` passed. Reviewed the gateway/router/classifier code, Docker packaging and report builder, and consulted the official Excalidraw export and FastAPI static-files references linked in the plan. Runtime tests were not rerun for this documentation-only update; web acceptance checks remain future work.

## Completed Pilot: Bounded Real-Data Subset

This followed M51. The learning website backlog did not block it. Learning question: how can real images, trustworthy labels and fixed splits enter the existing manifest/lazy-loader pipeline reproducibly?

Implementation deliverables:

1. Inspect the configured data drive, available space and file layout; choose a small explicit sample/byte budget.
2. Inspect candidate source documentation and labels, record provenance/usage terms, and define a defensible mapping to the supported classes. Verify that the intended classes can actually be derived; do not assume four clean labels or start with the long-term 50k-image aspiration.
3. Implement reproducible bounded acquisition/intake with source IDs, checksums and repeatable selection. Keep data paths configurable and bulk images outside Git.
4. Produce train/validation/test manifests with object-level separation, duplicate checks, class counts and the existing accepted/skipped/rejected quality report.
5. Verify the lazy Dataset/DataLoader on the accepted subset and run a small training/evaluation/checkpoint smoke test. Record the command, input dimensions, counts and outcomes.

Completion evidence: the official GZ2 Hart et al. metadata and Zenodo mapping were downloaded to Kingston; `scripts/prepare_galaxy_zoo_subset.py` created a deterministic 80-row pilot with 20 examples per label and 56/12/12 splits; all 80 SDSS DR7 cutouts were downloaded as validated 424x424 JPEGs; the strict loader accepted every row; and the existing PyTorch baseline ran 2 epochs with average real loss 1.4140 -> 1.4069. Train, validation and test accuracy were each 0.2500, and checkpoint reload matched logits/probabilities. The lenticular label is explicitly a derived proxy rather than an official GZ2 class. A useful accuracy result is a following baseline/evaluation milestone, then checkpoint/API parity, an honest Classify interface and CI for R1. W1 can support that interface; W2/W3 and video stay optional learning work.


## Active Next Milestone: Real-Data Baseline And Evaluation

Review the 80-image pilot and its derived four-label policy, then expand only if the sample and labels are defensible. Add class-wise metrics, a confusion matrix, error examples, repeatable seed/configuration and a comparison against a simple baseline. Do not claim useful accuracy from the current 25% smoke test. After the label/sample decision, run a larger bounded baseline before wiring the checkpoint into the Classify path.


## Completed Real-Data Baseline And Evaluation

Implemented the next real-data milestone locally. The existing baseline
trainer now accepts an explicit seed and learning rate and can write a
structured JSON report. The report records split/class counts, quality counts,
configuration, training summary, per-class precision/recall/F1, confusion
matrices, bounded misclassified image examples, and a majority-class baseline.

Command used:

    .venv/bin/python scripts/train_galaxy_cnn_baseline.py       /media/h1dr0/KINGSTON/cosmosai-data/galaxy/manifests/gz2_pilot_80.csv       --galaxy-data-root /media/h1dr0/KINGSTON/cosmosai-data/galaxy       --epochs 2 --batch-size 8 --learning-rate 0.1 --seed 7 --strict       --checkpoint-path /media/h1dr0/KINGSTON/cosmosai-data/galaxy/checkpoints/gz2_pilot_80.pt       --evaluation-report-path /media/h1dr0/KINGSTON/cosmosai-data/galaxy/manifests/gz2_pilot_80.evaluation.json

The 80-image pilot completed with 14 real optimizer updates, average loss
1.4140 -> 1.4069, and 0.2500 accuracy on train, validation and test. The
CNN predicted elliptical for every evaluated image and matched the balanced
majority baseline. The result confirms the evaluation/reporting path but does
not establish useful galaxy-classification accuracy. Focused verification:
22 existing CNN baseline tests plus 3 evaluation-report tests passed.


## M53a: Local Image Prediction And Upload Foundation

24 September 2026. R1 implementation, following the owner's request to make
concrete demo progress while continuing concept review and model improvement.
Scope/learning question: can an image without a manifest or known label use the
saved checkpoint without changing weights? See [commands, concept refresher and
function flow](local_classifier_demo.md).

Implemented:

- `GalaxyInferenceSample` carries only an image identifier and prepared pixels.
- `predict_image_from_checkpoint()` reuses saved preprocessing and the existing
  forward pass; the CLI exposes it through `--image-path`.
- `POST /classify/image` accepts raw JPEG/PNG bytes and returns all four
  probabilities/logits, input shape, checkpoint name and preprocessing recipe.
  It requires a real checkpoint; configuration failures return 503, invalid
  images fail clearly, and temporary image files are removed.
- Uploads are bounded at 5 MiB and 1,048,576 pixels, with one upload prediction
  at a time per process. The verification runs limited CPU threads to two.
- Existing training equations, saved weights, manifest predictions and legacy
  JSON endpoints are preserved. A custom UI and gateway upload forwarding are
  still pending. No model training or larger download was performed.

Verification command:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_galaxy_image_prediction.py tests/test_predict_galaxy_checkpoint.py \
  tests/test_checkpoint_preprocessing.py tests/test_service_contracts.py \
  tests/test_galaxy_torch_dataset.py tests/test_galaxy_cnn_baseline_skeleton.py
```

Verification: **73 focused tests passed**; `git diff --check` passed.
The focused suite covers image/manifest/API parity, exact checkpoint bytes and
in-memory weight stability, native/fixed input sizes, temporary-file cleanup,
bounded uploads, missing/corrupt/legacy checkpoints and existing training/service
contracts. A live temporary Uvicorn service returned HTTP 200 for the real
424x424 `gz2-587722983908311227.jpg`; HTTP probabilities/logits exactly matched
file prediction and the checkpoint checksum stayed unchanged. Elliptical scored
0.290915, spiral 0.286476, lenticular 0.198925, irregular 0.223683. This single
prediction is not an evaluation result. Invalid image bytes returned 422. The
temporary service was stopped after verification.

## Active Next Milestone: M53b Local Classify Page

Wire bounded uploads through the gateway/router and add the custom Classify UI
with image preview, real probabilities, loading/errors and the pilot's measured
limitations. Correct the enclosing router status for real checkpoint results.
Prepare the local launch and README/screenshots for a portfolio demo.

Model quality proceeds alongside the demo: review errors and the lenticular
proxy, choose a defensible label policy and larger bounded baseline, and use
validation results to select changes. The pilot still scores 25% and predicts
elliptical for every evaluated image. Do not describe it as a useful morphology
classifier yet. Concept review can continue one topic at a time without blocking
the application work. No commit or push was performed.


## M53b Completed: Local Upload And Prediction UI

24 September 2026. R1 scope: complete the usable local browser path while keeping
model limitations visible. Learning question: how does a new image reach saved
weights through the gateway/router, and what is different from training?

Implemented the Classify page, real image preview, four existing GZ2 samples,
probability bars, error/loading states and JSON download. Gateway/router forward
bounded raw image bytes using a shared HTTP helper; errors and real inference
status propagate. `GET /model` describes the loaded checkpoint and attaches an
evaluation summary only when its checksum matches. The launcher starts/stops all
three localhost services with limited CPU threads. A 3.4 KB pilot checkpoint is
bundled for drive-independent inference. No training dataset was copied.

Added optional standalone Compose packaging, kept direct service-directory
startup working, and retained existing JSON API behavior except for correcting
the enclosing real-inference status. Existing private/user changes remain.

Verification: 35 focused tests passed, including actual gateway -> router ->
classifier requests, file/API numerical parity, metric-checksum mismatch,
streamed byte limits, invalid images, upstream failures and direct startup.
Playwright passed uploads, sample selection, JSON downloads, invalid image/size,
error/retry and stale-response tests. Screenshots/pixel and geometry checks at
1440, 768, 390 and 320 pixels found no blank media, bar-value mismatch, browser
errors or horizontal/text overflow. Compose config validated; full container
builds were not run. The Python demo runs locally at port 8080.

One separate validation experiment trained the existing CNN for 30 epochs with
64x64 input, seed 7, SGD 0.1 and batch size 8. Train and validation stayed at 25%
and every prediction remained elliptical. Test records were excluded before
loading, and the original serving checkpoint was preserved. Artifacts are in
`models/gz2-validation-30epochs-64px/` (ignored by Git). No accuracy improvement
is claimed, and changing both resize and duration does not isolate causality.

Commands and the browser/function flow are in [local_classifier_demo.md](local_classifier_demo.md).
Public deployment planning is in [public_demo_plan.md](public_demo_plan.md).
No cloud resources, commit or push were created.

## Active Next Milestone: M54 Model Diagnosis And Better Baseline

Inspect the actual cutouts and labels, including the derived lenticular proxy.
Check whether a candidate can fit a small training subset; compare a more
expressive image baseline before scaling data. Expand a balanced, defensibly
labelled subset and use validation for decisions. Reserve the final test
comparison for the selected candidate.

In parallel, prepare the public reviewer demo by verifying the container build,
packaging the checkpoint/evaluation artifact without external drive mounts,
choosing hosting/account limits and publishing an approved HTTPS endpoint.
The UI/backend workflow can be reused unchanged when a compatible model improves.

## M54 Initial Diagnosis Completed; Serving Model Unchanged

24 September 2026. Scope: try a small bounded improvement, verify local browser
use and record public hosting as the next demo-delivery milestone.

Verified that all seven training batches contain the expected balanced labels:
14 examples per class. Inspected the four labelled preview JPEGs; target objects
are small relative to the surrounding field. This is a framing observation, not
independent verification of morphology labels.

Added `scripts/check_galaxy_learning.py`, reusing the shared data loader,
preprocessing, evaluator and tiny-model checkpoint writer. With two CPU threads:

- A separate 2,420-parameter spatial CNN memorized eight TRAIN examples (two per
  class) in a fixed 200-step diagnostic. It scored 8/8 on those same examples;
  no generalization or deployability is claimed for that network.
- The existing TinyGalaxyCNN trained with Adam 0.01, shuffled batches of eight,
  seed 7 and 30 epochs at 64x64 scored 15/56 train (26.8%) and 4/12 validation
  (33.3%), versus 14/56 and 3/12 in the earlier SGD trial. It still predicts spiral
  for 10/12 validation images. The one-image gain does not justify promotion.
- Test images were excluded before loading. There was no download or change to
  the serving checkpoint. Experimental artifacts remain separate under
  `models/gz2-learning-check-adam/`.

M54 remains active: review target framing and label provenance, compare the
spatial model on the full training split, then expand balanced data. A different
architecture needs explicit checkpoint metadata and serving parity tests first.
See the [measured learning check](galaxy_learning_check.md).

Verification: 36 focused tests passed, including test-pixel exclusion, balanced
training-only subset selection, checkpoint compatibility, bounds and overwrite
protection, unchanged serving bytes and existing upload/service contracts.
The live browser smoke check passed real uploads, examples, downloads, invalid
input, upstream error/retry, stale-response handling and nonblank media; no
JavaScript or layout-overflow errors at 1440, 768, 390 and 320 pixels. The running
demo still reports the original checkpoint checksum and 25% metrics on port 8080.
`git diff --check` passed. The README/local guide now include exact browser steps.

## M55 Planned: Public Reviewer Demo

Next demo-delivery work: verify local container builds and real uploads, package
the checkpoint without external-drive mounts, agree on hosting/account/spending
constraints, then expose only the gateway over HTTPS. Test from another device,
retain upload/privacy protections and the experimental-model warning, and
document teardown. Acceptance is in the [M55 plan](public_demo_plan.md).
No cloud resources, public endpoint, commit or push were created in this update.

## W1 Completed Locally: CNN Learning Replay Navigation

24 September 2026. Added CNN Learning Replay beside Model details in the local
Classify navigation. The gateway serves the existing replay at `/learn/`, with
relative assets, return links and a short description identifying recorded
teaching calculations rather than current-upload training. Docker packaging now
includes replay assets and `COSMOSAI_REPLAY_ROOT`; container runtime verification
remains future work. Standalone HTML exports hide unavailable hosted links.

Browser testing also found existing narrow-screen replay overflow. Pooled/logit
summaries now use two columns on mobile; wide matrices/gradient diagrams scroll
inside their sections and matrix blocks no longer shrink over each other.
Recorded calculations, trace data, training and serving checkpoint are unchanged.

Verification: 21 focused replay/gateway/service tests passed, including the HTTP
route, slash redirect, asset parity, path isolation and standalone report build.
Playwright verified navigation, direct refresh, replay controls, rendered diagram
connections, a return to real prediction, and working file-mode replay without
model requests. Screenshots/layout checks passed at 1440, 768, 390 and 320 pixels
with no page overflow or JavaScript errors. Only the existing local demo services
were reloaded; the PC was not restarted. W2-W4 and M55 remain planned.

## W2-W3 Completed Locally: Concepts And Diagrams

24 September 2026. Added Concepts Explanations and Diagrams beside Galaxy
Classifier and CNN Learning Replay. An explicit allowlist publishes all 67
concept topics / 139 headings from the canonical Markdown and 28 original
Excalidraw scenes. Structured Markdown parsing preserves formulas, code and
stable anchors; supported internal links become web links. Private documents,
editor URLs and workstation paths are not exposed. Original notes/scenes remain
unchanged. Generated exports ship with the app, so serving needs no authoring tools.

The official Excalidraw exporter produces SVGs with embedded fonts, attribution,
accessible titles, concept links, zoom/fit controls and downloadable scene copies.
Replay links cover convolution, ReLU, pooling, matrix multiplication, logits/loss,
backward, gradients and SGD. Return navigation restores the selected calculation
and imported trace using tab-local session storage, with an explicit fallback
when the saved state is unavailable. Standalone replay exports are preserved.

Verification: all 140 CosmosAI tests passed. Repository-wide discovery also
collected the unrelated assignment dashboard, whose httpx/redis/sqlalchemy
dependencies are absent; that project was not modified. Playwright checked
custom-trace return, search, 28 nonblank SVG pixel renderings, no external asset
requests/model calls from learning pages, and four pages at 320/390/768/1440px.
Screenshots confirmed readable mobile zoom and no page overflow. Source hashes
and anchor/path tests detect stale or unsafe generated content. No model changes.

## M55 Local Preparation Verified: Portable Release

The owner requested preparation before choosing a hosting account or budget.
Added `docker-compose.release.yml`, a bundled-checkpoint release Dockerfile,
`scripts/check_release_demo.py` and `deploy/README.md`. Built and ran all three
images with no bind mounts or Kingston dependency. The gateway includes all
four pages. Only its localhost port 8090 was published. Non-root UID 10001,
read-only roots, dropped capabilities, temporary storage and CPU/memory limits
were verified; model bytes are not writable and uploaded files are cleaned up.

The real JPEG/checksum/model-card/error-limit smoke check passed before and after
a restart, as did release-browser checks. One post-upload memory snapshot was
39.54 MiB gateway + 41.03 MiB router + 221.1 MiB classifier; warm JPEG requests in
the two smoke runs took 0.228 and 0.229 seconds. These are local observations,
not peak-load or cloud cold-start benchmarks. The experimental pilot stays unchanged.

Stopped/removed only the rehearsal containers/network; the normal local demo
remains on 8080. M55 public deployment is still pending account, budget, HTTPS,
access/rate controls and separate-device verification. No cloud resource, paid
service, image push, commit or GitHub publication was created by this work.
