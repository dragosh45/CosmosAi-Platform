# Architecture

CosmosAI is designed as a small AI inference platform using astronomy as the domain.

This file describes what is actually implemented and verified. Future roadmap items live in [portfolio_ai_project_plan_ultimate.md](portfolio_ai_project_plan_ultimate.md), and master's-course reuse ideas live in [master_project_integration.md](master_project_integration.md).

## Cuprins

- [[#Step 0: Local Service Skeleton And Mock Routing|Step 0: Local Service Skeleton And Mock Routing]]
- [[#Step 1: Galaxy Data Intake Proof|Step 1: Galaxy Data Intake Proof]]
- [[#Step 2: Tiny Model-Ready Dataset Proof|Step 2: Tiny Model-Ready Dataset Proof]]
- [[#Step 3: CNN Baseline Training Skeleton|Step 3: CNN Baseline Training Skeleton]]
- [[#Step 4: PyTorch CNN Forward Pass Proof|Step 4: PyTorch CNN Forward Pass Proof]]
- [[#Step 5: Tiny PyTorch Weight Update Proof|Step 5: Tiny PyTorch Weight Update Proof]]
- [[#Step 6: Tiny PyTorch Training Loop Proof|Step 6: Tiny PyTorch Training Loop Proof]]
- [[#Step 7: Tiny PyTorch Evaluation Proof|Step 7: Tiny PyTorch Evaluation Proof]]
- [[#Step 8: Tiny PyTorch Checkpoint Save/Load Proof|Step 8: Tiny PyTorch Checkpoint Save/Load Proof]]
- [[#Step 9: Tiny Checkpoint Inference Proof|Step 9: Tiny Checkpoint Inference Proof]]
- [[#Step 10: Optional Galaxy Service Checkpoint Inference|Step 10: Optional Galaxy Service Checkpoint Inference]]
- [[#Step 11: Optional Docker Checkpoint Compose Path|Step 11: Optional Docker Checkpoint Compose Path]]
- [[#Step 12: Shared Galaxy Runtime Package|Step 12: Shared Galaxy Runtime Package]]
- [[#Step 13: Shared Galaxy Package Review Checkpoint|Step 13: Shared Galaxy Package Review Checkpoint]]
- [[#High-level architecture|High-level architecture]]
- [[#Main services|Main services]]
- [[#Infrastructure layer|Infrastructure layer]]
- [[#Build principle|Build principle]]

## Step 0: Local Service Skeleton And Mock Routing

Step 0 is the foundation built through milestones 1-13. It gives us the local service layout, Docker containers, health checks, and mocked routing between services.

At this stage, routing works, but the classifiers are still stubs. That means the API flow is real HTTP between services, while the classification results are placeholder responses.

### Step 0 API flow

Code touchpoints: [api-gateway route()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:67) receives the public request, [inference-router route()](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:70) chooses the downstream service, then either [galaxy classify()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:160) or [stellar classify()](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:48) returns a response.

```text
User or client
    |
    | POST http://127.0.0.1:8000/route
    v
api-gateway container
    |
    | forwards request to:
    | http://inference-router:8000/route
    v
inference-router container
    |
    | if input_type == "galaxy_image"
    | calls http://galaxy-classifier-service:8000/classify
    |
    | if input_type == "stellar_spectrum"
    | calls http://stellar-classifier-service:8000/classify
    v
mock classifier service
    |
    | returns stub classification result
    v
inference-router
    |
    | returns selected service + classification result
    v
api-gateway
    |
    | returns final API response to the user
    v
User or client
```

### Step 0 Docker Compose infrastructure

Docker wiring is defined in [docker-compose.yml](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:1), with [api-gateway](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:4), [inference-router](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:20), [galaxy-classifier-service](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:36), and [stellar-classifier-service](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:52).

```text
Host machine
    |
    | published ports
    |
    | 127.0.0.1:8000 -> api-gateway:8000
    | 127.0.0.1:8001 -> inference-router:8000
    | 127.0.0.1:8002 -> galaxy-classifier-service:8000
    | 127.0.0.1:8003 -> stellar-classifier-service:8000
    v
Docker Compose network
    |
    +-- api-gateway
    |       internal URL: http://api-gateway:8000
    |
    +-- inference-router
    |       internal URL: http://inference-router:8000
    |
    +-- galaxy-classifier-service
    |       internal URL: http://galaxy-classifier-service:8000
    |
    +-- stellar-classifier-service
            internal URL: http://stellar-classifier-service:8000
```

### Step 0 service discovery rule

Inside Docker Compose, each service name becomes a private hostname. That is why [api-gateway](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:4) can call `http://inference-router:8000/route`: Docker Compose resolves [inference-router](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:20) to the inference-router container.

From the host machine, use the published localhost ports. From one container to another container, use the Compose service name and the internal container port.

### Step 0 code connection

The [API Gateway route()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:67) does not import the [inference-router route()](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:70) Python function. These are separate FastAPI apps, so they connect by HTTP.

The gateway uses [INFERENCE_ROUTER_URL](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:22), which points to the Docker Compose hostname for the router.

```text
apps/api-gateway/main.py
    INFERENCE_ROUTER_URL = "http://inference-router:8000/route"
    requests.post(INFERENCE_ROUTER_URL, json=request.model_dump(), timeout=5)

matches this endpoint:

apps/inference-router/main.py
    @app.post("/route")
    def route(request: RouteRequest) -> RouteResponse:
```

The connection is:

```text
URL host: inference-router
    -> Docker Compose service name
    -> inference-router container

URL port: 8000
    -> internal port used by the FastAPI app inside that container

URL path: /route
    -> FastAPI endpoint registered with @app.post("/route")
```

### Step 0 Clickable Code References

Use these links to jump from the architecture overview into the code. They use `vscode://file/...` links because this Obsidian vault is rooted at `docs/`, while the code files live outside the vault.

Service wiring:

- [docker-compose.yml](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:1): defines the Step 0 services and host ports.
- [api-gateway service](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:4): exposes the public API on host port 8000.
- [inference-router service](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:20): exposes the router on host port 8001.
- [galaxy-classifier-service service](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:36): exposes the galaxy stub on host port 8002.
- [stellar-classifier-service service](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:52): exposes the stellar stub on host port 8003.

API Gateway:

- [apps/api-gateway/main.py](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:1): public FastAPI app.
- [INFERENCE_ROUTER_URL](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:22): internal Docker Compose URL for the inference-router.
- [RouteRequest](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:26): request body accepted by `POST /route`.
- [RouteResponse](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:40): response body returned by `POST /route`.
- [health()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:59): `GET /health` endpoint.
- [route()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:67): forwards `POST /route` to inference-router.

Inference Router:

- [apps/inference-router/main.py](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:1): routing FastAPI app.
- [GALAXY_CLASSIFIER_URL](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:14): internal URL for the galaxy classifier stub.
- [STELLAR_CLASSIFIER_URL](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:17): internal URL for the stellar classifier stub.
- [SERVICE_BY_INPUT_TYPE](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:22): maps input types to service names.
- [health()](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:62): `GET /health` endpoint.
- [route()](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:70): chooses the downstream classifier service.

Classifier stubs:

- [apps/galaxy-classifier-service/main.py](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:1): galaxy classifier FastAPI stub.
- [galaxy health()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:152): galaxy `GET /health` endpoint.
- [galaxy classify()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:160): galaxy `POST /classify` endpoint.
- [apps/stellar-classifier-service/main.py](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:1): stellar classifier FastAPI stub.
- [stellar health()](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:40): stellar `GET /health` endpoint.
- [stellar classify()](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:48): stellar `POST /classify` stub.

## Step 1: Galaxy Data Intake Proof

Step 1 covers milestones 17-22. This step does not train a CNN and does not use the real Galaxy Zoo dataset yet.

The goal is to prove the first local data path:

Code touchpoints: [validat
e_manifest()](vscode://file/opt/projects/cosmosai-platform/scripts/validate_galaxy_manifest.py:44) checks the CSV contract, [load_manifest()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:89) creates [GalaxyManifestRecord](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:30) objects, [resolve_image_path()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:48) resolves image paths, and [load_image_for_record()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:106) loads the tiny image into a [GalaxyImage](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:27) object.

```text
sample manifest CSV
  -> validate CSV contract
  -> load rows into Python records
  -> resolve image paths
  -> optionally check image existence
  -> load one tiny sample image
```

This is not full preprocessing yet. It is a small proof that the project can move from dataset metadata toward image pixels in a controlled way.

### Step 1 Files

Step 1 files: [sample manifest CSV](vscode://file/opt/projects/cosmosai-platform/data/samples/galaxy_manifest_sample.csv:1), [tiny PPM image](vscode://file/opt/projects/cosmosai-platform/data/samples/images/processed/images_224/gz2-000001.ppm:1), [validator script](vscode://file/opt/projects/cosmosai-platform/scripts/validate_galaxy_manifest.py:1), [manifest loader script](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:1), and [image loader script](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:1).

```text
data/samples/galaxy_manifest_sample.csv
  tiny sample manifest with image_id, image_path, label, split, source

data/samples/images/processed/images_224/gz2-000001.ppm
  tiny local image used only for image-loading proof

scripts/validate_galaxy_manifest.py
  checks manifest columns, labels, splits, duplicates, and empty values

scripts/load_galaxy_manifest.py
  validates the manifest, loads rows into GalaxyManifestRecord objects,
  groups rows by split, resolves paths, and can check missing images

scripts/load_galaxy_image.py
  loads one tiny PPM image from a manifest record and reports dimensions
```

### Step 1 Data Contract Flow

The validator starts at [validate_manifest()](vscode://file/opt/projects/cosmosai-platform/scripts/validate_galaxy_manifest.py:44), using [REQUIRED_COLUMNS](vscode://file/opt/projects/cosmosai-platform/scripts/validate_galaxy_manifest.py:19), [ALLOWED_LABELS](vscode://file/opt/projects/cosmosai-platform/scripts/validate_galaxy_manifest.py:28), and [ALLOWED_SPLITS](vscode://file/opt/projects/cosmosai-platform/scripts/validate_galaxy_manifest.py:36).

```text
galaxy_manifest_sample.csv
    |
    | read by csv.DictReader
    v
validate_galaxy_manifest.py
    |
    | checks required columns:
    | image_id, image_path, label, split, source
    |
    | checks allowed labels:
    | elliptical, spiral, lenticular, irregular
    |
    | checks allowed splits:
    | train, val, test
    v
valid manifest contract
```

### Step 1 Python Object Flow

The manifest loader uses [load_manifest()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:89) to turn CSV rows into [GalaxyManifestRecord](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:30) objects.

```text
validated manifest CSV row
    |
    v
load_galaxy_manifest.py
    |
    | creates:
    v
GalaxyManifestRecord
    |
    | image_id = "gz2-000001"
    | image_path = "processed/images_224/gz2-000001.ppm"
    | label = "spiral"
    | split = "train"
    | source = "galaxy_zoo_2"
    v
structured Python object for future training code
```

### Step 1 Image Path Resolution Flow

Path logic lives in [resolve_image_path()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:48), [resolved_image_paths_by_id()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:61), and [missing_image_paths()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_manifest.py:73).

```text
galaxy data root
    |
    | data/samples/images
    |
    + manifest image_path
      processed/images_224/gz2-000001.ppm
    |
    v
resolved image path
    |
    | data/samples/images/processed/images_224/gz2-000001.ppm
    v
optional existence check
```

Path resolution means turning a manifest-relative path into a full usable filesystem path. It is not duplicate detection. Duplicate `image_id` values are checked by the validator.

### Step 1 Tiny Image Loading Flow

Image-loading logic lives in [read_ppm_tokens()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:45), [load_ppm_image()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:64), and [load_image_for_record()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:106).

```text
GalaxyManifestRecord
    |
    | resolve image_path against galaxy data root
    v
data/samples/images/processed/images_224/gz2-000001.ppm
    |
    | load PPM text tokens
    v
GalaxyImage
    |
    | width = 3
    | height = 3
    | channels = 3
    | pixel_values = 27
    v
proof that manifest metadata can reach image pixel values
```

The tiny image is a plain-text PPM file so this proof can run without image libraries such as Pillow. Later milestones can replace this with normal image loading for real JPEG/PNG data.

### Step 1 Code Flow Diagram

The detailed function flow, command examples, helper notes, and path examples are in [manifest_tooling_code_flow.excalidraw](excalidraw/manifest_tooling_code_flow.excalidraw). The architecture page stays high-level; the diagram is for reading the code flow.

The diagram has clickable links to the same scripts/functions.

### Step 1 PPM Token Note

The tiny sample image uses plain-text PPM `P3` format. Here, a token just means one whitespace-separated piece from that image file, such as `P3`, `3`, `255`, or an RGB value.

Example:

```text
P3
3 3
255
0 0 0  64 64 64  255 255 255
```

This is not the same as GPT/tokenizer tokens. Future JPEG/PNG loading will probably use an image library to decode pixels directly:

```text
JPEG/PNG file
  -> image library
  -> RGB pixel numbers
```

### Step 1 Test Coverage

```text
tests/test_galaxy_manifest_validator.py
  validates the sample manifest
  rejects invalid labels
  rejects invalid split values

tests/test_galaxy_manifest_loader.py
  loads records
  groups train / val / test splits
  resolves relative and absolute image paths
  reports missing image files

tests/test_galaxy_image_loader.py
  loads the tiny sample image through a manifest record
  verifies width, height, channels, and pixel value count
  fails clearly for missing image files
```

### Step 1 Boundary

Step 1 proves:

```text
CSV metadata can be validated
CSV rows can become Python objects
relative image paths can become full paths
missing images can be reported
one tiny image can be loaded from a manifest row
```

Step 1 does not do:

```text
real dataset download
real image preprocessing
tensor conversion
CNN architecture
model training
model serving
```

## Step 2: Tiny Model-Ready Dataset Proof

Step 2 covers milestones 24-28. It extends Step 1 from "we can load one tiny image" to "we can create model-ready sample objects grouped by dataset split."

This still does not train a CNN and still does not use the real full Galaxy Zoo dataset. The current purpose is to prove the local data pipeline shape before adding ML training code.

Code touchpoints: [preprocess_image()](vscode://file/opt/projects/cosmosai-platform/scripts/preprocess_galaxy_image.py:49) converts [GalaxyImage](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_image.py:27) pixels into [GalaxyImageTensor](vscode://file/opt/projects/cosmosai-platform/scripts/preprocess_galaxy_image.py:26), [create_training_sample_for_record()](vscode://file/opt/projects/cosmosai-platform/scripts/create_galaxy_training_sample.py:67) creates [GalaxyTrainingSample](vscode://file/opt/projects/cosmosai-platform/scripts/create_galaxy_training_sample.py:38), and [create_dataset_splits()](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_dataset_splits.py:48) creates [GalaxyDatasetSplits](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_dataset_splits.py:29).

```text
validated manifest rows
  -> load tiny image pixels
  -> normalize pixels to 0.0-1.0 values
  -> attach known label and numeric label_id
  -> group samples into train, val, and test buckets
```

### Step 2 Files

Step 2 files: [preprocessor script](vscode://file/opt/projects/cosmosai-platform/scripts/preprocess_galaxy_image.py:1), [training sample script](vscode://file/opt/projects/cosmosai-platform/scripts/create_galaxy_training_sample.py:1), [dataset split loader script](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_dataset_splits.py:1), [training sample tests](vscode://file/opt/projects/cosmosai-platform/tests/test_galaxy_training_sample.py:1), and [dataset split tests](vscode://file/opt/projects/cosmosai-platform/tests/test_galaxy_dataset_splits.py:1).

```text
scripts/preprocess_galaxy_image.py
  turns raw RGB pixel values into normalized model-ready numbers

scripts/create_galaxy_training_sample.py
  combines image tensor data with the known manifest label

scripts/load_galaxy_dataset_splits.py
  creates train, val, and test buckets for future training code
```

### Step 2 Current Sample Behavior

The current sample manifest has four rows, but only one tiny local image file exists. The default split loader skips rows whose image files are not present yet, while `--strict` fails immediately on the first missing image.

```text
train samples: 1
val samples: 0
test samples: 0
skipped train: 1
skipped val: 1
skipped test: 1
```

The `spiral` label is not predicted. It is read from the manifest as the known training target and converted to `label_id = 1`.

### Step 2 Code Flow Diagram

The detailed module dependency diagram, object/class diagram, function call graph, sequence diagram, and per-script notes are in [galaxy_training_sample_flow.excalidraw](excalidraw/galaxy_training_sample_flow.excalidraw).

The diagram contains clickable links to the scripts, functions, and objects used in this step.

### Step 2 Boundary

Step 2 proves:

```text
raw pixels can become normalized model-ready values
labels can become numeric class IDs
one manifest row can become one training sample
available samples can be grouped by train, val, and test
missing local sample images can be skipped or treated as strict errors
```

Step 2 does not do:

```text
real dataset download
image augmentation
CNN architecture
loss calculation
backpropagation
model checkpointing
model serving
```

## Step 3: CNN Baseline Training Skeleton

Step 3 covers milestones 30-32. It extends Step 2 from "we can create model-ready samples" to "we can run those samples through a training-shaped loop."

This is still not a real CNN and still does not use PyTorch. The current purpose is to prove the training code shape before adding the real ML framework.

Code touchpoints: [load_dataset_splits_from_manifest()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:302) loads the existing split data, [PlaceholderGalaxyModel](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:231) provides a fake model-like `forward()` method, [run_placeholder_training_step()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:760) turns logits into probabilities and loss, and [run_training_loop()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:794) repeats over epochs and train samples.

```text
GalaxyDatasetSplits
  -> train samples
  -> PlaceholderGalaxyModel.forward()
  -> fake logits
  -> softmax probabilities
  -> cross-entropy-style loss
  -> training summary
```

### Step 3 Files

Step 3 files: [training skeleton script](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:1), [training skeleton tests](vscode://file/opt/projects/cosmosai-platform/tests/test_galaxy_cnn_baseline_skeleton.py:1), and [[concepts_explanations#CNN-Shaped Training Loop Skeleton|concept notes]].

```text
scripts/train_galaxy_cnn_baseline.py
  reuses the current data pipeline and runs a placeholder training-loop shape

tests/test_galaxy_cnn_baseline_skeleton.py
  checks the placeholder model interface, epoch loop, probabilities, and errors

docs/concepts_explanations.md
  explains logits, softmax, loss, epochs, and what is mocked
```

### Step 3 Current Behavior

The current training skeleton uses only the one tiny local training image. It does not train on more data because the other sample manifest rows point to image files that do not exist locally yet.

```text
train samples: 1
val samples: 0
test samples: 0
skipped rows: 3
```

The script calculates fake logits from simple hand-written math. It then calculates softmax probabilities and placeholder loss, but it does not update weights.

```text
status: skeleton only - no weights updated
```

### Step 3 Code Flow Diagram

The detailed module dependency diagram, object/class diagram, function call graph, sequence diagram, and per-script notes are in [galaxy_cnn_training_skeleton_flow.excalidraw](excalidraw/galaxy_cnn_training_skeleton_flow.excalidraw).

The diagram contains clickable links to the training skeleton classes, functions, and related data-pipeline code.

The concept-focused diagram that connects the same code to logits, softmax, loss, epochs, and placeholder-vs-real training is in [galaxy_training_concepts_in_code.excalidraw](excalidraw/galaxy_training_concepts_in_code.excalidraw).

### Step 3 Boundary

Step 3 proves:

```text
the training script can reuse the existing dataset split loader
the loop can run for one or more epochs
one model-like object can produce logits
logits can become probabilities
probabilities can be compared with the known label_id
the output can be summarized for inspection
```

Step 3 does not do:

```text
real CNN layers
PyTorch tensors
backpropagation
optimizer step
weight updates
checkpoint saving
real model accuracy
model serving
```

## Step 4: PyTorch CNN Forward Pass Proof

Step 4 covers milestone 33. It extends Step 3 from "training-shaped placeholder math" to "one real PyTorch CNN forward pass on one tiny galaxy sample."

This is still not real training. The model has real PyTorch layers, but its weights are only initial random weights. No backpropagation or optimizer step happens yet.

Code touchpoints: [TinyGalaxyCNN](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:261) defines the tiny CNN layers, [galaxy_tensor_to_torch_image()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:319) converts our project tensor into PyTorch `NCHW` shape, and [run_torch_forward_pass()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:387) runs the real PyTorch forward pass without updating weights.

```text
GalaxyTrainingSample
  -> GalaxyImageTensor shape (height, width, channels)
  -> torch.Tensor shape (batch, channels, height, width)
  -> TinyGalaxyCNN real layers
  -> real PyTorch logits
  -> torch.softmax probabilities
  -> printed forward-pass proof
```

### Step 4 Files

Step 4 files: [training script](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:1), [requirements.txt](vscode://file/opt/projects/cosmosai-platform/requirements.txt:1), [training tests](vscode://file/opt/projects/cosmosai-platform/tests/test_galaxy_cnn_baseline_skeleton.py:1), and [[concepts_explanations#PyTorch CNN Forward Pass Proof|concept notes]].

```text
requirements.txt
  adds CPU-only PyTorch for local ML code

scripts/train_galaxy_cnn_baseline.py
  now contains both the placeholder training rehearsal and one real PyTorch forward proof

tests/test_galaxy_cnn_baseline_skeleton.py
  checks tensor conversion shape and real CNN output shape
```

### Step 4 Current Behavior

The command now prints two sections. The first section is still the placeholder skeleton from Step 3. The second section is the new PyTorch proof:

```text
PyTorch forward proof
  input_shape: (1, 3, 3, 3)
  logits: [0.1853, 0.1107, -0.3626, -0.2596]
  probabilities: [0.3177, 0.2949, 0.1837, 0.2036]
  predicted_label_id: 0
  status: real PyTorch forward pass only - no weights updated
```

The prediction is not meaningful yet because the tiny CNN has not learned from data. The milestone only proves that real PyTorch layers can receive our sample tensor and produce class scores.

### Step 4 Boundary

Step 4 proves:

```text
PyTorch is available locally
our normalized image values can become a torch.Tensor
the tensor can be reshaped from HWC to NCHW
a tiny real CNN can run one forward pass
the real CNN returns one logit per galaxy label
torch.softmax can turn those logits into probabilities
```

Step 4 does not do:

```text
loss.backward()
optimizer.step()
weight updates
training over a real dataset
validation accuracy
checkpoint saving
serving from galaxy-classifier-service
```

## Step 5: Tiny PyTorch Weight Update Proof

Step 5 covers milestone 34. It extends Step 4 from "real PyTorch forward pass only" to "one real PyTorch optimizer step changes model weights."

This is the first proof that the CNN can learn mechanically. It is still not useful galaxy training because it uses only one tiny local sample.

Code touchpoints: [run_torch_training_step()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:428) computes real PyTorch loss, runs backpropagation, calls `optimizer.step()`, and proves the classifier weight changed. [TorchTrainingStepResult](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:106) stores the before/after values for inspection.

```text
GalaxyTrainingSample
  -> torch.Tensor input
  -> TinyGalaxyCNN logits
  -> CrossEntropyLoss with true label_id
  -> loss.backward() calculates gradients
  -> optimizer.step() updates weights
  -> same sample is checked again
```

### Step 5 Current Behavior

The training command now prints a third section:

```text
PyTorch training step proof
  true_label: spiral
  true_label_id: 1
  loss_before: 1.2211
  loss_after: 1.1285
  probabilities_before: [0.3177, 0.2949, 0.1837, 0.2036]
  probabilities_after: [0.3006, 0.3235, 0.1766, 0.1992]
  weight_changed: True
```

For this one tiny example, the correct `spiral` probability increased and the loss decreased after one update.

### Step 5 Code Flow Diagram

The visual diagram for how one tiny image changes the CNN weights is in [galaxy_pytorch_weight_update_flow.excalidraw](excalidraw/galaxy_pytorch_weight_update_flow.excalidraw).

The diagram links to the training-step code, the tensor conversion code, and the concept notes for gradients, backpropagation, and optimizer steps.

### Step 5 Boundary

Step 5 proves:

```text
PyTorch can calculate loss from logits and the known label_id
backpropagation can calculate gradients
SGD can update at least one model weight
the command output can show before/after learning signals
```

Step 5 does not do:

```text
real dataset training
mini-batches
many useful epochs
validation accuracy
checkpoint saving
classifier service using a trained model
```

## Step 6: Tiny PyTorch Training Loop Proof

Step 6 covers milestone 35. It extends Step 5 from "one optimizer step" to "a tiny real PyTorch training loop over the available train split for the requested epoch count."

This is the first real loop shape for CNN training. It is still tiny and local because only one sample image exists right now.

Code touchpoints: [run_torch_training_loop()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:525) keeps one model and one optimizer alive across epochs, [TorchTrainingLoopSummary](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:157) stores the loop result, and [TorchTrainingEpochResult](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:144) stores one epoch loss summary.

```text
GalaxyDatasetSplits
  -> train samples only
  -> for each epoch
  -> for each train sample
  -> forward pass
  -> CrossEntropyLoss
  -> loss.backward()
  -> optimizer.step()
  -> track average real loss
```

Validation and test samples are loaded, but they are not used for weight updates.

### Step 6 Current Behavior

With `--epochs 2` and one usable train sample, the loop runs two real optimizer steps:

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
```

For this tiny sample, the loss decreased over repeated updates and the correct `spiral` probability increased.

### Step 6 Code Flow Diagram

The visual diagram for the tiny real PyTorch training loop is in [galaxy_pytorch_training_loop_flow.excalidraw](excalidraw/galaxy_pytorch_training_loop_flow.excalidraw).

The diagram links to the loop code, epoch result objects, tensor conversion, training concepts, and run observations.

### Step 6 Boundary

Step 6 proves:

```text
the project can run a real PyTorch training loop shape
epochs repeat passes over the train split
one model and one optimizer persist across loop steps
real loss can be tracked per epoch and overall
validation/test samples remain read-only
```

Step 6 does not do:

```text
real dataset download
mini-batches
useful validation accuracy
checkpoint saving
classifier service using a trained model
```

## Step 7: Tiny PyTorch Evaluation Proof

Step 7 covers milestone 36. It extends Step 6 from "train loop only" to "train, then evaluate the current model weights without updating them."

This is still tiny and local. The evaluation shape is real, but the current val/test splits have zero usable image files, so real validation accuracy is not available yet.

Code touchpoints: [evaluate_torch_split()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:682) runs read-only predictions for one split, [evaluate_torch_model()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:761) evaluates train/val/test with the same trained model, [TorchEvaluationSplitResult](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:205) stores accuracy/loss metrics, and [print_torch_evaluation_results()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:1122) prints the metrics.

```text
trained TinyGalaxyCNN
  -> model.eval()
  -> torch.no_grad()
  -> train split read-only metrics
  -> val split read-only metrics
  -> test split read-only metrics
  -> no optimizer.step()
```

### Step 7 Current Behavior

With the current tiny data, evaluation can inspect the one train sample. Val and test are reported as empty instead of pretending accuracy exists.

```text
PyTorch evaluation proof
  train: samples=1, correct=1, accuracy=1.0000, average_loss=1.0440
    first_true_label_id=1, first_predicted_label_id=1, first_correct_probability=0.3521
  val: samples=0, correct=0, accuracy=N/A, average_loss=0.0000
  test: samples=0, correct=0, accuracy=N/A, average_loss=0.0000
  status: read-only evaluation completed - no weights updated
```

### Step 7 Code Flow Diagram

The visual diagram for the read-only evaluation pass is in [galaxy_pytorch_evaluation_flow.excalidraw](excalidraw/galaxy_pytorch_evaluation_flow.excalidraw).

The diagram links to evaluation functions, the training loop, and the concept notes for train/validation/test and accuracy.

### Step 7 Boundary

Step 7 proves:

```text
the trained tiny model can be reused for read-only evaluation
evaluation uses model.eval() and torch.no_grad()
accuracy is calculated only when a split has samples
empty val/test splits are reported clearly
evaluation does not update weights
```

Step 7 does not do:

```text
real validation accuracy
full Galaxy Zoo data
checkpoint saving
model selection
classifier service using the trained model
```

## Step 8: Tiny PyTorch Checkpoint Save/Load Proof

Step 8 covers milestone 37. It extends Step 7 from "evaluate the trained model in memory" to "save the trained weights to disk and load them into a fresh model."

This is the first model artifact proof. The checkpoint is still tiny and local, but the flow is the same idea serving will need later: train somewhere, save weights, load weights somewhere else for prediction.

Code touchpoints: [save_torch_checkpoint()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:777) writes a PyTorch `state_dict`, [load_torch_checkpoint()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:800) loads that state into a fresh [TinyGalaxyCNN](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:261), [run_torch_checkpoint_round_trip()](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:820) proves loaded predictions match saved-model predictions, and [TorchCheckpointRoundTripResult](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:233) stores the proof result.

```text
trained TinyGalaxyCNN in memory
  -> model.state_dict()
  -> torch.save(...)
  -> checkpoint file under ignored models/
  -> fresh TinyGalaxyCNN()
  -> load_state_dict(...)
  -> same image produces matching logits/probabilities
```

### Step 8 Current Behavior

The current command saves a tiny checkpoint and proves the loaded model behaves like the trained model on the same sample:

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

### Step 8 Code Flow Diagram

The visual diagram for the checkpoint round-trip is in [galaxy_pytorch_checkpoint_flow.excalidraw](excalidraw/galaxy_pytorch_checkpoint_flow.excalidraw).

The diagram links to checkpoint save/load functions, the training loop, and the concept notes for `state_dict` and model artifacts.

### Step 8 Boundary

Step 8 proves:

```text
trained weights can be saved to disk
a fresh model can load saved weights
the loaded model can reproduce the trained model prediction
the checkpoint path is under models/, which is ignored by git
```

Step 8 does not do:

```text
real useful model checkpoint
checkpoint versioning
MLflow model registry
classifier service loading the checkpoint
cloud artifact storage
```

## Step 9: Tiny Checkpoint Inference Proof

Step 9 covers milestone 38. It extends Step 8 from "checkpoint save/load round-trip inside the training script" to "a separate local inference command loads a checkpoint and predicts one image."

This is still not API serving. It is the local command-line shape that the future galaxy classifier service will reuse conceptually.

Code touchpoints: [predict_from_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:99) loads one sample and checkpoint, [load_sample_for_prediction()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:48) reuses the manifest/image preprocessing path, [predict_sample_from_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:72) runs the loaded model forward pass, and [GalaxyCheckpointPrediction](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:21) stores the prediction result. The command wrapper is [scripts/predict_galaxy_checkpoint.py](vscode://file/opt/projects/cosmosai-platform/scripts/predict_galaxy_checkpoint.py:1).

```text
saved checkpoint file
  -> load_torch_checkpoint()
  -> manifest image_id
  -> existing preprocessing pipeline
  -> loaded TinyGalaxyCNN forward pass
  -> predicted label_id, label, logits, probabilities
```

### Step 9 Current Behavior

The current command loads `models/tiny_galaxy_cnn_baseline.pt` and predicts the tiny sample image:

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

The `manifest_label_for_learning` is printed only for learning and sanity checking. The model prediction comes from the checkpoint weights and the image tensor.

### Step 9 Code Flow Diagram

The visual diagram for checkpoint inference is in [galaxy_checkpoint_inference_flow.excalidraw](excalidraw/galaxy_checkpoint_inference_flow.excalidraw).

The diagram links to the prediction command, checkpoint loader, preprocessing path, and concept notes for inference.

### Step 9 Boundary

Step 9 proves:

```text
a saved checkpoint can be used without retraining
one local command can load checkpoint + image data
the loaded model returns prediction values
the prediction output includes label_id, label, logits, and probabilities
```

Step 9 does not do:

```text
FastAPI classifier service integration
real uploaded image inference
real useful galaxy model
batch inference
model registry
```

## Step 10: Optional Galaxy Service Checkpoint Inference

Step 10 covers milestone 39. It extends Step 9 from "checkpoint inference exists as a local command" to "galaxy-classifier-service can optionally call that same checkpoint inference helper."

The default service behavior is still safe stub mode. The service only uses checkpoint inference when all required local environment variables are configured.

Code touchpoints: [classify()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:160) keeps the public FastAPI endpoint, [classify_with_optional_checkpoint()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:105) tries the optional checkpoint path, [get_checkpoint_inference_config()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:86) reads the environment config, and [predict_from_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:99) runs the shared checkpoint inference path.

```text
POST /classify
  -> classify()
  -> classify_with_optional_checkpoint()
       |
       | if config is missing
       v
     return None -> old stub response

       |
       | if config exists and image_id is present
       v
     predict_from_checkpoint()
       -> manifest + image data
       -> saved checkpoint weights
       -> loaded TinyGalaxyCNN prediction
       -> ClassifyResponse(status="checkpoint_inference")
```

### Step 10 Optional Configuration

The service checks these environment variables:

```text
COSMOSAI_GALAXY_CHECKPOINT_PATH
  path to saved TinyGalaxyCNN checkpoint

COSMOSAI_GALAXY_MANIFEST_PATH
  path to the manifest CSV

COSMOSAI_GALAXY_DATA_ROOT
  folder used to resolve manifest image_path values
```

Example local values:

```bash
export COSMOSAI_GALAXY_CHECKPOINT_PATH=models/tiny_galaxy_cnn_baseline.pt
export COSMOSAI_GALAXY_MANIFEST_PATH=data/samples/galaxy_manifest_sample.csv
export COSMOSAI_GALAXY_DATA_ROOT=data/samples/images
```

### Step 10 Current Behavior

Without environment config, the endpoint still returns the original stub:

```json
{
  "image_id": "demo-galaxy-001",
  "image_uri": null,
  "label": "spiral",
  "confidence": 0.0,
  "status": "stub"
}
```

With environment config and `image_id="gz2-000001"`, the service can return the checkpoint prediction:

```json
{
  "image_id": "gz2-000001",
  "image_uri": null,
  "label": "spiral",
  "confidence": 0.3521,
  "status": "checkpoint_inference"
}
```

The confidence comes from the predicted class probability produced by softmax.

### Step 10 Boundary

Step 10 proves:

```text
galaxy-classifier-service can keep stub fallback behavior
optional checkpoint inference can be enabled with explicit local config
the service can reuse the existing manifest/preprocessing/checkpoint path
tests protect both stub fallback and optional checkpoint inference
```

Step 10 does not do:

```text
Docker Compose checkpoint integration
real uploaded image inference
real useful Galaxy Zoo-trained model
batch inference
model registry
cloud artifact loading
```

## Step 11: Optional Docker Checkpoint Compose Path

Step 11 covers milestone 40. It extends Step 10 from "the service can use checkpoint inference when configured" to "the full Docker Compose service chain can run that optional checkpoint path."

The normal [docker-compose.yml](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:1) stays stub-safe. The checkpoint path is enabled only by adding [docker-compose.checkpoint.yml](vscode://file/opt/projects/cosmosai-platform/docker-compose.checkpoint.yml:1) as an override.

Code touchpoints: [Dockerfile.checkpoint](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/Dockerfile.checkpoint:1) builds a checkpoint-capable galaxy service image, [requirements-checkpoint.txt](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/requirements-checkpoint.txt:1) adds CPU PyTorch, [docker-compose.checkpoint.yml](vscode://file/opt/projects/cosmosai-platform/docker-compose.checkpoint.yml:1) sets the checkpoint paths, and [smoke_test_compose_checkpoint.sh](vscode://file/opt/projects/cosmosai-platform/scripts/smoke_test_compose_checkpoint.sh:1) verifies the full path.

```text
docker-compose.yml
  -> normal service wiring and host ports

docker-compose.checkpoint.yml
  -> overrides only galaxy-classifier-service
  -> uses Dockerfile.checkpoint
  -> copies cosmosai/ shared package into the service image
  -> mounts ./models read-only
  -> mounts ./data/samples read-only
  -> sets checkpoint/manifest/data-root environment variables
```

### Step 11 Runtime Flow

```text
User or smoke test
  -> POST http://127.0.0.1:8000/route
  -> api-gateway
  -> inference-router
  -> galaxy-classifier-service container
  -> classify()
  -> classify_with_optional_checkpoint()
  -> predict_from_checkpoint()
  -> mounted checkpoint + mounted sample image data
  -> ClassifyResponse(status="checkpoint_inference")
```

### Step 11 Current Behavior

The optional smoke script builds and runs the override stack:

```bash
scripts/smoke_test_compose_checkpoint.sh
```

Current observed galaxy response through the public API gateway:

```json
{
  "input_type": "galaxy_image",
  "selected_service": "galaxy-classifier-service",
  "status": "stub",
  "classification": {
    "image_id": "gz2-000001",
    "image_uri": null,
    "label": "spiral",
    "confidence": 0.352059543132782,
    "status": "checkpoint_inference"
  }
}
```

The outer route status is still `stub` because the router itself is still a simple router. The nested classification status shows the galaxy service used checkpoint inference.

### Step 11 Boundary

Step 11 proves:

```text
the default Docker Compose path stays simple and stub-safe
an optional Compose override can enable checkpoint inference locally
the galaxy service container can access the copied cosmosai/ shared package
the galaxy service container can read mounted sample data and model artifacts
the full api-gateway -> inference-router -> galaxy-classifier-service path works with checkpoint_inference
```

Step 11 does not do:

```text
real useful Galaxy Zoo-trained model
arbitrary uploaded image_uri inference
production model packaging
model registry
cloud deployment
Kubernetes
Triton or edge deployment
```

## Step 12: Shared Galaxy Runtime Package

Step 12 covers milestones 41-42. It refactors the checkpoint inference and training helper path so the CLI commands, training script, and FastAPI galaxy service use shared package code instead of duplicating model/data logic in `scripts/`.

The package is under [cosmosai/galaxy](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/__init__.py:1). This moves reusable runtime code toward normal application modules, while keeping the old command-line scripts available as learning and inspection entrypoints.

### Step 12 Package Flow

```text
scripts/predict_galaxy_checkpoint.py
  -> thin CLI wrapper
  -> cosmosai.galaxy.checkpoint_inference.predict_from_checkpoint()

apps/galaxy-classifier-service/main.py
  -> FastAPI /classify endpoint
  -> classify_with_optional_checkpoint()
  -> cosmosai.galaxy.checkpoint_inference.predict_from_checkpoint()

scripts/train_galaxy_cnn_baseline.py
  -> training CLI and learning output
  -> imports shared dataset/model/checkpoint helpers from cosmosai.galaxy

apps/galaxy-classifier-service/Dockerfile.checkpoint
  -> copies cosmosai/ into the checkpoint-capable service image
```

### Step 12 Shared Modules

```text
cosmosai/galaxy/labels.py
  shared label IDs and reverse label lookup

cosmosai/galaxy/manifest.py
  manifest validation, loading, split grouping, and path resolution

cosmosai/galaxy/image_loader.py
  tiny PPM image loading helpers

cosmosai/galaxy/preprocessing.py
  raw pixel values -> normalized tensor-like values

cosmosai/galaxy/training_sample.py
  manifest record + tensor + label_id sample object

cosmosai/galaxy/dataset_splits.py
  train/val/test sample buckets used by the training script

cosmosai/galaxy/model.py
  TinyGalaxyCNN, tensor conversion, checkpoint save/load, forward pass helpers

cosmosai/galaxy/checkpoint_inference.py
  checkpoint + manifest + image_id -> prediction result
```

### Step 12 Code Touchpoints

Shared inference:

- [GalaxyCheckpointPrediction](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:21)
- [load_sample_for_prediction()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:48)
- [predict_sample_from_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:72)
- [predict_from_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:99)

Shared model helpers:

- [TinyGalaxyCNN](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/model.py:62)
- [galaxy_tensor_to_torch_image()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/model.py:101)
- [run_torch_forward_pass()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/model.py:131)
- [save_torch_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/model.py:170)
- [load_torch_checkpoint()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/model.py:192)

Shared dataset helpers:

- [GalaxyDatasetSplits](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/dataset_splits.py:22)
- [create_dataset_splits()](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/dataset_splits.py:41)

Entrypoints using shared inference:

- [CLI wrapper](vscode://file/opt/projects/cosmosai-platform/scripts/predict_galaxy_checkpoint.py:1)
- [service checkpoint helper](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:106)
- [checkpoint Dockerfile](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/Dockerfile.checkpoint:1)

Entrypoints using shared training/model helpers:

- [training CLI wrapper](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:1)
- [dataset split CLI wrapper](vscode://file/opt/projects/cosmosai-platform/scripts/load_galaxy_dataset_splits.py:1)

### Step 12 Boundary

Step 12 proves:

```text
checkpoint inference code can live in a shared importable package
the CLI prediction script can reuse the package helper
the galaxy FastAPI service can reuse the same package helper
the training script can reuse shared dataset/model/checkpoint helpers
the dataset split script can stay runnable while re-exporting package logic
the optional checkpoint Docker image no longer needs to copy scripts/
the checkpoint route still keeps stub fallback behavior
```

Step 12 does not do:

```text
full cleanup of all older learning CLI scripts
moving the whole training loop into package code
real dataset download
real useful Galaxy Zoo-trained model
production model packaging
cloud, Kubernetes, Triton, or edge deployment
```

## Step 13: Shared Galaxy Package Review Checkpoint

Step 13 covers milestone 43. It is a documentation checkpoint after the shared-package/OOP refactor, not a new runtime feature.

The detailed visual map is in [[excalidraw/shared_galaxy_package_oop_flow.excalidraw|shared_galaxy_package_oop_flow.excalidraw]]. It explains the module dependencies, object models, function call graph, sequence flow, and lazy import path with clickable links back to the Python files.

### Step 13 Review Focus

```text
scripts are now mostly command-line entrypoints for humans
cosmosai.galaxy is where reusable package logic lives
dataclasses carry structured data between pipeline steps
TinyGalaxyCNN is the current small PyTorch model class
the galaxy service imports checkpoint inference lazily only when optional checkpoint config is present
```

### Step 13 Code Touchpoints

- [training CLI imports shared helpers](vscode://file/opt/projects/cosmosai-platform/scripts/train_galaxy_cnn_baseline.py:36)
- [shared dataset split object](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/dataset_splits.py:22)
- [shared tiny CNN model](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/model.py:62)
- [shared checkpoint prediction object](vscode://file/opt/projects/cosmosai-platform/cosmosai/galaxy/checkpoint_inference.py:24)
- [lazy checkpoint import in service](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:106)

### Step 13 Boundary

Step 13 proves:

```text
the refactored code structure is documented visually
the documentation links back to concrete files, classes, and functions
the project has a review point before adding more model complexity
```

Step 13 does not do:

```text
new model behavior
more training data
accuracy improvement
production deployment
```

## High-level architecture

```text
User / Frontend
      ↓
FastAPI API Gateway
      ↓
Inference Router
      ├── Galaxy Classifier Service
      ├── Stellar Classifier Service
      └── RAG Assistant Service
```

Current local Docker Compose ports:

Service definitions are in [docker-compose.yml](vscode://file/opt/projects/cosmosai-platform/docker-compose.yml:1).

```text
api-gateway                  -> http://127.0.0.1:8000
inference-router             -> http://127.0.0.1:8001
galaxy-classifier-service    -> http://127.0.0.1:8002
stellar-classifier-service   -> http://127.0.0.1:8003
```

Current public entry point:

[API Gateway route()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:67) handles this public entry point.

```text
POST http://127.0.0.1:8000/route
```

## Main services

### [API Gateway](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:1)

Public entry point for the application.

Code entry points: [health()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:59), [route()](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:67), [INFERENCE_ROUTER_URL](vscode://file/opt/projects/cosmosai-platform/apps/api-gateway/main.py:22).

Responsibilities:

- expose external API endpoints
- validate requests
- call the inference router
- return responses to the user

Current endpoints:

```text
GET  /health
POST /route
```

Current route request examples:

```json
{"input_type":"galaxy_image","image_id":"demo-galaxy-001"}
{"input_type":"galaxy_image","image_uri":"file:///tmp/demo-galaxy.jpg"}
{"input_type":"stellar_spectrum","spectrum_id":"demo-star-001"}
{"input_type":"stellar_spectrum","spectrum_uri":"file:///tmp/demo-spectrum.csv"}
```

### [Inference Router](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:1)

Routes requests to the correct AI service.

Code entry points: [health()](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:62), [route()](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:70), [SERVICE_BY_INPUT_TYPE](vscode://file/opt/projects/cosmosai-platform/apps/inference-router/main.py:22).

```text
image request        -> galaxy-classifier-service
stellar features     -> stellar-classifier-service
astronomy question   -> rag-assistant-service
```

Current behavior:

```text
galaxy_image      -> calls galaxy-classifier-service /classify
stellar_spectrum  -> calls stellar-classifier-service /classify
unknown input     -> returns unsupported
```

Current response shape:

```json
{
  "input_type": "galaxy_image",
  "selected_service": "galaxy-classifier-service",
  "status": "stub",
  "classification": {
    "image_id": "demo-galaxy-001",
    "image_uri": null,
    "label": "spiral",
    "confidence": 0.0,
    "status": "stub"
  }
}
```

### [Galaxy Classifier Service](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:1)

Input:

Code entry points: [health()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:152), [classify()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:160), [classify_with_optional_checkpoint()](vscode://file/opt/projects/cosmosai-platform/apps/galaxy-classifier-service/main.py:105).

```text
galaxy image
```

Output:

```text
galaxy morphology class + confidence
```

Possible classes:

```text
spiral
elliptical
irregular
```

Model approach:

```text
MVP: simple PyTorch CNN or pretrained model
Later: fine-tuned ViT / ResNet
```

Current endpoints:

```text
GET  /health
POST /classify
```

Current default stub response:

```json
{
  "image_id": "demo-galaxy-001",
  "image_uri": null,
  "label": "spiral",
  "confidence": 0.0,
  "status": "stub"
}
```

Optional local checkpoint response:

```json
{
  "image_id": "gz2-000001",
  "image_uri": null,
  "label": "spiral",
  "confidence": 0.3521,
  "status": "checkpoint_inference"
}
```

The optional checkpoint path is local-only for now and requires explicit environment variables. Without that config, the service remains in stub mode.

### [Stellar Classifier Service](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:1)

Input:

Code entry points: [health()](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:40), [classify()](vscode://file/opt/projects/cosmosai-platform/apps/stellar-classifier-service/main.py:48).

```text
stellar features or spectrum data
```

Output:

```text
stellar spectral class
```

Possible classes:

```text
O / B / A / F / G / K / M
```

Model approach:

```text
MVP: tabular baseline / PyTorch MLP
Later: 1D CNN over spectral data
```

Current endpoints:

```text
GET  /health
POST /classify
```

Current stub response:

```json
{
  "spectrum_id": "demo-star-001",
  "spectrum_uri": null,
  "spectral_type": "G",
  "confidence": 0.0,
  "status": "stub"
}
```

### RAG Assistant Service

Input:

```text
astronomy question
```

Output:

```text
grounded answer using retrieved astronomy documents
```

Planned components:

```text
Qdrant vector database
embeddings
LLM provider through OpenAI API / LiteLLM
LangChain or LangGraph
```

## Infrastructure layer

Planned infrastructure:

```text
Docker
Docker Compose
MLflow
Prometheus
Grafana
Cloud Run
Kubernetes later
Triton / vLLM later
```

## Build principle

Start small:

```text
1. local FastAPI service
2. Dockerized API Gateway
3. Docker Compose
4. inference-router service
5. classifier service skeletons
6. stub request/response contracts
7. service-to-service routing
8. real model baseline
9. RAG
10. observability
11. cloud deployment
12. Kubernetes / advanced serving
```
