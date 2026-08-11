# CosmosAI Master's Project Integration

This document maps possible Applied Artificial Intelligence in Mechatronics and Robotics master's topics onto CosmosAI.

It is a companion to:

```text
docs/portfolio_ai_project_plan_ultimate.md
```

The portfolio roadmap remains canonical. This file is not a second roadmap.

## Purpose

CosmosAI is primarily a job-market / AI-infrastructure portfolio project. The master's programme can still be useful if academic work deepens the same technical direction instead of creating unrelated duplicate projects.

Good master's integration means:

```text
learn the academic concept
build something technically real
strengthen CosmosAI
preserve the AI engineering / AI infrastructure story
```

## Current Source Status

The course/topic list is provisional. It is based on planning notes, not confirmed full syllabi.

Before using CosmosAI for a course project, check:

```text
official syllabus
official project brief
required algorithms or tools
grading rubric
whether the course actually requires a project
whether the work is individual or team-based
hardware or software constraints
AI-assistance disclosure policy
```

Not every master's course will require a project. If a course has no project, this document can still guide learning, but it should not invent work just to force a match.

## Core Rule

```text
Does the master's topic naturally belong to an existing CosmosAI phase?
  -> yes: deepen that phase
  -> no: does it add useful AI / hardware / research value?
       -> yes: optional CosmosAI extension
       -> no: separate university project
```

Do not force robotics, SCADA, automotive, safety, or biomimetic topics into CosmosAI if the technical overlap is weak.

## Current CosmosAI Baseline

Already implemented:

```text
api-gateway
  -> inference-router
      -> galaxy-classifier-service stub
      -> stellar-classifier-service stub

galaxy data/model proof path:
  manifest validation/loading
  tiny image loading
  preprocessing
  model-ready samples
  train/val/test split buckets
  TinyGalaxyCNN
  forward pass
  CrossEntropyLoss
  backpropagation
  optimizer updates
  tiny training loop
  read-only evaluation
  checkpoint save/load
  checkpoint inference command
```

Still future:

```text
real Galaxy Zoo training
useful validation/test accuracy
stellar model
RAG
LangGraph orchestration
MLflow
GPU/CUDA benchmarks
ONNX/TensorRT/Triton
Kubernetes
edge deployment
multimodal data fusion
```

## Mapping Matrix

| Course / topic | Classification | CosmosAI fit |
|---|---|---|
| Systems and Applications for Data Processing | DIRECT INTEGRATION | Deepen the existing data validation, preprocessing, splits, statistics, and dataset-quality path. |
| Machine Vision & AI | DIRECT INTEGRATION | Galaxy morphology classification is a natural image-classification project. |
| Artificial Intelligence Applications | DIRECT INTEGRATION | Galaxy/stellar supervised learning fits a broad AI applications course. |
| Intelligent Systems Architectures | DIRECT INTEGRATION | The service topology is already a distributed intelligent-system architecture. |
| Software & Hardware Architectures | DIRECT INTEGRATION | CPU/GPU/ONNX/Triton benchmarking fits hardware-aware AI work. |
| Embedded AI | OPTIONAL COSMOSAI EXTENSION | ONNX/TensorRT/Jetson inference can be coherent after a useful model exists. |
| Human-Machine Interaction / Advanced Interfaces | OPTIONAL COSMOSAI EXTENSION | A future UI could show predictions, confidence, explanations, Grad-CAM, and RAG sources. |
| AI for Data Security | OPTIONAL COSMOSAI EXTENSION | Could study adversarial examples, API abuse, or RAG prompt-injection if the syllabus allows. |
| New Technologies in AI & Robotics | SYLLABUS-DEPENDENT | Could fit RAG, LangGraph, GPU, edge, or multimodal work depending on the semester topic. |
| Multi-Agent Learning | SYLLABUS-DEPENDENT | LangGraph/tool agents are not automatically Multi-Agent Reinforcement Learning. |
| Sensor Fusion | SYLLABUS-DEPENDENT | Astronomy multimodal fusion is real, but robotics courses may require Kalman/state estimation. |
| Virtual Engineering | SYLLABUS-DEPENDENT | Could fit deployment simulation or digital-twin-style performance modeling only if allowed. |
| Biomimetic Design | SEPARATE PROJECT | Weak fit for an astronomy inference platform. |
| SCADA | SEPARATE PROJECT | Prometheus/Grafana observability is not SCADA. |
| Automotive Sensor Networks | SEPARATE PROJECT | CAN/ECU networking does not belong in CosmosAI. |
| Functional Safety | SEPARATE PROJECT | API reliability is not ISO-style functional safety. |
| AI & ML in CAE/CAM | SEPARATE PROJECT | Manufacturing/engineering simulation is not the CosmosAI domain. |

## Direct Integrations

### Galaxy Vision Research Track

Possible course fit:

```text
Machine Vision & AI
Artificial Intelligence Applications
Systems and Applications for Data Processing
```

Candidate academic project:

```text
Comparative Deep Learning Approaches for Galaxy Morphology Classification
```

CosmosAI components reused:

```text
manifest validation
image loading/preprocessing
dataset splits
TinyGalaxyCNN proof path
future MLflow/evaluation path
```

Future additions:

```text
real Galaxy Zoo ingestion
dataset statistics
class imbalance analysis
augmentation
CNN baseline
ResNet/EfficientNet transfer learning
ViT / LoRA experiment
accuracy, precision, recall, F1
confusion matrix
Grad-CAM or another explainability view
```

### Intelligent-System Architecture Track

Possible course fit:

```text
Intelligent Systems Architectures
Software & Hardware Architectures
New Technologies in AI & Robotics, if the syllabus fits
```

Candidate academic project:

```text
Architecture and Evaluation of a Distributed Multi-Model Astronomy Intelligence Platform
```

CosmosAI components reused:

```text
api-gateway
inference-router
galaxy-classifier-service
stellar-classifier-service
future RAG assistant
Docker Compose
service contracts
health checks
```

Future additions:

```text
fault handling
model version metadata
service dependency diagrams
observability
scalability study
deployment comparison
```

## Optional CosmosAI Extensions

### GPU / Hardware-Aware AI Track

Possible course fit:

```text
Software & Hardware Architectures
Embedded AI, if the syllabus fits
New Technologies in AI & Robotics, if the syllabus fits
```

Candidate project:

```text
Hardware-Aware Performance Analysis of Deep Learning Inference for Astronomical Image Classification
```

Progression:

```text
PyTorch CPU
  -> PyTorch CUDA
  -> profiling
  -> mixed precision
  -> ONNX
  -> TensorRT
  -> Triton
```

Metrics:

```text
training time per epoch
inference latency
throughput
batch-size scaling
RAM / VRAM use
model size
accuracy impact after optimization
```

Local NVIDIA GPU is not required. A cloud NVIDIA GPU machine is acceptable and may be better for learning production-style infrastructure.

### Edge / Embedded AI Track

Candidate project:

```text
Edge Deployment and Optimization of a Galaxy Vision Model
```

Possible flow:

```text
trained galaxy checkpoint
  -> ONNX
  -> TensorRT
  -> Jetson or other edge target
  -> benchmark local inference
```

This should wait until a useful model exists. Do not add Jetson/TensorRT/C++ code before there is a concrete edge milestone.

### Multimodal Astronomy / Data Fusion Track

Candidate project:

```text
Multimodal Classification of Astronomical Objects Using Imaging, Spectral, and Photometric Data
```

Possible architecture:

```text
image encoder --------┐
spectrum encoder -----|
redshift -------------|-> fusion model
photometry -----------┘
```

This is scientifically coherent as multimodal astronomy. It is not automatically valid for a robotics Sensor Fusion course, because that may require Kalman filters, IMU/GPS/radar/camera fusion, or state estimation.

### Human-AI Interface Track

Candidate project:

```text
Interactive Explainable Astronomy AI Interface
```

Possible features:

```text
upload image/spectrum
show predicted class and confidence
show uncertainty
show Grad-CAM or visual explanation
show RAG-grounded explanation with sources
collect user feedback
```

## Syllabus-Dependent Topics

### Multi-Agent Learning

CosmosAI may later use LangGraph for LLM/tool orchestration. That is valuable engineering work, but it is not automatically academic Multi-Agent Learning.

Important distinction:

```text
LangGraph software/tool agents
  !=
Multi-Agent Reinforcement Learning
```

If the course is MARL, likely separate project ideas are better:

```text
multi-robot coordination
telescope scheduling agents
cooperative observation planning
traffic/resource allocation
```

### Sensor Fusion

CosmosAI multimodal astronomy could be valid only if the course accepts learned multimodal fusion. If the course requires robotics state estimation, create a separate project.

### Virtual Engineering

Possible only if deployment simulation, digital-twin-style system modeling, or virtual performance/cost analysis is accepted.

## Topics To Keep Separate By Default

Keep these outside CosmosAI unless an official brief creates a very clear bridge:

```text
Biomimetic Design
SCADA
Automotive Sensor Networks
Functional Safety
CAE/CAM-specific AI
robot control / actuators
MARL if CosmosAI cannot provide a real learning-agent environment
robotics Sensor Fusion if Kalman/state estimation is required
```

Separate focused university projects are better than weakening CosmosAI with artificial integrations.

## Potential Dissertation Direction

Strong long-term direction:

```text
CosmosAI as a production-oriented multimodal astronomy intelligence platform
```

Possible research/engineering question:

```text
How do model architecture, hardware backend, and serving topology affect accuracy,
latency, cost, and explainability for multimodal astronomy AI?
```

This could combine:

```text
galaxy vision model
stellar spectral model
RAG explanations
GPU/Triton serving
observability
optional multimodal fusion
```

## AI-Assisted Development Note

CosmosAI uses Codex/AI assistance for planning, code generation, debugging, tests, documentation, and concept explanation.

Preferred framing:

```text
AI suggestions are reviewed, run, tested, and understood before acceptance.
```

For university work, follow the university AI-use policy and disclose AI assistance when required.

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-08-11 | Keep `portfolio_ai_project_plan_ultimate.md` canonical | The project remains job-market / AI-infrastructure driven. |
| 2026-08-11 | Add `master_project_integration.md` as companion mapping | Master's work can deepen CosmosAI without becoming a competing roadmap. |
| 2026-08-11 | Preserve implemented service topology | The repo already uses api-gateway -> inference-router -> domain classifier services. |
| 2026-08-11 | Add CPU -> CUDA -> profiling -> ONNX/TensorRT -> Triton learning path | Safer learning progression before production GPU serving. |
| 2026-08-11 | Keep edge and multimodal tracks optional | They are coherent but should wait for useful models and official course requirements. |
