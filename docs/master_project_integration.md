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

Updated 9 September 2026: the **official AAIMR 2025-2027 curriculum** has now been checked, including its project-hour columns. The full course/assessment evidence and research limits are in [Craiova AAIMR course research](craiova_master_course_research.md). Your entry cohort and the applicability of this plan still need confirmation.

Course existence and scheduled project activity are now source-backed; **the mapping of CosmosAI to an accepted assignment remains provisional**. No current complete assignment briefs or grading rubrics were located. Not all courses allocate separate project hours, and laboratory work can still be assessed.

The research also corrects two earlier assumptions: the architecture and embedded-AI options are specifically **automotive architectures** and **autonomous driving**, not generic software architecture or edge inference. A benchmark alone may not satisfy them.

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

For the maintained implementation snapshot, read [current_status.md](current_status.md). M48 feeds the real training loop through DataLoader; M49 saves the checkpoint's input recipe and verifies consistent CLI/API preparation; M50 makes malformed manifests and accepted/skipped/rejected image rows explicit. This supports reproducible experiments, but is not yet a real-data academic result. The lists below distinguish mechanics already demonstrated from future useful models.

Learning addition approved 16 September 2026: [M50-L1/L2 CNN Learning Replay](cnn_learning_replay_plan.md), before M51. A measured trace and local HTML/JavaScript viewer will support worked convolution, derivative and SGD exercises alongside implementation. It is planned teaching material, potentially an appendix to a course submission, not a confirmed mandatory project or evidence of useful accuracy. The preferred academic investigation remains two models on identical real-data splits with error analysis and accuracy/latency measurements, subject to the actual brief. This addition does not replace that investigation or introduce robotics/hardware scope.

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
  DataLoader-driven training (already-loaded samples, not lazy image files)
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

DIRECT below means a natural technical overlap, **not confirmed course-project acceptance**. Full abbreviated curriculum, optional variants, credits and project hours: [[craiova_master_course_research#Confirmed Curriculum|research tables]].

| Course / topic | Classification | CosmosAI fit |
|---|---|---|
| Systems and Applications for Data Processing | DIRECT INTEGRATION | Deepen the existing data validation, preprocessing, splits, statistics, and dataset-quality path. |
| Machine Vision & AI | DIRECT INTEGRATION | Galaxy morphology classification is a natural image-classification project. |
| Artificial Intelligence Applications | DIRECT INTEGRATION | Galaxy/stellar supervised learning fits a broad AI applications course. |
| Intelligent Systems Architectures in Mechatronics and Robotics | SYLLABUS-DEPENDENT | Service boundaries and inference flow offer a candidate study; robotics/control requirements may not fit. |
| System, Software and Hardware Architectures in Automotive | SYLLABUS-DEPENDENT | Optional automotive course; a generic CPU/GPU benchmark is not automatically sufficient. |
| Embedded AI in Autonomous Driving | SYLLABUS-DEPENDENT | Optional driving-focused course; edge inference is a candidate only if domain/hardware rules allow it. |
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

### Delivery And Academic Reuse Rule

Delivery is now **R1 galaxy classifier -> R2 astronomy assistant -> R3 serving/performance**, as defined in the [roadmap](portfolio_ai_project_plan_ultimate.md). The stellar model remains a later extension, not a blocker for R1. General AI software jobs matter alongside master's reuse: retrieval, LLM APIs, evaluation, tests, observability and deployment have their own release evidence.

The preferred academic proposal is the [[craiova_master_course_research#Recommended CosmosAI Investigation|bounded two-model investigation]]: same object-level splits, a small CNN versus one transfer-learning model, validation-based selection, error analysis and measured accuracy/latency tradeoffs. Deliver configurations, commands, metrics, limitations and a report, not just a UI or an architecture diagram.

This study can extend a useful galaxy baseline; it does not require all optional tools or a stellar model. Before treating it as coursework, obtain the full brief, rubric, hardware constraints and approval to reuse existing work. Record the distinct new contribution for each submission and applicable AI-assistance disclosure rules. Hardware/control-specific assignments can remain separate projects.

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

Bounded additions for the proposed study (subject to the assignment):

```text
real Galaxy Zoo ingestion
dataset statistics
class imbalance analysis
augmentation
CNN baseline
one transfer-learning comparison model
accuracy, precision, recall, F1
confusion matrix
inference latency under a documented measurement setup
optional Grad-CAM or another explainability view
```

### Intelligent-System Architecture Track

Possible course fit:

```text
Intelligent Systems Architectures
Automotive system/software/hardware architectures, only if assignment permits
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
Automotive system/software/hardware architectures, if the syllabus fits
Embedded AI in autonomous driving, if domain/hardware constraints fit
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
