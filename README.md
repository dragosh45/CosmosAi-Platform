# CosmosAI Inference Platform

CosmosAI is a portfolio AI project that uses **astronomy** as the domain for building a small, production-style **AI inference platform**.

The project is designed to demonstrate both applied AI and AI infrastructure skills: model serving, API design, multi-model routing, RAG, observability, Docker, and cloud/Kubernetes deployment.

## What the project does

CosmosAI will support three main AI use cases:

```text
1. Galaxy image classification
   Upload a galaxy image → predict galaxy type

2. Stellar spectral classification
   Provide stellar features/spectrum data → predict stellar class

3. Astronomy RAG assistant
   Ask astronomy questions → retrieve context → generate grounded answer
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
              ↓
        Qdrant Vector DB + LLM
```

## Tech stack

### Backend and APIs

- Python
- FastAPI
- Pydantic
- Uvicorn

### Machine Learning

- PyTorch
- torchvision
- scikit-learn for metrics and baselines
- pandas / NumPy / matplotlib

### LLM and RAG

- LangChain / LangGraph
- Qdrant
- sentence-transformers
- OpenAI API / LiteLLM

### MLOps and observability

- MLflow
- Prometheus
- Grafana
- Langfuse

### Infrastructure and deployment

- Docker
- Docker Compose
- GitHub Actions
- Google Cloud Run
- Kubernetes / GKE later
- Triton / vLLM later

## Planned services

```text
apps/
  api-gateway/
  inference-router/
  galaxy-classifier-service/
  stellar-classifier-service/
  rag-assistant-service/

infra/
  docker/
  k8s/
  monitoring/

mlops/
  training/
  evaluation/
  mlflow/
```

## Current build status

The project is currently in the initial setup phase.

First milestone:

```text
FastAPI API Gateway
GET /health endpoint
Dockerfile
local run instructions
first working commit
```

## Project goal

The goal is to build more than a simple AI demo.

CosmosAI is intended to show how AI models can be packaged, served, routed, monitored, versioned, and deployed like production systems.

In short:

```text
AI application
+
model-serving infrastructure
+
cloud-ready deployment
```
