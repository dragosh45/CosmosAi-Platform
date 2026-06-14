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

The project has completed the local service foundation step.

Current completed foundation:

```text
FastAPI API Gateway
Inference Router
Galaxy Classifier Service stub
Stellar Classifier Service stub
Docker Compose local service network
Mocked service-to-service routing
Basic pytest service contract tests
```

## Run the API Gateway locally

```bash
cd /opt/projects/cosmosai-platform
source .venv/bin/activate
cd apps/api-gateway
uvicorn main:app --reload
```

Health check:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","service":"api-gateway"}
```

## Run local tests

Install the lightweight development test dependency:

```bash
cd /opt/projects/cosmosai-platform
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run the service contract tests:

```bash
cd /opt/projects/cosmosai-platform
source .venv/bin/activate
pytest
```

Expected result:

```text
4 passed
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
