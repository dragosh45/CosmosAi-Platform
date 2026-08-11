# Portfolio AI Project — CosmosAI: Astronomy Intelligence System

**Author:** Dragos-Nicolae Preda
**Created:** June 2026
**Status:** Active development
**Current implementation focus:** galaxy classifier model/inference integration
**Roadmap:** evolving with verified implementation milestones
**Companion to:** `AI_skills_through_self_learn_cannot_do_atwork.md` (skills this covers)
**Also covers:** `master/general_all_masters/project/master_decide_project.md` (replaces the plain CNN project)

---

## Master's Integration

CosmosAI may also become the technical foundation for suitable projects in the Applied Artificial Intelligence in Mechatronics and Robotics master's programme.

This document remains the canonical job-market / AI-infrastructure roadmap. Master's work should deepen CosmosAI only when the overlap is real, useful, and coherent. Detailed course/project mappings live in:

```text
docs/master_project_integration.md
```

Important rule:

```text
course/topic overlaps a current CosmosAI phase -> deepen that phase
course/topic adds useful AI/hardware/research depth -> optional CosmosAI extension
course/topic is unrelated or syllabus-specific -> separate university project
```

Do not assume every master's course requires a project. When a course does require one, compare the official syllabus, assignment brief, required technologies, and grading rubric before deciding whether CosmosAI is the right base.

---

## What This Is

A publicly deployed AI system with two classification tasks: galaxy morphology (from images) and stellar spectral type (from spectrum data). Both feed into the same RAG + LLM pipeline that answers natural language questions about astronomy, grounded in retrieved scientific literature.

Two models, one router, one generation pipeline — this is the multi-model serving architecture the AI infrastructure job market requires.

**The user experience in 60 seconds — two paths:**

**Path A — Galaxy image:**
1. Upload a galaxy image (from NASA/SDSS/personal)
2. See: galaxy type (Elliptical / Spiral / Lenticular / Irregular) + confidence score
3. See: LLM explanation streaming — *"This is a barred spiral galaxy (SBb type). The central bar is visible..."*
4. Ask: *"What would happen to this galaxy in 5 billion years?"*
5. Get a RAG-grounded answer from retrieved papers + Wikipedia

**Path B — Stellar spectrum:**
1. Upload a stellar spectrum CSV (wavelength + flux columns, from SDSS)
2. See: spectral type (O / B / A / F / G / K / M) + confidence score
3. See: LLM explanation streaming — *"This is a G-type main sequence star, like our Sun. Surface temperature ~5,500K..."*
4. Ask: *"How long will a G-type star like this remain on the main sequence?"*
5. Get a RAG-grounded answer from retrieved stellar physics papers

That's the demo. Two input types, two fine-tuned models, one routing layer, one generation pipeline — it answers every job interview question about multi-model AI in production.

---

## Why This Project (Not ContabilAI, Not a Ticket Labeler)

| Criterion | Ticket labeler | ContabilAI | CosmosAI |
|---|---|---|---|
| Public demo URL | No (PDQ data) | Possible | ✅ Yes |
| Genuinely interesting domain | No | Somewhat | ✅ Astronomy = rare |
| Covers LLM deployment gap | Partially | Yes | ✅ Yes, fully |
| Works for master's applications | No | No | ✅ Yes — Leiden + all others |
| Differentiates from other AI candidates | No | Somewhat | ✅ Multi-model routing + RAG + LLM + infra = rare combo |
| Not tied to PDQ work | No | Yes | ✅ Yes — 100% personal |

---

## Skill Gap Coverage

Every skill listed in `AI_skills_through_self_learn_cannot_do_atwork.md` is mapped here.

### Priority 1 — All covered

| Skill (from self-learn doc) | How CosmosAI covers it | Component |
|---|---|---|
| **1. LangChain** | Orchestrates the full RAG chain: document loader → text splitter → embeddings → retriever → LLM chain | Phase 3 |
| **2. LangGraph** | Multi-node agent: router detects input type → galaxy or stellar classifier → retrieve → generate → evaluate confidence loop | Phase 3 |
| **3. LLM APIs production** | Gemini/Anthropic API, streaming, structured output (Pydantic), retries, token tracking — deployed at Cloud Run URL | Phase 3+4 |
| **4. Vector databases** | Qdrant Cloud stores embeddings of arXiv papers + Wikipedia galaxy articles | Phase 2 |
| **5. RAG full pipeline** | Ingest arXiv PDFs → chunk by section → embed with sentence-transformers → store in Qdrant → retrieve on query → generate answer | Phase 2+3 |
| **6. HuggingFace Transformers** | `sentence-transformers` for document embeddings; `google/vit-base-patch16-224` fine-tuned for galaxy morphology; 1D-CNN for stellar spectra — two distinct architectures | Phase 1+2 |

### Priority 2 — All covered

| Skill | How CosmosAI covers it | Component |
|---|---|---|
| **7. FastAPI** | `POST /classify` (image upload → classification + explanation), `POST /ask` (Q&A), streaming SSE endpoint | Phase 4 |
| **8. Cloud AI services (GCP)** | GCP Cloud Run deployment + Gemini API (same GCP stack as PDQ — relevant experience) | Phase 4 |
| **9. AI Observability — Langfuse** | Tracks every request: image processed, retrieved docs, LLM prompt+response, latency, token cost, user rating | Phase 3 |
| **10. RAG Evaluation — RAGAS** | 20 astronomy Q&A pairs with ground-truth answers; measure faithfulness + answer relevance + context recall. Publish scores in README. | Phase 2 |
| **11. Document ingestion pipelines** | arXiv PDFs (scientific papers) via `pypdf`/`unstructured`; Wikipedia astronomy articles via `wikipedia-api`; chunking strategy for academic content (by section, not fixed tokens) | Phase 2 |
| **12. LlamaIndex** | Evaluate alongside LangChain for the RAG indexing layer — compare retrieval quality. Frame in README as "evaluated both frameworks." | Phase 2 |
| **13. Multi-LLM (LiteLLM)** | LiteLLM gateway: swap between Gemini Flash (free tier, fast), Anthropic Sonnet (quality), Mistral (open) without changing application code | Phase 3 |
| **14. Streaming (SSE)** | FastAPI `StreamingResponse` streams LLM explanation tokens to the frontend as they arrive | Phase 4 |

### Priority 3 — Partially covered (optional depth)

| Skill | How CosmosAI covers it | Priority |
|---|---|---|
| **16. MLflow** | Track CNN and ViT training experiments: architecture variant, hyperparameters, validation accuracy, confusion matrix. Publish best run in README. | Phase 1 — do this |
| **17. Multi-agent systems** | LangGraph graph: ClassifierAgent → RetrievalAgent → SynthesisAgent → optional EvaluationAgent loop | Phase 3 — do this |
| **18. Fine-tuning (LoRA/QLoRA)** | LoRA fine-tuning of `vit-base-patch16-224` on Galaxy Zoo 2 data — shows fine-tuning workflow even on a small model | Phase 1 optional |
| **15. n8n** | Scheduled workflow: fetch new arXiv papers tagged `astro-ph.GA` weekly → trigger re-indexing pipeline | Phase 5 optional |

**Coverage summary: 14 of 18 skills fully demonstrated, 4 partially/optionally.**

### Infrastructure / MLOps — Added for AI Infrastructure Engineer roles

| Skill | How CosmosAI covers it | Component |
|---|---|---|
| **Model serving (vLLM)** | Self-host Phi-3 Mini on a RunPod GPU instance via vLLM; LiteLLM switches from Gemini API to local endpoint with zero code change | Phase 5b |
| **Model serving (Triton/BentoML)** | Export both ViT (galaxy) and 1D-CNN (stellar) to ONNX; serve via Triton on same GPU instance; router directs each request to the correct model | Phase 5b |
| **Multi-model serving + routing** | Two models on one Triton instance; application-layer router detects input type and calls the right Triton model endpoint | Phase 5b |
| **Model registry** | Register both models in Vertex AI Model Registry with version metadata linked to MLflow training runs | Phase 5b |
| **Blue/green ML deployments** | Deploy galaxy_vit_v2 alongside v1 on Vertex AI endpoint, shift traffic 10% → 50% → 100% | Phase 5b |
| **Inference observability** | vLLM exposes Prometheus metrics at `/metrics` (latency p50/p95/p99, tokens/sec, GPU memory); connect to Grafana | Phase 5b |
| **GPU infrastructure ownership** | Raw VM on RunPod: install CUDA runtime, vLLM, Triton — no managed abstraction | Phase 5b |

---

## Architecture

Current implemented baseline:

```text
User / Frontend
      ↓
API Gateway
      ↓
Inference Router
      ├── Galaxy Classifier Service
      ├── Stellar Classifier Service
      └── RAG Assistant Service later
```

Planned mature end-state:

```
                            USER
                              │
               ┌──────────────┴───────────────┐
               │ upload galaxy image (.jpg)    │ upload stellar spectrum (.csv)
               └──────────────┬───────────────┘
                              │ HTTPS
                    ┌─────────▼──────────────────────────┐
                    │   FastAPI Service (GCP Cloud Run)   │
                    │   POST /classify                    │
                    │   POST /ask                         │
                    │   GET  /stream (SSE)                │
                    └─────────┬──────────────────────────┘
                              │
                    ┌─────────▼──────────────┐
                    │   LangGraph Agent       │
                    │                        │
                    │   Node 0: router        │  detects input type
                    └──────┬─────────────────┘
              ┌────────────┴─────────────────┐
          image input                  spectrum input
              │                              │
    ┌─────────▼──────────┐      ┌────────────▼───────┐
    │  Node 1a:          │      │  Node 1b:           │
    │  galaxy_classify   │      │  stellar_classify   │
    │  (ViT fine-tuned)  │      │  (1D-CNN on flux)   │
    │  → Elliptical      │      │  → O B A F G K M    │
    │  → Spiral          │      │    spectral type     │
    │  → Lenticular      │      └──────┬──────────────┘
    │  → Irregular       │             │
    └──────┬─────────────┘             │
           └──────────────┬────────────┘
                          │  {object_type, classification, confidence}
                 ┌────────▼─────────────────┐
                 │  Node 2: retrieve        │
                 │  literature (Qdrant)     │
                 │  galaxy or stellar docs  │
                 └────────┬─────────────────┘
                 ┌────────▼─────────────────┐
                 │  Node 3: generate +      │
                 │  stream (LiteLLM)        │
                 │  Gemini / self-hosted    │
                 └────────┬─────────────────┘
                 ┌────────▼─────────────────┐
                 │  Langfuse                │
                 │  traces, cost, latency   │
                 └──────────────────────────┘

TRAINING (offline, tracked by MLflow):
  Galaxy Zoo 2 images (50k labeled)
    → CNN baseline (PyTorch)
    → ViT fine-tuned with LoRA (HuggingFace + PEFT)
    → Best galaxy model checkpoint saved

  SDSS DR17 stellar spectra
    → 1D-CNN on flux arrays (wavelength → intensity)
    → Classes: O B A F G K M (Harvard spectral types, ~7 classes)
    → Best stellar model checkpoint saved

SERVING (Phase 5b — RunPod GPU):
  Both models exported to ONNX
    → Triton Inference Server hosts galaxy_vit + stellar_cnn
    → Application router calls correct Triton endpoint per request
  vLLM hosts Phi-3 Mini for generation step

INDEXING (offline, optional n8n automation):
  arXiv: astro-ph.GA (galaxy morphology) + astro-ph.SR (stellar physics)
  Wikipedia: galaxy types + Harvard spectral classification articles
    → section-aware chunking → sentence-transformers → Qdrant Cloud
```

---

## Build Phases

### Phase 0 — Setup (0.5 weekend)

**Why each piece:**

| Step | What it is | Why you need it |
|---|---|---|
| **GitHub repo `cosmosai`** | Public code repository | This IS the portfolio piece — a hiring manager clicks this link. MIT license means anyone can use/fork it. |
| **GCP project + Cloud Run** | Cloud Run = Google's serverless container platform. You push a Docker image, it runs at a public URL, scales to zero when no traffic (free). | This is what makes it "deployed in production" instead of "running on my laptop." The URL is the proof. |
| **GCP Artifact Registry** | A private Docker image store on GCP — like Docker Hub but inside your GCP project. | Cloud Run pulls your container image from here. You build locally → push to Artifact Registry → Cloud Run deploys it. |
| **Qdrant Cloud (free tier)** | Qdrant is a vector database — stores embeddings (number arrays that represent meaning). The cloud version is persistent and accessible from Cloud Run. | Your FastAPI service running on Cloud Run needs to query the vector DB. It can't use a local Docker container for that — it needs a cloud-hosted DB it can reach over the internet. |
| **Langfuse (free tier)** | Observability tool for LLM applications — tracks every request: what prompt was sent, what the LLM returned, how many tokens it used, how long it took. | You can't improve what you can't measure. Also: "I have observability on this" is a strong interview answer. |
| **MLflow (local or Dagshub)** | Experiment tracking for ML training runs — logs hyperparameters, metrics, model artifacts per run. | When you train the CNN and ViT, you run multiple experiments. Without MLflow you forget what you tried. With it you have a table: "run 3 was best — 89% accuracy, these params." |
| **Galaxy Zoo 2 dataset** | A citizen science project where 300,000 volunteers manually classified galaxy images. The result is a labeled dataset: each image has a type (Elliptical, Spiral, etc.) with confidence scores. | You need labeled data to train a supervised classifier. Galaxy Zoo 2 is the standard benchmark for galaxy morphology — using it means your results are comparable to published papers. |
| **SDSS DR17 stellar spectra** | The Sloan Digital Sky Survey — a telescope survey with spectra for millions of stars. Each spectrum is a 1D array: wavelength (3800–9200Å) vs flux (brightness). Stars are labeled with spectral type (O/B/A/F/G/K/M). Free via SkyServer SQL query or Kaggle mirrors. | Training data for the stellar classifier. The Harvard spectral classification (O through M) is 100+ years old and well-validated — your model's predictions can be directly compared to known classifications. |

**Concrete steps:**
```bash
# 1. GitHub
gh repo create cosmosai --public --license mit --clone

# 2. GCP (assuming gcloud CLI installed — same as PDQ infra work)
gcloud projects create cosmosai-dragos
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# 3. Qdrant Cloud — go to cloud.qdrant.io, create free cluster, save API key to .env

# 4. Langfuse — go to cloud.langfuse.com, create free account, save keys to .env

# 5. MLflow — run locally (simplest start)
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000   # runs at localhost:5000

# 6. Galaxy dataset
# Go to kaggle.com/datasets/jaimetrickz/galaxy-zoo-the-galaxy-challenge
# Download images_training_rev1.zip (~450MB subset is enough)

# 7. Stellar spectra dataset — Kaggle mirror of SDSS (free, no API key needed)
# kaggle.com/datasets/muhakabartay/sloan-digital-sky-survey-dr16
# Filter to class='STAR', gives you spectra with O/B/A/F/G/K/M labels
# ~10k rows is enough to train a good 1D-CNN
```

---

### Phase 1 — Two Classifiers (1.5 weekends)

**Goal:** Two fine-tuned models: one that classifies galaxy images, one that classifies stellar spectra. Both tracked in MLflow. Different input types, different architectures — this is what justifies the multi-model routing layer in Phase 3 and the Triton setup in Phase 5b.

**Current repository status:** Phase 1 has already started. The repo now contains a tiny but real galaxy PyTorch proof path:

```text
manifest validation/loading
  -> tiny image loading
  -> preprocessing
  -> model-ready samples
  -> train/val/test split buckets
  -> TinyGalaxyCNN forward pass
  -> CrossEntropyLoss
  -> backpropagation
  -> optimizer updates
  -> tiny training loop
  -> read-only evaluation
  -> checkpoint save/load
  -> separate checkpoint inference command
```

This is not yet useful full Galaxy Zoo training. It is the learning/proof foundation that should be preserved and deepened instead of rewritten from scratch.

**Recommended Phase 1 progression from the current code:**

```text
1A. Tiny local proof path — completed through current milestones
1B. Real Galaxy Zoo ingestion and dataset-quality checks
1C. Useful CNN baseline on real train/val/test splits
1D. Metrics, confusion matrix, class imbalance handling, MLflow tracking
1E. Transfer learning baseline, e.g. ResNet/EfficientNet
1F. ViT / LoRA experiment after the baseline is understood
1G. Stellar spectral classifier after the galaxy path is stable
```

**Data:** Galaxy Zoo 2 — 50k images labeled into 4 main types:
- **Elliptical / Smooth** — round, featureless blob, older stars, no structure
- **Spiral** — rotating disk with spiral arms (like the Milky Way)
- **Disk / Lenticular** — disk shape but no spiral arms — between elliptical and spiral
- **Irregular / Merger** — no clear structure, often two galaxies colliding

**What is a CNN (Convolutional Neural Network)?**
A CNN is a type of neural network designed for images. Instead of processing all pixels at once, it slides small filters (e.g. 3×3 pixel windows) across the image to detect edges, then shapes, then higher-level patterns. "3-layer conv" means 3 of these filter layers stacked — each one detects progressively more complex features. It's the standard approach for image classification since 2012. Think of it as: layer 1 detects edges → layer 2 detects curves and corners → layer 3 detects galaxy-level structures (arms, cores, rings).

Why build a baseline CNN first: it's simple, fast to train, gives you a performance number to beat. If CNN achieves 82% accuracy and ViT achieves 89%, you can say "ViT outperforms CNN by 7 points on this task" — that's a concrete, publishable finding.

**What is a ViT (Vision Transformer)?**
A Vision Transformer is a newer approach (2020) that applies the same "attention" mechanism used in ChatGPT/Claude to images. Instead of sliding filters, it cuts the image into patches (e.g. 16×16 pixel squares) and treats each patch like a word in a sentence — learning which patches "attend to" (relate to) each other. `google/vit-base-patch16-224` means: Google's base-size ViT, with 16×16 patches, trained on 224×224 images. It was pre-trained on ImageNet (millions of photos), so it already understands shapes, textures, and objects. You fine-tune it on galaxy images — meaning you start from this pre-trained knowledge and adjust the weights using your galaxy data.

**What is LoRA (Low-Rank Adaptation)?**
Fine-tuning a full ViT model requires a lot of GPU memory — you'd be updating 86 million parameters. LoRA is a technique that freezes most of the model and only trains small "adapter" matrices inserted into each layer. You get 95% of the fine-tuning quality with ~1% of the parameters updated. `lora_r=16` means the rank of those adapter matrices — lower = less memory, less capacity. For galaxy images on a Mac/free Colab, LoRA is what makes fine-tuning feasible.

**What to build:**
```python
# Step 1 — Baseline CNN (PyTorch)
# 3 convolutional layers → max pooling → fully connected → 4 output classes
# Train for 10 epochs on 40k images, validate on 10k
# Target: ~78-84% (measure and publish actual result)

# Step 2 — ViT fine-tuned with LoRA (HuggingFace + PEFT)
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "value"])
model = get_peft_model(model, peft_config)
# Train for 5 epochs — starts from ImageNet knowledge so converges faster
# Target: ~86-92% (measure and publish actual result)

# Step 3 — Metrics with scikit-learn (training eval only, not model building)
from sklearn.metrics import confusion_matrix, classification_report
import mlflow

preds = model.predict(val_images)
print(classification_report(val_labels, preds))  # F1, precision, recall per class
cm = confusion_matrix(val_labels, preds)         # confusion matrix → artifact

# Step 4 — Log both to MLflow
with mlflow.start_run(run_name="cnn-baseline"):
    mlflow.log_metric("val_accuracy", cnn_acc)
    mlflow.log_artifact("cnn_confusion_matrix.png")

with mlflow.start_run(run_name="vit-lora-r16"):
    mlflow.log_param("lora_r", 16)
    mlflow.log_metric("val_accuracy", vit_acc)
    mlflow.log_artifact("vit_confusion_matrix.png")

# Step 5 — Save best model checkpoint
torch.save(best_model.state_dict(), "galaxy_model_checkpoint.pt")
```

**scikit-learn is used only for metrics** (confusion matrix, classification report, train/test split). All model building is PyTorch + HuggingFace.

---

**Model 2 — Stellar Spectral Classifier (0.5 weekend)**

**Data:** SDSS DR17 stellar spectra — ~10k stars labeled O/B/A/F/G/K/M (Harvard spectral sequence, 7 classes). Each sample is a 1D array of ~3800 flux values (one per wavelength bin from 3800Å to 9200Å).

**What the spectral types mean:**
- **O** — hottest, blue, >30,000K (rare, massive, short-lived)
- **B** — blue-white, 10,000–30,000K (Rigel)
- **A** — white, 7,500–10,000K (Sirius, Vega)
- **F** — yellow-white, 6,000–7,500K
- **G** — yellow, 5,200–6,000K (our Sun)
- **K** — orange, 3,700–5,200K (Arcturus)
- **M** — red, coolest, <3,700K (most common, Proxima Centauri)

**What is a 1D-CNN for spectral data?**
Same convolution idea as image CNNs, but on a 1D signal (the spectrum) instead of a 2D image. Filters slide along the wavelength axis and detect absorption line patterns — specific dips in the spectrum that correspond to elements (hydrogen Balmer lines for A-stars, calcium lines for K-stars, etc.). 3 conv layers → pooling → fully connected → 7-class softmax.

```python
# Model 2 — 1D-CNN (PyTorch)
import torch.nn as nn

class StellarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
        )
        self.fc = nn.Sequential(nn.Linear(128*32, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):  # x: (batch, 1, 3800)
        return self.fc(self.conv(x).flatten(1))

# Train for 20 epochs on ~8k samples, validate on 2k
# Expected accuracy: ~92-96% (spectral types are quite distinct in absorption line patterns)

# Log to MLflow
with mlflow.start_run(run_name="stellar-1d-cnn"):
    mlflow.log_param("conv_layers", 3)
    mlflow.log_metric("val_accuracy", stellar_acc)
    mlflow.log_artifact("stellar_confusion_matrix.png")

torch.save(stellar_model.state_dict(), "stellar_model_checkpoint.pt")
```

**Output:** Two trained model checkpoints + MLflow experiments — galaxy (CNN vs ViT) and stellar (1D-CNN)

---

### Master's alignment

Relevant areas:

```text
Machine Vision & AI
Artificial Intelligence Applications
Systems and Applications for Data Processing
```

Possible academic deepening:

```text
Comparative Deep Learning Approaches for Galaxy Morphology Classification
```

This should reuse the current CosmosAI galaxy data/model pipeline. It should not create a second unrelated galaxy classifier unless an official assignment forces a separate structure.

### Phase 2 — RAG Pipeline (1 weekend)

**Goal:** A retrieval system over astronomical literature that answers questions about galaxy types.

**Data sources:**
- arXiv papers: `astro-ph.GA` (galaxy morphology, Hubble sequence, galaxy evolution) + `astro-ph.SR` (stellar physics, spectral classification, stellar evolution) — use `arxiv` Python library, 200-500 papers per category
- Wikipedia: galaxy type articles + Harvard spectral classification + stellar evolution articles via `wikipedia-api`
- Tag each chunk with `domain: galaxy` or `domain: stellar` in metadata — Qdrant retrieval filters on this so a stellar query doesn't pull galaxy papers

**Chunking strategy for scientific papers:**
```python
# Split by section heading, not fixed tokens
# Scientific papers: Abstract → Introduction → Methods → Results → Discussion
# Keep section context in metadata: {"source": "arxiv:2301.12345", "section": "Introduction"}
# Chunk size: ~500 tokens, overlap: 50 tokens
```

**Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (fast, free, runs locally for indexing)

**Vector store:** Qdrant Cloud — free tier, persistent, accessible from Cloud Run

**Evaluation — RAGAS on 20 Q&A pairs:**
```
Q: "What makes a spiral galaxy different from an elliptical?"
Ground truth: "Spiral galaxies have rotating disks with spiral arms..."
Run RAG → measure: faithfulness, answer_relevance, context_recall
Publish scores in README: "RAG pipeline: faithfulness 0.87, context recall 0.79"
```

**LlamaIndex comparison:** Index the same corpus with LlamaIndex, run same 20 queries, compare retrieval quality to LangChain RAGAS scores. 1-2h additional. Shows framework evaluation skill.

---

### Phase 3 — LangGraph Agent + LLM Layer (1 weekend)

**Goal:** Multi-node agent that classifies, retrieves, generates, and streams — with Langfuse tracking.

**Agent state:**
```python
class AstronomyState(TypedDict):
    raw_input: bytes          # image bytes OR spectrum CSV bytes
    input_type: str           # "galaxy" | "stellar" — set by router
    classification: str       # galaxy type OR spectral type
    confidence: float
    retrieved_docs: list[str]
    explanation: str
    follow_up_question: Optional[str]
    answer: Optional[str]
```

**Graph nodes:**
```
Node 0: router
  → inspects input file type / MIME type / shape
  → image (.jpg/.png/.fits) → conditional edge to Node 1a
  → tabular flux array (.csv, 1D) → conditional edge to Node 1b

Node 1a: classify_galaxy
  → runs ViT model on the image
  → sets classification (Elliptical/Spiral/Lenticular/Irregular) + confidence
  → conditional edge: if confidence < 0.65 → Node 1a_retry (low-confidence path)
                       if confidence ≥ 0.65 → Node 2

Node 1a_retry: retrieve_similar_examples (low-confidence path)
  → retrieves 3 similar galaxy images from training set for context
  → re-runs classifier → back to Node 2

Node 1b: classify_stellar
  → runs 1D-CNN model on the flux array
  → sets classification (O/B/A/F/G/K/M) + confidence
  → goes to Node 2 (stellar spectra are distinctive — low-confidence path rarely needed)

Node 2: retrieve_literature
  → queries Qdrant with filter: domain=input_type (galaxy or stellar)
  → top-5 docs relevant to classification
  → sets retrieved_docs

Node 3: generate_explanation
  → LiteLLM call (Gemini Flash by default, or self-hosted vLLM in Phase 5b)
  → prompt: input_type + classification + confidence + retrieved_docs → explanation
  → streams response via SSE

Node 4 (optional loop): evaluate_confidence
  → if explanation contains uncertainty markers → flag for human review
  → otherwise → END
```

**LiteLLM for multi-LLM:**
```python
from litellm import completion

response = completion(
    model="gemini/gemini-1.5-flash",   # or "claude-3-haiku" or "openai/phi3" (self-hosted vLLM)
    messages=[{"role": "user", "content": prompt}],
    stream=True
)
```
Swapping the model string is the only change needed. In Phase 5b, `"openai/phi3"` routes to the self-hosted vLLM endpoint instead of an external API — same application code, zero changes. Shows both multi-LLM competence and the managed-vs-self-hosted trade-off.

**Langfuse integration:**
```python
# Every request tracked: image_id, galaxy_type, retrieved_doc_ids,
# prompt_tokens, completion_tokens, cost_usd, latency_ms, user_rating
langfuse.trace(name="galaxy_classification", input=image_id, output=explanation)
```

---

### Phase 4 — API Gateway + Inference Router + Classifier Services + Docker + Cloud Run (1 week)

**Goal:** Deploy the already-started service topology at a public URL. The API gateway handles external requests, the inference-router chooses the domain service, and each classifier service owns its request contract, preprocessing, model/backend selection, and response shape.

This replaces the older single generic `model-server` assumption in this roadmap. The implemented repository already uses a more explicit topology:

```text
api-gateway/
    -> inference-router/
        -> galaxy-classifier-service/
        -> stellar-classifier-service/
        -> rag-assistant-service/ later
```

**Classifier service endpoints:**
```python
# galaxy-classifier-service/main.py
POST /classify   # image_id/image_uri -> {label, confidence, model_version, status}
GET  /health

# stellar-classifier-service/main.py
POST /classify   # spectrum_id/spectrum_uri -> {spectral_type, confidence, model_version, status}
GET  /health
```

**API gateway endpoints:**
```python
# api-gateway/main.py — calls inference-router over HTTP
POST /route            # request -> router -> correct classifier/RAG service
POST /classify         # later public UX endpoint
POST /ask              # {question} → RAG + LLM → streamed answer
GET  /health
```

**Future model/backend routing inside classifier services:**
```python
# 90% traffic to stable model v1, 10% to experimental model v2
import random
model_version = "v2" if random.random() < 0.10 else "v1"
model = galaxy_models[model_version]
```

**Local dev with Docker Compose:**
```yaml
# docker-compose.yml
services:
  api-gateway:
    build: ./apps/api-gateway
    ports: ["8000:8000"]
  inference-router:
    build: ./apps/inference-router
    ports: ["8001:8000"]
  galaxy-classifier-service:
    build: ./apps/galaxy-classifier-service
    ports: ["8002:8000"]
  stellar-classifier-service:
    build: ./apps/stellar-classifier-service
    ports: ["8003:8000"]
```

**Deploy to Cloud Run:**
```bash
# Deploy internal services first
gcloud run deploy cosmosai-inference-router \
  --image gcr.io/cosmosai/inference-router \
  --no-allow-unauthenticated --memory 4Gi

# Deploy api-gateway (public)
gcloud run deploy cosmosai-api \
  --image gcr.io/cosmosai/api-gateway \
  --allow-unauthenticated --memory 2Gi \
  --set-env-vars INFERENCE_ROUTER_URL=https://cosmosai-inference-router-xxxx.run.app/route
```

**Result:** Public URL → `https://cosmosai-api-xxxx.run.app`. The gateway stays thin, routing stays explicit, and classifier services remain the domain boundary where PyTorch now and Triton later can be selected behind the same API contract.

---

### Phase 5a — Polish + Optional Depth (0.5 weekend)

- [ ] Clean GitHub README: project overview, architecture diagram, demo GIF, RAGAS scores, MLflow best run link
- [ ] n8n workflow (optional): weekly arXiv fetch → trigger `/reindex` endpoint → Qdrant updated automatically
- [ ] Simple frontend: Streamlit or plain HTML with fetch — just enough to demo the upload + streaming
- [ ] CV entry and master's paragraph (see framings below)

---

### Phase 5a.5 — GPU/CUDA Learning Milestone (optional, before production GPU serving)

**Goal:** Learn GPU execution on the project’s own model before jumping to Triton or Kubernetes.

This phase should compare the same galaxy model path on:

```text
PyTorch CPU
  vs
PyTorch CUDA on an NVIDIA GPU
```

Measure:

```text
training time per epoch
inference latency
throughput / images per second
batch-size scaling
RAM vs VRAM use where practical
```

Then optionally compare:

```text
FP32
  vs
FP16 / automatic mixed precision
  vs
ONNX Runtime GPU
  vs
TensorRT
```

Learning progression:

```text
PyTorch CPU
  -> PyTorch CUDA
  -> profiling / mixed precision
  -> ONNX / TensorRT
  -> Triton
  -> Kubernetes GPU scheduling
```

Local NVIDIA hardware is not required. The default development path stays CPU-friendly; cloud NVIDIA GPU machines can be used later for CUDA experiments.

Master's alignment:

```text
Software & Hardware Architectures
Embedded AI / hardware-aware AI, if the official syllabus fits
New Technologies in AI & Robotics, if the semester topic fits
```

---

### Phase 5b — Model Serving Infrastructure + Kubernetes (3 weeks)

**Goal:** Replace the FastAPI model server from Phase 4 with production-grade serving: Triton for vision models, vLLM for the LLM, both running on GPU hardware and orchestrated by Kubernetes. Two sub-stories: raw GPU (RunPod) for learning, managed (Vertex AI) for the production story.

#### Sub-story 1: Self-hosted vLLM on RunPod (0.5 week)

**Why RunPod:** decentralized GPU cloud ($0.20–0.35/hr for RTX 3090, 24GB VRAM). This is the same category of infrastructure that companies like Pragmatike build and operate — not a managed abstraction, but raw GPU compute you configure yourself.

**What to build:**
```bash
# Rent an RTX 3090 on RunPod (hourly, tear down when done)
pip install vllm

# Serve Phi-3 Mini 3.8B (quantized — fits in 6GB VRAM, cheap to run)
vllm serve microsoft/Phi-3-mini-4k-instruct \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85

# vLLM starts an OpenAI-compatible HTTP server
# LiteLLM in Phase 3 calls it with model="openai/phi3-mini" — zero app code change
```

Export **both models** to ONNX and serve via **Triton Inference Server** on the same instance — this is the multi-model serving architecture:
```bash
# Export galaxy ViT to ONNX
torch.onnx.export(galaxy_model, image_dummy, "model_repository/galaxy_vit/1/model.onnx")

# Export stellar 1D-CNN to ONNX
torch.onnx.export(stellar_model, spectrum_dummy, "model_repository/stellar_cnn/1/model.onnx")

# Triton serves both models simultaneously from one server
docker run --gpus all -p 8001:8001 \
  -v ./model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Application router calls correct model:
# image input  → POST http://triton:8001/v2/models/galaxy_vit/infer
# spectrum input → POST http://triton:8001/v2/models/stellar_cnn/infer
```

Configure on the raw VM: GPU memory fraction split between Triton (both vision models) and vLLM (LLM), batching policy per model, max concurrent requests.

**Inference metrics** (vLLM exposes these automatically at `/metrics`):
- `vllm:e2e_request_latency_seconds` (p50 / p95 / p99)
- `vllm:gpu_cache_usage_perc`
- `vllm:num_requests_running`
- Token throughput (tokens/second)

Scrape with Prometheus, visualise in Grafana. This is an "observability system for inference" — literally line 5 of the job requirements.

---

#### Sub-story 2: Kubernetes on GKE with GPU node pool (1 week)

**Goal:** Move from Docker Compose (local) and Cloud Run (managed serverless) to Kubernetes — the industry standard for orchestrating GPU workloads. This is what "container orchestration for GPU-based workloads in production" means.

```bash
# Create GKE cluster with a GPU node pool
gcloud container clusters create cosmosai-cluster \
  --region europe-west1 --num-nodes 1

gcloud container node-pools create gpu-pool \
  --cluster cosmosai-cluster \
  --num-nodes 1 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4
```

**Deploy Triton and vLLM as Kubernetes workloads:**
```yaml
# triton-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-server
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: triton
        image: nvcr.io/nvidia/tritonserver:24.01-py3
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /models
---
apiVersion: v1
kind: Service
metadata:
  name: triton-service
spec:
  selector:
    app: triton-server
  ports:
  - port: 8001
```

**Auto-scaling with HPA:**
```yaml
# hpa.yaml — scale Triton pods based on request queue depth
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: triton-server
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
```

The api-gateway and rag-service run as separate K8s Deployments (no GPU needed). Triton + vLLM pods claim GPU resources via `nvidia.com/gpu: 1` limits. K8s scheduler ensures only one GPU workload per node unless explicitly allowed otherwise.

---

#### Sub-story 3: Vertex AI Model Registry (0.5 week)

**Why Vertex AI:** managed model serving on GCP (same cloud as the rest of CosmosAI). Gives you the model registry + blue/green story for MLOps/AI Engineer roles at larger companies.

```bash
# Register ViT model with metadata linked to MLflow training run
gcloud ai models upload \
  --region=europe-west1 \
  --display-name=vit-galaxy-v1 \
  --artifact-uri=gs://cosmosai-models/vit/v1/ \
  --container-image-uri=europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest

# Deploy to endpoint with auto-scaling
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --model=MODEL_ID \
  --min-replica-count=1 \
  --max-replica-count=3 \
  --accelerator=type=nvidia-tesla-t4,count=1

# Blue/green: deploy vit_v2 alongside vit_v1, shift traffic gradually
gcloud ai endpoints deploy-model ENDPOINT_ID \
  --model=VIT_V2_MODEL_ID \
  --traffic-split=VIT_V1_ID=90,VIT_V2_ID=10
# → test → shift to 50/50 → shift to 0/100
```

---

#### Sub-story 4: Comparison document (0.5 week)

One README section comparing the two approaches — this is what an infrastructure engineer thinks about, and publishing it demonstrates the mindset:

| Dimension | Self-hosted (RunPod + vLLM) | Managed (Vertex AI) |
|---|---|---|
| Ops burden | Full ownership — you configure everything | GCP manages scaling, health, updates |
| Inference latency (p50) | Measure and publish | Measure and publish |
| Cost at 100 req/day | Measure and publish | Measure and publish |
| GPU control | Full — memory split, batching policy, concurrency | Abstracted |
| Model registry | Manual (MLflow artifacts on GCS) | Vertex AI Model Registry — versioned, auditable |
| Blue/green rollout | Manual traffic split via nginx/load balancer | Native Vertex AI traffic split |
| Best for | GPU infrastructure roles, startups building their own compute | Enterprise MLOps, AI Engineer roles |

**Cost for Phase 5b:** RunPod dev time (~5 evenings) ≈ $8–12 total. Vertex AI endpoint (1hr/day for a month) ≈ $10–15. Total: under $30.

---

### Phase 5c — Optional Edge / Embedded AI Extension

**Goal:** Test whether a trained astronomy vision model can run on edge-style hardware.

Possible flow:

```text
trained galaxy checkpoint
  -> ONNX
  -> TensorRT
  -> NVIDIA Jetson or another suitable edge target
  -> local edge inference benchmark
```

Compare if hardware is available:

```text
laptop CPU
cloud NVIDIA GPU
edge GPU / Jetson
```

Measure:

```text
latency
throughput
memory use
model size
accuracy change after optimization/quantization
power/efficiency only if measurement is reliable
```

C/C++ should only be added if it has a real technical purpose, such as TensorRT C++ inference or embedded runtime integration. Do not add C++ just to make the project look more embedded.

Master's alignment:

```text
Embedded AI
Software & Hardware Architectures
New Technologies in AI & Robotics, if the syllabus fits
```

---

### Phase 6 — Optional Multimodal Astronomy / Data Fusion

**Goal:** Combine multiple astronomy data types after the galaxy and stellar models work independently.

Possible architecture:

```text
galaxy image -> vision encoder --------┐
                                      |
spectrum -> spectral encoder ----------|
                                      |--> fusion model --> prediction/explanation
redshift -----------------------------|
magnitude / colour indices -----------┘
```

Potential research question:

```text
Does combining image, spectral, and photometric data improve astronomy classification
compared with image-only or spectrum-only models?
```

This is scientifically coherent as multimodal astronomy/data fusion. It should still be treated as optional and late. A robotics Sensor Fusion course may require Kalman filters, IMU/GPS/camera/radar fusion, or state estimation, so do not assume this satisfies that course without the official syllabus.

---

## Full Tech Stack

```
Model 1 (galaxy):      PyTorch + HuggingFace ViT (vit-base-patch16-224) + LoRA (PEFT)
Model 2 (stellar):     PyTorch 1D-CNN on SDSS spectral flux arrays
Metrics (eval only):   scikit-learn — confusion_matrix, classification_report, train_test_split
Experiment tracking:   MLflow (both models, all runs)
Embeddings:            sentence-transformers (all-MiniLM-L6-v2)
Vector store:          Qdrant Cloud (tagged by domain: galaxy / stellar)
RAG framework:         LangChain (primary) + LlamaIndex (comparison)
RAG evaluation:        RAGAS
Agent:                 LangGraph (router node + two classifier branches + shared generate)
LLM gateway:           LiteLLM → Gemini Flash / Anthropic Haiku / self-hosted vLLM
Observability (app):   Langfuse (LLM traces, token cost, RAG quality)
Observability (infra): Prometheus + Grafana (inference latency, GPU utilization, throughput)
API service:           FastAPI + Pydantic v2 (api-gateway + inference-router + classifier services)
Streaming:             SSE (Server-Sent Events)
Containerization:      Docker + Docker Compose (local multi-service dev)
Model serving (Phase 4):    Classifier services load PyTorch checkpoints or call model backends behind /classify
Model serving (Phase 5b):   Triton Inference Server — optimized ONNX backend behind classifier services
Model serving (LLM):   vLLM (self-hosted on RunPod GPU, Phi-3 Mini)
Routing layer:         inference-router by input type; classifier-level model/backend selection later
Container orchestration: Kubernetes / GKE with GPU node pool (HPA auto-scaling)
Model registry:        MLflow (Phase 1-4) + Vertex AI Model Registry (Phase 5b)
GPU compute:           RunPod RTX 3090 (dev/learning, $0.25/hr) + GCP GKE GPU node + Vertex AI
Cloud:                 GCP Cloud Run (Phase 4) → GKE (Phase 5b) + Artifact Registry
Automation:            n8n (optional weekly re-index from arXiv)
Data:                  Galaxy Zoo 2 (Kaggle), SDSS DR17 stellar spectra (Kaggle), arXiv API, Wikipedia API
Language:              Python 3.11
```

---

## What You Can Say in an Interview

**"Experience building and deploying LLM-powered applications in production":**
> "I built CosmosAI — an astronomy AI system with two classification tasks deployed on GCP Cloud Run. It uses a fine-tuned ViT for galaxy morphology and a 1D-CNN for stellar spectral classification, a LangGraph router-agent that directs each request to the right model, Qdrant for RAG over scientific literature, and LiteLLM to swap between Gemini and self-hosted backends. The API is live at [URL], streams responses via SSE, tracks every request in Langfuse, and evaluates retrieval quality with RAGAS."

**"LangChain / LangGraph experience":**
> "The retrieval and generation pipeline is built with LangChain. The agent logic uses LangGraph — a multi-node state machine with a conditional low-confidence path that retrieves similar training examples before generating an explanation."

**"Cloud AI services":**
> "Deployed on GCP Cloud Run, using the Gemini API via LiteLLM. I also evaluated LlamaIndex vs LangChain for the RAG indexing layer and documented the comparison in the repo."

**"Multi-model serving architecture + intelligent request routing":**
> "CosmosAI runs two vision models — a fine-tuned ViT for galaxy images and a 1D-CNN for stellar spectra — both exported to ONNX and served via Triton Inference Server on the same RunPod GPU instance. An application-layer router in the LangGraph agent detects input type and calls the correct Triton model endpoint. The LLM generation step runs via self-hosted vLLM. I configured GPU memory partitioning between Triton and vLLM, set batching policies per model, and expose Prometheus metrics for latency, throughput, and GPU utilization visualised in Grafana. Both models are also registered in Vertex AI Model Registry with version metadata linked to MLflow runs, with blue/green rollout demonstrated."

**"RAG pipeline":**
> "Full RAG pipeline: ingest arXiv PDFs using section-aware chunking, embed with sentence-transformers, store in Qdrant Cloud, retrieve top-5 docs per query, generate grounded explanations. RAGAS evaluation published in the README."

---

## Master's Application Framings

**Leiden (astronomy + data science):**
> "To demonstrate both my computational skills and genuine engagement with astronomical data, I built CosmosAI — a galaxy morphology classifier trained on Galaxy Zoo 2, extended with a RAG system over galaxy formation literature. The project is available at [GitHub URL] and deployed publicly. This reflects the type of computational astrophysics work I want to pursue at Leiden."

**Georgia Tech / Bologna / OMSCS (ML + AI engineering):**
> "I built and deployed an end-to-end AI system combining fine-tuned vision models (ViT/LoRA), RAG over scientific literature (LangChain + Qdrant), LLM integration (Gemini/Anthropic via LiteLLM), and a production FastAPI service on GCP Cloud Run — with MLflow experiment tracking, RAGAS evaluation, and Langfuse observability."

---

## Build Timeline

**Pace:** 6–7 hours per week of focused work.

| Week | Phase | What you finish |
|---|---|---|
| 1 | Phase 0 | GCP setup, Qdrant, Langfuse, MLflow, datasets downloaded |
| 2 | Phase 1 pt.1 | Galaxy CNN baseline + ViT/LoRA trained, MLflow runs, metrics logged |
| 3 | Phase 1 pt.2 | Stellar 1D-CNN trained, both models saved as checkpoints |
| 4 | Phase 2 | RAG pipeline: arXiv + Wikipedia indexed in Qdrant, RAGAS evaluated |
| 5 | Phase 3 | LangGraph agent: router + two branches + generate + Langfuse tracing |
| 6 | Phase 4 | api-gateway + inference-router + classifier services, Docker Compose local, deployed to Cloud Run |
| 7 | Phase 5a | Clean README, architecture diagram, demo GIF, RAGAS scores published |
| 8 | Phase 5b pt.1 | vLLM on RunPod + Triton serving both ONNX models + Prometheus/Grafana |
| 9 | Phase 5b pt.2 | GKE cluster + GPU node pool, Triton + vLLM as K8s Deployments, HPA |
| 10 | Phase 5b pt.3 | Vertex AI Model Registry + blue/green + comparison doc in README |

**Total: 10 weeks ≈ 2.5 months at 6–7h/week.**

Start after CNRED submitted (July) — target: end of September / early October 2026.

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| Jun 2026 | Galaxy morphology chosen over exoplanet transit | Images are more demo-friendly; Galaxy Zoo 2 is a well-known dataset; visual output works better for a portfolio URL |
| Jun 2026 | LiteLLM for multi-LLM instead of single provider | Shows multi-LLM competence (item 13 in self-learn doc) for free — same code, different model string |
| Jun 2026 | Qdrant Cloud over pgvector | Dedicated vector DB = cleaner architecture story; pgvector is the PDQ-adjacent option so keeping them separate |
| Jun 2026 | Added Phase 5b: model serving infrastructure layer | Two reasons: (1) Pragmatike interview — they build GPU cloud, need raw infrastructure story not just Cloud Run; (2) broadens target job tier from AI Engineer (Tier 3) to AI Infrastructure / MLOps (Tier 3-4). RunPod chosen for raw GPU (cheapest, no managed abstraction, closest to what GPU infra companies operate); Vertex AI chosen for managed story (existing GCP knowledge, model registry + blue/green built-in). LiteLLM already in Phase 3 means swapping from Gemini API to self-hosted vLLM endpoint needs zero app code change. |
| Jun 2026 | Added stellar spectral classifier (1D-CNN) as second model | Enables multi-model serving architecture — directly required by Pragmatike JD ("multi-model serving architectures and intelligent request routing layers"). Galaxy ViT (image input) and stellar 1D-CNN (spectrum input) are genuinely different architectures on different data types, which makes the routing layer real and not artificial. Both served via Triton on same GPU instance. SDSS DR17 data is free and well-labeled. Adds only 0.5 weeks to Phase 1. |
| Jun 2026 | scikit-learn for metrics only, PyTorch for all model building | Matches market expectation — job descriptions want PyTorch/HuggingFace, not sklearn classifiers. sklearn stays for confusion_matrix, classification_report, train_test_split only. |
| Jun 2026 | Phase 4 split into api-gateway + model-server | Key infrastructure pattern: gateway never loads models directly. Enables canary routing (v1/v2 galaxy models) at the model-server layer. Aligns with ChatGPT v2 plan's service-split approach. Docker Compose for local multi-service dev. |
| Jun 2026 | Kubernetes (GKE) added to Phase 5b | Job description requires "container orchestration and operating GPU-based workloads in production." GKE with GPU node pool is the direct answer. HPA auto-scaling on GPU utilization. Triton + vLLM run as K8s Deployments claiming nvidia.com/gpu resources. |
| Aug 2026 | Roadmap reconciled with implemented service topology | The actual repo evolved into api-gateway -> inference-router -> galaxy/stellar classifier services. Future PyTorch/Triton backends should sit behind classifier services instead of forcing the code back into one generic model-server. |
| Aug 2026 | Added master's integration companion document | `docs/master_project_integration.md` maps possible AAIMR master's topics onto CosmosAI while keeping this roadmap canonical. Course fit remains provisional until official syllabi/project briefs are known. |
| Aug 2026 | Added CPU -> CUDA -> profiling -> ONNX/TensorRT -> Triton progression | This gives a safer learning bridge before production GPU serving and keeps local CPU development supported. |
| Jun 2026 | Timeline switched to per-week at 6-7h/week | More realistic planning unit. 10 weeks total ≈ 2.5 months. Starts after CNRED (July), done by end of September / early October 2026. |

---

*Related: `AI_skills_through_self_learn_cannot_do_atwork.md` — the full skill list this project covers*
*Related: `master/general_all_masters/project/master_decide_project.md` — master's project requirements*
*Related: `overall_ai_project_plan.md` — the PDQ-adjacent projects (pdq-rag, classifier, incident intelligence)*
*Last updated: June 2026*
