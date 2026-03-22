# llm-runtime

Universal orchestration layer for running AI models locally or on GPU. Abstracts model downloading, runtime initialization, inference engines, and API access into a single unified interface.

## Architecture

Four main services + infrastructure:

```
Streamlit UI (8502)
      ↓
API Dispatcher (8005)   ←→   Control Plane (8004)
      ↓                              ↓
MRM (8010)                     PostgreSQL (55432)
      ↓                         Alembic migrations
vLLM containers
      ↓
LiteLLM (4000) — OpenAI-compatible proxy
      ↓
Nginx (8081) — gateway
```

Supporting: Redis (runtime state), MLflow (experiment tracking), Embeddings/Infinity (7997), Neo4j (RAG, optional).

## Services

### Model Runtime Manager (MRM) — `model_runtime_manager/` port 8010
Manages vLLM Docker container lifecycle, GPU reservation, idle eviction. Redis is canonical truth; Docker is a sensor. Spec: `model_runtime_manager/MRM_SPEC.md`.

Key invariants:
- Only MRM manages Docker lifecycle for inference containers
- `ensure` + `touch` contract: callers must touch to prevent eviction
- Base models vs LoRA adapters are separate concerns

Key endpoints: `POST /models/ensure`, `/models/touch`, `/models/stop`, `/models/remove`, `GET /models/status`

### Control Plane — `control_plane/` port 8004
Policy engine only — decides, does not execute. All ML work delegated to workers. Spec: `control_plane/CORE_SPEC.md`.

Core entity: `Run` (immutable orchestration instance with state, contract, artifacts).

Supported contracts: `dataset.build.v1`, `train.qlora.v1`, `train.dpo.v1`, `eval.standard.v1`

State machine: `CREATED → DATASET_RUNNING → DATASET_READY → TRAIN_RUNNING → TRAIN_READY → DONE/FAILED`

Stack: FastAPI + SQLAlchemy 2.0 + asyncpg + Alembic

### API Dispatcher — `api/` port 8005
Thin execution bridge. Routes requests, calls MRM ensure/touch, handles LoRA materialization from S3, RAG context injection (Neo4j).

Key endpoints: `POST /chat`, dataset building

### Frontend — `frontend/` port 8502
Streamlit UI. Pages: home, chat, gpu_monitor, model_registry, training, deployment_manager, orchestration, prompt_studio, hf_registry, dataset_build, docs_tools.

## Running

```bash
docker compose -f docker-compose.dev.yml up
```

## Stack

- Python 3.10–3.12, FastAPI, Uvicorn, Pydantic
- PostgreSQL, Redis, SQLAlchemy 2.0, Alembic
- Docker SDK (Python), NVIDIA/CUDA, pynvml
- vLLM (inference), LiteLLM (proxy), Infinity (embeddings)
- S3/boto3 (artifacts), HuggingFace Hub, MLflow, Neo4j

## Chat Request Flow

1. Streamlit → Dispatcher: send message
2. Dispatcher → MRM: `ensure(base_model)` → get `model_alias` + `api_base`
3. Dispatcher → MRM: `touch(base_model)`, materialize LoRA if needed, inject RAG context
4. Dispatcher → LiteLLM → vLLM: execute inference
5. Response returned to UI
