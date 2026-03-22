# Architecture Rules

Service boundary rules for the distributed AI inference platform.
These rules enforce clean separation of concerns and prevent responsibility leakage.

---

## Service Permissions

### Gateway (`gateway/`)

✅ CAN:
- Accept and validate OpenAI-compatible API requests
- Call MRM `/models/ensure` to guarantee a model is running
- Call Scheduler `/schedule/ensure` in distributed mode
- Select a runtime instance via `ModelRouter` (least_loaded, round_robin, random)
- Proxy inference requests to vLLM `api_base`
- Proxy embedding requests to Infinity
- Record per-request metrics in MLflow (fire-and-forget, non-blocking)
- Expose Prometheus `/metrics` and OpenTelemetry traces

❌ CANNOT:
- Decide which node a model runs on (that is Scheduler's responsibility)
- Manage Docker containers or GPU allocation
- Call Control Plane, Dispatcher, or the training pipeline
- Store persistent state (Redis, database)
- Retry failed upstream requests (caller is responsible for retries)

---

### Scheduler (`scheduler/`)

✅ CAN:
- Maintain the NodeRegistry (which nodes are alive, their GPU capacity)
- Maintain PlacementStore (model → node mapping in Redis)
- Select a node for a model placement using a pluggable strategy
- Forward `ensure` to the selected Node Agent
- Expire dead nodes based on heartbeat TTL

❌ CANNOT:
- Start or stop Docker containers (delegate to MRM via Node Agent)
- Accept inference requests (no proxying, no routing of user traffic)
- Call Gateway or Control Plane
- Own or interpret model content (treat model IDs as opaque strings)

---

### Node Agent (`node_agent/`)

✅ CAN:
- Report GPU state to the Scheduler via heartbeat
- Forward `/local/ensure` to the local MRM
- Forward `/local/stop` to the local MRM
- Expose GPU metrics via Prometheus

❌ CANNOT:
- Make placement decisions (that is Scheduler's responsibility)
- Start Docker containers directly (delegate to MRM)
- Call other Node Agents
- Call Gateway or Control Plane

---

### Model Runtime Manager (`model_runtime_manager/`)

✅ CAN:
- Start, stop, and remove vLLM Docker containers
- Allocate and release GPU slots
- Manage model lifecycle state in Redis
- Enforce the ensure+touch contract (idle eviction)
- Report GPU memory metrics

❌ CANNOT:
- Route inference requests
- Decide which node a model runs on (it knows nothing about other nodes)
- Accept inference traffic (no `/v1/chat/completions`)
- Call Gateway, Scheduler, or Control Plane

---

### API Dispatcher (`api/`)

✅ CAN:
- Route chat requests that require LoRA adapters or RAG context
- Call MRM `ensure` + `touch` for model warm-up
- Materialize LoRA adapters from S3
- Retrieve RAG context from Neo4j
- Forward completed inference requests to LiteLLM
- Trigger dataset construction and training job execution
- Report completion events to Control Plane

❌ CANNOT:
- Own orchestration policy (that is Control Plane's responsibility)
- Store Run state (delegate to Control Plane)
- Manage Docker containers
- Accept external API traffic (Gateway is the public ingress)

---

### Control Plane (`control_plane/`)

✅ CAN:
- Accept and validate contracts (versioned intent payloads)
- Create and store immutable Run records
- Apply state machine rules and determine next actions
- Accept events from workers (facts, never opinions)
- Report run status and history

❌ CANNOT:
- Execute any ML work (no training, no inference, no Docker)
- Call MRM directly
- Access GPU hardware or model files
- Modify events after recording (events are immutable facts)

---

### Frontend (`frontend/`)

✅ CAN:
- Call API Dispatcher for chat and dataset building
- Call Control Plane for run management and contract submission
- Call Gateway for model listing (`GET /v1/models`)
- Display GPU metrics (read-only from MRM or Gateway)

❌ CANNOT:
- Call MRM directly for lifecycle operations (ensure, stop, remove)
- Call Scheduler directly
- Call Node Agent directly
- Store persistent state (the UI is stateless between page loads)

---

## Communication Rules

### Synchronous Calls (HTTP)

| Caller | Callee | Allowed? | Notes |
|--------|--------|----------|-------|
| Gateway | MRM | ✅ | ensure only |
| Gateway | Scheduler | ✅ | distributed mode only |
| Gateway | vLLM | ✅ | inference proxy |
| Gateway | Infinity | ✅ | embeddings |
| Gateway | MLflow | ✅ | async metrics, non-blocking |
| Dispatcher | MRM | ✅ | ensure + touch |
| Dispatcher | LiteLLM | ✅ | inference |
| Dispatcher | Control Plane | ✅ | events only |
| Dispatcher | Neo4j | ✅ | RAG context |
| Dispatcher | S3 | ✅ | adapter materialization |
| Control Plane | Dispatcher | ✅ | trigger execution |
| Scheduler | Node Agent | ✅ | ensure + stop |
| Node Agent | MRM | ✅ | ensure + stop |
| Node Agent | Scheduler | ✅ | heartbeat |
| Frontend | Dispatcher | ✅ | chat, dataset build |
| Frontend | Control Plane | ✅ | runs, contracts |
| Frontend | Gateway | ✅ | GET /v1/models only |
| Frontend | MRM | ⚠️ | read-only status only (refactor target) |
| Frontend | Scheduler | ❌ | never |
| Frontend | Node Agent | ❌ | never |
| MRM | anything | ❌ | MRM is a leaf node |
| vLLM | anything | ❌ | vLLM is a leaf node |

### Async / Fire-and-Forget

| Source | Destination | Pattern |
|--------|-------------|---------|
| Gateway | MLflow | POST metrics after response sent |
| Node Agent | Scheduler | Heartbeat every N seconds |
| Any | Jaeger | OTel span export |

---

## Data Ownership Rules

| Data | Owner | Storage | Access |
|------|-------|---------|--------|
| Model runtime state | MRM | Redis | MRM reads/writes; others read-only via MRM API |
| GPU slot allocation | MRM | Redis | MRM only |
| Node registry | Scheduler | Redis | Scheduler reads/writes; Node Agent writes via heartbeat |
| Placement records | Scheduler | Redis | Scheduler reads/writes |
| Run records | Control Plane | PostgreSQL | Control Plane reads/writes; others read-only via API |
| Orchestration events | Control Plane | PostgreSQL | Control Plane only (append-only) |
| LoRA adapter artifacts | S3 | S3 | Dispatcher downloads; Control Plane stores paths in Run |
| Request metrics | Gateway | In-memory + MLflow + Prometheus | Gateway writes; Grafana reads |
| GPU metrics | Node Agent | Prometheus | Node Agent writes |
| Routing metrics | Gateway | In-memory + Prometheus | Gateway writes |

---

## Naming Conventions

| Concept | Convention | Example |
|---------|------------|---------|
| Model identifier | HuggingFace repo ID | `meta-llama/Llama-2-7b-hf` |
| Model alias | lowercase, hyphens | `llama-2-7b-hf` |
| Node identifier | descriptive slug | `gpu-node-01` |
| Container name | `mrm-{model_alias}` | `mrm-llama-2-7b-hf` |
| Redis key (MRM) | `mrm:{entity}:{id}` | `mrm:model:meta-llama/Llama-2-7b-hf` |
| Redis key (Scheduler) | `scheduler:{entity}:{id}` | `scheduler:placement:meta-llama/Llama-2-7b-hf` |
| Contract type | `{domain}.{action}.v{N}` | `train.qlora.v1` |
| Event type | `SCREAMING_SNAKE_CASE` | `TRAIN_COMPLETED` |

---

## Prohibited Patterns

1. **Hardcoded URLs** — all inter-service URLs must come from environment variables or config.

2. **Synchronous `requests` library** in async services — use `httpx.AsyncClient` only.

3. **Direct Docker SDK calls outside MRM** — only MRM may call the Docker daemon.

4. **Direct Redis access outside the owning service** — services must use the owning service's HTTP API, not Redis directly.

5. **Shared Python imports between services** — each service is a separate Docker image. No `from gateway.x import y` in `scheduler/`. Tests may share `tests/utils/`.

6. **Print statements in service code** — use structured logging (structlog) with context binding.

7. **Bare `except Exception`** — catch specific exceptions or re-raise. Never silently swallow errors.

8. **State in request handlers** — handler functions must be stateless. State lives in Redis, PostgreSQL, or explicitly managed application-level singletons.
