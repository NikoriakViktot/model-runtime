---
id: spec.mrm.v1
title: Model Runtime Manager (MRM) — Runtime & Container Lifecycle Specification (v1)
owner: AI Control Plane
scope: runtime / inference
status: draft
last_updated: 2026-01-13
tags:
  - ai-control-plane
  - runtime
  - vllm
  - docker
  - gpu
  - redis
  - dispatcher-integration
llm_context:
  priority: HIGH
  use_as: "system_constraints_for_codegen"
---

# Model Runtime Manager (MRM) — Runtime & Container Lifecycle Specification (v1)

## 0. Summary (One Paragraph)
MRM is the single component that owns GPU runtime for inference and manages vLLM container 
lifecycle for **base models**. It exposes a small HTTP API used by 
the Dispatcher to **ensure** a base model is running and to **touch** 
it on every request to prevent eviction. 
MRM stores runtime state in Redis, uses locks to serialize per-model operations, 
and runs an idle reaper that stops containers when unused.

---

## 1. Role in the System (Boundaries)
### 1.1 MRM Owns
- Docker lifecycle for vLLM containers (start/stop/remove)
- GPU reservation / release
- readiness healthchecks
- idleness policy and eviction (stop when idle)
- runtime status truth (runtime only)

### 1.2 MRM Does NOT Own
- contracts, orchestration, state machine, policy (Control Plane owns)
- artifact URIs for datasets/LoRA/metrics (Control Plane owns)
- downloading datasets or adapters from S3 (Dispatcher owns)
- RAG / indexing / Neo4j logic (Dispatcher owns)
- UI logic (Streamlit owns only rendering + calls Dispatcher)

> Architectural axiom:
> **Control Plane decides. Dispatcher routes. MRM owns runtime.**

---

## 2. LLM-Friendly Invariants (Do Not Violate)
These are hard constraints for code generation and refactors.

### 2.1 Single Source of Runtime Control
Only MRM may call Docker start/stop/remove for inference containers.  
No other module may manage vLLM container lifecycle.

### 2.2 Runtime Truth: Redis is Canonical, Docker is Sensor
MRM runtime state is stored in Redis as the canonical record:
- `mrm:model:<base_model>`  (hash) — per model runtime state
- `mrm:gpu:<gpu>`           (set)  — GPU occupancy by base_model (v1: one model per GPU)
- `mrm:lock:<base_model>`   (string TTL) — per model operation lock

Docker inspection is treated as a **sensor**.
Only MRM may reconcile Redis with Docker observation and persist (in `mrm:model:<base_model>`):
- `observed_container_state`
- `observed_at`
- `last_error` (optional)

Outside MRM, do not derive runtime truth from container logs or by polling Docker (e.g. `docker ps`).

### 2.3 Canonical Activity Contract: ensure + touch
- `ensure(base_model)` guarantees model is running or starts it.
- `touch(base_model)` updates `last_used` to prevent eviction.

If Dispatcher does not call touch, idle reaper is allowed to stop the container.

### 2.4 Separation: Base Model vs LoRA Adapter
- MRM handles **base model runtime only**.
- LoRA adapters are artifacts downloaded/materialized by Dispatcher and passed as request payload (`lora_adapter`).

MRM must not download adapters, scan S3, or decide which adapter to load.

### 2.5 HF Auth Token for Gated/Private Models
If a base model is gated/private on Hugging Face, MRM must pass an HF auth token to the vLLM container
(e.g., `HF_TOKEN` and/or `HUGGINGFACE_HUB_TOKEN` via container env).
Without the token, vLLM may exit immediately and MRM must surface an actionable error.
---

## 3. Placement (Request Flow)
```

UI / Client
↓  POST /chat (Dispatcher)
Dispatcher
↓  ensure(base_model)
↓  touch(base_model)
↓  (optional) materialize LoRA from S3
↓  (optional) RAG inject context
↓  POST /chat/completions (LiteLLM)
LiteLLM
↓  routes to vLLM runtime endpoint
vLLM container (per base_model)

````

Key rule:
> Dispatcher must call **ensure + touch** on every inference request.

---

## 4. Public API (MRM HTTP)
### 4.1 `GET /health`
Returns `{ "status": "ok" }`.

### 4.2 `POST /models/ensure`
Request:
```json
{ "base_model": "Qwen/Qwen2.5-7B-Instruct" }
````


Response:

```json
{
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "model_alias": "qwen-7b-instruct",
  "api_base": "http://vllm_qwen_7b_instruct:8000/v1",
  "container": "vllm_qwen_7b_instruct",
  "gpu": "0",
  "state": "READY"
}
```
Error Response Schema (MRM)
For actionable failures (e.g., health timeout, immediate exit), MRM returns HTTP 409 with:

```json
{
  "error": "MODEL_START_FAILED",
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "container": "vllm_qwen_7b_instruct",
  "gpu": "0",
  "hint": "Possible causes: missing HF token, OOM, CUDA error, invalid vLLM flags",
  "logs_tail": ["... last lines when available ..."]
}
```

### 4.3 `POST /models/touch`
Request:
```json
{ "base_model": "Qwen/Qwen2.5-7B-Instruct" }
```
Response:
```json
{ "base_model": "Qwen/Qwen2.5-7B-Instruct", "touched": true }
```
Updates `last_used` in Redis. No heavy logic.

### 4.4 `POST /models/stop`

Request:

```json
{ "base_model": "Qwen/Qwen2.5-7B-Instruct" }
```

Stops container and releases GPU.

### 4.5 `POST /models/remove`

Stops + removes container; clears presence.

### 4.6 `GET /models/status/{base_model}` and `GET /models/status`

Truthful status from Redis + docker inspection (inside MRM only).

---

## 5. Runtime Policy (Deterministic)

### 5.1 Idle Eviction

MRM runs an internal loop:

* every `sweep_interval_sec`
* checks `last_used`
* if `now - last_used > idle_timeout_sec` → stop model

This is the only eviction mechanism.

### 5.2 Concurrency Lock

Per base model lock:

* `SET mrm:lock:<base_model> 1 NX EX <ttl>`

Only one of ensure/stop/remove can run per base_model concurrently.

**Lock TTL policy (v1)**
- Operations must complete within lock TTL.
- If `ensure()` times out or fails, MRM must perform cleanup (remove container if created, release GPU)
  and transition state to `ABSENT` with `last_error`.

### 5.3 GPU Reservation

GPU occupancy:

* reserve: `SADD mrm:gpu:<gpu> <base_model>`
* release: `SREM mrm:gpu:<gpu> <base_model>`

GPU selection:

* first free GPU from `allowed_gpus` in ModelSpec

**Invariant (v1): one GPU hosts at most one base_model at a time.**
Although Redis uses a set (`mrm:gpu:<gpu>`), MRM enforces `SCARD==0` before reserving.
MRM is not a scheduler in v1 and does not support multi-tenant GPU packing.

### 5.4 Runtime State Machine (v1)

MRM maintains a small deterministic state machine in Redis (`mrm:model:<base_model>.state`).
MRM is the only component allowed to reconcile this state with Docker reality.

**States**
- `ABSENT`   — no container exists (or removed)
- `STARTING` — container is being started / healthcheck pending
- `READY`    — container is running and healthy
- `STOPPING` — stop in progress
- `STOPPED`  — container exists but not running

**Primary transitions (v1)**

| Trigger / Actor | From → To | Notes |
|---|---|---|
| `ensure(base_model)` | `ABSENT → STARTING → READY` | start container fresh, wait health |
| `ensure(base_model)` | `STOPPED → STARTING → READY` | start existing container, healthcheck |
| `stop(base_model)` / idle reaper | `READY → STOPPING → STOPPED` | must release GPU reservation |
| `remove(base_model)` | `* → ABSENT` | stop+remove container, clear state, release GPU |
| hard failure (health timeout / crash) | `STARTING → ABSENT` | container removed (if exists), GPU released |
| `status()` reconcile | `* → READY` or `* → STOPPED/ABSENT` | only MRM may reconcile Redis state with Docker inspection |

---

## 6. Model Registry (Runtime Spec)

MRM uses a deterministic registry of `ModelSpec` entries.

A ModelSpec defines:

* base_model (HF id)
* model_alias (served name)
* vLLM image + container_name
* vLLM flags (max_model_len, lora settings, batching)
* allowed_gpus
* volumes + env (hf cache, artifacts)
* health path + port
* shm/ipc config

Merge strategy:

* default registry + env overrides (e.g. `MRM_MODEL_REGISTRY`)

Hard rule:

> MRM does not invent runtime config dynamically.
> It only executes ModelSpec.

MRM must pass HF auth token to vLLM containers when models are gated/private.

**MRM input env (service runtime)**
- `HF_TOKEN` (preferred)
- `HUGGINGFACE_HUB_TOKEN` (also supported)

**Env passed to vLLM container**
- MRM passes both `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` (when present) to maximize compatibility.
- Tokens must never be logged.
---

## 7. Integration Contract: Dispatcher ↔ MRM

### 7.1 Dispatcher MUST

1. Resolve base_model:

   * base request: `base:<hf_model>` or default
   * LoRA request: fetch run contract from Control Plane, take `base_model`

2. Call:

   * `POST MRM /models/ensure`
   * `POST /models/touch (body)` (every request)

3. Use `model_alias` returned by MRM as the model identifier for LiteLLM routing.

### 7.2 Dispatcher MUST NOT

* call Docker directly
* guess container names
* guess GPU availability
* assume model is running without calling ensure

### ModelSpec Shape (v1)
```yaml
base_model: "Qwen/Qwen2.5-7B-Instruct"
model_alias: "qwen-7b-instruct"
container_name: "vllm_qwen_7b_instruct"
image: "vllm/vllm-openai:latest"
allowed_gpus: ["0"]
port: 8000
health_path: "/health"
env:
  HF_TOKEN: "${HF_TOKEN}"
  HUGGINGFACE_HUB_TOKEN: "${HUGGINGFACE_HUB_TOKEN}"
volumes:
  "${MRM_HF_CACHE_HOST_PATH}": "/root/.cache/huggingface"
  "${MRM_ARTIFACTS_HOST_PATH}": "/app/artifacts"
```
---

## 8. LLM Context: Constraints for Code Generation (Copy-Paste Block)

Use this as a “guardrail prompt” for any LLM that generates code in this repo.

### Allowed

* call MRM API from Dispatcher
* read run contract / artifacts from Control Plane
* materialize datasets/adapters from S3 inside Dispatcher
* keep UI thin: call Dispatcher only

### Forbidden

* any Docker/GPU logic outside MRM
* any runtime lifecycle control in UI / Dispatcher / Workers
* any attempt to derive runtime truth from logs
* mixing RAG / indexing / dataset logic into MRM

### Canonical ordering for `/chat` in Dispatcher

1. resolve base_model
2. ensure(base_model) via MRM
3. touch(base_model) via MRM
4. optional: materialize LoRA adapter from S3
5. optional: RAG retrieve + inject context
6. forward request to LiteLLM

---

## 9. Operational Notes (Non-Goals)

* MRM is runtime-only; it is not a scheduler.
* Redis state is runtime cache/truth for runtime only.
* Control Plane remains the single source of truth for orchestration and artifacts.

Crash-on-start diagnostics

If a vLLM container exits immediately after start or healthcheck times out,
MRM should include a short tail of container logs in the error response **when available**
(e.g., last 200 lines) to make failures actionable (auth / OOM / CUDA / config errors).
---

## 10. Minimal Acceptance Tests (Behavior)

* Calling /models/ensure on a stopped model starts a container and returns READY.
* If Dispatcher calls touch every request, model stays alive under idle timeout.
* If touch is missing, model can be stopped by idle reaper.
* Stop/remove must free GPU reservation in Redis.
* Two concurrent ensure calls for same base_model do not start two containers (lock works).

---

# Streamlit Frontend — Chat / Inference UI Specification (v1)

## 1. Purpose

Streamlit UI is a thin human interface for:

* selecting base model or LoRA run_id
* sending messages
* displaying responses
* validating end-to-end path: UI → Dispatcher → MRM → vLLM → LiteLLM

## 2. UI Boundaries (Do/Don't)

UI may:

* call Dispatcher APIs
* render list of trained runs and metrics
* keep chat history in `st.session_state`

UI must not:

* call MRM directly
* call Control Plane directly (except via Dispatcher proxy endpoints)
* download artifacts from S3
* assume which vLLM container is running

## 3. Model Identifiers

* Base model is represented as an opaque string like `base:<HF_MODEL>` or a UI alias mapped to it.
* LoRA option is represented as `run_id` (truth in Control Plane), not a path and not a container name.

## 4. UI → Dispatcher Contract (Chat Request)

Request shape:

```json
{
  "model": "<base:... | run_id>",
  "messages": [{"role":"user","content":"..."}],
  "temperature": 0.7,
  "max_tokens": 512
}
```

UI assumes Dispatcher will:

* resolve base_model
* ensure + touch via MRM
* materialize LoRA from S3 if needed
* inject RAG if enabled
* proxy to LiteLLM

## 5. LLM Context for UI Codegen

* Keep UI thin and state minimal.
* Do not embed orchestration.
* Treat Dispatcher as the only backend dependency.