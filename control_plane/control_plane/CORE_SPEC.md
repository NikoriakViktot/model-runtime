
---

# AI Control Plane — Core Architecture Specification (v1)

## 1. Purpose of the Core

The **AI Control Plane** is a **policy engine and single source of truth** governing the lifecycle of all AI operations:

* dataset construction
* model training (SFT / QLoRA / DPO / SimPO)
* evaluation
* deployment
* inference enablement

The core **does not execute ML workloads**.
It **does not train**, **does not build datasets**, **does not own GPUs**.

> **Execution is external.
> Truth, policy, and state are internal.**

---

## 2. Fundamental Design Principles

### 2.1 Source of Truth

All system truth lives in **persistent state**, not in workers, queues, or logs.

* No in-memory orchestration
* No hidden execution logic
* No implicit pipelines

If the system restarts, **state reconstruction is complete from the database**.

---

### 2.2 Separation of Concerns

| Layer      | Responsibility                         |
| ---------- | -------------------------------------- |
| Core       | Policy, state transitions, truth       |
| Dispatcher | Translate actions into execution calls |
| Workers    | Execute tasks, produce artifacts       |
| UI         | Human interaction only                 |

No layer leaks responsibilities upward or downward.

---

## 3. Canonical Domain Model (Minimal v1)

The **entire orchestration kernel** is built around **one entity**.

### 3.1 `Run`

A **Run** represents a **single orchestration instance**.

A Run stores only:

* `id` — immutable identity
* `state` — current lifecycle state
* `contract` — the original declared intent
* `artifacts` — facts produced during execution

That is **intentionally sufficient**.

> If something cannot be expressed as
> **state, contract, or artifact**,
> it does not belong in the core.

---

## 4. High-Level Architecture

```
        UI / Client
             |
        POST /contracts
             |
      ┌───────────────┐
      │ Control Plane │   ← policy & truth
      └───────┬───────┘
              |
        next_action()
              |
      ┌───────────────┐
      │ Dispatcher    │   ← thin execution bridge
      └───────┬───────┘
              |
        API Dispatcher / Workers
              |
        Heavy execution
              |
        POST /events
              |
      ┌───────────────┐
      │ Control Plane │
      └───────────────┘
```

---

## 5. Protocols

The system uses **two and only two protocols**:

1. **Contract Protocol** — what should be done
2. **Event Protocol** — what has happened

Both are **explicit**, **versioned**, and **immutable**.

---

## 6. Contract Protocol

### 6.1 Contract Envelope (Canonical)

```json
{
  "type": "dataset.build.v1",
  "spec_version": "v1",
  "idempotency_key": "optional",
  "payload": { }
}
```

**Properties:**

* `type` — globally unique contract identifier
* `spec_version` — pinned schema version
* `payload` — validated, canonical intent

The core **never interprets free-form intent**.

---

## 7. Canonical Contracts (v1)

### 7.1 `dataset.build.v1`

**Semantic intent:**
Declare dataset construction.

```json
{
  "type": "dataset.build.v1",
  "spec_version": "v1",
  "payload": {
    "db_id": "string",
    "target_name": "string",
    "dataset_type": "sft | prefs | graph",
    "options": {
      "generate_rejected": true,
      "rejected_max_items": 500
    },
    "output": {
      "base_uri": "file:///artifacts/datasets"
    }
  }
}
```

**Expected artifacts:**

* `dataset_uri`
* `dataset_version`
* `manifest_uri`

---

### 7.2 `train.qlora.v1`

```json
{
  "type": "train.qlora.v1",
  "spec_version": "v1",
  "payload": {
    "target_slug": "string",
    "base_model": "string",
    "dataset": {
      "uri": "string"
    },
    "training": {
      "epochs": 1,
      "learning_rate": 0.000005
    },
    "output": {
      "lora_base_uri": "file:///artifacts/lora"
    }
  }
}
```

**Expected artifacts:**

* `lora_uri`
* `metrics_uri`
* `mlflow_run_id`

---

### 7.3 `train.dpo.v1`

```json
{
  "type": "train.dpo.v1",
  "spec_version": "v1",
  "payload": {
    "target_slug": "string",
    "base_model": "string",
    "prefs_dataset": {
      "uri": "string"
    },
    "training": {
      "epochs": 1,
      "learning_rate": 0.000005,
      "beta": 0.1
    },
    "output": {
      "lora_base_uri": "file:///artifacts/lora"
    }
  }
}
```

---

### 7.4 `eval.standard.v1`

```json
{
  "type": "eval.standard.v1",
  "spec_version": "v1",
  "payload": {
    "model": {
      "kind": "base | lora",
      "name": "string",
      "artifact_uri": "optional"
    },
    "dataset": {
      "uri": "string"
    },
    "metrics": ["loss", "accuracy", "f1"],
    "output": {
      "metrics_uri": "file:///artifacts/metrics"
    }
  }
}
```

---

## 8. Event Protocol

### 8.1 Event Envelope

```json
{
  "type": "event.v1",
  "run_id": "uuid",
  "event": "DATASET_BUILT",
  "payload": {
    "artifacts": { }
  }
}
```

Events are **facts**, not commands.

---

## 9. Run State Machine

### 9.1 States

* `CREATED`
* `DATASET_RUNNING`
* `DATASET_READY`
* `TRAIN_RUNNING`
* `TRAIN_READY`
* `EVAL_RUNNING`
* `DONE`
* `FAILED`

---

### 9.2 Transition Policy (Declarative)

| Current State   | Event             | Next State      | Action        |
| --------------- | ----------------- | --------------- | ------------- |
| CREATED         | CONTRACT_ACCEPTED | DATASET_RUNNING | DATASET_BUILD |
| DATASET_RUNNING | DATASET_BUILT     | TRAIN_RUNNING   | TRAIN         |
| TRAIN_RUNNING   | TRAIN_COMPLETED   | DONE            | —             |
| *               | FAILED            | FAILED          | —             |

**Policy is data, not code.**

---

## 10. Orchestration Rule

> **The core never executes.
> The core only decides.**

Workers:

* execute tasks
* produce artifacts
* emit events

The core:

* validates contracts
* stores truth
* applies policy
* issues next actions

---

## 11. Why This Design

### 11.1 Why Only `Run`

Because **orchestration is about state**, not entities.

Everything else (jobs, datasets, models) can be **derived views**, not truth.

---

### 11.2 Why JSON Contracts + Pydantic

Because contracts must be:

* machine-verifiable
* schema-exportable
* versionable
* LLM-readable

A canonical JSON contract is **both protocol and documentation**.

---

### 11.3 Why Events Instead of Polling

Events are **facts**.
Polling is **guessing**.

The system reacts to reality, not assumptions.

---

## 12. LLM Context Guarantee

Providing this specification to an LLM guarantees:

* correct mental model
* no invented pipelines
* no hidden orchestration
* no mixed responsibilities

The model understands **what the system is and is not allowed to do**.

---

## 13. Final Principle

> **The system is explainable by its data alone.
> If you need to read code to understand state, the architecture is wrong.**

---


