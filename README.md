# llm-runtime

Universal runtime for AI models — from download to inference in one command.

---

## Overview

llm-runtime is a lightweight orchestration layer for running AI models locally or on GPU.

It abstracts:

- model downloading
- runtime initialization
- inference engines (vLLM, llama.cpp, etc.)
- API access

---

## Problem

Running AI models typically requires:

- manual Docker configuration
- GPU setup
- different inference engines
- inconsistent APIs

This leads to high complexity and low reproducibility.

---

## Solution

llm-runtime provides:

- unified interface for model execution
- automatic runtime orchestration
- OpenAI-compatible API
- support for multiple backends

---

## Features

- run models from HuggingFace or local storage
- support for multiple inference engines
- Docker-based isolation
- model lifecycle management
- simple HTTP API

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/llm-runtime
cd llm-runtime

docker compose up
````

---

## Run a model

```bash
curl http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral"}'
```

---

## Chat example

```bash
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "message": "Hello"
  }'
```

---

## Architecture

```
User Request
     ↓
API Layer (FastAPI)
     ↓
Runtime Manager
     ↓
Engine Adapter (vLLM / llama.cpp)
     ↓
Docker Container
     ↓
Model
```

---

## Use Cases

* local AI development
* self-hosted inference
* rapid prototyping
* MLOps pipelines

---

## License

MIT


