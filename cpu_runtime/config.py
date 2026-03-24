"""
cpu_runtime/config.py

Configuration for the CPU inference service.
All values are read from environment variables with CPU_RUNTIME_ prefix.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CPU_RUNTIME_",
        env_file=".env",
        extra="ignore",
    )

    # Path to the GGUF model file inside the container
    model_path: str = "/models/model.gguf"

    # Alias reported in /v1/models and used as model name in responses
    model_alias: str = "cpu-model"

    # llama.cpp runtime parameters
    n_ctx: int = 4096           # context window (tokens)
    n_threads: int = 4          # inference threads
    n_batch: int = 512          # prompt eval batch size
    n_gpu_layers: int = 0       # 0 = pure CPU; set >0 to offload layers to GPU

    # Generation defaults (overridable per-request)
    max_tokens_default: int = 512
    temperature_default: float = 0.7
    top_p_default: float = 0.95
    repeat_penalty: float = 1.1

    # Server
    port: int = 8090
    host: str = "0.0.0.0"

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "cpu-runtime"
    log_format: str = "json"

    # Queue: max concurrent inference requests (llama.cpp is single-threaded)
    max_queue_depth: int = 16


settings = Settings()
