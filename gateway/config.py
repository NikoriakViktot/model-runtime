"""
gateway/config.py

All configuration is read from environment variables with GATEWAY_ prefix.
Sane defaults work for local docker-compose development.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        extra="ignore",
    )

    # --- Upstream services ---
    mrm_url: str = "http://model_runtime_manager:8010"
    embeddings_url: str = "http://embeddings_api:7997"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "gateway-requests"
    mlflow_enabled: bool = True

    # --- Timeouts ---
    # MRM_HEALTH_TIMEOUT_SEC in docker-compose is 600.  Give ourselves a
    # small buffer on top so the gateway doesn't cut the connection before
    # MRM finishes starting the model.
    mrm_ensure_timeout: float = 660.0

    # Inference proxy: large models can produce long completions.
    proxy_timeout: float = 300.0

    # Separate connect / read timeouts for the inference proxy.
    # connect_timeout: TCP handshake + TLS, should be short.
    # read_timeout: per-read timeout; for SSE streams this is between chunks.
    connect_timeout: float = 5.0
    read_timeout: float = 300.0

    # --- Auto-provision ---
    # If True and a model is not yet in the MRM registry, the gateway will
    # attempt to register it from HuggingFace before ensuring.
    auto_provision: bool = False
    default_preset: str = "small_chat"
    default_gpu: str = "0"

    # --- Routing ---
    # Strategy for selecting one instance when MRM/Scheduler returns multiple.
    # Options: "least_loaded" | "round_robin" | "random"
    routing_strategy: str = "least_loaded"

    # --- Distributed mode ---
    # Set to True to route through the Scheduler instead of local MRM.
    # False = single-node mode (current behaviour); True = distributed mode.
    use_scheduler: bool = False
    scheduler_url: str = "http://scheduler:8030"
    # How long to cache a placement locally before re-asking the Scheduler.
    # 0 = always ask the Scheduler (safest); >0 = cache in memory (faster).
    placement_cache_ttl_sec: float = 30.0

    # --- CPU / GPU hybrid routing ---
    # Comma-separated map of GPU model → CPU variant.
    # Format: "gpu_model_id:cpu_model_id,..."
    # Example: "mistral-7b-instruct:mistral-7b-gguf,llama3-8b:llama3-8b-gguf"
    cpu_model_map: str = ""

    # Requests with max_tokens <= this threshold are eligible for CPU routing
    # when runtime_preference="auto". 0 = disable auto-CPU routing.
    cpu_routing_max_tokens_threshold: int = 300

    # When True, streaming requests are always sent to GPU regardless of
    # runtime_preference="auto" (CPU latency is too high for real-time SSE).
    cpu_routing_block_streaming: bool = True

    # --- Circuit breaker (CPU runtime) ---
    # Number of consecutive errors before the circuit opens.
    cpu_cb_failure_threshold: int = 5
    # Seconds to wait in OPEN state before probing again (HALF_OPEN).
    cpu_cb_reset_timeout_sec: float = 30.0

    # CORS: comma-separated list of allowed origins, or "*" to allow all
    cors_origins: str = "*"

    # --- Observability ---
    # OTel OTLP/gRPC endpoint (e.g. "http://jaeger:4317").
    # Empty string disables tracing.
    otel_endpoint: str = ""
    otel_service_name: str = "gateway"


settings = Settings()
