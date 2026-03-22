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

    # --- Observability ---
    # OTel OTLP/gRPC endpoint (e.g. "http://jaeger:4317").
    # Empty string disables tracing.
    otel_endpoint: str = ""
    otel_service_name: str = "gateway"


settings = Settings()
