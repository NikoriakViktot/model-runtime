"""
scheduler/config.py

All configuration is read from environment variables with SCHEDULER_ prefix.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER_",
        env_file=".env",
        extra="ignore",
    )

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/3"

    # --- Node health ---
    # A node's Redis key expires after this many seconds.
    # Set to ~3× heartbeat interval so one missed beat isn't fatal.
    node_ttl_sec: int = 60

    # Threshold for marking a node STALE vs DEAD in the health enum.
    stale_threshold_sec: int = 30

    # --- Placement ---
    # "least_loaded" | "first_fit"
    placement_strategy: str = "least_loaded"

    # --- Per-node ensure timeout ---
    # Must exceed MRM's MRM_HEALTH_TIMEOUT_SEC (default 600s).
    node_ensure_timeout_sec: float = 700.0

    # --- Observability ---
    otel_endpoint: str = ""
    otel_service_name: str = "scheduler"


settings = Settings()
