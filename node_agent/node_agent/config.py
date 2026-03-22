"""
node_agent/config.py

All configuration is read from environment variables with NODE_AGENT_ prefix.

Each physical server gets its own NODE_AGENT_NODE_ID and NODE_AGENT_AGENT_URL.
Everything else defaults sensibly for a co-located deployment (MRM on localhost).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NODE_AGENT_",
        env_file=".env",
        extra="ignore",
    )

    # --- Identity ---
    # Unique name for this server in the cluster.
    # Defaults to hostname-based auto-detection in app.py if left empty.
    node_id: str = ""

    # Full URL at which this agent is reachable from the scheduler.
    # Must be set — the scheduler uses this to call /local/ensure.
    agent_url: str = "http://localhost:8020"

    # --- Upstream services ---
    scheduler_url: str = "http://scheduler:8030"
    mrm_url: str = "http://localhost:8010"

    # --- Heartbeat ---
    heartbeat_interval_sec: int = 15

    # --- Observability ---
    otel_endpoint: str = ""
    otel_service_name: str = "node-agent"


settings = Settings()
