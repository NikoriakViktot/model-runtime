from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from pydantic import Field

from typing import Dict, List
import os


class ModelSpec(BaseModel):
    base_model: str
    model_alias: str
    container_name: str
    image: str
    launch_mode: str = "openai"
    hf_model: str
    served_model_name: str
    gpu_memory_utilization: float = 0.6
    max_model_len: int = 2048
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 1024
    enable_lora: bool = True
    max_loras: int = 30
    max_lora_rank: int = 32
    enforce_eager: bool = False
    port: int = 8000
    allowed_gpus: List[str] = ["0"]
    quantization: str | None = None
    volumes: Dict[str, str] = {}
    env: Dict[str, str] = {}
    ipc_host: bool = True
    dtype: str = "auto"
    shm_size: str = "8gb"
    health_path: str = "/health"

    # --- Hybrid runtime extension ---
    # "gpu" → vLLM container (existing behaviour, default)
    # "cpu" → cpu_runtime container (llama.cpp / GGUF)
    runtime: str = "gpu"

    # CPU-specific: path to GGUF file inside the mounted models volume
    gguf_path: str = ""
    # CPU resources allocated to the container
    cpu_cores: int = 4       # passed as CPU_RUNTIME_N_THREADS
    memory_mb: int = 6144    # container memory limit in MiB


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRM_", env_file=".env", extra="ignore")
    hf_token: str = ""

    redis_url: str = "redis://redis:6379/2"
    docker_network: str = "ai_net"
    idle_timeout_sec: int = 900
    sweep_interval_sec: int = 30
    health_timeout_sec: int = 500
    health_poll_interval_sec: float = 1.0

    artifacts_host_path: str = "./artifacts"
    hf_cache_host_path: str = "./hf_cache"
    lora_root_host_path: str | None = None
    model_registry: Dict[str, ModelSpec] = {}
    keep_failed_containers: bool = Field(default=False, env="MRM_KEEP_FAILED_CONTAINERS")
    litellm_config_path: str = "/shared/litellm.generated.yaml"
    litellm_container_name: str = "dev_litellm"
    litellm_autowrite: bool = True
    litellm_restart_on_write: bool = False

    # ── CPU auto-fallback ──────────────────────────────────────────────
    # When set, MRM will redirect to this URL instead of failing if
    # the Docker nvidia runtime is unavailable (LITE / no-GPU mode).
    # Example: "http://cpu_runtime:8090"
    auto_fallback_cpu_url: str = ""
    # The model alias reported when serving via the CPU fallback.
    # Should match CPU_RUNTIME_MODEL_ALIAS env var of the cpu_runtime container.
    auto_fallback_cpu_alias: str = "cpu-model"

    def get_lora_host_path(self) -> str:
        """
        Гарантує, що ми завжди повернемо валідний шлях до адаптерів на хості.
        """
        if self.lora_root_host_path and self.lora_root_host_path.strip():
            return self.lora_root_host_path

        # Fallback: якщо змінна не задана, беремо artifacts_host_path + /adapters
        # Це "Single Source of Truth" логіка
        return os.path.join(self.artifacts_host_path, "adapters")

    def load_default_registry(self) -> Dict[str, ModelSpec]:
        base_env = {
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", ""),
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "HUGGINGFACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
            "HUGGING_FACE_HUB_TOKEN": os.getenv("HF_TOKEN", ""),
        }

        def vols() -> Dict[str, str]:
            art = os.path.abspath(self.artifacts_host_path)
            hf = os.path.abspath(self.hf_cache_host_path)
            return {
                art: "/app/artifacts",
                hf: "/root/.cache/huggingface",
            }
        return {
            "Qwen/Qwen1.5-1.8B-Chat": ModelSpec(
                base_model="Qwen/Qwen1.5-1.8B-Chat",
                model_alias="qwen-1.8b-chat",
                container_name="vllm_qwen_1p8b_chat",
                image="vllm/vllm-openai:v0.13.0",
                hf_model="Qwen/Qwen1.5-1.8B-Chat",
                served_model_name="qwen-1.8b-chat",
                allowed_gpus=["0"],
                volumes=vols(),
                env=base_env,
                gpu_memory_utilization=0.60,
                max_model_len=2048,
                enable_lora=True,
                max_loras=30,
                max_lora_rank=64,
                enforce_eager=True,
                port=8000,
            ),
            # "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
            #     base_model="Qwen/Qwen2.5-7B-Instruct",
            #     model_alias="qwen-7b-instruct",
            #     container_name="vllm_qwen_7b_instruct",
            #     image="vllm/vllm-openai:v0.13.0",
            #     hf_model="Qwen/Qwen2.5-7B-Instruct-AWQ",
            #     served_model_name="qwen-7b-instruct",
            #     quantization="awq",
            #     allowed_gpus=["0"],
            #     volumes=vols(),
            #     env=base_env,
            #     gpu_memory_utilization=0.90,
            #     max_model_len=256,
            #     max_num_seqs=1,
            #     max_num_batched_tokens=1024,
            #     enable_lora=True,
            #     max_loras=2,
            #     max_lora_rank=32,
            #     port=8000,
            # ),

            # "mistralai/Mistral-Nemo-Instruct-2407": ModelSpec(
            #     base_model="mistralai/Mistral-Nemo-Instruct-2407",
            #     model_alias="mistral-nemo-instruct",
            #     container_name="vllm_mistral_nemo_instruct",
            #     image="vllm/vllm-openai:latest",
            #     hf_model="mistralai/Mistral-Nemo-Instruct-2407",
            #     served_model_name="mistral-nemo-instruct",
            #     allowed_gpus=["0"],
            #     volumes=vols(),
            #     env=base_env,
            #     gpu_memory_utilization=0.80,
            #     max_model_len=1024,
            #     enable_lora=True,
            #     max_loras=2,
            #     max_lora_rank=32,
            #     enforce_eager=True,
            #     port=8000,
            # ),
            # "mistralai/Mistral-7B-Instruct-v0.3": ModelSpec(
            #     base_model="mistralai/Mistral-7B-Instruct-v0.3",
            #     model_alias="mistral-7b-instruct",
            #     container_name="vllm_mistral_7b_instruct",
            #     image="vllm/vllm-openai:v0.13.0",
            #     launch_mode="openai",
            #     hf_model="mistralai/Mistral-7B-Instruct-v0.3",
            #     served_model_name="mistral-7b-instruct",
            #     allowed_gpus=["0"],
            #     volumes=vols(),
            #     env=base_env,
            #     gpu_memory_utilization=0.80,
            #     max_model_len=512,
            #     enable_lora=True,
            #     max_loras=2,
            #     max_lora_rank=32,
            #     enforce_eager=True,
            #     port=8000,
            # ),

        }
