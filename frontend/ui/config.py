#frontend/ui/config.py

import os

CP_URL = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")
DISPATCHER_URL = os.getenv("API_DISPATCHER_URL", "http://api_dispatcher:8005")
S3_BUCKET = os.getenv("AWS_S3_BUCKET") or os.getenv("AWS_BUCKET_NAME")

NEO4J_UI_URL = os.getenv("NEO4J_UI_URL", "http://localhost:7474")

MLFLOW_UI_PATH = os.getenv("MLFLOW_UI_PATH", "/mlflow/")
CP_SWAGGER_PATH = os.getenv("CP_SWAGGER_PATH", "/api/docs")
DISPATCH_SWAGGER_PATH = os.getenv("DISPATCH_SWAGGER_PATH", "/dispatch/docs")
VLLM_DOCS_PATH = os.getenv("VLLM_DOCS_PATH", "/docs")
LITELLM_BASE_PATH = os.getenv("LITELLM_BASE_PATH", "/v1/models")
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000")
MRM_BASE_URL = "http://model_runtime_manager:8010"

MODEL_MAP = {
  "qwen-1.8b-chat": "Qwen/Qwen1.5-1.8B-Chat",
  # "qwen-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
  # "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
}