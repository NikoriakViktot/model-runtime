FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -U pip

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the full package so that `..` relative imports in mrm/ resolve correctly.
# mrm/ is a subpackage of model_runtime_manager/, not a top-level package.
COPY __init__.py         /app/model_runtime_manager/__init__.py
COPY mrm                 /app/model_runtime_manager/mrm
COPY backends            /app/model_runtime_manager/backends
COPY core                /app/model_runtime_manager/core
COPY engines             /app/model_runtime_manager/engines
COPY infra               /app/model_runtime_manager/infra
COPY model_variant_registry.py  /app/model_runtime_manager/model_variant_registry.py
COPY execution_selector.py      /app/model_runtime_manager/execution_selector.py
COPY vram_estimator.py          /app/model_runtime_manager/vram_estimator.py
COPY model_inspector.py         /app/model_runtime_manager/model_inspector.py

# /app is on PYTHONPATH so `import model_runtime_manager` resolves
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "model_runtime_manager.mrm.app:app", "--host", "0.0.0.0", "--port", "8010"]
