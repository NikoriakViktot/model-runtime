FROM python:3.11-slim

# Build-time deps for llama-cpp-python (C/C++ compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CMAKE_ARGS enables OpenBLAS for faster CPU matrix ops
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8090

CMD ["uvicorn", "cpu_runtime.app:app", "--host", "0.0.0.0", "--port", "8090"]
