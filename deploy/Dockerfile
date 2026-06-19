# ── KOIB RAG — серверная сборка (CPU, под 2 ГБ ОЗУ) ──
# Опциональный путь развёртывания. Образ ставит CPU-сборку torch и
# (по умолчанию) запекает модель эмбеддингов, чтобы рантайму не нужна была сеть.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Системные библиотеки для faiss/numpy (минимум)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU-torch отдельным слоем (иначе подтянется CUDA-сборка ~2.5 ГБ)
COPY requirements-torch-cpu.txt .
RUN pip install -r requirements-torch-cpu.txt

# 2) Остальные зависимости
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

# 3) Код
COPY . .

# 4) (опционально) запечь модель эмбеддингов в образ для офлайн-старта.
#    Отключить:  docker build --build-arg BAKE_MODEL=0 ...
ARG BAKE_MODEL=1
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-small')" ; \
    fi

# Непривилегированный пользователь
RUN useradd -r -u 10001 koib && chown -R koib:koib /app
USER koib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
