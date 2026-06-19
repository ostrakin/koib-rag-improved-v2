#!/usr/bin/env bash
# Запуск KOIB RAG API с настройками под небольшой сервер (2 ГБ ОЗУ).
set -euo pipefail
cd "$(dirname "$0")"

# ── Ограничиваем число потоков BLAS/OpenMP/torch: на маленькой машине это и
#    экономит ОЗУ, и убирает бессмысленную конкуренцию за CPU. ──
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Меньше арен glibc → ниже резидентная память (RSS).
export MALLOC_ARENA_MAX=2

# Виртуальное окружение, если есть
if [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Один воркер: модель + индекс загружаются один раз на процесс.
exec uvicorn api.app:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 30
