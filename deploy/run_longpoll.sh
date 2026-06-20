#!/usr/bin/env bash
# Запуск VK Long Poll бота КОИБ с настройками под небольшой сервер (2 ГБ ОЗУ).
set -euo pipefail
cd "$(dirname "$0")/.."

# Ограничиваем число потоков BLAS/OpenMP/torch — экономия ОЗУ и CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=2

# Виртуальное окружение, если есть
if [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

exec python run_longpoll.py
