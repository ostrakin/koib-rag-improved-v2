# KOIB RAG — единый репозиторий, две сборки

RAG-система вопросов и ответов по документации КОИБ. Код хранится в одном месте
(никаких дублей функций) и раскладывается на два приложения:

- **koib-indexer** — индексация на мощном компьютере (парсинг, OCR, чанкинг,
  построение FAISS + BM25 + DocStore). Точка входа: `index_app.py`.
- **koib-server** — обслуживание запросов на сервере (2 ГБ ОЗУ), без тяжёлых
  зависимостей. Точка входа: `serve_app.py`.

Подробности рефакторинга и список устранённых дублей — в `ANALYSIS.md`.

## Структура

```
.
├── config.py                 # общая конфигурация (.env)
├── index_app.py              # ▶ приложение ИНДЕКСАЦИИ
├── serve_app.py              # ▶ приложение СЕРВЕРА
├── make_apps.py              # сборка dist/koib-indexer и dist/koib-server
├── src/                      # единственная копия всех модулей
├── api/                      # FastAPI (deps.py + routes)
├── tools/index_transfer.py   # verify / export / pack / unpack индекса
├── deploy/                   # Dockerfile, docker-compose, koib.service, run_server.sh
├── extras/                   # необязательное (не входит в сборки)
├── .env.indexer.example
├── .env.server.example
├── requirements-indexer.txt
├── requirements-server.txt
└── requirements-torch-cpu.txt
```

## Быстрый старт

### 1. Индексация (на компьютере)
```bash
cp .env.indexer.example .env
pip install -r requirements-indexer.txt      # нужен системный tesseract-ocr +rus
# положить документы в data/docs/
python index_app.py ingest --rebuild
python index_app.py verify
python index_app.py pack --out index_bundle.zip
```

### 2. Перенос индекса
```bash
scp index_bundle.zip user@server:/opt/koib-server/
```

### 3. Сервер
```bash
cp .env.server.example .env                   # модель/префиксы как при индексации!
pip install -r requirements-torch-cpu.txt
pip install -r requirements-server.txt
python -m tools.index_transfer unpack --archive index_bundle.zip --output-dir ./output
python serve_app.py --serve                   # или uvicorn api.app:app (см. deploy/)
```

## Сборка двух приложений отдельно
```bash
python make_apps.py --zip
# → dist/koib-indexer(.zip), dist/koib-server(.zip)
```
