# KOIB RAG — серверная сборка (v4.12)

Готовая к развёртыванию сборка, которая **только обслуживает запросы**: гибридный
поиск (FAISS + BM25) по заранее построенному индексу и генерация ответа через
GigaChat. Тяжёлая индексация (парсинг PDF/DOCX, OCR, vision-описания) сюда **не
входит** — её модули и зависимости удалены.

Рассчитана на небольшой сервер: **2 ГБ ОЗУ, CPU, без GPU**.

---

## Что внутри

```
koib-server/
├── api/                     FastAPI: /health /ready /query /vk_callback
├── src/                     серверное ядро (поиск, генерация, валидация, VK-бот)
├── config.py                конфигурация (читает .env)
├── main.py                  CLI: --serve и --query (проверочный)
├── export_index.py          проверка/экспорт загруженного индекса (только stdlib)
├── requirements-torch-cpu.txt   CPU-сборка torch (ставится первой!)
├── requirements-server.txt      остальные зависимости (без OCR/парсинга)
├── .env.server.example      шаблон конфигурации → скопировать в .env
├── run_server.sh            запуск с лимитами потоков под 2 ГБ
├── setup_swap.sh            создание 2 ГБ swap (страховка от OOM)
├── koib.service             unit для systemd с лимитами памяти
├── Dockerfile / docker-compose.yml   опциональный путь через контейнер
├── docs/VK_BOT_SETUP.md     настройка VK Callback API
└── output/                  ← СЮДА загрузить готовый индекс (см. PLACE_INDEX_HERE.txt)
```

**Удалено из серверной сборки** (нужно только для индексации/оценки):
`src/parsing.py`, `src/chunking.py`, `src/figure_captioning.py`, `src/artifacts.py`,
`src/jsonl_repair.py`, `src/evaluation.py`, `src/logging_module.py`,
`batch_ingest.py`, `build_ideal_index.py`, `tests/`. За счёт этого с сервера ушли
тяжёлые зависимости `pymupdf`, `python-docx`, `pillow`, `pytesseract`.

---

## Память: почему 2 ГБ хватает (и где грань)

Весь веб-слой импортируется без тяжёлых библиотек — `torch`, `faiss`,
`sentence-transformers`, `langchain` подгружаются **лениво, при первом запросе**.
Поэтому сервер стартует мгновенно и легко, а память расходуется так:

| Компонент                                | ОЗУ (примерно) |
|------------------------------------------|----------------|
| Python + uvicorn + FastAPI               | ~150 МБ        |
| PyTorch (CPU) + sentence-transformers    | ~350–500 МБ    |
| Модель `multilingual-e5-small` (384-dim) | ~450 МБ        |
| FAISS-индекс в ОЗУ (≈1.5 КБ/чанк)         | ~80–300 МБ     |
| numpy / langchain / pymorphy2            | ~150 МБ        |
| **Итого в установившемся режиме**        | **~1.2–1.6 ГБ**|

Грань близко, поэтому в сборке предусмотрены три страховки: CPU-сборка torch,
ограничение потоков (1 поток BLAS/OpenMP), один воркер uvicorn и swap-файл.

**Чего НЕ включать на 2 ГБ** (каждый пункт грузит ещё одну torch-модель → OOM):
`USE_RERANKER`, `USE_ONNX_RERANKER`, `USE_HYDE`, `FIGURE_CAPTIONING_ENABLED`,
несколько воркеров uvicorn. В `.env.server.example` они уже выключены.

---

## Установка (нативно, рекомендуется)

```bash
# 0) распаковать архив, например в /opt
sudo mkdir -p /opt/koib-server && sudo tar -xf koib-server.tar.gz -C /opt/koib-server --strip-components=1
cd /opt/koib-server

# 1) swap — обязательно на 2 ГБ ОЗУ
sudo ./setup_swap.sh

# 2) окружение
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# 3) сначала CPU-torch (иначе подтянется CUDA-сборка на ~2.5 ГБ!)
pip install -r requirements-torch-cpu.txt
# 4) затем остальное
pip install -r requirements-server.txt

# 5) конфигурация
cp .env.server.example .env
nano .env          # вписать GIGACHAT_CREDENTIALS и VK_* ; проверить префиксы эмбеддингов

# 6) положить индекс
#    распакуйте вашу папку output/ так, чтобы получилось ./output/index/*, ./output/docstore/*
python export_index.py --output-dir ./output     # проверка целостности

# 7) пробный запрос (загрузит модель — первый раз медленно, скачает ~470 МБ)
python main.py --query "Как включить КОИБ?"

# 8) запуск сервера
./run_server.sh
```

Проверка:

```bash
curl http://localhost:8000/health   # {"status":"ok",...}
curl http://localhost:8000/ready    # покажет, какие индексы найдены
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"query":"Как включить КОИБ?","top_k":4}'
```

### Офлайн-режим после первого запуска

Модель скачивается из Hugging Face **один раз** и кэшируется. После первого
успешного запроса можно отключить сеть к HF, поставив в `.env`:

```
HF_OFFLINE_MODE=true
```

---

## Запуск как сервис (systemd)

```bash
sudo useradd -r -s /usr/sbin/nologin koib
sudo chown -R koib:koib /opt/koib-server
sudo cp /opt/koib-server/koib.service /etc/systemd/system/koib.service
sudo systemctl daemon-reload
sudo systemctl enable --now koib
sudo systemctl status koib
journalctl -u koib -f
```

Unit ограничивает память (`MemoryMax=1800M`) и при распухании перезапускает
процесс, а не подвешивает хост.

---

## Запуск в Docker (опционально)

```bash
cp .env.server.example .env && nano .env
# индекс — в ./output (монтируется внутрь контейнера)
docker compose up -d --build
docker compose logs -f
```

Образ по умолчанию запекает модель эмбеддингов (офлайн-старт). Лимиты памяти под
2 ГБ заданы в `docker-compose.yml`. На самой машине swap всё равно желателен.

---

## Эндпоинты API

| Метод | Путь           | Назначение                                            |
|-------|----------------|-------------------------------------------------------|
| GET   | `/health`      | живость сервиса                                       |
| GET   | `/ready`       | готовность: какие индексы найдены в `output/`         |
| GET   | `/`            | информация о сервисе, ссылка на `/docs`               |
| POST  | `/query`       | вопрос → ответ + источники                            |
| POST  | `/vk_callback` | webhook VK Callback API (см. `docs/VK_BOT_SETUP.md`)  |

Тело `POST /query`:

```json
{ "query": "текст вопроса", "top_k": 4, "model_filter": "", "use_memory": false, "validate": true }
```

---

## Важные замечания

- **Префиксы эмбеддингов.** `PASSAGE_PREFIX` / `QUERY_PREFIX` / `LOCAL_EMBEDDING_MODEL`
  в `.env` обязаны совпадать с теми, что были при индексации. Это требование
  модели e5: иначе вектор запроса не ляжет в то же пространство, что векторы
  индекса, и релевантность упадёт. Менять — только если меняли при сборке индекса.
- **Первый запрос медленный** (10–30 с): грузится модель и FAISS. Можно сделать
  «прогрев» сразу после старта — отправить один `POST /query`.
- **GigaChat** работает по сети (`ngw.devices.sberbank.ru`, `gigachat.devices.sberbank.ru`);
  откройте исходящий HTTPS. Локальной памяти LLM не потребляет.
- **Известное предупреждение Pydantic.** Поле `validate` в модели запроса
  затеняет атрибут `BaseModel.validate` — это лишь `UserWarning`, на работу не
  влияет (поле принимается и используется). Если захотите убрать — переименуйте
  поле и в `api/routes/query.py`, и в вызове `pipeline.answer(...)`.
