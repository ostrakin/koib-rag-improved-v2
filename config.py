# -*- coding: utf-8 -*-
"""Централизованная конфигурация KOIB RAG."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "KOIB RAG API"
APP_VERSION = os.getenv("KOIB_VERSION", "4.12.0")

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = Path(os.getenv("KOIB_DOCS_DIR", str(DATA_DIR / "docs"))).expanduser()
ARTIFACTS_DIR = Path(os.getenv("KOIB_ARTIFACTS_DIR", str(DATA_DIR / "artifacts"))).expanduser()
OUTPUT_DIR = Path(os.getenv("KOIB_OUTPUT_DIR", str(BASE_DIR / "output"))).expanduser()
INDEX_DIR = OUTPUT_DIR / "index"
DOCSTORE_DIR = OUTPUT_DIR / "docstore"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"

# LLM providers: gigachat | openai | local
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat").lower().strip()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower().strip()

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "passage: ")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "query: ")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ВНИМАНИЕ: изменение размеров чанков требует ПЕРЕИНДЕКСАЦИИ (index_app.py ingest --rebuild).
# 500 токенов + Context Expansion даёт более точный поиск, чем крупные чанки по 800:
# мелкий чанк точнее матчится с вопросом, а соседи дотягиваются автоматически.
TEXT_CHUNK_SIZE = int(os.getenv("TEXT_CHUNK_SIZE", "500"))
TEXT_CHUNK_OVERLAP = int(os.getenv("TEXT_CHUNK_OVERLAP", "60"))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

VECTOR_SEARCH_K = int(os.getenv("VECTOR_SEARCH_K", "15"))
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "10"))
# С более мелкими чанками (500 ток.) в контекст берём больше фрагментов.
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "6"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "DiTy/ru-reranker-base")
USE_ONNX_RERANKER = os.getenv("USE_ONNX_RERANKER", "false").lower() == "true"
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"
BM25_USE_STOPWORDS = os.getenv("BM25_USE_STOPWORDS", "true").lower() == "true"
BM25_USE_LEMMATIZATION = os.getenv("BM25_USE_LEMMATIZATION", "true").lower() == "true"

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_TEMPERATURE = float(os.getenv("GIGACHAT_TEMPERATURE", "0.2"))
GIGACHAT_MAX_TOKENS = int(os.getenv("GIGACHAT_MAX_TOKENS", "1536"))
GIGACHAT_TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "45"))
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true"

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1536"))

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "IlyaGusev/saiga_mistral_7b")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434").rstrip("/")

VALIDATION_IGNORE_QUOTES = os.getenv("VALIDATION_IGNORE_QUOTES", "true").lower() == "true"
UNCERTAINTY_MIN_LENGTH = int(os.getenv("UNCERTAINTY_MIN_LENGTH", "50"))
VALIDATION_USE_LLM_JUDGE = os.getenv("VALIDATION_USE_LLM_JUDGE", "false").lower() == "true"
VALIDATION_CHECK_CITATIONS = os.getenv("VALIDATION_CHECK_CITATIONS", "true").lower() == "true"

OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "50"))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "80"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "80"))
PARSING_ENGINE = os.getenv("PARSING_ENGINE", "pymupdf")
ARTIFACT_ALLOW_BM25_FALLBACK = os.getenv("ARTIFACT_ALLOW_BM25_FALLBACK", "false").lower() == "true"
ARTIFACT_MIN_CONTENT_CHARS = int(os.getenv("ARTIFACT_MIN_CONTENT_CHARS", "30"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

VK_CONFIRM_CODE = os.getenv("VK_CONFIRM_CODE", "12345678")
VK_GROUP_ID = os.getenv("VK_GROUP_ID", "")
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN", "")
VK_SECRET_KEY = os.getenv("VK_SECRET_KEY", "")
VK_API_VERSION = os.getenv("VK_API_VERSION", "5.131")
VK_RATE_LIMIT_PER_MINUTE = int(os.getenv("VK_RATE_LIMIT_PER_MINUTE", "20"))
VK_GLOBAL_RATE_LIMIT_PER_MINUTE = int(os.getenv("VK_GLOBAL_RATE_LIMIT_PER_MINUTE", "120"))
VK_MAX_INCOMING_CHARS = int(os.getenv("VK_MAX_INCOMING_CHARS", "1500"))
VK_MAX_OUTGOING_CHARS = int(os.getenv("VK_MAX_OUTGOING_CHARS", "3900"))
VK_OUTBOUND_TIMEOUT = int(os.getenv("VK_OUTBOUND_TIMEOUT", "10"))
VK_DEDUP_TTL_SECONDS = int(os.getenv("VK_DEDUP_TTL_SECONDS", "86400"))
VK_SEND_TYPING = os.getenv("VK_SEND_TYPING", "true").lower() == "true"
VK_REPLY_IN_GROUP_CHATS = os.getenv("VK_REPLY_IN_GROUP_CHATS", "false").lower() == "true"
VK_BOT_MENTION_ALIASES = [
    alias.strip().lower()
    for alias in os.getenv("VK_BOT_MENTION_ALIASES", "коиб,koib").split(",")
    if alias.strip()
]
VK_ADMIN_IDS = {
    item.strip()
    for item in os.getenv("VK_ADMIN_IDS", "").split(",")
    if item.strip()
}
VK_ERROR_MESSAGE = os.getenv(
    "VK_ERROR_MESSAGE",
    "Не удалось подготовить ответ. Проверьте подключение к серверу и при необходимости обратитесь к администратору системы.",
)
PROCEDURAL_REMINDER_ENABLED = os.getenv("PROCEDURAL_REMINDER_ENABLED", "true").lower() == "true"

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_MAX_CANDIDATES = int(os.getenv("SEMANTIC_CACHE_MAX_CANDIDATES", "1000"))

MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "2"))
MAX_TABLE_ROWS_IN_PROMPT = int(os.getenv("MAX_TABLE_ROWS_IN_PROMPT", "30"))
MAX_TABLE_TOKENS_IN_PROMPT = int(os.getenv("MAX_TABLE_TOKENS_IN_PROMPT", "1500"))
USE_USHAPED_CONTEXT = os.getenv("USE_USHAPED_CONTEXT", "true").lower() == "true"

# ── Нарезка таблиц при индексации (Проблема 3) ──
# Слишком крупные таблицы ломают эмбеддинг и забивают контекст. Если таблица
# длиннее порога, она режется по строкам на несколько table-чанков; заголовок
# повторяется в каждой части, чтобы чанк оставался осмысленным. Рекорд в старом
# индексе — 26 918 символов в одной операционной таблице КОИБ.
TABLE_MAX_CHARS = int(os.getenv("TABLE_MAX_CHARS", "3500"))
TABLE_MIN_ROWS = int(os.getenv("TABLE_MIN_ROWS", "2"))
TABLE_MIN_COLS = int(os.getenv("TABLE_MIN_COLS", "2"))
TABLE_MIN_CONTENT_CHARS = int(os.getenv("TABLE_MIN_CONTENT_CHARS", "30"))

# ── OCR preprocessing (Проблема 2) ──
# Лёгкая чистка (замены + гомоглифы) применяется всегда. Глубокая (отсев мусора
# и разбиение слипшихся слов) включается только на индексации (мощная машина).
OCR_PREPROCESSING_ENABLED = os.getenv("OCR_PREPROCESSING_ENABLED", "true").lower() == "true"
OCR_DEEP_CLEAN = os.getenv("OCR_DEEP_CLEAN", "true").lower() == "true"
OCR_DROP_GARBAGE_LINES = os.getenv("OCR_DROP_GARBAGE_LINES", "true").lower() == "true"
OCR_SPLIT_GLUED_WORDS = os.getenv("OCR_SPLIT_GLUED_WORDS", "true").lower() == "true"

# ── Жёсткая фильтрация по модели КОИБ (Проблема 4) ──
# strict=true: если по нужной модели нашлось достаточно фрагментов, чанки с
# model='unknown' отбрасываются, чтобы не смешивать интерфейсы 2010/2017.
MODEL_FILTER_STRICT = os.getenv("MODEL_FILTER_STRICT", "true").lower() == "true"
MODEL_STRICT_MIN_EXACT = int(os.getenv("MODEL_STRICT_MIN_EXACT", "2"))

# ── Context Expansion (Проблема 5) ──
CONTEXT_EXPANSION_ENABLED = os.getenv("CONTEXT_EXPANSION_ENABLED", "true").lower() == "true"
CONTEXT_EXPANSION_WINDOW = int(os.getenv("CONTEXT_EXPANSION_WINDOW", "1"))
CONTEXT_EXPANSION_MAX_CHARS = int(os.getenv("CONTEXT_EXPANSION_MAX_CHARS", "2400"))

# ── Vision-описания рисунков (Проблема 7) ── только на этапе индексации
FIGURE_CAPTIONING_ENABLED = os.getenv("FIGURE_CAPTIONING_ENABLED", "false").lower() == "true"
FIGURE_CAPTION_PROVIDER = os.getenv("FIGURE_CAPTION_PROVIDER", "none").lower().strip()  # openai|local|none
FIGURE_CAPTION_MODEL = os.getenv("FIGURE_CAPTION_MODEL", "gpt-4o-mini")
FIGURE_CAPTION_MAX_TOKENS = int(os.getenv("FIGURE_CAPTION_MAX_TOKENS", "180"))
FIGURE_CAPTION_MAX_IMAGES = int(os.getenv("FIGURE_CAPTION_MAX_IMAGES", "0"))  # 0 = без лимита

# ── Стиль ответа и доработка LLM (улучшение UX) ──
# ANSWER_REFINE_ENABLED: второй проход LLM — анализирует вопрос и черновой ответ
# и переписывает его понятным дружелюбным языком (без сухой выдачи чанков).
# Сохраняет факты, цитаты [Документ: ...] и регламентное уведомление.
ANSWER_REFINE_ENABLED = os.getenv("ANSWER_REFINE_ENABLED", "true").lower() == "true"
ANSWER_REFINE_MAX_TOKENS = int(os.getenv("ANSWER_REFINE_MAX_TOKENS", "1024"))
ANSWER_REFINE_TEMPERATURE = float(os.getenv("ANSWER_REFINE_TEMPERATURE", "0.35"))

# ── Отправка релевантных рисунков пользователю ──
# Если среди найденных чанков есть figure с существующим файлом изображения,
# бот прикрепит его к ответу (VK: загрузка через photos.getMessagesUploadServer).
SEND_FIGURES_ENABLED = os.getenv("SEND_FIGURES_ENABLED", "true").lower() == "true"
MAX_FIGURES_PER_ANSWER = int(os.getenv("MAX_FIGURES_PER_ANSWER", "2"))

# ── Починка JSONL-артефактов (Проблема 1) ──
JSONL_REPAIR_ENABLED = os.getenv("JSONL_REPAIR_ENABLED", "true").lower() == "true"

INDEXING_BATCH_SIZE = int(os.getenv("INDEXING_BATCH_SIZE", "64"))
INDEXING_FLUSH_THRESHOLD = int(os.getenv("INDEXING_FLUSH_THRESHOLD", "2000"))
INDEXING_DEVICE = os.getenv("INDEXING_DEVICE", "cpu").lower().strip()
INGEST_MAX_WORKERS = int(os.getenv("INGEST_MAX_WORKERS", "0"))
HF_OFFLINE_MODE = os.getenv("HF_OFFLINE_MODE", "false").lower() == "true"


def get_device() -> str:
    """Определить устройство для embedding-модели."""
    if INDEXING_DEVICE != "auto":
        return INDEXING_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dirs() -> None:
    for d in [DOCS_DIR, ARTIFACTS_DIR, OUTPUT_DIR, INDEX_DIR, DOCSTORE_DIR, FIGURES_DIR, LOGS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def rel_figure_path(path) -> str:
    """Сохранить путь к рисунку относительно OUTPUT_DIR, с прямым слэшем.

    Раньше в метаданных чанков хранился абсолютный путь машины-индексатора
    (Q:\\...\\output\\figures\\x.png). После переноса индекса на сервер такой путь
    «висел» на чужой машине, и модальность рисунков не работала. Теперь хранится
    переносимое относительное значение «figures/x.png», а разрешается оно в
    рантайме через resolve_figure_path().
    """
    import os
    from pathlib import Path

    if not path:
        return ""
    p = Path(str(path))
    try:
        rel = p.resolve().relative_to(OUTPUT_DIR.resolve())
        return str(rel).replace(os.sep, "/")
    except Exception:
        # уже относительный или вне OUTPUT_DIR — берём как есть, нормализуя слэши
        return str(p).replace(os.sep, "/")


def resolve_figure_path(stored) -> str:
    """Разрешить сохранённый путь к рисунку в абсолютный путь на текущей машине.

    Принимает любой формат: «figures/x.png», «Q:\\...\\figures\\x.png», голое имя
    файла. Возвращает существующий файл, иначе пустую строку. Так индекс остаётся
    переносимым: относительный путь резолвится через текущий FIGURES_DIR, а
    унаследованный абсолютный — через basename, если он лежит в FIGURES_DIR.
    """
    import os

    if not stored:
        return ""
    raw = str(stored).strip()
    if not raw:
        return ""
    # 1) сам сохранённый путь как есть (абсолютный или относительный к CWD)
    if os.path.isfile(raw):
        return raw
    # 2) путь относительно текущего OUTPUT_DIR (канонический формат «figures/x.png»)
    cand = OUTPUT_DIR / raw
    if cand.is_file():
        return str(cand)
    # 3) только имя файла в текущем FIGURES_DIR (совместимость со старыми
    #    абсолютными путями и любым расположением figures/)
    base = os.path.basename(raw.replace("\\", "/"))
    if base:
        cand2 = FIGURES_DIR / base
        if cand2.is_file():
            return str(cand2)
    return ""
