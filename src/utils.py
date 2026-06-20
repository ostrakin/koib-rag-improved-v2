
# -*- coding: utf-8 -*-
"""
Koib-V-4.8 — Общие утилиты + Память диалога
============================================
★ База: clean_text, text_hash, estimate_tokens, детекция моделей, парсинг
★ НОВОЕ: ConversationMemory (SQLite) + Query Rewriting для RAG-пайплайна
"""
import re
import uuid
import hashlib
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import (
    METADATA_DIR,
    OCR_PREPROCESSING_ENABLED,
    OCR_DEEP_CLEAN,
    OCR_DROP_GARBAGE_LINES,
    OCR_SPLIT_GLUED_WORDS,
)
from .preprocessing import preprocess_light, deep_clean as _deep_clean

logger = logging.getLogger("koib.utils")


# ═══════════════════════════════════════════════════════════════
# Базовые утилиты
# ═══════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Очистка текста от мусорных символов."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(
        r'[^\w\s\-\+\=\*\/\(\)\[\]\{\}\$\<\>\,\.\;\:\!\?\%\&\|\^\~`\"\'\\@\#№°'
        r'±≥≤≈×÷→←↑↓∈∑∫∂∇∞≈≠√∏∝∧∨¬⊂⊃⊆⊇∅∩∪'
        r'\u0400-\u04FF\u2116\n\r\t]',
        '', text, flags=re.UNICODE
    )
    lines = [line.strip() for line in text.split('\n')]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)


def text_hash(text: str) -> str:
    """SHA-256 хэш текста, укороченный до 16 символов."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """
    Оценка количества токенов для русского (BPE).
    1 токен ≈ 2.5 символа (коэффициент 0.4).
    """
    if not text:
        return 0
    return max(1, int(len(text) * 0.4))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Обрезать текст до заданного количества токенов."""
    if not text:
        return ""
    max_chars = int(max_tokens * 2.5)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."


def generate_unique_id(prefix: str = "") -> str:
    """Сгенерировать уникальный 12-символьный ID."""
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}{uid}" if prefix else uid


# ═══════════════════════════════════════════════════════════════
# Детекция моделей КОИБ в тексте и именах файлов
# ═══════════════════════════════════════════════════════════════
KNOWN_MODELS = {"koib2010", "koib2017a", "koib2017b"}

# Человекочитаемые ярлыки для бота и ответов.
MODEL_LABELS: Dict[str, str] = {
    "koib2010": "КОИБ-2010",
    "koib2017a": "КОИБ-2017А",
    "koib2017b": "КОИБ-2017Б",
    "unknown": "не определена",
}

# «Сильные» сигналы — децимальные/модельные коды и титульные формулировки.
# Только они могут ЗАКРЕПИТЬ модель за документом (см. StickyModelTagger).
# Слабое упоминание соседней модели в тексте не должно перебивать титул.
STRONG_MODEL_PATTERNS: Dict[str, List[str]] = {
    "koib2010": [
        r"17404049\.438900\.001",
        r"1097746072797\.58\.29\.13\.000\.001",
        r"2010\s+МОДЕЛИ\s+17404049",
    ],
    "koib2017a": [
        r"17404049\.5013009\.008-01",
        r"1027700094949\.58\.29\.13\.000\.016",
        r"2017\s+МОДЕЛИ\s+17404049\.5013009",
    ],
    "koib2017b": [
        r"БАВУ\.201119",
        r"0912053",
    ],
}

# «Слабые» сигналы — текстовые упоминания; используются, только если сильного нет.
KOIB_MODEL_PATTERNS: Dict[str, List[str]] = {
    "koib2010": [
        r"КОИБ[-\s]?2010", r"КОИБ\s*2010", r"0912054",
        r"PRINT_KOIB2010", r"2010.*руководство",
        r"17404049\.438900\.001",
    ],
    "koib2017a": [
        r"КОИБ[-\s]?2017\s*[АA]\b", r"КОИБ[-\s]?2017А",
        r"17404049\.5013009\.008-01",
        r"17404049\.5013009", r"PRINT_KOIB2017[АA]",
    ],
    "koib2017b": [
        r"КОИБ[-\s]?2017\s*[БB]\b", r"КОИБ[-\s]?2017Б",
        r"БАВУ\.201119", r"0912053", r"PRINT_KOIB2017[БB]",
    ],
}

_STRONG_COMPILED = {
    model: [re.compile(p, re.IGNORECASE) for p in pats]
    for model, pats in STRONG_MODEL_PATTERNS.items()
}

_MODEL_PATTERNS = [
    re.compile(r'\b([A-ZА-Я]{2,}[\-\s]?\d{1,4}[A-ZА-Яа-я0-9\-/]*)\b'),
    re.compile(r'\b(модель\s+[A-ZА-Яа-я0-9\-/]+)\b', re.IGNORECASE),
]

_FILENAME_MODEL_PATTERNS = [
    re.compile(r'([A-ZА-ЯЁ]{2,}[\-]?\d{2,4}[A-ZА-ЯЁ0-9\-]*)', re.IGNORECASE),
]


class ModelDetection(tuple):
    """Tuple-compatible result: can be unpacked as (name, confidence) and compared to str."""
    def __new__(cls, name: str, confidence: float):
        return tuple.__new__(cls, (name, confidence))

    @property
    def name(self) -> str:
        return self[0]

    @property
    def confidence(self) -> float:
        return self[1]

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return tuple.__eq__(self, other)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"ModelDetection(name={self.name!r}, confidence={self.confidence!r})"


def detect_model_in_text(text: str) -> ModelDetection:
    """
    Определить модель КОИБ в тексте.
    Возвращает (model_name, confidence). Сильные сигналы (децимальные коды)
    дают высокую уверенность сразу.
    """
    if not text or len(text.strip()) < 5:
        return ModelDetection("unknown", 0.0)

    strong = detect_strong_model(text)
    if strong.name in KNOWN_MODELS:
        return strong

    scores: Dict[str, float] = {}
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        match_count = 0
        for pat in patterns:
            if re.findall(pat, text, re.IGNORECASE):
                match_count += 1
        if match_count > 0:
            scores[model_key] = match_count

    if scores:
        best = max(scores, key=scores.get)
        confidence = min(scores[best] / 3.0, 1.0)
        return ModelDetection(best, round(confidence, 3))

    for pattern in _MODEL_PATTERNS:
        match = pattern.search(text)
        if match:
            return ModelDetection(match.group(1).strip(), 0.3)

    return ModelDetection("unknown", 0.0)


def detect_model_from_filename(filename: str) -> str:
    """Определить модель КОИБ по имени файла."""
    fn = filename.lower()
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, fn, re.IGNORECASE):
                return model_key

    for pattern in _FILENAME_MODEL_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group(1).strip()

    return "unknown"


# Категории документов. Правовые и процедурные документы кросс-модельны (не
# привязаны к конкретной модели КОИБ-2010/2017), поэтому их нельзя класть в
# фасет ``model`` — для них заводим отдельный фасет ``doc_category``.
_CATEGORY_FILENAME_PATTERNS: Dict[str, List[str]] = {
    "legal": ["постанов", "цик", "пост-", "пост.", "пост ", "_114_", "896-8", "1148-7"],
    "procedure": ["инструкц", "порядк", "регламент", "139", "едг"],
    "form": ["акт", "протокол", "приложение", "zp21"],
}


def detect_doc_category(filename: str, text: str = "") -> str:
    """Грубая категория документа: ``manual`` / ``legal`` / ``procedure`` / ``form``.

    Используется как самостоятельный фасет фильтрации. Правовые и процедурные
    документы считаются кросс-модельными и не должны отсеиваться строгим
    фильтром по модели КОИБ (см. retrieval._apply_model_policy).
    """
    fn = (filename or "").lower()
    for cat in ("legal", "procedure", "form"):
        if any(p in fn for p in _CATEGORY_FILENAME_PATTERNS[cat]):
            return cat
    # руководства по эксплуатации с распознанной моделью — это manual
    if detect_model_from_filename(filename) in KNOWN_MODELS or "руковод" in fn or "эксплуат" in fn:
        return "manual"
    return "manual"


def detect_strong_model(text: str) -> ModelDetection:
    """Найти ТОЛЬКО сильный сигнал модели (децимальный код / титул).

    Возвращает (model, 0.95) при попадании, иначе ('unknown', 0.0). Используется
    «липким» теггером: только сильный сигнал закрепляет модель за документом.
    """
    if not text:
        return ModelDetection("unknown", 0.0)
    scores: Dict[str, int] = {}
    for model_key, patterns in _STRONG_COMPILED.items():
        hits = sum(1 for p in patterns if p.search(text))
        if hits:
            scores[model_key] = hits
    if scores:
        best = max(scores, key=scores.get)
        return ModelDetection(best, 0.95)
    return ModelDetection("unknown", 0.0)


def model_label(model: str) -> str:
    """Человекочитаемое имя модели для ответов и бота."""
    return MODEL_LABELS.get((model or "unknown"), model or "не определена")


class StickyModelTagger:
    """Назначает модель КОИБ страницам склеенного PDF, «залипая» на титулах.

    Логика: один раз увидев СИЛЬНЫЙ сигнал (децимальный код / титульный лист),
    закрепляем модель за всеми последующими страницами, пока не встретится новый
    сильный сигнал другой модели. Слабые текстовые упоминания соседних моделей
    (например, сравнение «КОИБ-2010, 2017Б» внутри руководства 2017А) НЕ
    перебивают закреплённую модель. Это устраняет преобладание model='unknown'
    и смешивание интерфейсов 2010/2017.
    """

    def __init__(self, default_model: str = "unknown"):
        self.current = default_model if default_model in KNOWN_MODELS else "unknown"

    def feed(self, text: str) -> str:
        """Обновить состояние по тексту страницы и вернуть модель для неё."""
        strong = detect_strong_model(text)
        if strong.name in KNOWN_MODELS and strong.confidence >= 0.9:
            self.current = strong.name
            return self.current
        if self.current in KNOWN_MODELS:
            return self.current
        # сильного сигнала ещё не было — пробуем слабую детекцию как временную
        weak = detect_model_in_text(text)
        if weak.name in KNOWN_MODELS and weak.confidence >= 0.6:
            return weak.name
        return "unknown"

    def assign_pages(self, pages: List[str]) -> List[str]:
        """Назначить модель списку страниц в порядке следования."""
        return [self.feed(p or "") for p in pages]


def resolve_model(raw_model: str, content: str, source: str = "") -> str:
    """Вернуть валидную модель КОИБ.

    Если в метаданных уже стоит известная модель — оставить её. Иначе определить
    по содержимому и имени источника. Мусорные «модели» (коды документов и т.п.)
    превращаются в 'unknown'.
    """
    m = (raw_model or "").strip()
    if m in KNOWN_MODELS:
        return m
    by_text = detect_model_in_text(content)
    if by_text.name in KNOWN_MODELS and by_text.confidence > 0.3:
        return by_text.name
    by_name = detect_model_from_filename(source)
    if by_name in KNOWN_MODELS:
        return by_name
    return "unknown"


# ═══════════════════════════════════════════════════════════════
# Парсинг заголовков и подписей к рисункам
# ═══════════════════════════════════════════════════════════════
_FIGURE_CAPTION_PATTERNS = [
    re.compile(r'(?:Рис\.|Рисунок)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Схема|схема)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(?:Чертёж|чертёж)\s*\d+[\.\:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
]


def find_figure_caption(text: str) -> str:
    """Найти подпись к рисунку в тексте."""
    for pattern in _FIGURE_CAPTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


_HEADING_PATTERNS = [
    re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$'),
    re.compile(r'^([А-ЯЁ][А-ЯЁ\s]{2,})$'),
]


def extract_headings(text: str) -> List[str]:
    """Извлечь заголовки из текста."""
    headings = []
    for line in text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 4:
            continue
        for pattern in _HEADING_PATTERNS:
            if pattern.match(line_stripped):
                headings.append(line_stripped)
                break
    return headings


# ═══════════════════════════════════════════════════════════════
# ★ НОВОЕ: Память диалога (SQLite для persistence)
# ═══════════════════════════════════════════════════════════════
class ConversationMemory:
    """
    Хранение истории диалога для Query Rewriting.
    SQLite для persistence между рестартами.
    """
    def __init__(self, db_path: Optional[Path] = None, max_history: int = 5):
        self.db_path = db_path or (METADATA_DIR / "conversation_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    user_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_timestamp
                ON history(user_id, timestamp)
            ''')

    async def add_message(self, user_id: str, role: str, content: str):
        """Добавить сообщение в историю (async wrapper)."""
        await asyncio.to_thread(self._sync_add, user_id, role, content)

    def _sync_add(self, user_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                'INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)',
                (user_id, role, content),
            )

    async def get_history(self, user_id: str) -> List[Dict[str, str]]:
        """Получить последние N сообщений пользователя."""
        return await asyncio.to_thread(self._sync_get, user_id)

    def _sync_get(self, user_id: str) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute('''
            SELECT role, content FROM history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, self.max_history))
        rows = cur.fetchall()
        return [{'role': row[0], 'content': row[1]} for row in reversed(rows)]

    async def clear_history(self, user_id: str):
        """Очистить историю пользователя."""
        await asyncio.to_thread(self._sync_clear, user_id)

    def _sync_clear(self, user_id: str):
        with self.conn:
            self.conn.execute(
                'DELETE FROM history WHERE user_id = ?',
                (user_id,),
            )


# ═══════════════════════════════════════════════════════════════
# ★ НОВОЕ: Query Rewriting (разрешение местоимений)
# ═══════════════════════════════════════════════════════════════
QUERY_REWRITE_PROMPT = """История диалога:
{history}

Текущий вопрос пользователя:
{query}

Переформулируй текущий вопрос в самостоятельный запрос для поиска по технической документации.
Раскрой все местоимения ("она", "его", "этот параметр") на основе контекста диалога.
Верни ТОЛЬКО переформулированный вопрос, без пояснений."""


async def rewrite_query(
    query: str,
    history: List[Dict[str, str]],
    llm_client,
) -> str:
    """
    Переформулировать запрос с учётом истории диалога.
    Разрешает местоимения: "а какая у неё мощность?" -> "какая мощность у КОИБ-2017?"

    Args:
        query: Текущий запрос пользователя
        history: История диалога от ConversationMemory
        llm_client: Экземпляр LLMClient для генерации

    Returns:
        Переформулированный запрос (или оригинальный, если rewriting не удался)
    """
    if not history or len(history) < 2:
        return query

    history_text = '\n'.join(
        f"{msg['role'].capitalize()}: {msg['content'][:200]}"
        for msg in history[-4:]
    )
    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, query=query)

    try:
        rewritten = await llm_client.generate_async(
            prompt, max_tokens=150, temperature=0.01
        )
        rewritten = rewritten.strip()
        if 10 < len(rewritten) < 500:
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as exc:
        logger.warning(f"Query rewrite failed: {exc}")

    return query


# ═══════════════════════════════════════════════════════════════
# OCR/RAG artifact normalization helpers
# ═══════════════════════════════════════════════════════════════
def strip_embedding_prefix(text: str) -> str:
    """Remove service prefixes (E5 passage/query) that must never be shown to users or stored in chunks."""
    if not text:
        return ""
    result = str(text)
    # Some old artifacts contain repeated prefixes after re-exporting vectorstore docs.
    for _ in range(3):
        stripped = result.lstrip()
        lowered = stripped.lower()
        if lowered.startswith("passage:"):
            result = stripped[len("passage:"):].lstrip()
            continue
        if lowered.startswith("query:"):
            result = stripped[len("query:"):].lstrip()
            continue
        break
    return result


def collapse_hyphenation(text: str) -> str:
    """Join OCR/PDF hyphenation split across line breaks without touching normal hyphens."""
    if not text:
        return ""
    text = re.sub(r"(?<=[А-Яа-яA-Za-z])[-¬]\s*\n\s*(?=[А-Яа-яA-Za-z])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_ocr_text(text: str) -> str:
    """Conservative text normalization for OCR/DOCX/CSV artifacts.

    Лёгкая предобработка (доменные замены komb→КОИБ, починка гомоглифов) включена
    всегда, когда OCR_PREPROCESSING_ENABLED. Тяжёлые шаги (отсев мусора, разбиение
    слипшихся слов) сюда НЕ входят — они в deep_clean_ocr_text для этапа индексации.
    """
    text = strip_embedding_prefix(text)
    text = collapse_hyphenation(text)
    # Drop OOXML fragments that sometimes leak from OCR-to-DOCX conversion.
    text = re.sub(r"</?w:[^>]+>", " ", text)
    text = re.sub(r"</?mc:[^>]+>", " ", text)
    text = re.sub(r"<[^>]{1,80}>", " ", text)
    if OCR_PREPROCESSING_ENABLED:
        text = preprocess_light(text)
    text = clean_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deep_clean_ocr_text(text: str) -> str:
    """Глубокая очистка OCR для этапа индексации (мощная машина).

    Поверх normalize_ocr_text дополнительно отсекает мусорные строки и разрезает
    слипшиеся слова. На боевом сервере не используется — текст уже очищен в индексе.
    """
    text = normalize_ocr_text(text)
    if OCR_PREPROCESSING_ENABLED and OCR_DEEP_CLEAN:
        text = _deep_clean(
            text,
            drop_garbage=OCR_DROP_GARBAGE_LINES,
            split_words=OCR_SPLIT_GLUED_WORDS,
        )
        # повторная нормализация пробелов после разбиения слов
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def looks_like_jsonl_artifact(path: Path) -> bool:
    """Detect old index artifacts so they are not indexed as plain text by accident."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="ignore") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    return False
                import json
                obj = json.loads(line)
                keys = set(obj.keys())
                return bool({"chunk_id", "content", "metadata"} <= keys or {"embedding", "content"} <= keys)
    except Exception:
        return False
    return False


