# -*- coding: utf-8 -*-
"""Indexing: FAISS + SQLite FTS5 + SQLite DocStore.

v4.11 fixes:
- E5 ``passage:`` prefix is applied only inside embed_documents(), not stored in FAISS docs;
- DocStore stores every chunk, not only structured chunks, so BM25 results can display original text;
- BM25 keeps a normalized FTS column and a raw display column separately;
- legacy BM25 schema is rebuilt automatically.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import (
    BM25_USE_LEMMATIZATION,
    BM25_USE_STOPWORDS,
    DOCSTORE_DIR,
    EMBEDDING_PROVIDER,
    INDEXING_BATCH_SIZE,
    INDEXING_FLUSH_THRESHOLD,
    INDEX_DIR,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    PASSAGE_PREFIX,
    get_device,
)
from .utils import normalize_ocr_text, strip_embedding_prefix

logger = logging.getLogger("koib.indexing")

_GLOBAL_EMBEDDINGS = None
_GLOBAL_EMBEDDINGS_LOCK = threading.Lock()


class PassagePrefixEmbeddings:
    """Apply E5 passage prefix during embedding while keeping stored page_content clean."""

    def __init__(self, inner, passage_prefix: str = PASSAGE_PREFIX):
        self.inner = inner
        self.passage_prefix = passage_prefix or ""

    def _prefix_doc(self, text: str) -> str:
        text = text or ""
        if not self.passage_prefix:
            return text
        stripped = text.lstrip()
        if stripped.lower().startswith("passage:") or stripped.lower().startswith("query:"):
            return text
        return self.passage_prefix + text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.inner.embed_documents([self._prefix_doc(t) for t in texts])

    def embed_query(self, text: str) -> List[float]:
        # Query prefix is still controlled by retrieval/rag_pipeline to preserve old behavior.
        return self.inner.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.inner, "aembed_documents"):
            return await self.inner.aembed_documents([self._prefix_doc(t) for t in texts])
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        if hasattr(self.inner, "aembed_query"):
            return await self.inner.aembed_query(text)
        return self.embed_query(text)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def get_global_embeddings():
    global _GLOBAL_EMBEDDINGS
    if _GLOBAL_EMBEDDINGS is not None:
        return _GLOBAL_EMBEDDINGS
    with _GLOBAL_EMBEDDINGS_LOCK:
        if _GLOBAL_EMBEDDINGS is not None:
            return _GLOBAL_EMBEDDINGS
        if EMBEDDING_PROVIDER == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            device = get_device()
            logger.info(
                "Загрузка embedding-модели '%s' на устройстве %s (batch_size=%s)",
                LOCAL_EMBEDDING_MODEL,
                device,
                INDEXING_BATCH_SIZE,
            )
            base = HuggingFaceEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": INDEXING_BATCH_SIZE,
                },
            )
            _GLOBAL_EMBEDDINGS = PassagePrefixEmbeddings(base)
        elif EMBEDDING_PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings

            _GLOBAL_EMBEDDINGS = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
        return _GLOBAL_EMBEDDINGS


RU_STOPWORDS = {
    "и", "в", "на", "с", "по", "для", "из", "к", "от", "о", "об", "а", "но", "да",
    "не", "что", "как", "это", "то", "же", "бы", "вы", "мы", "он", "она", "они", "оно",
    "я", "ты", "его", "её", "их", "мой", "твой", "наш", "ваш", "свой", "этот", "тот",
    "такой", "который", "весь", "все", "вся", "всё", "быть", "был", "была", "было",
    "были", "будет", "есть", "нет", "ещё", "уже", "только", "если", "или", "при",
    "про", "за", "до", "после", "между", "через", "над", "под", "перед", "так",
    "тоже", "лишь", "ведь", "вот", "даже", "ну", "ли", "ни", "тебя", "мне", "мной",
    "ним", "ней", "нами", "вам", "вас", "нас", "них", "чего", "чему", "чем", "кем",
    "ком", "где", "когда", "зачем", "почему", "куда", "откуда", "какой", "какая", "какие",
}
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_MORPH_ANALYZER = None
_MORPH_LOCK = threading.Lock()


def _get_morph():
    global _MORPH_ANALYZER
    if not BM25_USE_LEMMATIZATION:
        return None
    if _MORPH_ANALYZER is not None:
        return _MORPH_ANALYZER
    with _MORPH_LOCK:
        if _MORPH_ANALYZER is not None:
            return _MORPH_ANALYZER
        try:
            import pymorphy3

            _MORPH_ANALYZER = pymorphy3.MorphAnalyzer()
        except Exception as exc:
            logger.warning("pymorphy3 недоступен, BM25 будет без лемматизации: %s", exc)
            _MORPH_ANALYZER = None
        return _MORPH_ANALYZER


def _lemmatize_token(token: str) -> str:
    morph = _get_morph()
    if morph is None:
        return token
    try:
        return morph.parse(token)[0].normal_form
    except Exception:
        return token


def _get_processed_tokens(text: str) -> List[str]:
    if not text:
        return []
    text = strip_embedding_prefix(text)
    raw_tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]
    if BM25_USE_STOPWORDS:
        raw_tokens = [t for t in raw_tokens if t not in RU_STOPWORDS]
    return [_lemmatize_token(t) for t in raw_tokens] if BM25_USE_LEMMATIZATION else raw_tokens


def tokenize_ru(text: str) -> str:
    return " ".join(_get_processed_tokens(text))


def prepare_fts_query(query: str) -> str:
    tokens = _get_processed_tokens(query)
    if not tokens:
        return ""
    seen, unique = set(), []
    for token in tokens[:20]:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return " OR ".join(f'"{token}"' for token in unique)


def _chunk_display_content(chunk) -> str:
    return normalize_ocr_text(chunk.full_content or chunk.content or "")


class DocStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DOCSTORE_DIR / "docstore.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS docstore "
                "(chunk_id TEXT PRIMARY KEY, content TEXT, chunk_type TEXT, metadata TEXT)"
            )
            # Порядок чанков внутри документа — для Context Expansion (Проблема 5).
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS chunk_order "
                "(source TEXT, seq INTEGER, chunk_id TEXT, page INTEGER, "
                "chunk_type TEXT, model TEXT, PRIMARY KEY(source, seq, chunk_id))"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_order ON chunk_order(source, seq)"
            )

    def add_many(self, chunks) -> None:
        rows = []
        order_rows = []
        for c in chunks:
            content = _chunk_display_content(c)
            if not content:
                continue
            meta = dict(c.metadata or {})
            meta.setdefault("chunk_type", c.chunk_type)
            rows.append((c.chunk_id, content, c.chunk_type, json.dumps(meta, ensure_ascii=False)))
            if "seq" in meta:
                try:
                    order_rows.append(
                        (
                            str(meta.get("source", "")),
                            int(meta.get("seq")),
                            c.chunk_id,
                            int(meta.get("page", 0) or 0),
                            c.chunk_type,
                            str(meta.get("model", "unknown")),
                        )
                    )
                except Exception:
                    pass
        if rows:
            with self.lock, self.conn:
                self.conn.executemany("INSERT OR REPLACE INTO docstore VALUES (?, ?, ?, ?)", rows)
                if order_rows:
                    self.conn.executemany(
                        "INSERT OR REPLACE INTO chunk_order VALUES (?, ?, ?, ?, ?, ?)", order_rows
                    )

    def get_content(self, chunk_id: str) -> Optional[str]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT content FROM docstore WHERE chunk_id = ?", (chunk_id,))
            row = cur.fetchone()
        return row[0] if row else None

    def get_neighbors(self, source: str, seq: int, window: int = 1) -> List[Tuple[int, str, str, str]]:
        """Вернуть соседние чанки документа: (seq, chunk_id, content, chunk_type).

        Только текстовые соседи (таблицы/формулы/рисунки не «склеиваются»), в
        диапазоне seq ± window, исключая сам чанк, упорядоченные по seq.
        """
        if not source or seq is None or window <= 0:
            return []
        low, high = int(seq) - int(window), int(seq) + int(window)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT o.seq, o.chunk_id, d.content, o.chunk_type FROM chunk_order o "
                "JOIN docstore d ON d.chunk_id = o.chunk_id "
                "WHERE o.source = ? AND o.seq BETWEEN ? AND ? AND o.seq != ? "
                "AND o.chunk_type = 'text' ORDER BY o.seq",
                (source, low, high, int(seq)),
            )
            return [(int(r[0]), r[1], r[2], r[3]) for r in cur.fetchall()]

    def clear(self) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM docstore")
            self.conn.execute("DELETE FROM chunk_order")

    def count(self) -> int:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM docstore")
            row = cur.fetchone()
        return int(row[0]) if row else 0


class BM25FTSIndex:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (INDEX_DIR / "bm25_fts.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn:
            cur = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
            exists = bool(cur.fetchone())
            if exists:
                info = self.conn.execute("PRAGMA table_info(chunks_fts)").fetchall()
                columns = {row[1] for row in info}
                if "raw_content" not in columns or "fts_content" not in columns:
                    logger.warning("Обнаружена старая BM25/FTS схема; пересоздаю chunks_fts.")
                    self.conn.execute("DROP TABLE IF EXISTS chunks_fts")
                    exists = False
            if not exists:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                    "chunk_id UNINDEXED, fts_content, raw_content UNINDEXED, chunk_type UNINDEXED, "
                    "source UNINDEXED, page UNINDEXED, heading UNINDEXED, "
                    "model UNINDEXED, metadata UNINDEXED, "
                    "tokenize='unicode61 remove_diacritics 1')"
                )

    def add_chunks(self, chunks) -> None:
        rows = []
        for c in chunks:
            raw = _chunk_display_content(c)
            tokenized = tokenize_ru(raw)
            if tokenized:
                metadata = dict(c.metadata or {})
                metadata.setdefault("chunk_id", c.chunk_id)
                metadata.setdefault("chunk_type", c.chunk_type)
                rows.append(
                    (
                        c.chunk_id,
                        tokenized,
                        raw,
                        c.chunk_type,
                        metadata.get("source", ""),
                        str(metadata.get("page", 0)),
                        metadata.get("heading", ""),
                        metadata.get("model", "unknown"),
                        json.dumps(metadata, ensure_ascii=False),
                    )
                )
        if rows:
            with self.lock, self.conn:
                self.conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(r[0],) for r in rows])
                self.conn.executemany("INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        fts_query = prepare_fts_query(query)
        if not fts_query:
            return []
        try:
            with self.lock:
                cur = self.conn.cursor()
                cur.execute(
                    "SELECT chunk_id, raw_content, chunk_type, source, page, heading, model, metadata, "
                    "bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_query, k),
                )
                rows = cur.fetchall()
            results = []
            for row in rows:
                metadata = json.loads(row[7]) if row[7] else {}
                metadata.setdefault("chunk_id", row[0])
                metadata.setdefault("content", row[1])
                metadata.setdefault("chunk_type", row[2])
                metadata.setdefault("source", row[3])
                metadata.setdefault("page", int(row[4]) if row[4] else 0)
                metadata.setdefault("heading", row[5])
                metadata.setdefault("model", row[6])
                score = -float(row[8]) if row[8] is not None else 0.0
                results.append((metadata, score))
            return results
        except Exception as exc:
            logger.debug("BM25 search failed: %s", exc)
            return []

    def count(self) -> int:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM chunks_fts")
            row = cur.fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM chunks_fts")


class IndexBuilder:
    def __init__(self, index_dir: Optional[Path] = None, docstore_path: Optional[Path] = None, load_existing: bool = False):
        self.output_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docstore_path = docstore_path or (self.output_dir.parent / "docstore" / "docstore.db")
        self.text_vectorstore = None
        self.summary_vectorstore = None
        self.bm25 = BM25FTSIndex(self.output_dir / "bm25_fts.db")
        self.docstore = DocStore(self.docstore_path)
        self._text_docs = []
        self._summary_docs = []
        if load_existing:
            self.load()

    def add_chunks(self, chunks) -> None:
        if not chunks:
            return
        # Дедупликация по содержимому (Проблема 8): убираем повторы, оставляя
        # чанк с богатейшими метаданными, но сохраняя ссылки на все источники.
        # Импорт отложенный, чтобы избежать циклической зависимости chunking→indexing.
        try:
            from .chunking import deduplicate_chunks
            deduped = deduplicate_chunks(chunks)
            if len(deduped) < len(chunks):
                logger.info("Дедупликация: %s → %s чанков", len(chunks), len(deduped))
            chunks = deduped
        except Exception as exc:
            logger.debug("Дедупликация недоступна, чанки идут как есть: %s", exc)

        self.docstore.add_many(chunks)
        self.bm25.add_chunks(chunks)
        for chunk in chunks:
            lc_doc = chunk.to_langchain_doc()
            lc_doc.page_content = normalize_ocr_text(lc_doc.page_content)
            if not lc_doc.page_content:
                continue
            if chunk.chunk_type == "text":
                self._text_docs.append(lc_doc)
            else:
                self._summary_docs.append(lc_doc)
        if len(self._text_docs) + len(self._summary_docs) >= INDEXING_FLUSH_THRESHOLD:
            self._flush_vectorstores()

    def _flush_vectorstores(self) -> None:
        if not self._text_docs and not self._summary_docs:
            return
        embeddings = get_global_embeddings()
        from langchain_community.vectorstores import FAISS

        if self._text_docs:
            if self.text_vectorstore is None and (self.output_dir / "text_index.faiss").exists():
                self.text_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings, index_name="text_index", allow_dangerous_deserialization=True
                )
            if self.text_vectorstore is None:
                self.text_vectorstore = FAISS.from_documents(self._text_docs, embeddings)
            else:
                self.text_vectorstore.add_documents(self._text_docs)
            self.text_vectorstore.save_local(str(self.output_dir), index_name="text_index")
            self._text_docs = []

        if self._summary_docs:
            if self.summary_vectorstore is None and (self.output_dir / "summary_index.faiss").exists():
                self.summary_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings, index_name="summary_index", allow_dangerous_deserialization=True
                )
            if self.summary_vectorstore is None:
                self.summary_vectorstore = FAISS.from_documents(self._summary_docs, embeddings)
            else:
                self.summary_vectorstore.add_documents(self._summary_docs)
            self.summary_vectorstore.save_local(str(self.output_dir), index_name="summary_index")
            self._summary_docs = []

    def save(self) -> None:
        self._flush_vectorstores()

    def load(self) -> None:
        embeddings = get_global_embeddings()
        try:
            from langchain_community.vectorstores import FAISS

            if (self.output_dir / "text_index.faiss").exists():
                self.text_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings, index_name="text_index", allow_dangerous_deserialization=True
                )
            if (self.output_dir / "summary_index.faiss").exists():
                self.summary_vectorstore = FAISS.load_local(
                    str(self.output_dir), embeddings, index_name="summary_index", allow_dangerous_deserialization=True
                )
        except Exception as exc:
            logger.warning("Не удалось загрузить FAISS-индексы: %s", exc)
