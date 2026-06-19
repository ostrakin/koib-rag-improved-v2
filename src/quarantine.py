# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Модуль карантина
================================
Фильтрация и изоляция сомнительных чанков, которые содержат
недостоверную, противоречивую или устаревшую информацию.

Карантин работает на уровне метаданных чанка: каждый чанк может
получить статус 'quarantined', после чего он исключается из
результатов поиска до ручной проверки.
"""
import json
import logging
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from config import METADATA_DIR

logger = logging.getLogger("koib.quarantine")


# ═══════════════════════════════════════════════════════════════
# SQLite-хранилище карантина
# ═══════════════════════════════════════════════════════════════
class QuarantineStore:
    """
    SQLite-хранилище для карантинных чанков.

    Хранит информацию о заблокированных чанках:
    причину карантина, дату, источник.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or METADATA_DIR / "quarantine.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Создать таблицы карантина."""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS quarantine (
                    chunk_id TEXT PRIMARY KEY,
                    reason TEXT,
                    source TEXT,
                    quarantined_at TEXT,
                    reviewed INTEGER DEFAULT 0
                )
            ''')

    def add(
        self,
        chunk_id: str,
        reason: str,
        source: str = "",
    ) -> None:
        """Добавить чанк в карантин."""
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO quarantine (chunk_id, reason, source, quarantined_at) '
                'VALUES (?, ?, ?, ?)',
                (chunk_id, reason, source, datetime.now().isoformat()),
            )

    def is_quarantined(self, chunk_id: str) -> bool:
        """Проверить, находится ли чанк в карантине."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT COUNT(*) FROM quarantine WHERE chunk_id = ? AND reviewed = 0',
            (chunk_id,),
        )
        return cur.fetchone()[0] > 0

    def remove(self, chunk_id: str) -> None:
        """Удалить чанк из карантина (одобрить)."""
        with self.conn:
            self.conn.execute(
                'UPDATE quarantine SET reviewed = 1 WHERE chunk_id = ?',
                (chunk_id,),
            )

    def get_all(self, reviewed: bool = False) -> List[Dict[str, Any]]:
        """Получить все карантинные записи."""
        cur = self.conn.cursor()
        cur.execute(
            'SELECT chunk_id, reason, source, quarantined_at FROM quarantine '
            'WHERE reviewed = ?',
            (1 if reviewed else 0,),
        )
        return [
            {
                "chunk_id": row[0],
                "reason": row[1],
                "source": row[2],
                "quarantined_at": row[3],
            }
            for row in cur.fetchall()
        ]


# Глобальный экземпляр хранилища
_quarantine_store: Optional[QuarantineStore] = None


def _get_quarantine_store() -> QuarantineStore:
    """Получить глобальный экземпляр QuarantineStore."""
    global _quarantine_store
    if _quarantine_store is None:
        _quarantine_store = QuarantineStore()
    return _quarantine_store


# ═══════════════════════════════════════════════════════════════
# Фильтрация результатов
# ═══════════════════════════════════════════════════════════════
def filter_quarantined_chunks(results: list) -> list:
    """
    Отфильтровать карантинные чанки из результатов поиска.

    Проверяет каждый чанк на наличие в карантине и исключает
    заблокированные из выдачи.

    Args:
        results: Список RetrievalResult

    Returns:
        Отфильтрованный список без карантинных чанков
    """
    store = _get_quarantine_store()
    filtered = []
    for r in results:
        chunk_id = r.chunk_id if hasattr(r, 'chunk_id') else ""
        if chunk_id and store.is_quarantined(chunk_id):
            logger.debug(f"Чанк {chunk_id} в карантине, пропущен")
            continue
        filtered.append(r)
    return filtered


def quarantine_chunk(chunk_id: str, reason: str, source: str = "") -> None:
    """
    Отправить чанк в карантин.

    Args:
        chunk_id: Идентификатор чанка
        reason:   Причина карантина
        source:   Источник (имя файла)
    """
    store = _get_quarantine_store()
    store.add(chunk_id, reason, source)
    logger.info(f"Чанк {chunk_id} отправлен в карантин: {reason}")


def approve_chunk(chunk_id: str) -> None:
    """
    Одобрить карантинный чанк (убрать из карантина).

    Args:
        chunk_id: Идентификатор чанка
    """
    store = _get_quarantine_store()
    store.remove(chunk_id)
    logger.info(f"Чанк {chunk_id} одобрен и убран из карантина")


def list_quarantined(reviewed: bool = False) -> List[Dict[str, Any]]:
    """
    Получить список карантинных чанков.

    Args:
        reviewed: Включать ли уже проверенные чанки

    Returns:
        Список словарей с информацией о карантинных чанках
    """
    store = _get_quarantine_store()
    return store.get_all(reviewed=reviewed)
