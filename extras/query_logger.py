# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — JSONL логирование запросов
★ ВОЗВРАЩЕНО из V-4.3: структурированный JSONL с ротацией по дате.
Критично для диплома — даёт аналитику по usage.
"""
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("koib.query_logger")


class QueryLogger:
    def __init__(self, log_dir: Optional[Path] = None):
        from config import LOGS_DIR
        self.log_dir = log_dir or LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"queries_{self._current_date}.jsonl"

    def _check_date_rotation(self) -> None:
        current_date = datetime.now().strftime("%Y%m%d")
        if current_date != self._current_date:
            self._current_date = current_date
            self.log_file = self.log_dir / f"queries_{self._current_date}.jsonl"

    def _compute_query_hash(self, query: str) -> str:
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]

    def log(
        self,
        query: str,
        answer: str,
        model_type: str = "",
        sources: List[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        status: str = "approved",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            self._check_date_rotation()
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query_hash": self._compute_query_hash(query),
                "query": query,
                "model_type": model_type,
                "answer": answer,
                "sources": sources or [],
                "validation": validation_result or {},
                "status": status,
            }
            if extra_metadata:
                entry["metadata"] = extra_metadata
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            return True
        except Exception as exc:
            logger.error(f"Ошибка записи в лог: {exc}")
            return False

    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        entries = []
        try:
            if not self.log_file.exists():
                return entries
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
            entries.reverse()
        except Exception as exc:
            logger.warning(f"Ошибка чтения лога: {exc}")
        return entries


_global_logger: Optional[QueryLogger] = None


def get_query_logger() -> QueryLogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = QueryLogger()
    return _global_logger
