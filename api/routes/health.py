# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from config import APP_NAME, APP_VERSION, INDEX_DIR

router = APIRouter()
_start_time = time.time()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "version": APP_VERSION,
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    text_index = Path(INDEX_DIR) / "text_index.faiss"
    summary_index = Path(INDEX_DIR) / "summary_index.faiss"
    bm25_index = Path(INDEX_DIR) / "bm25_fts.db"
    ready = bm25_index.exists() and (text_index.exists() or summary_index.exists())
    return {
        "status": "ready" if ready else "not_ready",
        "version": APP_VERSION,
        "indexes": {
            "text_index": text_index.exists(),
            "summary_index": summary_index.exists(),
            "bm25_index": bm25_index.exists(),
        },
    }


@router.get("/")
async def root() -> Dict[str, str]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": "RAG-система вопросов и ответов по технической документации КОИБ",
        "docs": "/docs",
    }
