# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI

from api.middleware.logging import LoggingMiddleware
from api.deps import close_pipeline as close_vk_pipeline
from api.routes.health import router as health_router
from api.routes.query import router as query_router
from api.routes.vk_callback import router as vk_router
from config import APP_NAME, APP_VERSION, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("koib.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    app.state.vk_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100, ttl_dns_cache=300))
    try:
        from src.retrieval import SemanticCache

        purged = SemanticCache().purge_stale(days=30)
        if purged:
            logger.info("Очищено устаревших записей semantic cache: %s", purged)
    except Exception as exc:
        logger.warning("Не удалось очистить semantic cache: %s", exc)
    logger.info("%s v%s запущен", APP_NAME, APP_VERSION)
    try:
        yield
    finally:
        await close_vk_pipeline()
        if hasattr(app.state, "vk_session") and not app.state.vk_session.closed:
            await app.state.vk_session.close()
        logger.info("%s остановлен", APP_NAME)


app = FastAPI(
    title=APP_NAME,
    description="RAG-система для операторов КОИБ и технической документации",
    version=APP_VERSION,
    lifespan=lifespan,
)
app.add_middleware(LoggingMiddleware)
app.include_router(health_router, tags=["health"])
app.include_router(query_router, tags=["query"])
app.include_router(vk_router, tags=["vk"])
