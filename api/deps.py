# -*- coding: utf-8 -*-
"""Общие зависимости API: единственный экземпляр RAG-пайплайна и VK-сервиса.

Раньше функция ``_get_pipeline`` копировалась в ``routes/query.py`` и
``routes/vk_callback.py`` — два разных глобальных ``_pipeline`` означали, что
тяжёлая модель эмбеддингов и FAISS могли загрузиться в память ДВАЖДЫ.
Теперь источник истины один, и оба роутера берут пайплайн отсюда.
"""
from __future__ import annotations

from typing import Optional

from src.rag_pipeline import RAGPipeline
from src.vk_bot import VKBotService

_pipeline: Optional[RAGPipeline] = None
_vk_service: Optional[VKBotService] = None


def get_pipeline() -> RAGPipeline:
    """Ленивая инициализация единственного RAGPipeline на процесс."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_vk_service() -> VKBotService:
    """Ленивая инициализация VK-сервиса поверх того же пайплайна."""
    global _vk_service
    if _vk_service is None:
        _vk_service = VKBotService(pipeline_factory=get_pipeline)
    return _vk_service


async def close_pipeline() -> None:
    """Аккуратно закрывает VK-сервис и пайплайн при остановке приложения."""
    global _pipeline, _vk_service
    if _vk_service is not None:
        _vk_service.close()
        _vk_service = None
    if _pipeline is not None:
        await _pipeline.close()
        _pipeline = None
