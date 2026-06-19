# -*- coding: utf-8 -*-
"""FastAPI route для VK Callback API.

Route отвечает VK строкой "ok" максимально быстро. Вся тяжелая работа
(RAG, GigaChat, отправка сообщений) уходит в BackgroundTasks.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Request

from api.deps import get_vk_service

logger = logging.getLogger("koib.api.vk")
router = APIRouter()


@router.post("/vk_callback")
async def vk_webhook(request: Request, background_tasks: BackgroundTasks) -> str:
    service = get_vk_service()
    try:
        raw_data = await request.json()
    except Exception:
        logger.warning("VK callback без валидного JSON")
        return "ok"

    if not service.validate_callback(raw_data):
        return "ok"

    if service.is_confirmation(raw_data):
        return service.confirmation_code()

    message = service.parse_message(raw_data)
    if message is None:
        return "ok"

    should_process, reason = service.should_process(message)
    if not should_process:
        logger.debug("VK callback пропущен: %s", reason)
        return "ok"

    session = request.app.state.vk_session
    background_tasks.add_task(service.process_message, message, session)
    return "ok"
