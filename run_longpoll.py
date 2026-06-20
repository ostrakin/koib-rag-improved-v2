# -*- coding: utf-8 -*-
"""KOIB RAG — запуск VK-бота на Bots Long Poll API.

Использует ГОТОВЫЙ индекс из ./output (переносится с машины-индексатора через
tools/index_transfer.py) и тот же RAGPipeline, что и HTTP-сервер. Индексации
здесь нет — только обслуживание запросов из VK.

Запуск:
    python run_longpoll.py

Переменные окружения берутся из .env (см. .env.longpoll.example):
    VK_ACCESS_TOKEN, VK_GROUP_ID   — токен сообщества (scope: messages+manage) и id группы
    GIGACHAT_CREDENTIALS           — Base64-ключ авторизации GigaChat
    KOIB_VK_CUSTOM_PROMPT=true      — (опц.) включить усиленный промпт из src/vk_prompt.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import HF_OFFLINE_MODE, ensure_dirs  # noqa: E402

if HF_OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib.longpoll.main")


def _maybe_apply_custom_prompt() -> None:
    """По флагу KOIB_VK_CUSTOM_PROMPT подменить системный промпт GigaChat."""
    if os.getenv("KOIB_VK_CUSTOM_PROMPT", "false").lower() != "true":
        return
    try:
        import src.generation as generation
        from src.vk_prompt import VK_BOT_SYSTEM_PROMPT

        generation.SYSTEM_PROMPT = VK_BOT_SYSTEM_PROMPT
        logger.info("Включён усиленный системный промпт VK-бота (src/vk_prompt.py).")
    except Exception as exc:
        logger.warning("Не удалось применить кастомный промпт, использую базовый: %s", exc)


def _warn_if_index_incompatible() -> None:
    """Необязательная проверка манифеста индекса при старте."""
    try:
        import json

        from config import OUTPUT_DIR
        from tools.index_transfer import MANIFEST_REL, _check_compat

        mpath = Path(OUTPUT_DIR) / MANIFEST_REL
        if mpath.exists():
            _check_compat(json.loads(mpath.read_text(encoding="utf-8")))
    except Exception:
        pass


async def _amain() -> None:
    ensure_dirs()
    _maybe_apply_custom_prompt()
    _warn_if_index_incompatible()

    # Ленивая инициализация единственного пайплайна на процесс (как в api/deps).
    from api.deps import get_pipeline
    from src.vk_longpoll import VKLongPollRunner
    from src.vk_menu_bot import VKMenuBotService

    service = VKMenuBotService(pipeline_factory=get_pipeline)
    runner = VKLongPollRunner(service)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, runner.stop)
        except NotImplementedError:  # Windows
            pass

    try:
        await runner.run()
    finally:
        service.close()
        try:
            await get_pipeline().close()
        except Exception:
            pass


def main() -> None:
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        logger.info("Прерывание с клавиатуры — выход.")


if __name__ == "__main__":
    main()
