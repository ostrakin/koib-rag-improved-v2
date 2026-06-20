# -*- coding: utf-8 -*-
"""VK-сервис с меню-навигацией поверх существующего VKBotService.

Что добавляет к базовому сервису (src/vk_bot.py), НЕ дублируя его логику:
  • главное меню (постоянная нижняя клавиатура);
  • раздел «Выбрать модель КОИБ» с кнопкой «Назад»;
  • режим «Общие вопросы» (любая модель, без жёсткого фильтра);
  • раздел «Частые вопросы» — кнопки прогоняются через тот же RAG-пайплайн.

Реализовано через переопределение двух точек расширения базового класса:
  • _handle_model_payload — перехватывает payload-кнопки меню (nav/help/faq),
    а set_model отдаёт родителю (там уже корректно обрабатывается отложенный
    вопрос и подтверждение);
  • _handle_command — добавляет текстовые команды меню (/menu, «назад»,
    «частые вопросы» и т.п.), при /start показывает приветствие с клавиатурой.

Вся «тяжёлая» логика (rate-limit, дедупликация, выбор модели, RAG, отправка,
загрузка рисунков) наследуется без изменений.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import aiohttp

from src.vk_bot import VKBotService, VKIncomingMessage, _MODEL_ANY
from src.vk_keyboards import (
    FAQ_MENU_TEXT,
    MAIN_MENU_TEXT,
    MODELS_MENU_TEXT,
    WELCOME_TEXT,
    faq_menu_keyboard,
    main_menu_keyboard,
    models_menu_keyboard,
)
from src.vk_faq import get_faq
from src.utils import KNOWN_MODELS, model_label

logger = logging.getLogger("koib.vk_menu")


class VKMenuBotService(VKBotService):
    """VKBotService + навигация по меню. Совместим с Callback API и Long Poll."""

    # ─────────────────────────── payload-кнопки ───────────────────────────
    async def _handle_model_payload(self, message: VKIncomingMessage,
                                    session: aiohttp.ClientSession) -> bool:
        """Перехват кнопок меню. True — если payload распознан и обработан."""
        try:
            data = json.loads(message.payload)
        except Exception:
            return False
        if not isinstance(data, dict):
            return False

        cmd = str(data.get("cmd") or "").strip()

        if cmd == "nav":
            await self._show_menu(str(data.get("to") or "main"), message, session)
            return True

        if cmd == "help":
            await self.send_message(message.peer_id, self._help_text(), session,
                                    keyboard=main_menu_keyboard())
            return True

        if cmd == "faq":
            await self._handle_faq(str(data.get("id") or ""), message, session)
            return True

        if cmd == "set_model":
            return await self._apply_model(message, str(data.get("model") or ""), session)

        return False

    async def _apply_model(self, message: VKIncomingMessage, model: str,
                           session: aiohttp.ClientSession) -> bool:
        """Сохранить выбор модели, подтвердить и вернуть пользователя в меню.

        Зеркалит поведение базового VKBotService (включая ответ на отложенный
        вопрос), но добавляет дружелюбное подтверждение с главным меню.
        """
        model = model.strip()
        if model not in KNOWN_MODELS and model != _MODEL_ANY:
            return False

        user_key = str(message.user_id)
        self.prefs.set_model(user_key, model)
        label = "любая модель (общий режим)" if model == _MODEL_ANY else model_label(model)
        await self.send_message(
            message.peer_id,
            f"Принято ✅ Текущая модель: {label}.\n"
            "Задайте вопрос текстом или откройте «Частые вопросы». "
            "Сменить модель можно в любой момент.",
            session, keyboard=main_menu_keyboard(),
        )

        # Если выбор модели был ответом на отложенный вопрос — отвечаем сразу.
        pending = self.prefs.pop_pending(user_key)
        if pending:
            model_filter = "" if model == _MODEL_ANY else model
            await self._answer_query(message, pending, model_filter, session)
        return True

    # ─────────────────────────── текстовые команды ───────────────────────────
    async def _handle_command(self, query: str, message: VKIncomingMessage,
                              session: aiohttp.ClientSession) -> Optional[str]:
        command = re.sub(r"\s+", " ", (query or "").lower().strip())

        if command in {"/start", "start", "начать", "/menu", "menu", "меню", "главное меню"}:
            await self.send_message(message.peer_id, WELCOME_TEXT, session,
                                    keyboard=main_menu_keyboard())
            return ""

        if command in {"назад", "⬅️ назад", "back", "/back"}:
            await self.send_message(message.peer_id, MAIN_MENU_TEXT, session,
                                    keyboard=main_menu_keyboard())
            return ""

        if command in {"частые вопросы", "❓ частые вопросы", "faq", "/faq", "вопросы"}:
            await self._show_menu("faq", message, session)
            return ""

        if command in {"🗳 выбрать модель коиб", "выбрать модель коиб", "выбрать модель"}:
            await self._show_menu("models", message, session)
            return ""

        if command in {"💬 общие вопросы", "общие вопросы", "общий режим"}:
            self.prefs.set_model(str(message.user_id), _MODEL_ANY)
            await self.send_message(
                message.peer_id,
                "Режим общих вопросов включён: отвечаю по всем моделям КОИБ. "
                "Задайте вопрос текстом или откройте «Частые вопросы».",
                session, keyboard=main_menu_keyboard(),
            )
            return ""

        # Остальное (/help, /model, /reset, /health) — штатно у родителя.
        return await super()._handle_command(query, message, session)

    # ─────────────────────────── вспомогательное ───────────────────────────
    async def _show_menu(self, to: str, message: VKIncomingMessage,
                         session: aiohttp.ClientSession) -> None:
        if to == "models":
            await self.send_message(message.peer_id, MODELS_MENU_TEXT, session,
                                    keyboard=models_menu_keyboard())
        elif to == "faq":
            await self.send_message(message.peer_id, FAQ_MENU_TEXT, session,
                                    keyboard=faq_menu_keyboard())
        else:
            await self.send_message(message.peer_id, MAIN_MENU_TEXT, session,
                                    keyboard=main_menu_keyboard())

    async def _handle_faq(self, faq_id: str, message: VKIncomingMessage,
                          session: aiohttp.ClientSession) -> None:
        item = get_faq(faq_id)
        if item is None:
            await self.send_message(
                message.peer_id,
                "Не нашёл этот вопрос. Откройте «Частые вопросы» заново.",
                session, keyboard=faq_menu_keyboard(),
            )
            return
        # Отвечаем с учётом уже выбранной модели; если не выбрана — «любая».
        stored = self.prefs.get_model(str(message.user_id))
        model_filter = stored if stored in KNOWN_MODELS else ""
        await self._answer_query(message, item.question, model_filter, session)
