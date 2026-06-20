# -*- coding: utf-8 -*-
"""Клавиатуры и тексты меню для VK Long Poll бота КОИБ.

Навигация сделана БЕЗ серверного состояния: каждая кнопка несёт в payload
описание действия (cmd/nav/faq/set_model). Кнопка «Назад» — это просто переход
в нужное меню. Так нет рассинхронизации «в каком меню сейчас пользователь».

Используются ОБЫЧНЫЕ (bottom) клавиатуры с текстовыми кнопками (type=text):
они остаются внизу чата, и при нажатии VK присылает message_new с payload —
то есть всё проходит через штатный разбор сообщения, без message_event.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.vk_faq import FAQ_ITEMS

# Спецзначение «любая модель / общий режим» — совпадает с _MODEL_ANY в vk_bot.py
MODEL_ANY = "any"


def _btn(label: str, payload: Dict[str, Any], color: str = "secondary") -> Dict[str, Any]:
    """Текстовая кнопка VK. payload сериализуется в строку (требование API)."""
    return {
        "action": {
            "type": "text",
            "label": label,
            "payload": json.dumps(payload, ensure_ascii=False),
        },
        "color": color,
    }


def main_menu_keyboard() -> str:
    """Главное меню (постоянная нижняя клавиатура)."""
    kb = {
        "one_time": False,
        "inline": False,
        "buttons": [
            [_btn("🗳 Выбрать модель КОИБ", {"cmd": "nav", "to": "models"}, "primary")],
            [_btn("💬 Общие вопросы", {"cmd": "set_model", "model": MODEL_ANY}, "secondary")],
            [_btn("❓ Частые вопросы", {"cmd": "nav", "to": "faq"}, "secondary")],
            [_btn("ℹ️ Помощь", {"cmd": "help"}, "secondary")],
        ],
    }
    return json.dumps(kb, ensure_ascii=False)


def models_menu_keyboard() -> str:
    """Меню выбора модели КОИБ с кнопкой «Назад»."""
    kb = {
        "one_time": False,
        "inline": False,
        "buttons": [
            [_btn("КОИБ-2010", {"cmd": "set_model", "model": "koib2010"}, "primary"),
             _btn("КОИБ-2017А", {"cmd": "set_model", "model": "koib2017a"}, "primary")],
            [_btn("КОИБ-2017Б", {"cmd": "set_model", "model": "koib2017b"}, "primary"),
             _btn("Не знаю / любая", {"cmd": "set_model", "model": MODEL_ANY}, "secondary")],
            [_btn("⬅️ Назад", {"cmd": "nav", "to": "main"}, "secondary")],
        ],
    }
    return json.dumps(kb, ensure_ascii=False)


def faq_menu_keyboard() -> str:
    """Меню частых вопросов: по кнопке на вопрос + «Назад»."""
    rows: List[List[Dict[str, Any]]] = [
        [_btn(item.short, {"cmd": "faq", "id": item.id}, "secondary")]
        for item in FAQ_ITEMS
    ]
    rows.append([_btn("⬅️ Назад", {"cmd": "nav", "to": "main"}, "secondary")])
    kb = {"one_time": False, "inline": False, "buttons": rows}
    return json.dumps(kb, ensure_ascii=False)


# ─────────────────────────── Тексты меню ───────────────────────────
WELCOME_TEXT = (
    "Здравствуйте! 👋 Я — помощник по технической эксплуатации КОИБ.\n\n"
    "Помогу разобраться с подготовкой, работой и нештатными ситуациями. "
    "Можно просто написать вопрос своими словами — например «КОИБ не принимает бюллетень».\n\n"
    "Для точных ответов выберите модель: интерфейсы различаются "
    "(у КОИБ-2010 — физические кнопки, у КОИБ-2017 — сенсорный экран).\n\n"
    "Выберите раздел в меню ниже 👇"
)

MAIN_MENU_TEXT = "Главное меню. Выберите раздел или просто задайте вопрос текстом 👇"

MODELS_MENU_TEXT = (
    "Выберите модель КОИБ. Ответы будут фильтроваться по ней, чтобы инструкции "
    "2010 и 2017 не смешивались. Если не уверены — нажмите «Не знаю / любая»."
)

FAQ_MENU_TEXT = (
    "Частые вопросы. Нажмите на нужный — я отвечу по документации с учётом "
    "выбранной модели. Или вернитесь «Назад» и задайте свой вопрос."
)
