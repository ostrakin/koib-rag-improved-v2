# -*- coding: utf-8 -*-
"""Частые вопросы (FAQ) для VK Long Poll бота КОИБ.

Каждый пункт — это кнопка с коротким ярлыком (label ≤ 40 символов, лимит VK)
и полный текст вопроса, который уходит в RAG-пайплайн ровно так же, как если бы
пользователь набрал его руками. Никакой отдельной «базы ответов» нет — ответы
всегда генерируются по реальному индексу, поэтому FAQ не устаревает.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FAQItem:
    id: str          # стабильный идентификатор (уходит в payload кнопки)
    short: str       # ярлык кнопки, ≤ 40 символов
    question: str    # полный вопрос, который прогоняется через RAG


# Порядок = порядок кнопок сверху вниз. Держите ≤ 8 пунктов: так меню остаётся
# читаемым, а клавиатура — в пределах лимитов VK.
FAQ_ITEMS: List[FAQItem] = [
    FAQItem(
        id="power_on",
        short="🔌 КОИБ не включается",
        question="Что делать, если КОИБ не включается при подготовке к работе?",
    ),
    FAQItem(
        id="reject_ballot",
        short="📄 Не принимает бюллетень",
        question="КОИБ не принимает бюллетень и возвращает его обратно. Что делать?",
    ),
    FAQItem(
        id="jam",
        short="⚙️ Замятие бюллетеня",
        question="Как устранить замятие бюллетеня в приёмном устройстве КОИБ?",
    ),
    FAQItem(
        id="prepare",
        short="🟢 Подготовка к работе",
        question="Как правильно подготовить КОИБ к работе перед началом голосования?",
    ),
    FAQItem(
        id="print_protocol",
        short="🖨 Печать протокола",
        question="Как распечатать протокол об итогах голосования на КОИБ?",
    ),
    FAQItem(
        id="seal_drive",
        short="🔒 Опечатывание накопителя",
        question="Как опечатать накопитель (флеш-карту) КОИБ после голосования?",
    ),
    FAQItem(
        id="error_codes",
        short="❗ Ошибка на экране",
        question="На экране КОИБ появилось сообщение об ошибке. Как понять причину и что делать?",
    ),
    FAQItem(
        id="end_voting",
        short="🏁 Завершение голосования",
        question="Какие действия нужно выполнить на КОИБ при завершении голосования?",
    ),
]

_FAQ_BY_ID: Dict[str, FAQItem] = {item.id: item for item in FAQ_ITEMS}


def get_faq(faq_id: str) -> Optional[FAQItem]:
    """Вернуть пункт FAQ по id или None."""
    return _FAQ_BY_ID.get((faq_id or "").strip())
