# -*- coding: utf-8 -*-
"""Регламентные правила для ответов оператору КОИБ.

Главная задача модуля — гарантировать, что при ответах по нештатным
ситуациям оператор получает не только технические шаги, но и обязательное
процессуальное напоминание: председатель УИК + горячая линия технической
поддержки.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Pattern


PROCEDURAL_REMINDER = (
    "Важно: при устранении нештатной ситуации оператор обязан "
    "проинформировать председателя участковой комиссии и сообщить об инциденте "
    "на горячую линию технической поддержки в порядке, предусмотренном регламентом ЦИК."
)

PROCEDURAL_REMINDER_TITLE = "Регламентное уведомление"


@dataclass(frozen=True)
class IncidentPattern:
    code: str
    description: str
    patterns: List[Pattern[str]]


# Нештатные ситуации и технические инциденты, по которым бот должен отвечать,
# но обязательно добавлять процессуальное уведомление.
INCIDENT_PATTERNS: List[IncidentPattern] = [
    IncidentPattern(
        code="general_failure",
        description="общий технический сбой или нештатная ситуация",
        patterns=[
            re.compile(r"\bнештатн\w*\b", re.IGNORECASE),
            re.compile(r"\bинцидент\w*\b", re.IGNORECASE),
            re.compile(r"\bтехническ\w*\s+сбо[йя]\b", re.IGNORECASE),
            re.compile(r"\bсбо[йя]\b", re.IGNORECASE),
            re.compile(r"\bотказ\w*\b", re.IGNORECASE),
            re.compile(r"\bнеисправн\w*\b", re.IGNORECASE),
            re.compile(r"\bавари[яий]\w*\b", re.IGNORECASE),
            re.compile(r"\bошибк[аиу]\b", re.IGNORECASE),
        ],
    ),
    IncidentPattern(
        code="device_not_working",
        description="оборудование не работает или не включается",
        patterns=[
            re.compile(r"\bне\s+работа\w*\b", re.IGNORECASE),
            re.compile(r"\bне\s+включа\w*\b", re.IGNORECASE),
            re.compile(r"\bне\s+запуска\w*\b", re.IGNORECASE),
            re.compile(r"\bне\s+печата\w*\b", re.IGNORECASE),
            re.compile(r"\bне\s+сканиру\w*\b", re.IGNORECASE),
            re.compile(r"\bне\s+считыва\w*\b", re.IGNORECASE),
            re.compile(r"\bне\s+принима\w*\s+бюллетен\w*\b", re.IGNORECASE),
        ],
    ),
    IncidentPattern(
        code="jam_or_stuck",
        description="замятие, застревание или блокировка бюллетеня/механизма",
        patterns=[
            re.compile(r"\bзастр\w*\b", re.IGNORECASE),
            re.compile(r"\bзамяти\w*\b", re.IGNORECASE),
            re.compile(r"\bзажевал\w*\b", re.IGNORECASE),
            re.compile(r"\bзавис\w*\b", re.IGNORECASE),
            re.compile(r"\bблокиров\w*\b", re.IGNORECASE),
        ],
    ),
    IncidentPattern(
        code="power_or_connection",
        description="питание, ИБП, кабель или соединение",
        patterns=[
            re.compile(r"\bпитани\w*\b", re.IGNORECASE),
            re.compile(r"\bибп\b", re.IGNORECASE),
            re.compile(r"\bаккумулятор\w*\b", re.IGNORECASE),
            re.compile(r"\bкабел\w*\b", re.IGNORECASE),
            re.compile(r"\bсоединени\w*\b", re.IGNORECASE),
            re.compile(r"\bиндикатор\w*\b", re.IGNORECASE),
        ],
    ),
]


def detect_incident(text: str) -> bool:
    """Вернуть True, если текст похож на запрос/ответ о нештатной ситуации."""
    if not text:
        return False
    for group in INCIDENT_PATTERNS:
        if any(pattern.search(text) for pattern in group.patterns):
            return True
    return False


def incident_reasons(text: str) -> List[str]:
    """Список причин, по которым запрос классифицирован как технический инцидент."""
    if not text:
        return []
    reasons: List[str] = []
    for group in INCIDENT_PATTERNS:
        if any(pattern.search(text) for pattern in group.patterns):
            reasons.append(group.description)
    return reasons


def contains_procedural_reminder(text: str) -> bool:
    """Проверить, есть ли в ответе напоминание о председателе УИК и горячей линии."""
    value = (text or "").lower()
    return (
        ("председател" in value and ("уик" in value or "участков" in value or "комисс" in value))
        and ("горяч" in value and "лини" in value)
        and ("цик" in value or "регламент" in value)
    )


def ensure_procedural_reminder(answer: str, query: str = "", force: bool = False) -> str:
    """Добавить обязательное уведомление, если вопрос/ответ связан с инцидентом.

    Args:
        answer: готовый ответ модели или пайплайна.
        query: исходный вопрос оператора.
        force: добавить уведомление независимо от классификации.
    """
    answer = (answer or "").strip()
    must_add = force or detect_incident(query) or detect_incident(answer)
    if not must_add or contains_procedural_reminder(answer):
        return answer
    if answer:
        return f"{answer}\n\n**{PROCEDURAL_REMINDER_TITLE}:** {PROCEDURAL_REMINDER}"
    return f"**{PROCEDURAL_REMINDER_TITLE}:** {PROCEDURAL_REMINDER}"


def build_incident_instruction(query: str) -> str:
    """Короткая инструкция для prompt builder, если вопрос про инцидент."""
    if not detect_incident(query):
        return ""
    reasons = "; ".join(incident_reasons(query)) or "нештатная ситуация"
    return (
        "Дополнительное обязательное требование: вопрос классифицирован как "
        f"технический инцидент ({reasons}). Если ты даёшь шаги устранения, "
        "в конце ответа отдельным блоком 'Регламентное уведомление' дословно напомни: "
        f"{PROCEDURAL_REMINDER}"
    )


def as_markdown_list(items: Iterable[str]) -> str:
    """Утилита для человекочитаемой выдачи причин в диагностике."""
    return "\n".join(f"- {item}" for item in items)
